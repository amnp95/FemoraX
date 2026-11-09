#include "exec/implicit_stage.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#include "exec/matrix_free_operator.hpp"
#include "exec/runtime.hpp"
#include "exec/tcl_time.hpp"
#include "io/h5.hpp"
#include "solvers/chebyshev.hpp"
#include "solvers/jacobi.hpp"
#include "solvers/krylov.hpp"
#include "solvers/vector_ops.hpp"

namespace femorax::exec {

namespace {

using Index = std::int32_t;
using solvers::VectorOps;

// Newmark::newStep (displ == 1): U_t = U, Udot_t = Udot, Udotdot_t = Udotdot;
// Udot = a1 Udot + a2 Udotdot_t, Udotdot = a4 Udotdot + a3 Udot_t (Vector::addVector order).
void newmark_predict(const Vec& u, const Vec& v, const Vec& a, const Vec& ut, const Vec& vt, const Vec& at,
                     double a1, double a2, double a3, double a4) {
  Kokkos::parallel_for(
      "newmark_predict", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, static_cast<Index>(u.extent(0))),
      KOKKOS_LAMBDA(const Index q) {
        ut(q) = u(q);
        vt(q) = v(q);
        at(q) = a(q);
        v(q) = v(q) * a1 + at(q) * a2;
        a(q) = a(q) * a4 + vt(q) * a3;
      });
}

// Newmark::update (displ == 1): U += dU, Udot += c2 dU, Udotdot += c3 dU.
void newmark_update(const Vec& du, const Vec& u, const Vec& v, const Vec& a, double c2, double c3) {
  Kokkos::parallel_for(
      "newmark_update", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, static_cast<Index>(u.extent(0))),
      KOKKOS_LAMBDA(const Index q) {
        u(q) += du(q);
        v(q) += du(q) * c2;
        a(q) += du(q) * c3;
      });
}

// B = -f - (ma: M a, or nothing) - sum over active patterns of ag_p M r_(dof p).
void form_residual(const Vec& b, const Vec& f, const Vec& ma, bool inertia, const Kokkos::View<double*, Space>& ag,
                   const Kokkos::View<int*, Space>& dof, int npat, const Vec& mr0, const Vec& mr1, const Vec& mr2) {
  Kokkos::parallel_for(
      "implicit_residual", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, static_cast<Index>(b.extent(0))),
      KOKKOS_LAMBDA(const Index q) {
        double r = -f(q);
        if (inertia) r -= ma(q);
        for (int p = 0; p < npat; ++p) {
          const int d = dof(p);
          const double m = d == 0 ? mr0(q) : d == 1 ? mr1(q) : mr2(q);
          r -= ag(p) * m;
        }
        b(q) = r;
      });
}

}  // namespace

struct ImplicitStage::Impl {
  InternalForce& force;
  const compile::Compiled& c;
  Vec u, v_committed, a;
  ForceSettings fs;
  OperatorSettings os;
  int lanes;
  std::int64_t n;
  Vec ut, vt, at, du, du0, b, tmp, mr[3];
  std::unique_ptr<VectorOps> ops;

  Impl(InternalForce& f, const compile::Compiled& cc, Vec u_, Vec v_, Vec a_, const ForceSettings& s)
      : force(f), c(cc), u(std::move(u_)), v_committed(std::move(v_)), a(std::move(a_)), fs(s),
        os(f.resolve(OperatorSettings{})), lanes(simd_width()), n(f.num_equations()) {
    const auto un = static_cast<std::size_t>(n);
    ut = Vec("implicit_ut", un);
    vt = Vec("implicit_vt", un);
    at = Vec("implicit_at", un);
    du = Vec("implicit_du", un);
    du0 = Vec("implicit_du0", un);
    b = Vec("implicit_b", un);
    tmp = Vec("implicit_tmp", un);
    ops = std::make_unique<VectorOps>(n, lanes);
    if (os.tangent == TangentSource::Stored) force.enable_stored_tangents();
    // M r for a unit ground acceleration in each direction (the element's own mass, the
    // moving base included, plus the nodal masses): the UniformExcitation load is -ag M r.
    for (int d = 0; d < 3; ++d) {
      mr[d] = Vec("implicit_mr", un);
      force.mass_times_ground(d, os, u, mr[d]);
    }
  }
};

ImplicitStage::ImplicitStage(InternalForce& force, const compile::Compiled& c, Vec u, Vec v, Vec a,
                             const ForceSettings& force_settings)
    : impl_(std::make_unique<Impl>(force, c, std::move(u), std::move(v), std::move(a), force_settings)) {}

ImplicitStage::~ImplicitStage() = default;

ImplicitStageReport ImplicitStage::run(const io::AnalysisStep& an, double& time, GroundMotions& motions,
                                       Recorders& recorders, bool verbose) {
  Impl& s = *impl_;
  ImplicitStageReport rep;
  const bool is_static = an.type == io::AnalysisType::Static;
  InternalForce& force = s.force;
  const Vec& u = s.u;
  const Vec& v = force.v;  // the trial velocity the elements see
  const Vec& a = s.a;
  const double gamma = an.gamma, beta = an.beta;
  if (!is_static && an.newmark_form != "D")
    throw io::UnsupportedComponent("unsupported component: Newmark -form " + an.newmark_form + " in stage \"" +
                                   an.name + "\" (only the displacement form D so far)");

  const bool linear = an.algorithm == io::Algorithm::Linear;
  const bool modified = an.algorithm == io::Algorithm::ModifiedNewton;
  const bool factor_once = modified && an.modified_newton_factor_once;
  const bool initial = modified && an.modified_newton_initial;
  const bool stored = s.os.tangent == TangentSource::Stored;
  // Linear: OpenSees commits the stress of the state determination before the solve with
  // the strain after it (Linear::solveCurrentStep updates the domain without another
  // getStress); for a material with history that pair is inconsistent, so FemoraX refuses
  // it rather than reproduce it.
  if (linear && force.has_state())
    throw io::UnsupportedComponent("unsupported component: the Linear algorithm in stage \"" + an.name +
                                   "\" with a material that has history state (OpenSees would commit a stress "
                                   "inconsistent with the strain); use Newton or ModifiedNewton. Run this model "
                                   "with the OpenSees backend.");

  // Solver: automatic unless the model sets it.
  solvers::SolveOptions so;
  so.relative_tolerance = an.solver_tolerance.value_or(1e-12);
  so.max_iterations = an.solver_max_iterations.value_or(
      static_cast<int>(std::max<std::int64_t>(200, std::min<std::int64_t>(s.n, 10000))));
  // Integrator constants (Newmark::newStep, displ == 1) per step size: substeps change them.
  auto coefficients = [&](double h) -> OperatorCoefficients {
    // ModifiedNewton -initial: the initial stiffness K_0 in the stiffness term (the damping
    // keeps its own terms, as Newmark::formEleTangent with INITIAL_TANGENT).
    if (is_static) return {1.0, 0.0, 0.0, initial};
    return {1.0, gamma / (beta * h), 1.0 / (beta * h * h), initial};
  };
  MatrixFreeOperator op(force, u, coefficients(an.dt), s.os);
  // Preconditioner: Jacobi (the automatic choice today). Developer override for studies:
  // FEMORAX_PRECONDITIONER=chebyshev[:degree] (not a user setting; the model file's
  // preconditioner attribute is "auto" or "jacobi").
  solvers::Jacobi jacobi_pc;
  std::unique_ptr<solvers::Chebyshev> chebyshev;
  if (const char* e = std::getenv("FEMORAX_PRECONDITIONER"); e && std::string(e).rfind("chebyshev", 0) == 0) {
    solvers::ChebyshevOptions co;
    const std::string spec(e);
    if (const auto colon = spec.find(":"); colon != std::string::npos) co.degree = std::atoi(spec.c_str() + colon + 1);
    chebyshev = std::make_unique<solvers::Chebyshev>(co, s.lanes);
  }
  solvers::Preconditioner& jacobi = chebyshev ? static_cast<solvers::Preconditioner&>(*chebyshev)
                                              : static_cast<solvers::Preconditioner&>(jacobi_pc);
  solvers::KrylovSolver solver(so, s.lanes);
  rep.solver = op.info().symmetry == solvers::Symmetry::SymmetricPositiveDefinite ? "cg" : "gmres";
  rep.preconditioner = jacobi.name();
  bool have_tangent = false;  // the preconditioner and solver are set up for the current operator
  double tangent_h = an.dt;   // the step size of the operator's coefficients
  // ModifiedNewton -factoronce keeps the tangent across steps until an attempt fails its test
  // (ModifiedNewton::solveCurrentStep resets factorOnce) or the step size changes (Femora's
  // FemoraAnalyzeTransientStep issues the algorithm again).
  bool kept_tangent = false;
  auto update_tangent = [&](double h) {
    if (h != tangent_h) {
      op.set_coefficients(coefficients(h));
      tangent_h = h;
      have_tangent = false;
    }
    if (stored) force.store_tangents(u, s.os);
    if (!have_tangent || stored) {
      jacobi.setup(op);
      solver.setup(op, &jacobi);
    }
    have_tangent = true;
  };

  // Convergence test (OpenSees default without a test: NormUnbalance 1e-6, 25 iterations).
  const io::TestType test = an.test_type.value_or(io::TestType::NormUnbalance);
  const double tol = an.test_tol.value_or(1e-6);
  const int max_iter = an.test_max_iter.value_or(25);
  const char* test_name = test == io::TestType::NormUnbalance  ? "NormUnbalance"
                          : test == io::TestType::NormDispIncr ? "NormDispIncr"
                                                               : "EnergyIncr";

  // Velocities the elements see: the committed ones (Newmark), none (Static).
  if (is_static) Kokkos::deep_copy(v, 0.0);
  else Kokkos::deep_copy(v, s.v_committed);

  auto form_unbalance = [&] {
    force.compute(s.fs);  // f_int(u) + Rayleigh damping forces at v (v = 0 in static stages)
    if (!is_static) force.apply_operator({0.0, 0.0, 1.0}, s.os, u, a, s.tmp);
    form_residual(s.b, force.f, s.tmp, !is_static, motions.accel(), motions.dof(), motions.num_active(), s.mr[0],
                  s.mr[1], s.mr[2]);
  };
  // A failed solve fails the step (OpenSees: the algorithm returns an error, the analysis
  // reverts), so recovery can retry or substep it.
  std::string solve_failure;
  auto solve = [&]() -> bool {
    const auto r = solver.solve(s.b, s.du);  // initial guess: the previous increment
    rep.krylov_iterations += r.iterations;
    if (solver.fell_back()) rep.fell_back = true;
    // Linear takes the solve as the step's answer, so it must converge. Newton-type
    // algorithms accept an inexact increment (the outer test governs) once the solve has
    // reduced the residual by six orders.
    const bool good_enough = !linear && r.final_residual <= 1e-6 * r.rhs_norm;
    if (!r.converged && !good_enough) {
      solve_failure = "the " + solver.name() + " solve did not converge in " + std::to_string(r.iterations) +
                      " iterations (residual " + std::to_string(r.final_residual) + " of " +
                      std::to_string(r.rhs_norm) + ")";
      return false;
    }
    return true;
  };
  auto test_value = [&] {
    switch (test) {
      case io::TestType::NormUnbalance: return s.ops->norm2(s.b);
      case io::TestType::NormDispIncr: return s.ops->norm2(s.du);
      case io::TestType::EnergyIncr: break;
    }
    return 0.5 * std::abs(s.ops->dot(s.du, s.b));
  };

  // LoadControl state (one integrator per stage, as Femora writes one per analysis).
  double dlambda = an.load_increment;
  const double dmin = an.load_min_increment.value_or(an.load_increment);
  const double dmax = an.load_max_increment.value_or(an.load_increment);
  const double spec_iter = an.load_min_increment && an.load_max_increment ? an.load_num_iter : 1;
  double last_iter = spec_iter;

  // Developer switches (not user settings): every Newton iteration's test value, for
  // comparisons with OpenSees' test print flag; every retry and substep (stage, retry, limit,
  // step size as Femora's FEMORA_PROGRESS|RETRY lines print them, then the step's time).
  const bool trace_iterations = std::getenv("FEMORAX_TRACE_ITERATIONS") != nullptr;
  const bool trace_events = std::getenv("FEMORAX_TRACE_EVENTS") != nullptr;

  // Femora's retry and substep recovery (schema 0.5): a failed step goes back to the
  // committed state and is retried at the same size up to max_retries times; then, below
  // num_sublevels, it is split into num_substeps steps one level down, each with the same
  // recovery. Static stages retry only.
  const int max_retries = an.max_retries.value_or(0);
  const int num_sublevels = is_static ? 0 : an.num_sublevels.value_or(0);
  const int num_substeps = an.num_substeps.value_or(2);
  std::string failure;  // the last failed attempt
  double attempt_time = 0.0;

  // One step of size h (static: the load increment): newStep, the algorithm, then commit and
  // record. On failure everything returns to the committed state (Domain::revertToLastCommit,
  // Newmark::revertToLastStep) and the step returns false.
  auto attempt = [&](double h) -> bool {
    const double time0 = time;
    if (is_static) {
      dlambda *= spec_iter / last_iter;
      if (dlambda < dmin) dlambda = dmin;
      else if (dlambda > dmax) dlambda = dmax;
      Kokkos::deep_copy(s.ut, u);
      time += dlambda;
      last_iter = 0;
    } else {
      const double a1 = 1.0 - gamma / beta, a2 = h * (1.0 - 0.5 * gamma / beta);
      const double a3 = -1.0 / (beta * h), a4 = 1.0 - 0.5 / beta;
      newmark_predict(u, v, a, s.ut, s.vt, s.at, a1, a2, a3, a4);
      time += h;
    }
    attempt_time = time;
    Kokkos::deep_copy(s.du0, s.du);  // the solver's initial guess, restored on failure
    motions.evaluate(time);
    const OperatorCoefficients k = coefficients(h);
    auto update = [&] {
      if (is_static) VectorOps::axpy(1.0, s.du, u);
      else newmark_update(s.du, u, v, a, k.damping, k.mass);
    };
    int iterations = 0;
    bool ok = true;
    if (linear) {
      update_tangent(h);
      form_unbalance();
      ok = solve();
      if (ok) update();
      iterations = 1;
      if (!ok) failure = solve_failure;
    } else {
      form_unbalance();
      // ModifiedNewton: the tangent once per attempt, after the first unbalance; -factoronce:
      // once while it stays valid.
      if (modified && !(factor_once && kept_tangent && h == tangent_h)) update_tangent(h);
      kept_tangent = factor_once;
      int iter = 1;
      while (true) {
        if (!modified) update_tangent(h);
        if (!solve()) {
          ok = false;
          failure = solve_failure;
          break;
        }
        update();
        form_unbalance();
        const double value = test_value();
        iterations = iter;
        if (trace_iterations) std::printf("femorax-trace: %s step %lld iteration %d %.17g\n", an.name.c_str(),
                                          static_cast<long long>(rep.steps + 1), iter, value);
        if (value <= tol) break;
        if (iter >= max_iter) {
          char msg[200];
          std::snprintf(msg, sizeof(msg), "did not converge in %d iterations (%s = %.6g, tolerance %.6g)", iter,
                        test_name, value, tol);
          failure = msg;
          ok = false;
          kept_tangent = false;  // OpenSees: factorOnce = 1 after a failed test
          break;
        }
        ++iter;
      }
    }
    rep.newton_iterations += iterations;
    if (is_static) last_iter = iterations;  // LoadControl counts the failed attempt's updates too
    if (!ok) {
      if (is_static) Kokkos::deep_copy(u, s.ut);
      else {
        Kokkos::deep_copy(u, s.ut);
        Kokkos::deep_copy(v, s.vt);
        Kokkos::deep_copy(a, s.at);
      }
      Kokkos::deep_copy(s.du, s.du0);
      time = time0;
      ++rep.failed_attempts;
      return false;
    }
    rep.max_newton_iterations = std::max(rep.max_newton_iterations, iterations);
    ++rep.commits;
    // commit (the materials' trial state, from the last state determination at the
    // converged displacement), then record.
    force.commit_state();
    recorders.record(time, u, is_static ? s.v_committed : v, a);
    return true;
  };

  std::function<bool(double, int)> recover = [&](double h, int level) -> bool {
    if (attempt(h)) return true;
    for (int r = 1; r <= max_retries; ++r) {
      ++rep.retries;
      if (trace_events)
        std::printf("femorax-event|RETRY|%s|%d|%d|%.15g|%.17g\n", an.name.c_str(), r, max_retries, h, time + h);
      if (attempt(h)) return true;
    }
    if (level >= num_sublevels) return false;
    ++rep.substepped;
    const double hs = h / static_cast<double>(num_substeps);
    if (trace_events)
      std::printf("femorax-event|SUBSTEP|%s|%d|%.15g|%.17g\n", an.name.c_str(), level + 1, hs, time);
    for (int i = 0; i < num_substeps; ++i)
      if (!recover(hs, level + 1)) return false;
    return true;
  };

  const double final_time = an.final_time ? as_tcl_time(*an.final_time) : 0.0;
  auto more = [&] { return an.final_time ? as_tcl_time(time) < final_time : rep.steps < *an.num_steps; };
  while (more()) {
    if (!recover(an.dt, 0)) {
      char msg[512];
      std::snprintf(msg, sizeof(msg), "stage \"%s\": step %lld (t = %.10g) %s", an.name.c_str(),
                    static_cast<long long>(rep.steps + 1), attempt_time, failure.c_str());
      std::string m(msg);
      if (max_retries > 0 || num_sublevels > 0) {
        std::snprintf(msg, sizeof(msg), "; recovery exhausted (%d retries, %d sublevels of %d substeps)",
                      max_retries, num_sublevels, num_substeps);
        m += msg;
      }
      throw io::ModelError(m);
    }
    ++rep.steps;
  }
  if (!is_static) Kokkos::deep_copy(s.v_committed, v);
  Kokkos::fence();
  if (verbose && rep.steps > 0) {
    char recovery[160] = "";
    if (rep.retries > 0 || rep.substepped > 0)
      std::snprintf(recovery, sizeof(recovery), "; recovery: %lld retries, %lld steps split into substeps",
                    static_cast<long long>(rep.retries), static_cast<long long>(rep.substepped));
    std::printf("femorax: stage %s: %s, %s, %lld steps, %.2f Newton and %.1f %s+%s iterations per step%s%s\n",
                an.name.c_str(), io::to_string(an.integrator),
                linear ? "Linear" : modified ? "ModifiedNewton" : "Newton",
                static_cast<long long>(rep.steps), static_cast<double>(rep.newton_iterations) / rep.commits,
                static_cast<double>(rep.krylov_iterations) / std::max<std::int64_t>(1, rep.newton_iterations),
                rep.solver.c_str(), rep.preconditioner.c_str(), rep.fell_back ? " (CG fell back to GMRES)" : "",
                recovery);
  }
  return rep;
}

}  // namespace femorax::exec
