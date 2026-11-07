#include "exec/implicit_stage.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "exec/matrix_free_operator.hpp"
#include "exec/runtime.hpp"
#include "exec/tcl_time.hpp"
#include "io/h5.hpp"
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
  Vec ut, vt, at, du, b, tmp, mr[3];
  std::unique_ptr<VectorOps> ops;

  Impl(InternalForce& f, const compile::Compiled& cc, Vec u_, Vec v_, Vec a_, const ForceSettings& s)
      : force(f), c(cc), u(std::move(u_)), v_committed(std::move(v_)), a(std::move(a_)), fs(s),
        os(f.resolve(OperatorSettings{})), lanes(simd_width()), n(f.num_equations()) {
    const auto un = static_cast<std::size_t>(n);
    ut = Vec("implicit_ut", un);
    vt = Vec("implicit_vt", un);
    at = Vec("implicit_at", un);
    du = Vec("implicit_du", un);
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

  // Integrator constants (Newmark::newStep, displ == 1) or the static tangent.
  const double dt = an.dt, gamma = an.gamma, beta = an.beta;
  OperatorCoefficients k{1.0, 0.0, 0.0};
  double c2 = 0.0, c3 = 0.0;
  if (!is_static) {
    if (an.newmark_form != "D")
      throw io::UnsupportedComponent("unsupported component: Newmark -form " + an.newmark_form + " in stage \"" +
                                     an.name + "\" (only the displacement form D so far)");
    c2 = gamma / (beta * dt);
    c3 = 1.0 / (beta * dt * dt);
    k = {1.0, c2, c3};
  }
  const double a1 = 1.0 - gamma / beta, a2 = dt * (1.0 - 0.5 * gamma / beta);
  const double a3 = -1.0 / (beta * dt), a4 = 1.0 - 0.5 / beta;

  // Tangent policy.
  if (an.algorithm == io::Algorithm::ModifiedNewton && an.modified_newton_initial && !force.constant_tangents())
    throw io::UnsupportedComponent("unsupported component: ModifiedNewton -initial with a material whose tangent "
                                   "is not constant (stage \"" + an.name + "\")");
  const bool stored = s.os.tangent == TangentSource::Stored;

  // Solver: automatic unless the model sets it.
  solvers::SolveOptions so;
  so.relative_tolerance = an.solver_tolerance.value_or(an.algorithm == io::Algorithm::Linear ? 1e-12 : 1e-10);
  so.max_iterations = an.solver_max_iterations.value_or(
      static_cast<int>(std::max<std::int64_t>(200, std::min<std::int64_t>(s.n, 10000))));
  MatrixFreeOperator op(force, u, k, s.os);
  solvers::Jacobi jacobi;
  solvers::KrylovSolver solver(so, s.lanes);
  rep.solver = op.info().symmetry == solvers::Symmetry::SymmetricPositiveDefinite ? "cg" : "gmres";
  bool have_tangent = false;
  auto update_tangent = [&] {
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

  // Velocities the elements see: the committed ones (Newmark), none (Static).
  if (is_static) Kokkos::deep_copy(v, 0.0);
  else Kokkos::deep_copy(v, s.v_committed);

  auto form_unbalance = [&] {
    force.compute(s.fs);  // f_int(u) + Rayleigh damping forces at v (v = 0 in static stages)
    if (!is_static) force.apply_operator({0.0, 0.0, 1.0}, s.os, u, a, s.tmp);
    form_residual(s.b, force.f, s.tmp, !is_static, motions.accel(), motions.dof(), motions.num_active(), s.mr[0],
                  s.mr[1], s.mr[2]);
  };
  auto solve = [&] {
    const auto r = solver.solve(s.b, s.du);  // initial guess: the previous increment
    rep.krylov_iterations += r.iterations;
    if (solver.fell_back()) rep.fell_back = true;
    if (!r.converged) {
      throw io::ModelError("stage \"" + an.name + "\": the " + solver.name() + " solve did not converge in " +
                           std::to_string(r.iterations) + " iterations (residual " + std::to_string(r.final_residual) +
                           " of " + std::to_string(r.rhs_norm) + ")");
    }
  };
  auto update = [&] {
    if (is_static) VectorOps::axpy(1.0, s.du, u);
    else newmark_update(s.du, u, v, a, c2, c3);
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

  const double final_time = an.final_time ? as_tcl_time(*an.final_time) : 0.0;
  auto more = [&] { return an.final_time ? as_tcl_time(time) < final_time : rep.steps < *an.num_steps; };
  const bool stage_tangent_once =
      an.algorithm == io::Algorithm::ModifiedNewton && (an.modified_newton_factor_once || an.modified_newton_initial);

  while (more()) {
    // newStep
    if (is_static) {
      dlambda *= spec_iter / last_iter;
      if (dlambda < dmin) dlambda = dmin;
      else if (dlambda > dmax) dlambda = dmax;
      time += dlambda;
      last_iter = 0;
    } else {
      newmark_predict(u, v, a, s.ut, s.vt, s.at, a1, a2, a3, a4);
      time += dt;
    }
    motions.evaluate(time);

    int iterations = 0;
    if (an.algorithm == io::Algorithm::Linear) {
      update_tangent();
      form_unbalance();
      solve();
      update();
      iterations = 1;
    } else {
      form_unbalance();
      if (an.algorithm == io::Algorithm::ModifiedNewton && !(stage_tangent_once && have_tangent)) update_tangent();
      int iter = 1;
      while (true) {
        if (an.algorithm == io::Algorithm::Newton) update_tangent();
        solve();
        update();
        form_unbalance();
        const double value = test_value();
        iterations = iter;
        if (value <= tol) break;
        if (iter >= max_iter) {
          char msg[320];
          std::snprintf(msg, sizeof(msg),
                        "stage \"%s\": step %lld (t = %.10g) did not converge in %d iterations (%s = %.6g, "
                        "tolerance %.6g)%s",
                        an.name.c_str(), static_cast<long long>(rep.steps + 1), time, iter,
                        test == io::TestType::NormUnbalance  ? "NormUnbalance"
                        : test == io::TestType::NormDispIncr ? "NormDispIncr"
                                                             : "EnergyIncr",
                        value, tol,
                        (an.max_retries.value_or(0) > 0 || an.num_substeps.value_or(0) > 0)
                            ? "; Femora's retry and substep recovery is not available yet"
                            : "");
          throw io::ModelError(msg);
        }
        ++iter;
      }
    }
    if (is_static) last_iter = iterations;
    rep.newton_iterations += iterations;
    rep.max_newton_iterations = std::max(rep.max_newton_iterations, iterations);
    ++rep.steps;
    // commit (materials without state: nothing to swap), then record.
    recorders.record(time, u, is_static ? s.v_committed : v, a);
  }
  if (!is_static) Kokkos::deep_copy(s.v_committed, v);
  Kokkos::fence();
  if (verbose && rep.steps > 0) {
    std::printf("femorax: stage %s: %s, %s, %lld steps, %.2f Newton and %.1f %s iterations per step%s\n",
                an.name.c_str(), io::to_string(an.integrator),
                an.algorithm == io::Algorithm::Linear     ? "Linear"
                : an.algorithm == io::Algorithm::Newton ? "Newton"
                                                        : "ModifiedNewton",
                static_cast<long long>(rep.steps), static_cast<double>(rep.newton_iterations) / rep.steps,
                static_cast<double>(rep.krylov_iterations) / std::max<std::int64_t>(1, rep.newton_iterations),
                rep.solver.c_str(), rep.fell_back ? " (CG fell back to GMRES)" : "");
  }
  return rep;
}

}  // namespace femorax::exec
