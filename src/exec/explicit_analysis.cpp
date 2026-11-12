#include "exec/explicit_analysis.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

#include "compile/batches.hpp"
#include "compile/registry.hpp"
#include "compile/uniform_damping_fit.hpp"
#include "exec/device_memory.hpp"
#include "exec/drm_load.hpp"
#include "exec/force_settings.hpp"
#include "exec/ground_motion.hpp"
#include "exec/internal_force.hpp"
#include "exec/recorders.hpp"
#include "exec/runtime.hpp"
#include "exec/tcl_time.hpp"
#include "exec/implicit_stage.hpp"
#include "io/h5.hpp"

namespace femorax::exec {

namespace {

using Space = Kokkos::DefaultExecutionSpace;
using Index = std::int32_t;
using IVec = Kokkos::View<int*, Space>;
using CVec = Kokkos::View<std::int8_t*, Space>;

double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <class T>
Kokkos::View<T*, Space> upload(const char* label, const std::vector<T>& h) {
  Kokkos::View<T*, Space> d(std::string(label), h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

// Effective ground-motion load on equation q: -m * sum over active patterns acting in q's DOF.
KOKKOS_INLINE_FUNCTION double ground_load(double m, int dof_q, const Vec& accel, const IVec& dof,
                                          int npat) {
  double r = 0.0;
  for (int p = 0; p < npat; ++p)
    if (dof(p) == dof_q) r -= m * accel(p);
  return r;
}

// --- ExplicitDifference ----------------------------------------------------------------
// newStep: Utdot += dt * Utdotdot; Ut += dt * Utdot  (Vector::addVector(1.0, x, dt)).
void ed_predict(const Vec& u, const Vec& vh, const Vec& a, double dt, Index neq) {
  Kokkos::parallel_for(
      "ed_predict", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq), KOKKOS_LAMBDA(const Index q) {
        vh(q) += a(q) * dt;
        u(q) += vh(q) * dt;
      });
}

// solve and update: a(n+1) = R / m; recorded v = vh + (dt/8) (3 a(n+1) + a(n)).
void ed_correct(const Vec& f, const Vec& m, const CVec& eq_dof, const Vec& g_accel, const IVec& g_dof,
                int npat, const Vec& vh, const Vec& a, const Vec& v_rec, double dt, Index neq) {
  const double halfT = dt * 0.125;
  Kokkos::parallel_for(
      "ed_correct", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq), KOKKOS_LAMBDA(const Index q) {
        const double R = -f(q) + ground_load(m(q), eq_dof(q), g_accel, g_dof, npat);
        const double a_new = R / m(q);
        const double t = 3.0 * a_new + a(q);  // Utdotdot1 = 3 Udotdot + Utdotdot
        v_rec(q) = vh(q) + t * halfT;         // Utdot1 = Utdot + halfT * Utdotdot1
        a(q) = a_new;
      });
}

// --- CentralDifference -----------------------------------------------------------------
// newStep: garbage velocity Utdot = -c2 * Utm1 (the element damping force uses it).
void cd_garbage_velocity(const Vec& v_elem, const Vec& um, double c2, Index neq) {
  Kokkos::parallel_for(
      "cd_garbage", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq),
      KOKKOS_LAMBDA(const Index q) { v_elem(q) = um(q) * -c2; });
}

// Solve the diagonal system and update (CentralDifference::update, then Utm1 = Ut; Ut = U).
void cd_solve(const Vec& f, const Vec& m, const Vec& cm, const CVec& eq_dof, const Vec& g_accel,
              const IVec& g_dof, int npat, const Vec& u, const Vec& um, const Vec& v_rec,
              const Vec& a, double c2, double c3, double dt, Index neq) {
  Kokkos::parallel_for(
      "cd_solve", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq), KOKKOS_LAMBDA(const Index q) {
        // Utdotdot = -2 c3 Ut + c3 Utm1 (the garbage acceleration in the inertia term)
        const double ag = u(q) * (-2.0 * c3) + um(q) * c3;
        const double R = -f(q) - m(q) * ag + ground_load(m(q), eq_dof(q), g_accel, g_dof, npat);
        const double A = m(q) * c3 + cm(q) * c2;  // tangent c3 M + c2 C, diagonal
        const double U = R / A;
        double Udot = U * 3.0;  // Udot = 3U - 4Ut + Utm1, then * c2
        Udot += u(q) * -4.0;
        Udot += um(q);
        Udot *= c2;
        const double Udotdot = (Udot - v_rec(q)) / dt;  // (Udot - Utdot) / dt
        um(q) = u(q);
        u(q) = U;
        v_rec(q) = Udot;
        a(q) = Udotdot;
      });
}

bool all_finite(const Vec& u) {
  int bad = 0;
  Kokkos::parallel_reduce(
      "nan_check", Kokkos::RangePolicy<Space>(0, static_cast<int>(u.extent(0))),
      KOKKOS_LAMBDA(const int q, int& b) {
        if (!Kokkos::isfinite(u(q))) ++b;  // count (a Max reduction starts at INT_MIN)
      },
      bad);
  return bad == 0;
}

std::string gb(std::int64_t b) {
  char s[32];
  std::snprintf(s, sizeof(s), "%.2f GB", static_cast<double>(b) / 1e9);
  return s;
}

}  // namespace

std::int64_t estimate_device_bytes(const io::Model& model, const compile::Compiled& c) {
  const std::int64_t neq = c.dofs.num_equations;
  std::int64_t nslots = 0, geometry = 0, tangents = 0;
  bool non_spd = false;
  for (const auto& b : c.batches) {
    nslots += static_cast<std::int64_t>(b.ndof) * b.padded;
    geometry += static_cast<std::int64_t>(b.geometry_doubles) * b.padded * 8;
    tangents += static_cast<std::int64_t>(b.tangent_doubles) * b.padded * 8;
    if (b.tangent != TangentSymmetry::SymmetricPositiveDefinite &&
        b.tangent != TangentSymmetry::SymmetricPositiveSemidefinite)
      non_spd = true;
  }
  std::int64_t bytes = compile::memory_estimate_bytes(c);
  bytes += 2 * nslots * 8 + (neq + 1) * 8;  // gather: slots, their list, each equation's start
  bytes += geometry;                        // geometry cache and element constants
  bytes += 3 * neq * 8;                     // velocity, previous displacement, acceleration
  bool implicit = false, gmres = non_spd;
  for (const auto& s : model.process)
    if (const auto* a = std::get_if<io::AnalysisStep>(&s.step); a && !io::is_explicit(a->integrator)) {
      implicit = true;
      // A static stage has no mass term: a semi-definite tangent gives a non-SPD operator.
      if (a->type == io::AnalysisType::Static) gmres = gmres || tangents > 0;
    }
  if (implicit) {
    bytes += 10 * neq * 8;  // Newmark predictor, increment, residual, M r (3), work
    bytes += 7 * neq * 8;   // CG (4), its initial guess, Jacobi diagonal and inverse
    bytes += tangents;      // stored Gauss-point tangents
    if (gmres) bytes += (2 * 30 + 3) * neq * 8;  // GMRES(30): basis V, Z, w, r
  }
  bytes += Recorders::device_bytes_estimate(model, c);
  for (const auto& p : model.drm_patterns) {  // host work only: the header and the matching
    const io::DrmFile f = io::read_drm_header(p.path);
    const compile::DrmSetup s = compile::setup_drm(p, f, c);
    std::int64_t dofs = 0;
    for (const auto& e : s.elements) dofs += c.batches[static_cast<std::size_t>(e.batch)].ndof;
    bytes += DrmLoad::device_bytes(static_cast<std::int64_t>(s.node.size()), dofs, f.samples, c.dofs.num_equations);
  }
  return bytes;
}

RunReport run_model(const io::Model& model, const RunOptions& opt) {
  RunReport rep;
  const double t_start = now();

  // Compile (host). Coloring only if the developer override asks for the colored scatter.
  compile::CompileOptions co;
  co.renumbering = opt.renumbering;
  co.simd_width = simd_width();
  if (const char* o = std::getenv("FEMORAX_FORCE_SETTINGS"); o && std::string(o).find("colored") != std::string::npos)
    co.build_coloring = true;
  const compile::Compiled c = compile::compile_model(model, co);
  rep.compile_seconds = now() - t_start;
  rep.equations = c.dofs.num_equations;

  // Memory check before allocating anything on the device.
  const std::int64_t neq = c.dofs.num_equations;
  rep.device_bytes_estimate = estimate_device_bytes(model, c);
  rep.device_bytes_free = device_free_memory_bytes();
  if (rep.device_bytes_free > 0 && rep.device_bytes_estimate > rep.device_bytes_free / 10 * 9) {
    throw io::ModelError("the model needs about " + gb(rep.device_bytes_estimate) + " of device memory, but " +
                         default_execution_space() + " has " + gb(rep.device_bytes_free) +
                         " free. Use a device with more memory, or (later) more GPUs.");
  }

  const double t_setup = now();
  InternalForce force(c, GeometryCache::Auto);
  const Choice choice = choose_force_settings(force, c);
  rep.force_settings = to_string(choice.settings);
  rep.force_settings_source = choice.source;
  rep.elements = force.num_elements();

  const Vec m = upload("lumped_mass", c.lumped_mass);
  const Vec cm = upload("lumped_alpha_mass", c.lumped_alpha_mass);
  const CVec eq_dof = upload("eq_dof", c.eq_dof);
  const Vec& u = force.u;       // displacement
  const Vec& v_elem = force.v;  // velocity the elements see (ED: v(n+1/2); CD: garbage)
  Vec v_rec("velocity", static_cast<std::size_t>(neq));     // committed (recorded) velocity
  Vec a("acceleration", static_cast<std::size_t>(neq));     // committed acceleration
  Vec um("previous_displacement", static_cast<std::size_t>(neq));
  Kokkos::deep_copy(u, 0.0);
  GroundMotions motions(model);
  std::vector<std::unique_ptr<DrmLoad>> drm;
  for (const auto& p : model.drm_patterns) drm.push_back(std::make_unique<DrmLoad>(p, c, force));
  Recorders recorders(model, c, opt.vtkhdf_single_precision);
  recorders.attach(force);
  rep.setup_seconds = now() - t_setup;
  if (opt.verbose) {
    std::printf("femorax: %lld elements, %lld equations, force kernel %s (%s), device %s\n",
                static_cast<long long>(rep.elements), static_cast<long long>(neq), rep.force_settings.c_str(),
                rep.force_settings_source.c_str(), default_execution_space().c_str());
    // Uniform damping: the fit, what it gives over the band, and how to switch (docs/design.md).
    for (const auto& r : model.uniform) {
      const auto fl = registry::uniform_damping_filters(r);
      const double eta = uniform_damping_eta(r.ratio), two_pi = 2.0 * std::acos(-1.0);
      double lo = 1e300, hi = 0.0;
      for (int k = 0; k <= 50; ++k) {
        const double f = r.freq1 * std::pow(r.freq2 / r.freq1, k / 50.0);
        const double z = uniform_damping_mode(fl, eta, two_pi * f).zeta;
        lo = std::min(lo, z);
        hi = std::max(hi, z);
      }
      const double s1 = uniform_damping_mode(fl, eta, two_pi * r.freq1).frequency_ratio;
      const double s2 = uniform_damping_mode(fl, eta, two_pi * r.freq2).frequency_ratio;
      const bool femorax_fit = r.fit != "opensees";
      std::printf("femorax: Uniform damping %s (%zu elements): %.4g %% over %.4g-%.4g Hz, fit \"%s\", %d filters: "
                  "damping %.3g-%.3g %% over the band, frequency x%.3f at %.4g Hz and x%.3f at %.4g Hz%s; %s\n",
                  r.id.c_str(), r.elements.size(), 100.0 * r.ratio, r.freq1, r.freq2, r.fit.c_str(), fl.nFilter,
                  100.0 * lo, 100.0 * hi, s1, r.freq1, s2, r.freq2,
                  femorax_fit ? " (stiffness kept at the band's geometric mean)" : " (no stiffness correction)",
                  femorax_fit ? "fit = \"opensees\" reproduces OpenSees"
                              : "fit = \"femorax\" (the default) gives the target across the band");
    }
  }

  double time = 0.0;  // the domain time
  std::unique_ptr<ImplicitStage> implicit;  // created at the first implicit stage
  for (const auto& step : model.process) {
    if (const auto* p = std::get_if<io::PatternStep>(&step.step)) {
      bool drm_pattern = false;
      for (auto& d : drm)
        if (d->tag() == p->tag) {
          motions.activate_drm(d.get());
          drm_pattern = true;
        }
      if (drm_pattern) continue;
      motions.activate(p->tag);
      for (const auto& [dof, vel0] : motions.initial_velocities()) {
        // UniformExcitation::setDomain: committed velocity vel0 in the pattern's DOF.
        auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_rec);
        for (Index q = 0; q < static_cast<Index>(neq); ++q)
          if (c.eq_dof[q] == dof) h(q) = vel0;
        Kokkos::deep_copy(v_rec, h);
      }
    } else if (const auto* r = std::get_if<io::RecorderStep>(&step.step)) {
      if (!recorders.activate(r->name)) throw io::ModelError("process step names an unknown recorder " + r->name);
    } else if (const auto* s = std::get_if<io::SetTimeStep>(&step.step)) {
      time = s->time;
    } else if (const auto* ms = std::get_if<io::UpdateMaterialStageStep>(&step.step)) {
      // updateMaterialStage -material <tag> -stage <stage>, for each listed material. The
      // switch to the plastic stage (elast2Plast) happens in the next state determination.
      for (const auto tag : ms->materials)
        for (std::size_t i = 0; i < model.materials.size(); ++i)
          if (model.materials[i].tag == tag) force.set_material_stage(static_cast<std::int32_t>(i), ms->stage);
    } else if (const auto* lc = std::get_if<io::LoadConstStep>(&step.step)) {
      // loadConst [-time t]: the supported patterns (UniformExcitation) have no load
      // factor to freeze; the body force is constant anyway. Only the time changes.
      if (lc->time) time = *lc->time;
    } else if (const auto* ia = std::get_if<io::AnalysisStep>(&step.step); ia && !io::is_explicit(ia->integrator)) {
      const auto* an = ia;
      if (!implicit) implicit = std::make_unique<ImplicitStage>(force, c, u, v_rec, a, choice.settings);
      StageReport st;
      st.name = an->name;
      st.integrator = io::to_string(an->integrator);
      st.dt = an->dt;
      const double t0 = now();
      const ImplicitStageReport ir = implicit->run(*an, time, motions, recorders, opt.verbose);
      st.steps = ir.steps;
      st.seconds = now() - t0;
      st.newton_iterations = ir.newton_iterations;
      st.retries = ir.retries;
      st.substepped = ir.substepped;
      st.commits = ir.commits;
      st.krylov_iterations = ir.krylov_iterations;
      st.solver = ir.solver;
      if (!all_finite(u))
        throw io::ModelError("stage \"" + an->name + "\": displacements are not finite at the end of the stage");
      rep.stages.push_back(st);
    } else if (const auto* an = std::get_if<io::AnalysisStep>(&step.step)) {
      StageReport st;
      st.name = an->name;
      st.integrator = io::to_string(an->integrator);
      st.dt = an->dt;
      const bool ed = an->integrator == io::Integrator::ExplicitDifference;
      st.critical_dt = force.min_critical_dt(ed ? ExplicitScheme::ExplicitDifference : ExplicitScheme::CentralDifference);
      if (an->dt > st.critical_dt) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "stage \"%s\": dt = %.6g exceeds the stable step %.6g of %s",
                      an->name.c_str(), an->dt, st.critical_dt, st.integrator.c_str());
        if (!an->allow_unstable_dt) throw io::ModelError(std::string(msg) + " (reduce dt, or set allow_unstable_dt)");
        if (opt.verbose) std::printf("femorax: warning: %s\n", msg);
      }
      st.min_critical_dt = st.critical_dt;
      const bool recheck = opt.stable_step_check_interval > 0 && force.has_state();
      bool warned = false;
      // domainChanged: the integrator starts from the committed state.
      const double dt = an->dt;
      const Index n = static_cast<Index>(neq);
      if (ed) Kokkos::deep_copy(v_elem, v_rec);  // Utdot = committed velocity
      else Kokkos::deep_copy(um, u);             // Utm1 = Ut ("assuming Ut-1 = Ut")
      const double c2 = 0.5 / dt, c3 = 1.0 / (dt * dt);
      const double t0 = now();
      std::int64_t k = 0;
      const double final_time = an->final_time ? as_tcl_time(*an->final_time) : 0.0;
      auto more = [&] { return an->final_time ? as_tcl_time(time) < final_time : k < *an->num_steps; };
      while (more()) {
        if (ed) {
          ed_predict(u, v_elem, a, dt, n);
          time += dt;
          motions.evaluate(time);
          force.step = StepContext{time, dt, 1};
          force.compute(choice.settings);
          motions.apply_drm(time, force.f);
          ed_correct(force.f, m, eq_dof, motions.accel(), motions.dof(), motions.num_active(), v_elem, a,
                     v_rec, dt, n);
        } else {
          motions.evaluate(time);  // loads at t(n)
          cd_garbage_velocity(v_elem, um, c2, n);
          force.step = StepContext{time, dt, 1};
          force.compute(choice.settings);
          motions.apply_drm(time, force.f);
          cd_solve(force.f, m, cm, eq_dof, motions.accel(), motions.dof(), motions.num_active(), u, um, v_rec,
                   a, c2, c3, dt, n);
          time += dt;  // CentralDifference::commit
        }
        ++k;
        force.commit_state();  // Domain::commit: the materials' trial state becomes committed
        recorders.record(time, u, v_rec, a);
        if (recheck && k % opt.stable_step_check_interval == 0) {
          // History materials change their stiffness during the stage (PressureDependMultiYield
          // with confinement), so the stable step can fall below dt.
          const double dtc =
              force.min_critical_dt(ed ? ExplicitScheme::ExplicitDifference : ExplicitScheme::CentralDifference);
          st.min_critical_dt = std::min(st.min_critical_dt, dtc);
          if (dt > dtc && !warned) {
            char msg[384];
            std::snprintf(msg, sizeof(msg),
                          "stage \"%s\": at step %lld (t = %.6g) the stable step fell to %.6g (%.6g at the start of "
                          "the stage), below dt = %.6g: the material stiffened",
                          an->name.c_str(), static_cast<long long>(k), time, dtc, st.critical_dt, dt);
            if (!an->allow_unstable_dt) throw io::ModelError(std::string(msg) + " (reduce dt, or set allow_unstable_dt)");
            if (opt.verbose) std::printf("femorax: warning: %s\n", msg);
            warned = true;
          }
        }
        if (opt.nan_check_interval > 0 && k % opt.nan_check_interval == 0 && !all_finite(u)) {
          throw io::ModelError("stage \"" + an->name + "\": displacements are not finite at step " +
                               std::to_string(k) + " (t = " + std::to_string(time) +
                               "): the analysis is unstable");
        }
      }
      Kokkos::fence();
      if (!all_finite(u))
        throw io::ModelError("stage \"" + an->name + "\": displacements are not finite at the end of the stage");
      st.steps = k;
      st.seconds = now() - t0;
      rep.stages.push_back(st);
      if (opt.verbose) {
        char limit[96];
        if (st.min_critical_dt < st.critical_dt) {
          std::snprintf(limit, sizeof(limit), "stable step %.6g, smallest %.6g", st.critical_dt, st.min_critical_dt);
        } else {
          std::snprintf(limit, sizeof(limit), "stable step %.6g", st.critical_dt);
        }
        std::printf("femorax: stage %s: %s, %lld steps of %.6g (%s), %.3f s (%.3f ms/step)\n", st.name.c_str(),
                    st.integrator.c_str(), static_cast<long long>(st.steps), st.dt, limit, st.seconds,
                    st.steps ? 1e3 * st.seconds / st.steps : 0.0);
      }
    }
  }
  recorders.finish();
  rep.records = recorders.records_written();
  rep.total_seconds = now() - t_start;
  return rep;
}

}  // namespace femorax::exec
