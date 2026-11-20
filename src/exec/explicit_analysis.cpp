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
#include "compile/local_time_stepping.hpp"
#include "compile/lts_levels.hpp"
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
void ed_correct(const Vec& f, const Vec& m, const Vec& mg, const CVec& eq_dof, const Vec& g_accel, const IVec& g_dof,
                int npat, const Vec& vh, const Vec& a, const Vec& v_rec, double dt, Index neq, const CVec& skip) {
  const double halfT = dt * 0.125;
  const bool has_skip = skip.extent(0) > 0;
  Kokkos::parallel_for(
      "ed_correct", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq), KOKKOS_LAMBDA(const Index q) {
        if (has_skip && skip(q)) return;  // in a mass block (ed_block)
        const double R = -f(q) + ground_load(mg(q), eq_dof(q), g_accel, g_dof, npat);
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
void cd_solve(const Vec& f, const Vec& m, const Vec& mg, const Vec& cm, const CVec& eq_dof, const Vec& g_accel,
              const IVec& g_dof, int npat, const Vec& u, const Vec& um, const Vec& v_rec,
              const Vec& a, double c2, double c3, double dt, Index neq, const CVec& skip) {
  const bool has_skip = skip.extent(0) > 0;
  Kokkos::parallel_for(
      "cd_solve", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq), KOKKOS_LAMBDA(const Index q) {
        if (has_skip && skip(q)) return;  // in a mass block (cd_block)
        // Utdotdot = -2 c3 Ut + c3 Utm1 (the garbage acceleration in the inertia term)
        const double ag = u(q) * (-2.0 * c3) + um(q) * c3;
        const double R = -f(q) - m(q) * ag + ground_load(mg(q), eq_dof(q), g_accel, g_dof, npat);
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

// --- Mass blocks of exact constraints (rigid links, diaphragms; docs/design.md, "Exact
// constraints u = T q"): the equations coupled by T^T M T are solved per block with the
// inverse of the block's M (ExplicitDifference) or of c3 M + c2 C (CentralDifference),
// computed on the host once per stage; the diagonal kernels skip these equations.
constexpr int kMaxBlock = 12;

struct MassBlocks {
  int count = 0;
  IVec start, eq;
  Kokkos::View<std::int64_t*, Space> offset;
  Vec M, inv;
  CVec in_block;  // [neq]: 1 for the equations of a block
};

// The inverse of a symmetric positive definite matrix A (nb x nb, row major) by Cholesky;
// false when A is not positive definite.
bool invert_spd(int nb, const double* A, double* inv) {
  double L[kMaxBlock][kMaxBlock] = {};
  for (int j = 0; j < nb; ++j) {
    double d = A[j * nb + j];
    for (int k = 0; k < j; ++k) d -= L[j][k] * L[j][k];
    if (!(d > 0.0)) return false;
    L[j][j] = std::sqrt(d);
    for (int i = j + 1; i < nb; ++i) {
      double x = A[i * nb + j];
      for (int k = 0; k < j; ++k) x -= L[i][k] * L[j][k];
      L[i][j] = x / L[j][j];
    }
  }
  for (int col = 0; col < nb; ++col) {  // solve L L^T x = e_col
    double y[kMaxBlock], x[kMaxBlock];
    for (int i = 0; i < nb; ++i) {
      double r = i == col ? 1.0 : 0.0;
      for (int k = 0; k < i; ++k) r -= L[i][k] * y[k];
      y[i] = r / L[i][i];
    }
    for (int i = nb - 1; i >= 0; --i) {
      double r = y[i];
      for (int k = i + 1; k < nb; ++k) r -= L[k][i] * x[k];
      x[i] = r / L[i][i];
    }
    for (int i = 0; i < nb; ++i) inv[i * nb + col] = x[i];
  }
  return true;
}

MassBlocks upload_mass_blocks(const compile::Compiled& c) {
  MassBlocks b;
  const auto& mb = c.mass_blocks;
  b.count = static_cast<int>(mb.count());
  if (b.count == 0) return b;
  b.start = upload("block_start", std::vector<int>(mb.start.begin(), mb.start.end()));
  b.eq = upload("block_eq", std::vector<int>(mb.eq.begin(), mb.eq.end()));
  b.offset = upload("block_offset", mb.offset);
  b.M = upload("block_M", mb.M);
  b.inv = Vec("block_inverse", mb.M.size());
  std::vector<std::int8_t> in(static_cast<std::size_t>(c.dofs.num_equations), 0);
  for (const auto q : mb.eq) in[static_cast<std::size_t>(q)] = 1;
  b.in_block = upload("in_block", in);
  return b;
}

// The blocks' inverses for a stage: of M (c3 = 0) or of c3 M + c2 C.
void prepare_mass_blocks(const compile::Compiled& c, MassBlocks& b, double c3, double c2, const std::string& stage) {
  if (b.count == 0) return;
  const auto& mb = c.mass_blocks;
  std::vector<double> inv(mb.M.size());
  for (std::size_t k = 0; k < mb.count(); ++k) {
    const int nb = mb.start[k + 1] - mb.start[k];
    double A[kMaxBlock * kMaxBlock];
    for (int i = 0; i < nb * nb; ++i) {
      const auto o = static_cast<std::size_t>(mb.offset[k] + i);
      A[i] = c3 == 0.0 ? mb.M[o] : c3 * mb.M[o] + c2 * mb.C[o];
    }
    if (!invert_spd(nb, A, inv.data() + mb.offset[k])) {
      const auto q = mb.eq[static_cast<std::size_t>(mb.start[k])];
      std::string node = "?";
      for (std::int32_t i = 0; i < c.dofs.num_nodes() && node == "?"; ++i)
        for (int d = 0; d < c.dofs.ndf[static_cast<std::size_t>(i)]; ++d)
          if (c.dofs.equation(i, d) == q) node = std::to_string(c.dofs.node_tag[static_cast<std::size_t>(i)]);
      throw io::ModelError("stage \"" + stage + "\": the mass of the rigidly constrained group at node " + node +
                           " is not positive definite: an explicit stage needs mass on every free DOF (give the "
                           "master or its slaves mass, including rotary inertia where they rotate)");
    }
  }
  auto h = Kokkos::create_mirror_view(b.inv);
  for (std::size_t i = 0; i < inv.size(); ++i) h(i) = inv[i];
  Kokkos::deep_copy(b.inv, h);
}

// ExplicitDifference on the blocks: a(n+1) = M^-1 (-f + loads), as ed_correct per equation.
void ed_block(const MassBlocks& b, const Vec& f, const CVec& eq_dof, const Vec& g_accel, const IVec& g_dof, int npat,
              const Vec& vh, const Vec& a, const Vec& v_rec, double dt) {
  if (b.count == 0) return;
  const double halfT = dt * 0.125;
  const auto st = b.start, eq = b.eq;
  const auto off = b.offset;
  const auto M = b.M, inv = b.inv;
  Kokkos::parallel_for(
      "ed_block", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count), KOKKOS_LAMBDA(const Index k) {
        const int s0 = st(k), nb = st(k + 1) - s0;
        const auto o = off(k);
        double R[kMaxBlock];
        for (int i = 0; i < nb; ++i) {
          double r = -f(eq(s0 + i));
          for (int j = 0; j < nb; ++j) r += ground_load(M(o + i * nb + j), eq_dof(eq(s0 + j)), g_accel, g_dof, npat);
          R[i] = r;
        }
        for (int i = 0; i < nb; ++i) {
          double a_new = 0.0;
          for (int j = 0; j < nb; ++j) a_new += inv(o + i * nb + j) * R[j];
          const int q = eq(s0 + i);
          v_rec(q) = vh(q) + (3.0 * a_new + a(q)) * halfT;
          a(q) = a_new;
        }
      });
}

// CentralDifference on the blocks: U = (c3 M + c2 C)^-1 R, then cd_solve's updates.
void cd_block(const MassBlocks& b, const Vec& f, const CVec& eq_dof, const Vec& g_accel, const IVec& g_dof, int npat,
              const Vec& u, const Vec& um, const Vec& v_rec, const Vec& a, double c2, double c3, double dt) {
  if (b.count == 0) return;
  const auto st = b.start, eq = b.eq;
  const auto off = b.offset;
  const auto M = b.M, inv = b.inv;
  Kokkos::parallel_for(
      "cd_block", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count), KOKKOS_LAMBDA(const Index k) {
        const int s0 = st(k), nb = st(k + 1) - s0;
        const auto o = off(k);
        double R[kMaxBlock], U[kMaxBlock];
        for (int i = 0; i < nb; ++i) {
          double r = -f(eq(s0 + i));
          for (int j = 0; j < nb; ++j) {
            const int qj = eq(s0 + j);
            const double ag = u(qj) * (-2.0 * c3) + um(qj) * c3;
            const double mij = M(o + i * nb + j);
            r += -mij * ag + ground_load(mij, eq_dof(qj), g_accel, g_dof, npat);
          }
          R[i] = r;
        }
        for (int i = 0; i < nb; ++i) {
          double x = 0.0;
          for (int j = 0; j < nb; ++j) x += inv(o + i * nb + j) * R[j];
          U[i] = x;
        }
        for (int i = 0; i < nb; ++i) {
          const int q = eq(s0 + i);
          double Udot = U[i] * 3.0;
          Udot += u(q) * -4.0;
          Udot += um(q);
          Udot *= c2;
          const double Udotdot = (Udot - v_rec(q)) / dt;
          um(q) = u(q);
          u(q) = U[i];
          v_rec(q) = Udot;
          a(q) = Udotdot;
        }
      });
}

// --- Local time stepping (docs/design.md; compile/local_time_stepping.hpp, the host reference)
// One substep of the stabilized leapfrog LTS from the force f at the mixed field (the coarse
// equations at u_n, the fine ones at z_k): g = (loads - f - ext) / m = -(w + A Pi_f z_k), then
//   k = 0: z_1 = u_n + c g (c = 1/2 tau^2 2p^2/(omega delta)), z_0 = u_n;
//   k > 0: z_{k+1} = (1 + beta_k) z_k - beta_k z_{k-1} + c g (c = tau^2 2p^2/omega beta_{k+1/2});
// and the next mixed field into u.
void lts_substep(bool first, double a1, double a2, double c, const Vec& f, const Vec& ext, const Vec& m,
                 const Vec& mg, const CVec& eq_dof, const Vec& g_accel, const IVec& g_dof, int npat, const CVec& fine,
                 const Vec& un, const Vec& z, const Vec& zp, const Vec& u, Index neq) {
  Kokkos::parallel_for(
      "lts_substep", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq), KOKKOS_LAMBDA(const Index q) {
        const double g = (-f(q) - ext(q) + ground_load(mg(q), eq_dof(q), g_accel, g_dof, npat)) / m(q);
        if (first) {
          zp(q) = un(q);
          z(q) = un(q) + c * g;
        } else {
          const double next = a1 * z(q) - a2 * zp(q) + c * g;
          zp(q) = z(q);
          z(q) = next;
        }
        u(q) = fine(q) ? z(q) : un(q);
      });
}

// CentralDifference after the LTS substeps: u(n+1) = -u(n-1) + 2 z_p (no damping), then the
// velocity and acceleration as cd_solve.
void cd_lts_finish(const Vec& z, const Vec& un, const Vec& u, const Vec& um, const Vec& v_rec, const Vec& a,
                   double c2, double dt, Index neq) {
  Kokkos::parallel_for(
      "cd_lts_finish", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq), KOKKOS_LAMBDA(const Index q) {
        const double U = -um(q) + 2.0 * z(q);
        double Udot = U * 3.0;
        Udot += un(q) * -4.0;
        Udot += um(q);
        Udot *= c2;
        const double Udotdot = (Udot - v_rec(q)) / dt;
        um(q) = un(q);
        u(q) = U;
        v_rec(q) = Udot;
        a(q) = Udotdot;
      });
}

// ExplicitDifference after the LTS substeps: the step's acceleration 2 (z_p - u) / dt^2 in
// place of R / m, then as ed_correct.
void ed_lts_finish(const Vec& z, const Vec& un, const Vec& u, const Vec& vh, const Vec& a, const Vec& v_rec,
                   double dt, Index neq) {
  const double halfT = dt * 0.125, s = 2.0 / (dt * dt);
  Kokkos::parallel_for(
      "ed_lts_finish", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq), KOKKOS_LAMBDA(const Index q) {
        const double a_new = (z(q) - un(q)) * s;
        u(q) = un(q);
        v_rec(q) = vh(q) + (3.0 * a_new + a(q)) * halfT;
        a(q) = a_new;
      });
}

// What beams cannot do in explicit stages (docs/design.md, "Beams", step 4): Rayleigh stiffness
// damping (refused on beams by design: "Structures"). Implicit stages take it. Beams without a
// mass of their own are fine: their share of the nodal masses enters their stable step.
void check_beams(const io::Model& model, const compile::Compiled& c, const io::AnalysisStep& an) {
  if (!io::is_explicit(an.integrator)) return;
  for (const auto& b : c.batches) {
    if (b.element != io::ElementType::DispBeamColumn) continue;
    for (const auto& ps : b.params) {
      const auto& g = model.elements[static_cast<std::size_t>(ps.group)];
      if (ps.damping.betaK != 0.0 || ps.damping.betaKinit != 0.0 || ps.damping.betaKcomm != 0.0)
        throw io::ModelError("stage \"" + an.name + "\": element group \"" + g.id +
                             "\" (dispBeamColumn) has Rayleigh stiffness damping, which FemoraX refuses on beams in "
                             "explicit stages (it cuts the stable step and gives spurious damping forces in yielding "
                             "frames); use mass-proportional damping (docs/design.md, \"Structures\")");
    }
  }
}

// Every free DOF of an explicit stage needs mass (docs/design.md, "Structures", rule 5):
// names the first node and DOF without it.
void check_masses(const compile::Compiled& c, const io::AnalysisStep& an) {
  for (std::int32_t node = 0; node < c.dofs.num_nodes(); ++node)
    for (int d = 0; d < c.dofs.ndf[static_cast<std::size_t>(node)]; ++d) {
      const std::int32_t q = c.dofs.equation(node, d);
      // A constrained DOF's slot (embedded node): its mass is on its host's equations.
      if (q >= 0 && q < c.dofs.num_equations && !(c.lumped_mass[static_cast<std::size_t>(q)] > 0.0))
        throw io::ModelError("stage \"" + an.name + "\": node " + std::to_string(c.dofs.node_tag[static_cast<std::size_t>(node)]) +
                             " DOF " + std::to_string(d + 1) +
                             " has no mass: an explicit stage needs mass on every free DOF (give its elements a "
                             "density, or add a nodal mass)");
    }
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
    const auto nodes = static_cast<std::int64_t>(s.node.size());
    bytes += DrmLoad::device_bytes(nodes, dofs, DrmLoad::window_samples(nodes, f.samples), c.dofs.num_equations);
    // The DRM elements' own states (Uniform-damped elastic): both sides, both buffers.
    for (const auto& e : s.elements) bytes += 2 * 2 * static_cast<std::int64_t>(c.batches[static_cast<std::size_t>(e.batch)].nstate) * 8;
    bytes += 3 * nodes * 8;  // the free field's velocity
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
  // The ground motion's mass: without the rotary inertia the embedded rotation tie lumps onto
  // translations (exact constraints); the lumped mass otherwise.
  const Vec mg = c.ground_mass.empty() ? m : upload("ground_mass", c.ground_mass);
  const CVec eq_dof = upload("eq_dof", c.eq_dof);
  MassBlocks blocks = upload_mass_blocks(c);
  const Vec& u = force.u;       // displacement
  const Vec& v_elem = force.v;  // velocity the elements see (ED: v(n+1/2); CD: garbage)
  Vec v_rec("velocity", static_cast<std::size_t>(neq));     // committed (recorded) velocity
  Vec a("acceleration", static_cast<std::size_t>(neq));     // committed acceleration
  Vec um("previous_displacement", static_cast<std::size_t>(neq));
  // With exact constraints the recorders read the constrained nodes' slots: the velocity and
  // acceleration copied with their slots expanded (the displacement is the force object's u).
  Vec v_out = v_rec, a_out = a;
  if (force.has_constraints()) {
    v_out = Vec("recorded_velocity", force.u.extent(0));
    a_out = Vec("recorded_acceleration", force.u.extent(0));
  }
  Kokkos::deep_copy(u, 0.0);
  GroundMotions motions(model);
  std::vector<std::unique_ptr<DrmLoad>> drm;
  for (const auto& p : model.drm_patterns)
    drm.push_back(std::make_unique<DrmLoad>(p, c, force, opt.drm_window_samples));
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
      check_beams(model, c, *an);
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
      check_beams(model, c, *an);
      check_masses(c, *an);
      const bool ed = an->integrator == io::Integrator::ExplicitDifference;
      const ExplicitScheme scheme = ed ? ExplicitScheme::ExplicitDifference : ExplicitScheme::CentralDifference;
      // Batches whose stable step cannot change keep it for the whole stage: computed once
      // here; the periodic re-check below covers only the others, with the cheap bound.
      const double fixed_dt = force.min_critical_dt(scheme, InternalForce::Batches::Fixed);
      st.critical_dt = std::min(fixed_dt, force.min_critical_dt(scheme, InternalForce::Batches::Changing));
      // Local time stepping: the levels from every element's own stable step.
      bool lts = false;
      compile::LtsCoefficients lcoef;
      int ring = -1;
      CVec fine_d;
      Vec un, z, zp, ext;
      ForceSettings lts_settings = choice.settings;
      if (opt.local_time_stepping) {
        if (blocks.count > 0)
          throw io::UnsupportedComponent("unsupported component: local time stepping with rigid links or diaphragms "
                                         "(their block masses; embedded nodes are supported)");
        if (force.has_state())
          throw io::ModelError("stage \"" + an->name + "\": local time stepping does not support materials with state "
                               "or Uniform damping yet");
        for (const auto& rr : model.rayleigh)
          if (rr.alphaM != 0.0 || rr.betaK != 0.0 || rr.betaKinit != 0.0 || rr.betaKcomm != 0.0)
            throw io::ModelError("stage \"" + an->name + "\": local time stepping does not support Rayleigh damping yet");
        const compile::LtsLevels lv = compile::assign_lts_levels(c, force.element_critical_dts(scheme), an->dt, opt.lts);
        if (lv.too_fine > 0) {
          char msg[256];
          std::snprintf(msg, sizeof(msg),
                        "stage \"%s\": %lld elements need a step finer than %d local time stepping levels give "
                        "(their smallest stable step %.6g, dt = %.6g)",
                        an->name.c_str(), static_cast<long long>(lv.too_fine), opt.lts.max_levels, lv.min_element_dt,
                        an->dt);
          throw io::ModelError(msg);
        }
        st.lts_levels = lv.levels;
        if (lv.levels > 1) {
          lts = true;
          int ratio_p = 1;
          for (int l = 1; l < lv.levels; ++l) ratio_p *= opt.lts.ratio;
          lcoef = compile::lts_coefficients(ratio_p, opt.lts_nu);
          std::vector<std::int8_t> fine(lv.equation_level.size());
          for (std::size_t q = 0; q < fine.size(); ++q) {
            fine[q] = lv.equation_level[q] > 0 ? 1 : 0;
            st.lts_fine_equations += fine[q];
          }
          fine_d = upload("lts_fine", fine);
          std::vector<std::vector<std::uint8_t>> member(lv.touched_level.size());
          for (std::size_t b = 0; b < member.size(); ++b)
            for (const auto l : lv.touched_level[b]) member[b].push_back(l > 0 ? 1 : 0);
          ring = force.add_element_subset(member);
          st.lts_substeps = ratio_p;
          st.lts_substep_elements = force.subset_elements(ring);
          // Element evaluations per coarse step: a global step at dt / p runs every element p
          // times; the two-level scheme runs the substep elements p times and the others once.
          const double all = static_cast<double>(lv.elements), sub = static_cast<double>(st.lts_substep_elements);
          st.lts_speedup = all * ratio_p / (all - sub + sub * ratio_p);
          un = Vec("lts_u", static_cast<std::size_t>(neq));
          z = Vec("lts_z", static_cast<std::size_t>(neq));
          zp = Vec("lts_z_prev", static_cast<std::size_t>(neq));
          ext = Vec("lts_ext", static_cast<std::size_t>(neq));
          // The substeps re-evaluate only some elements and keep the others' slots: the gather
          // scatter, the fused material path.
          lts_settings.scatter = Scatter::Gather;
          lts_settings.material = MaterialPath::Fused;
        }
      }
      if (an->dt > st.critical_dt && !lts) {
        char msg[256];
        std::snprintf(msg, sizeof(msg), "stage \"%s\": dt = %.6g exceeds the stable step %.6g of %s",
                      an->name.c_str(), an->dt, st.critical_dt, st.integrator.c_str());
        if (!an->allow_unstable_dt) throw io::ModelError(std::string(msg) + " (reduce dt, or set allow_unstable_dt)");
        if (opt.verbose) std::printf("femorax: warning: %s\n", msg);
      }
      st.min_critical_dt = st.critical_dt;
      const bool recheck = opt.stable_step_check_interval > 0 && force.stability_may_change();
      if (recheck) force.prepare_stability_bounds();
      bool warned = false;
      // domainChanged: the integrator starts from the committed state.
      const double dt = an->dt;
      const Index n = static_cast<Index>(neq);
      // The equations only: u and v_elem also span the embedded nodes' slots (InternalForce).
      const std::pair<std::size_t, std::size_t> eqs{0, static_cast<std::size_t>(neq)};
      if (ed) Kokkos::deep_copy(Kokkos::subview(v_elem, eqs), v_rec);  // Utdot = committed velocity
      else Kokkos::deep_copy(um, Kokkos::subview(u, eqs));             // Utm1 = Ut ("assuming Ut-1 = Ut")
      const double c2 = 0.5 / dt, c3 = 1.0 / (dt * dt);
      if (ed) prepare_mass_blocks(c, blocks, 0.0, 0.0, an->name);
      else prepare_mass_blocks(c, blocks, c3, c2, an->name);
      const double t0 = now();
      std::int64_t k = 0;
      const double final_time = an->final_time ? as_tcl_time(*an->final_time) : 0.0;
      auto more = [&] { return an->final_time ? as_tcl_time(time) < final_time : k < *an->num_steps; };
      // The LTS substeps from u = u_n at time t: z_p in z, u_n in un, u holds the last mixed
      // field (the finish kernels restore it). The loads and the DRM forces are held at t.
      const double tau2 = (dt / lcoef.p) * (dt / lcoef.p);
      auto lts_substeps = [&](double t) {
        Kokkos::deep_copy(un, Kokkos::subview(u, std::pair<std::size_t, std::size_t>{0, static_cast<std::size_t>(neq)}));
        Kokkos::deep_copy(ext, 0.0);
        motions.apply_drm(t, ext);
        force.step = StepContext{t, dt, 1};
        for (int j = 0; j < lcoef.p; ++j) {
          if (j == 0) force.compute(lts_settings);
          else force.compute_subset(lts_settings, ring);
          const bool first = j == 0;
          const double bk = first ? 0.0 : lcoef.beta[static_cast<std::size_t>(j - 1)];
          const double cc = first ? 0.5 * tau2 * lcoef.c_first : tau2 * lcoef.c_rest * lcoef.beta_half[static_cast<std::size_t>(j - 1)];
          lts_substep(first, 1.0 + bk, bk, cc, force.f, ext, m, mg, eq_dof, motions.accel(), motions.dof(),
                      motions.num_active(), fine_d, un, z, zp, u, n);
        }
      };
      while (more()) {
        if (lts && ed) {
          ed_predict(u, v_elem, a, dt, n);
          time += dt;
          motions.evaluate(time);
          lts_substeps(time);
          ed_lts_finish(z, un, u, v_elem, a, v_rec, dt, n);
        } else if (lts) {
          motions.evaluate(time);
          cd_garbage_velocity(v_elem, um, c2, n);
          lts_substeps(time);
          cd_lts_finish(z, un, u, um, v_rec, a, c2, dt, n);
          time += dt;
        } else if (ed) {
          ed_predict(u, v_elem, a, dt, n);
          time += dt;
          motions.evaluate(time);
          force.step = StepContext{time, dt, 1};
          force.compute(choice.settings);
          motions.apply_drm(time, force.f);
          ed_correct(force.f, m, mg, eq_dof, motions.accel(), motions.dof(), motions.num_active(), v_elem, a,
                     v_rec, dt, n, blocks.in_block);
          ed_block(blocks, force.f, eq_dof, motions.accel(), motions.dof(), motions.num_active(), v_elem, a, v_rec,
                   dt);
        } else {
          motions.evaluate(time);  // loads at t(n)
          cd_garbage_velocity(v_elem, um, c2, n);
          force.step = StepContext{time, dt, 1};
          force.compute(choice.settings);
          motions.apply_drm(time, force.f);
          cd_block(blocks, force.f, eq_dof, motions.accel(), motions.dof(), motions.num_active(), u, um, v_rec, a,
                   c2, c3, dt);  // before cd_solve: it reads u and um of the block's equations only
          cd_solve(force.f, m, mg, cm, eq_dof, motions.accel(), motions.dof(), motions.num_active(), u, um, v_rec,
                   a, c2, c3, dt, n, blocks.in_block);
          time += dt;  // CentralDifference::commit
        }
        ++k;
        force.commit_state();  // Domain::commit: the materials' trial state becomes committed
        motions.commit_drm();
        if (force.has_constraints()) {  // the constrained nodes' values (their slots)
          force.expand_slots(u);
          force.expanded_copy(v_rec, v_out);
          force.expanded_copy(a, a_out);
        }
        recorders.record(time, u, v_out, a_out);
        if (recheck && k % opt.stable_step_check_interval == 0) {
          // History materials change their stiffness during the stage (PressureDependMultiYield
          // with confinement), so the stable step can fall below dt.
          // The bound never overstates the step; when it falls below dt, the exact check
          // decides, so a run stops only when the stable step itself is below dt.
          double dtc = std::min(fixed_dt, force.min_critical_dt_bound(scheme));
          if (dt > dtc) dtc = std::min(fixed_dt, force.min_critical_dt(scheme, InternalForce::Batches::Changing));
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
        if (lts)
          std::printf("femorax: stage %s: local time stepping, %d levels: %lld fine equations take %d substeps, "
                      "%lld elements evaluated in each substep, estimated gain over a global step %.2fx\n",
                      st.name.c_str(), st.lts_levels, static_cast<long long>(st.lts_fine_equations), st.lts_substeps,
                      static_cast<long long>(st.lts_substep_elements), st.lts_speedup);
      }
    }
  }
  recorders.finish();
  rep.records = recorders.records_written();
  rep.total_seconds = now() - t_start;
  return rep;
}

}  // namespace femorax::exec
