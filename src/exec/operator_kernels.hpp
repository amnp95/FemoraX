#pragma once

// Matrix-free operator kernels (docs/design.md, "The matrix-free operator"): y = K_eff x and
// diag(K_eff) element by element, plus the state determination that stores the Gauss-point
// tangent. Generic over the element (MatrixFreeElement), the geometry mode and the tangent
// source; results go through the same output modes and slots as the force kernels
// (exec/assembly.hpp), so merged equations from any constraint are handled the same way.
//
// Tangent sources:
//   Constant    one D per parameter set (materials with CONSTANT_TANGENT), set up once
//   Stored      D written per Gauss point by store_tangents (packed when declared symmetric)
//   Recomputed  the material is called again at the current displacement
// Materials with history state: the stored source writes the trial state as the force
// kernel does; the recomputed source is only for materials without state.
//
// Included only by exec .cpp files (it contains device lambdas).

#include "exec/element_kernels.hpp"
#include "physics/common/tangent_storage.hpp"
#include "physics/operators/element_operator.hpp"

namespace femorax::exec::kernels {

enum class Source { Constant = 0, Stored = 1, Recomputed = 2 };

template <class Elem>
using Storage = TangentStorage<Elem::Material::NSTRAIN, is_symmetric(Elem::Material::TANGENT)>;

// Per-batch operator data on the device.
template <class Elem>
struct DevTangents {
  Vec constant;  // [nparams * NS * NS]: D of each parameter set (Constant)
  Vec stored;    // per Gauss point, Storage::SIZE values: [k * (NGP * padded) + e * NGP + gp]
};

template <class Elem>
KOKKOS_INLINE_FUNCTION PointArray<const double> tangents_of(const Vec& t, Index e, Index padded) {
  return {t.data() + static_cast<std::ptrdiff_t>(e) * Elem::NGP, Elem::NGP * padded, 1};
}

// D of the material at the strain of ue at point s, from the point's committed state; the
// trial state is written (the same values the force kernel writes at this displacement).
template <class Elem>
KOKKOS_INLINE_FUNCTION void material_tangent(const StepContext& step, const typename Elem::Params& p,
                                             const typename Elem::PointShape& s, const double* ue,
                                             Strided<const double> committed, Strided<double> trial,
                                             double (*D)[Elem::Material::NSTRAIN]) {
  using Mat = typename Elem::Material;
  double eps[Mat::NSTRAIN], sig[Mat::NSTRAIN];
  Elem::point_strain(s, ue, eps);
  material_update<Mat>(step, p.mat, committed, trial, eps, sig, D);
}

// The recomputed source: materials without history state only (InternalForce::resolve
// falls back to the stored source for the others), so no state is passed. For a material
// with state the branch is never taken; it is not compiled either, which keeps the compiler
// from analysing an update on empty state views.
template <class Elem>
KOKKOS_INLINE_FUNCTION void recomputed_tangent(const StepContext& step, const typename Elem::Params& p,
                                               const typename Elem::PointShape& s, const double* ue,
                                               double (*D)[Elem::Material::NSTRAIN]) {
  if constexpr (Elem::Material::NSTATE == 0) {
    material_tangent<Elem>(step, p, s, ue, {}, {}, D);
  } else {
    (void)step, (void)p, (void)s, (void)ue, (void)D;
    Kokkos::abort("recomputed tangent for a material with history state");
  }
}

// One element: gather, operator or diagonal, emit. Diagonal = true writes diag(K_eff).
// unit_dir >= 0 replaces x by the rigid unit translation in that direction at every element
// DOF, fixed ones included (y = K_eff r: the UniformExcitation load -ag M r, which couples
// the free DOFs to the moving base through a consistent mass).
template <class Elem, bool Cached, int SourceI, bool Diagonal>
void launch_operator(const DevBatch<Elem>& b, const DevTangents<Elem>& t, const Vec& coords, Index N,
                     const OperatorCoefficients& k, const Vec& u, const Vec& x, const Vec& y,
                     const Vec& slots, Out out, Index begin, Index end, bool by_color, int unit_dir = -1) {
  static_assert(MatrixFreeElement<Elem>, "the matrix-free operator needs a MatrixFreeElement");
  using Mat = typename Elem::Material;
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM, NS = Mat::NSTRAIN;
  constexpr int kNodeNdf = Elem::NODE_NDF[0];  // a constant: no std::array call in device code
  const auto node = b.node, eq = b.eq, param = b.param, colors = b.color_elements;
  const auto params = b.params;
  const auto geo = b.geometry, cst = b.constants;
  const auto tc = t.constant, ts = t.stored;
  const auto sc = b.state[b.current];
  const Index P = b.padded;
  const std::int64_t base = b.slot_offset;
  const StepContext step = b.step;
  Kokkos::parallel_for(
      Diagonal ? "element_operator_diagonal" : "element_operator_apply",
      Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(begin, end), KOKKOS_LAMBDA(const Index i) {
        const Index e = by_color ? colors(i) : i;
        const Index pi = param(e);
        const DevParams<Elem>& pr = params(pi);
        double xe_e[NDOF], ue[NDOF], ye[NDOF];
        int eqe[NDOF];
        for (int a = 0; a < NDOF; ++a) {
          const Index q = eq(a * P + e);
          eqe[a] = q;
          if (unit_dir >= 0) xe_e[a] = (a % kNodeNdf) == unit_dir ? 1.0 : 0.0;
          else xe_e[a] = (!Diagonal && q >= 0) ? x(q) : 0.0;
          ue[a] = (SourceI == static_cast<int>(Source::Recomputed) && q >= 0) ? u(q) : 0.0;
        }
        double xel[NNODE][NDM], xl[3][8] = {};
        if (!Cached) {
          for (int a = 0; a < NNODE; ++a) {
            const Index n = node(a * P + e);
            for (int d = 0; d < NDM; ++d) xel[a][d] = coords(d * N + n);
          }
          Elem::local_coords(xel, xl);
        }
        auto tangent_at = [&](int gp, const typename Elem::PointShape& s, double (*D)[NS]) {
          if (SourceI == static_cast<int>(Source::Constant)) {
            for (int a = 0; a < NS; ++a)
              for (int c = 0; c < NS; ++c) D[a][c] = tc((static_cast<std::int64_t>(pi) * NS + a) * NS + c);
          } else if (SourceI == static_cast<int>(Source::Stored)) {
            Storage<Elem>::load(tangents_of<Elem>(ts, e, P).point(gp), D);
          } else {  // Recomputed: materials without history state only (InternalForce::resolve)
            recomputed_tangent<Elem>(step, pr.p, s, ue, D);
          }
        };
        // Committed tangent (betaKcomm): from the point's committed state.
        auto committed_at = [&](int gp, double (*D)[NS]) {
          Mat::committed_tangent(pr.p.mat, state_of<Elem>(sc, e, P).point(gp), D);
        };
        // Cached is a template constant (plain if: nvcc cannot capture inside if constexpr).
        // Only the source in use is built (a recomputed source may compute element constants).
        if (Cached) {
          const auto cached = cached_geometry<Elem>(geo, cst, e, P);
          if (Diagonal) element_operator_diagonal<Elem>(pr.p, pr.damp, k, cached, tangent_at, committed_at, eqe, ye);
          else element_operator_apply<Elem>(pr.p, pr.damp, k, cached, tangent_at, committed_at, xe_e, ye);
        } else {
          const auto recomputed = Elem::recomputed_geometry(xl, pr.p);
          if (Diagonal) element_operator_diagonal<Elem>(pr.p, pr.damp, k, recomputed, tangent_at, committed_at, eqe, ye);
          else element_operator_apply<Elem>(pr.p, pr.damp, k, recomputed, tangent_at, committed_at, xe_e, ye);
        }
        for (int a = 0; a < NDOF; ++a)
          emit(out, y, slots, base + static_cast<std::int64_t>(a) * P + e, eq(a * P + e), ye[a]);
      });
}

// H5DRM effective forces of the DRM elements of a batch (exec/drm_load.hpp): for element
// elems(i) with boundary-node mask masks(i), the free-field motion of its nodes (du, da at
// node_slot(node)), split into the exterior part (u_e, a_e) and the boundary part (u_b,
// a_b); y_e = K u_e + M a_e and y_b = K u_b + M a_b through the element's own operator;
// boundary nodes get -y_e, exterior nodes +y_b (OpenSees: Peff_b = -M_be a_e - K_be u_e,
// Peff_e = M_eb a_b + K_eb u_b). Written to slots(base + a * n + i), local DOF a; elastic
// materials only (the constant tangent).
template <class Elem, bool Cached>
void launch_drm_forces(const DevBatch<Elem>& b, const DevTangents<Elem>& t, const Vec& coords, Index N,
                       const Kokkos::View<Index*, Space>& elems, const Kokkos::View<std::uint32_t*, Space>& masks,
                       const Kokkos::View<Index*, Space>& node_slot, const Vec& du, const Vec& da, const Vec& dv,
                       bool damped, const Vec& slots, std::int64_t base) {
  static_assert(MatrixFreeElement<Elem>, "DRM forces need a MatrixFreeElement");
  using Mat = typename Elem::Material;
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM, NS = Mat::NSTRAIN;
  const auto node = b.node, param = b.param;
  const auto params = b.params;
  const auto geo = b.geometry, cst = b.constants;
  const auto tc = t.constant;
  const auto sc = b.state[b.current];
  const Index P = b.padded;
  const Index n = static_cast<Index>(elems.extent(0));
  if (n == 0) return;
  Kokkos::parallel_for(
      "element_drm_forces", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, n), KOKKOS_LAMBDA(const Index i) {
        const Index e = elems(i);
        const std::uint32_t mask = masks(i);
        const Index pi = param(e);
        const DevParams<Elem>& pr = params(pi);
        double ue_x[NDOF], ae_x[NDOF], ue_b[NDOF], ae_b[NDOF];
        for (int a = 0; a < NNODE; ++a) {
          const Index sl = node_slot(node(a * P + e));
          const bool boundary = ((mask >> a) & 1u) != 0;
          for (int c = 0; c < 3; ++c) {
            double uu = du(3 * static_cast<std::int64_t>(sl) + c), aa = da(3 * static_cast<std::int64_t>(sl) + c);
            if (damped) {
              // The layer's Rayleigh damping (docs/design.md, "Damped DRM layer"): for an
              // elastic layer C v = alphaM M v + (betaK + betaKinit + betaKcomm) K v, so the
              // operator below applies M (a + alphaM v) + K (u + beta v).
              const double vv = dv(3 * static_cast<std::int64_t>(sl) + c);
              uu += (pr.damp.betaK + pr.damp.betaKinit + pr.damp.betaKcomm) * vv;
              aa += pr.damp.alphaM * vv;
            }
            ue_x[3 * a + c] = boundary ? 0.0 : uu;
            ae_x[3 * a + c] = boundary ? 0.0 : aa;
            ue_b[3 * a + c] = boundary ? uu : 0.0;
            ae_b[3 * a + c] = boundary ? aa : 0.0;
          }
        }
        double xel[NNODE][NDM], xl[3][8] = {};
        if (!Cached) {
          for (int a = 0; a < NNODE; ++a) {
            const Index nd = node(a * P + e);
            for (int d = 0; d < NDM; ++d) xel[a][d] = coords(d * N + nd);
          }
          Elem::local_coords(xel, xl);
        }
        auto tangent_at = [&](int, const typename Elem::PointShape&, double (*D)[NS]) {
          for (int a = 0; a < NS; ++a)
            for (int c = 0; c < NS; ++c) D[a][c] = tc((static_cast<std::int64_t>(pi) * NS + a) * NS + c);
        };
        auto committed_at = [&](int gp, double (*D)[NS]) {
          Mat::committed_tangent(pr.p.mat, state_of<Elem>(sc, e, P).point(gp), D);
        };
        const Rayleigh none{};
        const OperatorCoefficients K{1.0, 0.0, 0.0}, M{0.0, 0.0, 1.0};
        double kx[NDOF], mx[NDOF], kb[NDOF], mb[NDOF];
        if (Cached) {
          const auto g = cached_geometry<Elem>(geo, cst, e, P);
          element_operator_apply<Elem>(pr.p, none, K, g, tangent_at, committed_at, ue_x, kx);
          element_operator_apply<Elem>(pr.p, none, M, g, tangent_at, committed_at, ae_x, mx);
          element_operator_apply<Elem>(pr.p, none, K, g, tangent_at, committed_at, ue_b, kb);
          element_operator_apply<Elem>(pr.p, none, M, g, tangent_at, committed_at, ae_b, mb);
        } else {
          const auto g = Elem::recomputed_geometry(xl, pr.p);
          element_operator_apply<Elem>(pr.p, none, K, g, tangent_at, committed_at, ue_x, kx);
          element_operator_apply<Elem>(pr.p, none, M, g, tangent_at, committed_at, ae_x, mx);
          element_operator_apply<Elem>(pr.p, none, K, g, tangent_at, committed_at, ue_b, kb);
          element_operator_apply<Elem>(pr.p, none, M, g, tangent_at, committed_at, ae_b, mb);
        }
        for (int a = 0; a < NDOF; ++a) {
          const bool boundary = ((mask >> (a / 3)) & 1u) != 0;
          // OpenSees' order: Peff_b = 0 - M_be a_e, then - K_be u_e; Peff_e = M_eb a_b + K_eb u_b.
          const double v = boundary ? (-mx[a] - kx[a]) : (mb[a] + kb[a]);
          slots(base + static_cast<std::int64_t>(a) * n + i) = v;
        }
      });
}

// A material whose response is linear but has state (Uniform damping on an elastic material:
// the state is the damping's filters): the DRM forces need the element's own force, not its
// constant tangent (docs/design.md, "Damped DRM layer").
template <class Mat>
inline constexpr bool kLinearWithState = [] {
  if constexpr (requires { typename Mat::Base; }) return Mat::Base::NSTATE == 0 && Mat::NSTATE > 0;
  else return false;
}();

// H5DRM effective forces of the DRM elements of a batch whose material is linear with state:
// the same split as launch_drm_forces, but y_e = F(u_e, v_e) + M a_e with F the element's own
// force (compute<false>: stiffness, Rayleigh damping, the material's damping stress) for the
// free field restricted to the exterior nodes, from its own state (sc / st: committed and
// trial, [state of DRM element i on the exterior side, then of i + n on the boundary side],
// the per-point layout of state_of with 2 n elements); y_b likewise. No body force.
template <class Elem>
void launch_drm_forces_compute(const DevBatch<Elem>& b, const Vec& coords, Index N,
                               const Kokkos::View<Index*, Space>& elems,
                               const Kokkos::View<std::uint32_t*, Space>& masks,
                               const Kokkos::View<Index*, Space>& node_slot, const Vec& du, const Vec& da,
                               const Vec& dv, const Vec& sc, const Vec& st, const Vec& slots, std::int64_t base) {
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM;
  const auto node = b.node, param = b.param;
  const auto params = b.params;
  const StepContext step = b.step;
  const Index P = b.padded;
  const Index n = static_cast<Index>(elems.extent(0));
  if (n == 0) return;
  const bool has_v = dv.extent(0) > 0;
  Kokkos::parallel_for(
      "element_drm_forces_compute", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, n),
      KOKKOS_LAMBDA(const Index i) {
        const Index e = elems(i);
        const std::uint32_t mask = masks(i);
        const DevParams<Elem>& pr = params(param(e));
        typename Elem::Params pz = pr.p;  // the DRM forces carry no body force
        pz.b[0] = pz.b[1] = pz.b[2] = 0.0;
        double ue_x[NDOF], ve_x[NDOF], ae_x[NDOF], ue_b[NDOF], ve_b[NDOF], ae_b[NDOF], xe[NNODE][NDM];
        for (int a = 0; a < NNODE; ++a) {
          const Index nd = node(a * P + e);
          for (int d = 0; d < NDM; ++d) xe[a][d] = coords(d * N + nd);
          const std::int64_t sl = 3 * static_cast<std::int64_t>(node_slot(nd));
          const bool boundary = ((mask >> a) & 1u) != 0;
          for (int c = 0; c < 3; ++c) {
            const double uu = du(sl + c), aa = da(sl + c), vv = has_v ? dv(sl + c) : 0.0;
            ue_x[3 * a + c] = boundary ? 0.0 : uu;
            ve_x[3 * a + c] = boundary ? 0.0 : vv;
            ae_x[3 * a + c] = boundary ? 0.0 : aa;
            ue_b[3 * a + c] = boundary ? uu : 0.0;
            ve_b[3 * a + c] = boundary ? vv : 0.0;
            ae_b[3 * a + c] = boundary ? aa : 0.0;
          }
        }
        double fx[NDOF], fb[NDOF];
        Elem::template compute<false>(pz, with_step(pr.damp, step), xe, ue_x, ve_x, state_of<Elem>(sc, i, 2 * n),
                                      state_of<Elem>(st, i, 2 * n), fx, nullptr);
        Elem::template compute<false>(pz, with_step(pr.damp, step), xe, ue_b, ve_b,
                                      state_of<Elem>(sc, n + i, 2 * n), state_of<Elem>(st, n + i, 2 * n), fb,
                                      nullptr);
        double Me[NDOF * NDOF];
        Elem::mass(pz, xe, Me);
        for (int a = 0; a < NDOF; ++a) {
          double mx = 0.0, mb = 0.0;
          for (int k = 0; k < NDOF; ++k) {
            mx += Me[a * NDOF + k] * ae_x[k];
            mb += Me[a * NDOF + k] * ae_b[k];
          }
          const bool boundary = ((mask >> (a / 3)) & 1u) != 0;
          slots(base + static_cast<std::int64_t>(a) * n + i) = boundary ? (-mx - fx[a]) : (mb + fb[a]);
        }
      });
}

// State determination for the Stored source: D at every point at displacement u.
template <class Elem, bool Cached>
void launch_store_tangents(const DevBatch<Elem>& b, const DevTangents<Elem>& t, const Vec& coords, Index N,
                           const Vec& u) {
  using Mat = typename Elem::Material;
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM, NS = Mat::NSTRAIN;
  const auto node = b.node, eq = b.eq, param = b.param;
  const auto params = b.params;
  const auto geo = b.geometry;
  const auto ts = t.stored;
  const auto sc = b.state[b.current], st = b.state[1 - b.current];
  const Index P = b.padded;
  const StepContext step = b.step;
  Kokkos::parallel_for(
      "element_store_tangents", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count),
      KOKKOS_LAMBDA(const Index e) {
        const DevParams<Elem>& pr = params(param(e));
        double ue[NDOF];
        for (int a = 0; a < NDOF; ++a) {
          const Index q = eq(a * P + e);
          ue[a] = q >= 0 ? u(q) : 0.0;
        }
        double xel[NNODE][NDM], xl[3][8];
        if (!Cached) {
          for (int a = 0; a < NNODE; ++a) {
            const Index n = node(a * P + e);
            for (int d = 0; d < NDM; ++d) xel[a][d] = coords(d * N + n);
          }
          Elem::local_coords(xel, xl);
        }
        const PointArray<double> out{ts.data() + static_cast<std::ptrdiff_t>(e) * Elem::NGP, Elem::NGP * P, 1};
        for (int gp = 0; gp < Elem::NGP; ++gp) {
          typename Elem::PointShape s;
          if (Cached) Elem::point_load_geometry(gp, cache_of<Elem>(geo, e, P).point(gp), s);
          else Elem::point_geometry(xl, gp, s);
          double D[NS][NS];
          material_tangent<Elem>(step, pr.p, s, ue, state_of<Elem>(sc, e, P).point(gp), state_of<Elem>(st, e, P).point(gp),
                                 D);
          Storage<Elem>::store(D, out.point(gp));
        }
      });
}

}  // namespace femorax::exec::kernels
