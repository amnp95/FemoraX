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
KOKKOS_INLINE_FUNCTION void material_tangent(const typename Elem::Params& p, const typename Elem::PointShape& s,
                                             const double* ue, Strided<const double> committed, Strided<double> trial,
                                             double (*D)[Elem::Material::NSTRAIN]) {
  using Mat = typename Elem::Material;
  double eps[Mat::NSTRAIN], sig[Mat::NSTRAIN];
  Elem::point_strain(s, ue, eps);
  Mat::update(p.mat, committed, trial, eps, sig, D);
}

// The recomputed source: materials without history state only (InternalForce::resolve
// falls back to the stored source for the others), so no state is passed. For a material
// with state the branch is never taken; it is not compiled either, which keeps the compiler
// from analysing an update on empty state views.
template <class Elem>
KOKKOS_INLINE_FUNCTION void recomputed_tangent(const typename Elem::Params& p, const typename Elem::PointShape& s,
                                               const double* ue, double (*D)[Elem::Material::NSTRAIN]) {
  if constexpr (Elem::Material::NSTATE == 0) {
    material_tangent<Elem>(p, s, ue, {}, {}, D);
  } else {
    (void)p, (void)s, (void)ue, (void)D;
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
  const auto node = b.node, eq = b.eq, param = b.param, colors = b.color_elements;
  const auto params = b.params;
  const auto geo = b.geometry, cst = b.constants;
  const auto tc = t.constant, ts = t.stored;
  const auto sc = b.state[b.current];
  const Index P = b.padded;
  const std::int64_t base = b.slot_offset;
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
          if (unit_dir >= 0) xe_e[a] = (a % Elem::NODE_NDF[0]) == unit_dir ? 1.0 : 0.0;
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
            recomputed_tangent<Elem>(pr.p, s, ue, D);
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
          material_tangent<Elem>(pr.p, s, ue, state_of<Elem>(sc, e, P).point(gp), state_of<Elem>(st, e, P).point(gp), D);
          Storage<Elem>::store(D, out.point(gp));
        }
      });
}

}  // namespace femorax::exec::kernels
