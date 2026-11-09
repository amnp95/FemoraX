#pragma once

// The compacted plastic-corrector path (docs/design.md, "Hard parts"; milestone 2): for an
// element batch whose material splits into an elastic predictor and a plastic corrector
// (SplitMaterial, PressureDependMultiYield), the force runs in four launches instead of one
// fused kernel:
//
//   1. predict  one thread per point: gather, strain, Mat::predict; a point that stays
//               elastic is finished (stress, tangent, trial state), a plastic one is
//               flagged; every point stores its strain;
//   2. compact  a scan of the flags lists the plastic points (one count to the host);
//   3. correct  one thread per listed point: Mat::correct (the sub-increment loop), so the
//               branchy, expensive work runs on full warps of plastic points only;
//   4. force    the element force kernel (any layout, scatter, geometry mode) with the
//               element instantiated on Replay<Mat>, whose update reads the stored stress
//               and tangent instead of computing them.
//
// Per point it stores the stress (6), the tangent (36) and the strain (6), in the per-point
// layout of the material state (value[k * (NGP * padded) + e * NGP + gp]), plus a flag and a
// list entry: 48 doubles and 8 bytes per point, allocated with the batch.
//
// Same bits as the fused path where the compiler does not contract (host backends: predict
// + correct give update's bits, tested in test_pdmy_reference, and Replay hands over the
// values unchanged); on GPUs the corrector is a separate kernel, which nvcc may fuse
// differently, so there it agrees to rounding (and plastic decisions may flip), like the
// two thread layouts. Kept as a variant for the TACC benchmarks; the auto-tune considers it
// where it is bit-identical.
//
// Included only by exec .cpp files (device lambdas).

#include <Kokkos_Core.hpp>

#include <cstdint>

#include "exec/element_kernels.hpp"
#include "physics/contracts.hpp"

namespace femorax::exec::kernels {

// The stored results of one point, as a material: update copies stress and tangent from
// the "trial" buffer the force kernel hands it (the results buffer); everything else is
// the real material's.
template <class Mat>
struct Replay {
  static constexpr MaterialKind KIND = Mat::KIND;
  static constexpr int NSTRAIN = Mat::NSTRAIN;
  static constexpr int NSTATE = NSTRAIN + NSTRAIN * NSTRAIN + NSTRAIN;  // stress, tangent, strain
  static constexpr int R_STRESS = 0, R_TANGENT = NSTRAIN, R_STRAIN = NSTRAIN + NSTRAIN * NSTRAIN;
  static constexpr TangentSymmetry TANGENT = Mat::TANGENT;
  static constexpr bool CONSTANT_TANGENT = Mat::CONSTANT_TANGENT;
  using Params = typename Mat::Params;

  FEMORAX_HD static void update(const Params&, Strided<const double>, Strided<double> results, const double*,
                                double sig[NSTRAIN], double (*D)[NSTRAIN]) {
    for (int a = 0; a < NSTRAIN; ++a) sig[a] = results[R_STRESS + a];
    for (int a = 0; a < NSTRAIN; ++a)
      for (int c = 0; c < NSTRAIN; ++c) D[a][c] = results[R_TANGENT + NSTRAIN * a + c];
  }
  FEMORAX_HD static void initial_tangent(const Params& p, double (*D)[NSTRAIN]) { Mat::initial_tangent(p, D); }
  FEMORAX_HD static void committed_tangent(const Params& p, Strided<const double> committed, double (*D)[NSTRAIN]) {
    Mat::committed_tangent(p, committed, D);
  }
  FEMORAX_HD static double density(const Params& p) { return Mat::density(p); }
};

template <class Elem>
using ReplayElement = typename Elem::template WithMaterial<Replay<typename Elem::Material>>;

template <class Elem>
inline constexpr bool kCompactable = SplitMaterial<typename Elem::Material> && RebindableElement<Elem> &&
                                     PointwiseElement<Elem>;

// The compacted path's device data of one batch.
template <class Elem>
struct DevCompact {
  Vec results;                               // Replay::NSTATE per point
  Kokkos::View<std::int32_t*, Space> flag;   // 1: the point needs the corrector
  Kokkos::View<std::int32_t*, Space> list;   // the plastic points (e * NGP + gp)
  Kokkos::View<DevParams<ReplayElement<Elem>>*, Space> params;  // the batch's parameter sets, retyped
};

// 1. Predictor: one thread per point (consecutive threads take the points of one element,
// as the team layout does; each gathers its element's nodal values).
template <class Elem, bool Cached>
void launch_predict(const DevBatch<Elem>& b, const DevCompact<Elem>& d, const Vec& coords, Index N, const Vec& u) {
  using Mat = typename Elem::Material;
  using R = Replay<Mat>;
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM, NGP = Elem::NGP, NS = Mat::NSTRAIN;
  const auto node = b.node, eq = b.eq, param = b.param;
  const auto params = b.params;
  const auto geo = b.geometry;
  const auto sc = b.state[b.current], st = b.state[1 - b.current];
  const auto res = d.results;
  const auto flag = d.flag;
  const Index P = b.padded;
  Kokkos::parallel_for(
      "element_predict", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count * NGP),
      KOKKOS_LAMBDA(const Index i) {
        const Index e = i / NGP;
        const int gp = static_cast<int>(i - e * NGP);
        double ue[NDOF];
        for (int k = 0; k < NDOF; ++k) {
          const Index q = eq(k * P + e);
          ue[k] = q >= 0 ? u(q) : 0.0;
        }
        typename Elem::PointShape s;
        if (Cached) {  // a constant (template argument); nvcc cannot capture inside if constexpr
          Elem::point_load_geometry(gp, cache_of<Elem>(geo, e, P).point(gp), s);
        } else {
          double xe[NNODE][NDM], xl[NDM][NNODE];
          for (int a = 0; a < NNODE; ++a) {
            const Index n = node(a * P + e);
            for (int dd = 0; dd < NDM; ++dd) xe[a][dd] = coords(dd * N + n);
          }
          Elem::local_coords(xe, xl);
          Elem::point_geometry(xl, gp, s);
        }
        double eps[NS], sig[NS], D[NS][NS];
        Elem::point_strain(s, ue, eps);
        const Strided<double> o = state_of<Elem>(res, e, P).point(gp);
        for (int a = 0; a < NS; ++a) o[R::R_STRAIN + a] = eps[a];
        const bool plastic = Mat::predict(SerialInner{}, params(param(e)).p.mat, state_of<Elem>(sc, e, P).point(gp),
                                          state_of<Elem>(st, e, P).point(gp), eps, sig, D);
        flag(i) = plastic ? 1 : 0;
        if (!plastic) {
          for (int a = 0; a < NS; ++a) o[R::R_STRESS + a] = sig[a];
          for (int a = 0; a < NS; ++a)
            for (int c = 0; c < NS; ++c) o[R::R_TANGENT + NS * a + c] = D[a][c];
        }
      });
}

// 2. Compaction: the plastic points, in point order. Returns their number.
template <class Elem>
Index compact_plastic_points(const DevBatch<Elem>& b, const DevCompact<Elem>& d) {
  const auto flag = d.flag;
  const auto list = d.list;
  Index total = 0;
  Kokkos::parallel_scan(
      "plastic_points", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count * Elem::NGP),
      KOKKOS_LAMBDA(const Index i, Index& offset, const bool final) {
        if (flag(i)) {
          if (final) list(offset) = i;
          ++offset;
        }
      },
      total);
  return total;
}

// 3. Corrector: one thread per plastic point.
template <class Elem>
void launch_correct(const DevBatch<Elem>& b, const DevCompact<Elem>& d, Index count) {
  using Mat = typename Elem::Material;
  using R = Replay<Mat>;
  constexpr int NGP = Elem::NGP, NS = Mat::NSTRAIN;
  if (count <= 0) return;
  const auto param = b.param;
  const auto params = b.params;
  const auto sc = b.state[b.current], st = b.state[1 - b.current];
  const auto res = d.results;
  const auto list = d.list;
  const Index P = b.padded;
  Kokkos::parallel_for(
      "element_correct", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, count),
      KOKKOS_LAMBDA(const Index i) {
        const Index pt = list(i), e = pt / NGP;
        const int gp = static_cast<int>(pt - e * NGP);
        const Strided<double> o = state_of<Elem>(res, e, P).point(gp);
        double eps[NS], sig[NS], D[NS][NS];
        for (int a = 0; a < NS; ++a) eps[a] = o[R::R_STRAIN + a];
        Mat::correct(SerialInner{}, params(param(e)).p.mat, state_of<Elem>(sc, e, P).point(gp),
                     state_of<Elem>(st, e, P).point(gp), eps, sig, D);
        for (int a = 0; a < NS; ++a) o[R::R_STRESS + a] = sig[a];
        for (int a = 0; a < NS; ++a)
          for (int c = 0; c < NS; ++c) o[R::R_TANGENT + NS * a + c] = D[a][c];
      });
}

// 4. The batch as the force kernels see it with the Replay material: the same maps,
// geometry and constants; the committed state is the real one, the "trial" buffer the
// results.
template <class Elem>
DevBatch<ReplayElement<Elem>> replay_batch(const DevBatch<Elem>& b, const DevCompact<Elem>& d) {
  DevBatch<ReplayElement<Elem>> r;
  r.count = b.count;
  r.padded = b.padded;
  r.node = b.node;
  r.eq = b.eq;
  r.param = b.param;
  r.color_elements = b.color_elements;
  r.color_start = b.color_start;
  r.params = d.params;
  r.geometry = b.geometry;
  r.constants = b.constants;
  r.current = 0;
  r.state[0] = b.state[b.current];
  r.state[1] = d.results;
  r.slot_offset = b.slot_offset;
  return r;
}

}  // namespace femorax::exec::kernels
