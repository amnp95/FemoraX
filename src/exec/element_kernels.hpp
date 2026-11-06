#pragma once

// Composable element kernels: one template per thread layout, generic over the element
// type (any registered element), the geometry mode (template bool) and the output mode
// (exec/assembly.hpp). A new element gets every variant without new kernel code; the team
// layout needs the PointwiseElement interface and the same DOF count at every node.
//
// Included only by exec .cpp files (it contains device lambdas).

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "exec/assembly.hpp"
#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"

namespace femorax::exec::kernels {

using Index = std::int32_t;

template <class Elem>
struct DevParams {
  typename Elem::Params p;
  Rayleigh damp;
};

// Device data of one batch. Plain structs (not class members): nvcc does not allow private
// types in device lambdas.
template <class Elem>
struct DevBatch {
  Index count = 0, padded = 0;
  Kokkos::View<Index*, Space> node, eq, param, color_elements;
  std::vector<Index> color_start;  // host: launch bounds per color
  Kokkos::View<DevParams<Elem>*, Space> params;
  // Geometry cache, per Gauss point: value[k * (NGP * padded) + e * NGP + gp], k < NGEOM.
  Vec geometry;
  // Slot of element e's k-th value: slot_offset + k * padded + e.
  std::int64_t slot_offset = 0;
};

// Team-layout scratch per element, in doubles: coordinates, u, v, per (point, node) the
// three contributions (internal, body, damping) per node DOF, and the element force.
template <class Elem>
struct TeamScratch {
  static constexpr int NDFN = Elem::NODE_NDF[0];
  static constexpr int X = Elem::NNODE * Elem::NDM;
  static constexpr int U = X, V = U + Elem::NDOF, C = V + Elem::NDOF;
  static constexpr int CONTRIB = Elem::NGP * Elem::NNODE * 3 * NDFN;
  static constexpr int FE = C + CONTRIB;
  static constexpr int PER_ELEMENT = FE + Elem::NDOF;
};

template <class Elem>
KOKKOS_INLINE_FUNCTION PointArray<const double> cache_of(const Vec& g, Index e, Index padded) {
  return {g.data() + static_cast<std::ptrdiff_t>(e) * Elem::NGP, Elem::NGP * padded, 1};
}

// Geometry cache on the device, by the same code path as recomputing.
template <class Elem>
void fill_geometry(const DevBatch<Elem>& b, const Vec& coords, Index N) {
  const auto node = b.node;
  const auto g = b.geometry;
  const Index P = b.padded;
  Kokkos::parallel_for(
      "element_geometry", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count),
      KOKKOS_LAMBDA(const Index e) {
        double xe[Elem::NNODE][Elem::NDM], xl[Elem::NDM][Elem::NNODE];
        for (int a = 0; a < Elem::NNODE; ++a) {
          const Index n = node(a * P + e);
          for (int d = 0; d < Elem::NDM; ++d) xe[a][d] = coords(d * N + n);
        }
        Elem::local_coords(xe, xl);
        const PointArray<double> out{g.data() + static_cast<std::ptrdiff_t>(e) * Elem::NGP,
                                     Elem::NGP * P, 1};
        for (int gp = 0; gp < Elem::NGP; ++gp) Elem::point_store_geometry(xl, gp, out.point(gp));
      });
}

// One thread per element.
template <class Elem, bool Cached>
void launch_flat(const DevBatch<Elem>& b, const Vec& coords, Index N, const Vec& u, const Vec& v,
                 const Vec& f, const Vec& slots, Out out, Index begin, Index end, bool by_color) {
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM;
  const auto node = b.node, eq = b.eq, param = b.param, colors = b.color_elements;
  const auto params = b.params;
  const auto geo = b.geometry;
  const Index P = b.padded;
  const std::int64_t base = b.slot_offset;
  Kokkos::parallel_for(
      "element_force_flat", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(begin, end),
      KOKKOS_LAMBDA(const Index i) {
        const Index e = by_color ? colors(i) : i;
        double ue[NDOF], ve[NDOF], fe[NDOF];
        for (int k = 0; k < NDOF; ++k) {
          const Index q = eq(k * P + e);
          ue[k] = q >= 0 ? u(q) : 0.0;
          ve[k] = q >= 0 ? v(q) : 0.0;
        }
        const DevParams<Elem>& pr = params(param(e));
        if (Cached) {  // a constant (template argument); nvcc cannot capture inside if constexpr
          Elem::template compute_cached<false>(pr.p, pr.damp, cache_of<Elem>(geo, e, P), ue, ve,
                                               {}, {}, fe, nullptr);
        } else {
          double xe[NNODE][NDM];
          for (int a = 0; a < NNODE; ++a) {
            const Index n = node(a * P + e);
            for (int d = 0; d < NDM; ++d) xe[a][d] = coords(d * N + n);
          }
          Elem::template compute<false>(pr.p, pr.damp, xe, ue, ve, {}, {}, fe, nullptr);
        }
        for (int k = 0; k < NDOF; ++k)
          emit(out, f, slots, base + static_cast<std::int64_t>(k) * P + e, eq(k * P + e), fe[k]);
      });
}

// One thread per Gauss point, ept elements per team.
template <class Elem, bool Cached>
void launch_team(const DevBatch<Elem>& b, const Vec& coords, Index N, const Vec& u, const Vec& v,
                 const Vec& f, const Vec& slots, Out out, Index begin, Index end, bool by_color,
                 int ept) {
  static_assert(PointwiseElement<Elem>, "the team layout needs the pointwise element interface");
  using S = TeamScratch<Elem>;
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM, NGP = Elem::NGP;
  constexpr int NDFN = S::NDFN, NSTRESS = Elem::Material::NSTRAIN;
  using Policy = Kokkos::TeamPolicy<Space>;
  using Member = Policy::member_type;
  using Scratch = Kokkos::View<double*, Space::scratch_memory_space, Kokkos::MemoryUnmanaged>;

  const Index n = end - begin;
  if (n <= 0) return;
  const int league = (n + ept - 1) / ept;
  const auto node = b.node, eq = b.eq, param = b.param, colors = b.color_elements;
  const auto params = b.params;
  const auto geo = b.geometry;
  const Index P = b.padded;
  const std::int64_t base = b.slot_offset;

  Policy probe(league, Kokkos::AUTO);
  const int team_size = std::min(
      ept * NGP, probe.team_size_max(KOKKOS_LAMBDA(const Member&){}, Kokkos::ParallelForTag()));
  Policy policy(league, std::max(team_size, 1));
  policy.set_scratch_size(0, Kokkos::PerTeam(static_cast<std::size_t>(ept) * S::PER_ELEMENT * sizeof(double)));

  Kokkos::parallel_for(
      "element_force_team", policy, KOKKOS_LAMBDA(const Member& m) {
        Scratch buf(m.team_scratch(0), static_cast<std::size_t>(ept) * S::PER_ELEMENT);
        const Index first = begin + m.league_rank() * ept;
        auto element = [&](int slot) -> Index {
          const Index i = first + slot;
          if (i >= end) return -1;
          return by_color ? colors(i) : i;
        };

        // 1. Gather: work item (slot, a) loads node a of element slot into scratch.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NNODE), [&](int t) {
          const int slot = t / NNODE, a = t % NNODE;
          const Index e = element(slot);
          if (e < 0) return;
          double* g = &buf(slot * S::PER_ELEMENT);
          if (!Cached) {
            const Index nd = node(a * P + e);
            for (int d = 0; d < NDM; ++d) g[S::X * 0 + NDM * a + d] = coords(d * N + nd);
          }
          for (int d = 0; d < NDFN; ++d) {
            const Index q = eq((NDFN * a + d) * P + e);
            g[S::U + NDFN * a + d] = q >= 0 ? u(q) : 0.0;
            g[S::V + NDFN * a + d] = q >= 0 ? v(q) : 0.0;
          }
        });
        m.team_barrier();

        // 2. Points: work item (slot, gp) updates point gp and writes its contribution to
        // every node.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NGP), [&](int t) {
          const int slot = t / NGP, gp = t % NGP;
          const Index e = element(slot);
          if (e < 0) return;
          double* g = &buf(slot * S::PER_ELEMENT);
          const double* ue = g + S::U;
          const double* ve = g + S::V;
          const DevParams<Elem>& pr = params(param(e));
          typename Elem::PointShape s;
          if (Cached) {  // a constant (template argument); nvcc cannot capture inside if constexpr
            Elem::point_load_geometry(gp, cache_of<Elem>(geo, e, P).point(gp), s);
          } else {
            double xl[NDM][NNODE];
            Elem::local_coords(reinterpret_cast<const double(*)[NDM]>(g), xl);
            Elem::point_geometry(xl, gp, s);
          }
          double D0[NSTRESS][NSTRESS];
          if (pr.damp.betaKinit != 0.0) Elem::Material::initial_tangent(pr.p.mat, D0);
          typename Elem::PointResult r;
          double dd[NSTRESS][NSTRESS];
          Elem::point_update_shape(pr.p, pr.damp, s, ue, ve, {}, {}, D0, r, dd);
          double* c = g + S::C + gp * NNODE * 3 * NDFN;
          for (int j = 0; j < NNODE; ++j)
            Elem::point_node_contribution(pr.p, pr.damp, r, ve, j, c + 3 * NDFN * j,
                                          c + 3 * NDFN * j + NDFN, c + 3 * NDFN * j + 2 * NDFN);
        });
        m.team_barrier();

        // 3. Nodes: work item (slot, j) adds node j's contributions in Gauss point order
        // (the order of Elem::compute).
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NNODE), [&](int t) {
          const int slot = t / NNODE, j = t % NNODE;
          if (element(slot) < 0) return;
          double* g = &buf(slot * S::PER_ELEMENT);
          const double* c = g + S::C;
          for (int q3 = 0; q3 < NDFN; ++q3) {
            double resid = 0.0, fdamp = 0.0;
            for (int gp = 0; gp < NGP; ++gp) {
              const double* cj = c + gp * NNODE * 3 * NDFN + 3 * NDFN * j;
              resid += cj[q3];
              resid -= cj[NDFN + q3];
              fdamp += cj[2 * NDFN + q3];
            }
            g[S::FE + NDFN * j + q3] = resid + fdamp;
          }
        });
        m.team_barrier();

        // 4. Out: one work item per element, its values in DOF order (as the flat kernel).
        // Tied nodes of one element share equations, so its writes must not run concurrently.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept), [&](int slot) {
          const Index e = element(slot);
          if (e < 0) return;
          const double* fe = &buf(slot * S::PER_ELEMENT) + S::FE;
          for (int k = 0; k < NDOF; ++k)
            emit(out, f, slots, base + static_cast<std::int64_t>(k) * P + e, eq(k * P + e), fe[k]);
        });
      });
}

// Smallest stable explicit step over the batch's elements (Elem::critical_dt; committed
// state empty: elements without history state so far).
template <class Elem>
double min_critical_dt(const DevBatch<Elem>& b, const Vec& coords, Index N, ExplicitScheme scheme) {
  const auto node = b.node, param = b.param;
  const auto params = b.params;
  const Index P = b.padded;
  double result = 0.0;
  Kokkos::parallel_reduce(
      "element_critical_dt", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count),
      KOKKOS_LAMBDA(const Index e, double& m) {
        double xe[Elem::NNODE][Elem::NDM];
        for (int a = 0; a < Elem::NNODE; ++a) {
          const Index n = node(a * P + e);
          for (int d = 0; d < Elem::NDM; ++d) xe[a][d] = coords(d * N + n);
        }
        const DevParams<Elem>& pr = params(param(e));
        const double dt = Elem::critical_dt(pr.p, pr.damp, scheme, xe, {});
        if (dt < m) m = dt;
      },
      Kokkos::Min<double>(result));
  return result;
}

}  // namespace femorax::exec::kernels
