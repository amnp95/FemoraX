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
#include "physics/common/element_term.hpp"
#include "physics/common/inner.hpp"
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
  // Element constants (ElementTermElement), per element: value[k * padded + e], k < NCONST.
  // Built with the geometry cache; without it the kernels recompute them.
  Vec constants;
  // Material history state, two buffers (committed and trial) of Mat::NSTATE values per
  // Gauss point: value[k * (NGP * padded) + e * NGP + gp]. state[current] is committed;
  // commit swaps the roles (the update writes every trial value). Empty for materials
  // without state.
  Vec state[2];
  int current = 0;
  // Slot of element e's k-th value: slot_offset + k * padded + e.
  std::int64_t slot_offset = 0;
};

// Whether the element has an element-level stiffness term (physics/contracts.hpp).
template <class Elem>
inline constexpr bool kHasTerm = ElementTermElement<Elem>;

// Team-layout scratch per element, in doubles: coordinates, u, v, per (point, node) the
// three contributions (internal, body, damping) per node DOF, the element force, and for
// elements with an element-level term its initial internal and damping force (R0).
template <class Elem>
struct TeamScratch {
  static constexpr int NDFN = Elem::NODE_NDF[0];
  static constexpr int X = Elem::NNODE * Elem::NDM;
  static constexpr int U = X, V = U + Elem::NDOF, C = V + Elem::NDOF;
  static constexpr int CONTRIB = Elem::NGP * Elem::NNODE * 3 * NDFN;
  static constexpr int FE = C + CONTRIB;
  static constexpr int R0 = FE + Elem::NDOF;
  static constexpr int PER_ELEMENT = R0 + (kHasTerm<Elem> ? 2 * Elem::NDOF : 0);
};

template <class Elem>
KOKKOS_INLINE_FUNCTION PointArray<const double> cache_of(const Vec& g, Index e, Index padded) {
  return {g.data() + static_cast<std::ptrdiff_t>(e) * Elem::NGP, Elem::NGP * padded, 1};
}

// Element e's points in a state buffer (the per-point layout of the geometry cache).
template <class Elem>
KOKKOS_INLINE_FUNCTION PointArray<double> state_of(const Vec& s, Index e, Index padded) {
  return {s.data() + static_cast<std::ptrdiff_t>(e) * Elem::NGP, Elem::NGP * padded, 1};
}

KOKKOS_INLINE_FUNCTION Strided<const double> constants_of(const Vec& c, Index e, Index padded) {
  return {c.data() + e, padded};
}

// The element's cached geometry source: the geometry cache and the element constants.
template <class Elem>
KOKKOS_INLINE_FUNCTION typename Elem::CachedGeometry cached_geometry(const Vec& g, const Vec& c, Index e,
                                                                     Index padded) {
  return Elem::cached_geometry(cache_of<Elem>(g, e, padded), constants_of(c, e, padded));
}

// K_el's initial internal and damping force of one element (element_term_force_node at
// every node) into r0[0..NDOF) and r0[NDOF..2 NDOF). Nothing for elements without a term.
template <class Elem, class Geometry>
KOKKOS_INLINE_FUNCTION void element_term_initial(const DevParams<Elem>& pr, const Geometry& geo,
                                                 const double* ue, const double* ve, double* r0) {
  if constexpr (kHasTerm<Elem>) {
    const Strided<const double> c = geo.constants();
    double zu[Elem::NTERM], zv[Elem::NTERM];
    Elem::element_term_reduce(pr.p, c, ue, zu);
    Elem::element_term_reduce(pr.p, c, ve, zv);
    for (int j = 0; j < Elem::NNODE; ++j)
      element_term_force_node<Elem>(pr.p, pr.damp, c, zu, zv, j, r0 + 3 * j, r0 + Elem::NDOF + 3 * j);
  } else {
    (void)pr;
    (void)geo;
    (void)ue;
    (void)ve;
    (void)r0;
  }
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

// Element constants on the device, by the same function as recomputing them.
template <class Elem>
void fill_constants(const DevBatch<Elem>& b, const Vec& coords, Index N) {
  if constexpr (kHasTerm<Elem>) {
    const auto node = b.node, param = b.param;
    const auto params = b.params;
    const auto c = b.constants;
    const Index P = b.padded;
    Kokkos::parallel_for(
        "element_constants", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count),
        KOKKOS_LAMBDA(const Index e) {
          double xe[Elem::NNODE][Elem::NDM], xl[Elem::NDM][Elem::NNODE];
          for (int a = 0; a < Elem::NNODE; ++a) {
            const Index n = node(a * P + e);
            for (int d = 0; d < Elem::NDM; ++d) xe[a][d] = coords(d * N + n);
          }
          Elem::local_coords(xe, xl);
          Elem::element_constants(params(param(e)).p, xl, Strided<double>{c.data() + e, P});
        });
  } else {
    (void)b;
    (void)coords;
    (void)N;
  }
}

// One thread per element.
template <class Elem, bool Cached>
void launch_flat(const DevBatch<Elem>& b, const Vec& coords, Index N, const Vec& u, const Vec& v,
                 const Vec& f, const Vec& slots, Out out, Index begin, Index end, bool by_color) {
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM;
  const auto node = b.node, eq = b.eq, param = b.param, colors = b.color_elements;
  const auto params = b.params;
  const auto geo = b.geometry, cst = b.constants;
  const auto sc = b.state[b.current], st = b.state[1 - b.current];
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
          typename Elem::Work work;
          Elem::template compute_with<false>(SerialInner{}, cached_geometry<Elem>(geo, cst, e, P), pr.p, pr.damp,
                                             ue, ve, state_of<Elem>(sc, e, P), state_of<Elem>(st, e, P), fe,
                                             nullptr, work);
        } else {
          double xe[NNODE][NDM];
          for (int a = 0; a < NNODE; ++a) {
            const Index n = node(a * P + e);
            for (int d = 0; d < NDM; ++d) xe[a][d] = coords(d * N + n);
          }
          Elem::template compute<false>(pr.p, pr.damp, xe, ue, ve, state_of<Elem>(sc, e, P), state_of<Elem>(st, e, P),
                                        fe, nullptr);
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
  const auto geo = b.geometry, cst = b.constants;
  const auto sc = b.state[b.current], st = b.state[1 - b.current];
  const Index P = b.padded;
  const std::int64_t base = b.slot_offset;
  constexpr bool HAS_TERM = kHasTerm<Elem>;

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
          Elem::point_update_shape(pr.p, pr.damp, s, ue, ve, state_of<Elem>(sc, e, P).point(gp),
                                   state_of<Elem>(st, e, P).point(gp), D0, r, dd);
          double* c = g + S::C + gp * NNODE * 3 * NDFN;
          for (int j = 0; j < NNODE; ++j)
            Elem::point_node_contribution(pr.p, pr.damp, r, ve, j, c + 3 * NDFN * j,
                                          c + 3 * NDFN * j + NDFN, c + 3 * NDFN * j + 2 * NDFN);
        });
        // 2b. Element term (elements with one): work item slot writes K_el's initial
        // internal and damping force (the values Elem::compute starts from).
        if (HAS_TERM) {  // a constant; nvcc cannot capture inside if constexpr
          Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept), [&](int slot) {
            const Index e = element(slot);
            if (e < 0) return;
            double* g = &buf(slot * S::PER_ELEMENT);
            const DevParams<Elem>& pr = params(param(e));
            if (Cached) {
              element_term_initial<Elem>(pr, cached_geometry<Elem>(geo, cst, e, P), g + S::U, g + S::V, g + S::R0);
            } else {
              double xl[NDM][NNODE];
              Elem::local_coords(reinterpret_cast<const double(*)[NDM]>(g), xl);
              element_term_initial<Elem>(pr, Elem::recomputed_geometry(xl, pr.p), g + S::U, g + S::V, g + S::R0);
            }
          });
        }
        m.team_barrier();

        // 3. Nodes: work item (slot, j) adds node j's contributions in Gauss point order
        // (the order of Elem::compute), after the element term's initial values.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NNODE), [&](int t) {
          const int slot = t / NNODE, j = t % NNODE;
          if (element(slot) < 0) return;
          double* g = &buf(slot * S::PER_ELEMENT);
          const double* c = g + S::C;
          for (int q3 = 0; q3 < NDFN; ++q3) {
            double resid = 0.0, fdamp = 0.0;
            if (HAS_TERM) {
              resid = g[S::R0 + NDFN * j + q3];
              fdamp = g[S::R0 + NDOF + NDFN * j + q3];
            }
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

// Smallest stable explicit step over the batch's elements (Elem::critical_dt at the
// committed state).
template <class Elem>
double min_critical_dt(const DevBatch<Elem>& b, const Vec& coords, Index N, ExplicitScheme scheme) {
  const auto node = b.node, param = b.param;
  const auto params = b.params;
  const auto sc = b.state[b.current];
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
        const double dt = Elem::critical_dt(pr.p, pr.damp, scheme, xe, state_of<Elem>(sc, e, P));
        if (dt < m) m = dt;
      },
      Kokkos::Min<double>(result));
  return result;
}

}  // namespace femorax::exec::kernels

namespace femorax::exec::kernels {

// The default state of every point (Material::initial_state if the material has one,
// zeros otherwise) in both buffers.
template <class Elem>
void init_state(DevBatch<Elem>& b) {
  using Mat = typename Elem::Material;
  if constexpr (Mat::NSTATE > 0) {
    const std::size_t n = static_cast<std::size_t>(Mat::NSTATE) * Elem::NGP * b.padded;
    for (auto& s : b.state) s = Vec("material_state", n);
    if constexpr (requires(const typename Mat::Params& q, Strided<double> v) { Mat::initial_state(q, v); }) {
      const auto params = b.params;
      const auto param = b.param;
      const auto s0 = b.state[0], s1 = b.state[1];
      const Index P = b.padded;
      Kokkos::parallel_for(
          "material_initial_state", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count),
          KOKKOS_LAMBDA(const Index e) {
            const auto& p = params(param(e)).p.mat;
            for (int gp = 0; gp < Elem::NGP; ++gp) {
              Mat::initial_state(p, state_of<Elem>(s0, e, P).point(gp));
              Mat::initial_state(p, state_of<Elem>(s1, e, P).point(gp));
            }
          });
    }
  } else {
    (void)b;
  }
}

}  // namespace femorax::exec::kernels

namespace femorax::exec::kernels {

// What a recorder reads of a material after a commit: getStress and getStrain. Materials
// with history state define recorded_stress / recorded_strain (from the committed state);
// for the others the stress is the update at the strain of the displacement and the strain
// is that strain.
template <class Mat>
KOKKOS_INLINE_FUNCTION void recorded_value(int which, const typename Mat::Params& p, Strided<const double> committed,
                                           const double* eps, double* out) {
  constexpr int NS = Mat::NSTRAIN;
  if (which == 0) {
    if constexpr (requires(Strided<double> s, const double* e, double* o) {
                    Mat::recorded_stress(p, committed, s, e, o);
                  }) {
      double scratch[Mat::NSTATE > 0 ? Mat::NSTATE : 1];
      Mat::recorded_stress(p, committed, Strided<double>{scratch, 1}, eps, out);
    } else {
      static_assert(Mat::NSTATE == 0, "a material with history state needs recorded_stress");
      double D[NS][NS];
      Mat::update(p, committed, {}, eps, out, D);
    }
  } else {
    if constexpr (requires(const double* e, double* o) { Mat::recorded_strain(p, committed, e, o); }) {
      Mat::recorded_strain(p, committed, eps, out);
    } else {
      for (int c = 0; c < NS; ++c) out[c] = eps[c];
    }
  }
}

// Element output (VTKHDF stress3D6 = which 0, strain3D6 = which 1) of the listed elements
// at displacement u and the committed state: out(offset + rows(i) * 6 + c), averaged over
// the points as the element does (Elem::average_stress / average_strain).
template <class Elem>
void launch_element_output(const DevBatch<Elem>& b, int which, const Kokkos::View<Index*, Space>& elems,
                           const Kokkos::View<Index*, Space>& rows, const Vec& out, std::int64_t offset,
                           const Vec& coords, Index N, const Vec& u) {
  using Mat = typename Elem::Material;
  constexpr int NNODE = Elem::NNODE, NDOF = Elem::NDOF, NDM = Elem::NDM, NS = Mat::NSTRAIN, NGP = Elem::NGP;
  const auto node = b.node, eq = b.eq, param = b.param;
  const auto params = b.params;
  const auto sc = b.state[b.current];
  const Index P = b.padded;
  Kokkos::parallel_for(
      "element_output", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, static_cast<Index>(elems.extent(0))),
      KOKKOS_LAMBDA(const Index i) {
        const Index e = elems(i);
        double ue[NDOF], xel[NNODE][NDM], xl[3][8];
        for (int a = 0; a < NDOF; ++a) {
          const Index q = eq(a * P + e);
          ue[a] = q >= 0 ? u(q) : 0.0;
        }
        for (int a = 0; a < NNODE; ++a) {
          const Index n = node(a * P + e);
          for (int d = 0; d < NDM; ++d) xel[a][d] = coords(d * N + n);
        }
        Elem::local_coords(xel, xl);
        const auto& p = params(param(e)).p;
        double vals[NGP][NS];
        for (int gp = 0; gp < NGP; ++gp) {
          typename Elem::PointShape s;
          Elem::point_geometry(xl, gp, s);
          double eps[NS];
          Elem::point_strain(s, ue, eps);
          recorded_value<Mat>(which, p.mat, state_of<Elem>(sc, e, P).point(gp), eps, vals[gp]);
        }
        double avg[NS];
        if (which == 0) Elem::average_stress(vals, avg);
        else Elem::average_strain(vals, avg);
        for (int c = 0; c < NS; ++c) out(offset + static_cast<std::int64_t>(rows(i)) * NS + c) = avg[c];
      });
}

}  // namespace femorax::exec::kernels
