#include "exec/explicit_force.hpp"

#include <algorithm>
#include <string>

#include "exec/device_memory.hpp"
#include "exec/runtime.hpp"

namespace femorax::exec {

const char* to_string(Layout l) { return l == Layout::Flat ? "flat" : "team"; }
const char* to_string(Scatter s) {
  return s == Scatter::Atomic ? "atomic" : s == Scatter::Colored ? "colored" : "gather";
}
const char* to_string(Geometry g) { return g == Geometry::Recompute ? "recompute" : "cached"; }

std::string to_string(const ForceSettings& s) {
  std::string r = std::string(to_string(s.layout)) + "_" + to_string(s.scatter);
  if (s.layout == Layout::Team) r += "_ept" + std::to_string(s.elements_per_team);
  if (s.geometry == Geometry::Cached) r += "_cached";
  return r;
}

namespace {

using detail::Brick;
using detail::DevBatch;
using detail::DevParams;
using detail::Space;
using Vec = Kokkos::View<double*, Space>;
using Index = std::int32_t;

constexpr int NNODE = 8, NDOF = 24, NGP = 8, NGEOM = Brick::NGEOM;

// How an element's force leaves the kernel.
enum Out : int { PLAIN = 0, ATOMIC = 1, SLOTS = 2 };

// Team-layout scratch per element, in doubles: xe (24), ue (24), ve (24), per (point, node)
// the contributions residJ, body, dampJ (8 x 8 x 9), and the element force fe (24).
constexpr int GATHER = 3 * NDOF;
constexpr int CONTRIB = NGP * NNODE * 9;
constexpr int FE = GATHER + CONTRIB;
constexpr int PER_ELEMENT = FE + NDOF;

template <class T>
Kokkos::View<T*, Space> upload(const char* label, const std::vector<T>& h) {
  Kokkos::View<T*, Space> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)),
                            h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

// The cached geometry of element e: point gp is at base + e * NGP + gp, values nPoints apart.
KOKKOS_INLINE_FUNCTION PointArray<const double> cache_of(const Vec& g, Index e, Index padded) {
  return {g.data() + static_cast<std::ptrdiff_t>(e) * NGP, NGP * padded, 1};
}

// Element e's force value k, out of the kernel.
KOKKOS_INLINE_FUNCTION void emit(int out, const Vec& f, const Vec& slots, std::int64_t slot_base,
                                 Index P, Index e, int k, Index q, double value) {
  if (out == SLOTS) {
    slots(slot_base + static_cast<std::int64_t>(k) * P + e) = value;
  } else if (q >= 0) {
    if (out == ATOMIC) Kokkos::atomic_add(&f(q), value);
    else f(q) += value;
  }
}

// Compute the geometry cache on the device, with the same code path as Geometry::Recompute.
void fill_geometry(const DevBatch& b, const Vec& coords, Index N) {
  const auto node = b.node;
  const auto g = b.geometry;
  const Index P = b.padded;
  Kokkos::parallel_for(
      "explicit_force_geometry", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, b.count),
      KOKKOS_LAMBDA(const Index e) {
        double xe[NNODE][3], xl[3][8];
        for (int a = 0; a < NNODE; ++a) {
          const Index n = node(a * P + e);
          for (int d = 0; d < 3; ++d) xe[a][d] = coords(d * N + n);
        }
        Brick::local_coords(xe, xl);
        const PointArray<double> out{g.data() + static_cast<std::ptrdiff_t>(e) * NGP, NGP * P, 1};
        for (int gp = 0; gp < NGP; ++gp) Brick::point_store_geometry(xl, gp, out.point(gp));
      });
}

// One thread per element.
template <bool Cached>
void launch_flat(const DevBatch& b, const Vec& coords, Index N, const Vec& u, const Vec& v,
                 const Vec& f, const Vec& slots, int out, Index begin, Index end, bool by_color) {
  const auto node = b.node, eq = b.eq, param = b.param, colors = b.color_elements;
  const auto params = b.params;
  const auto geo = b.geometry;
  const Index P = b.padded;
  const std::int64_t slot_base = b.slot_offset;
  Kokkos::parallel_for(
      "explicit_force_flat", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(begin, end),
      KOKKOS_LAMBDA(const Index i) {
        const Index e = by_color ? colors(i) : i;
        double ue[NDOF], ve[NDOF], fe[NDOF];
        for (int k = 0; k < NDOF; ++k) {
          const Index q = eq(k * P + e);
          ue[k] = q >= 0 ? u(q) : 0.0;
          ve[k] = q >= 0 ? v(q) : 0.0;
        }
        const DevParams& pr = params(param(e));
        if (Cached) {  // a constant (template argument); nvcc cannot capture inside if constexpr
          Brick::compute_cached<false>(pr.p, pr.damp, cache_of(geo, e, P), ue, ve, {}, {}, fe,
                                       nullptr);
        } else {
          double xe[NNODE][3];
          for (int a = 0; a < NNODE; ++a) {
            const Index n = node(a * P + e);
            for (int d = 0; d < 3; ++d) xe[a][d] = coords(d * N + n);
          }
          Brick::compute<false>(pr.p, pr.damp, xe, ue, ve, {}, {}, fe, nullptr);
        }
        for (int k = 0; k < NDOF; ++k) emit(out, f, slots, slot_base, P, e, k, eq(k * P + e), fe[k]);
      });
}

// One thread per Gauss point; ept elements per team.
template <bool Cached>
void launch_team(const DevBatch& b, const Vec& coords, Index N, const Vec& u, const Vec& v,
                 const Vec& f, const Vec& slots, int out, Index begin, Index end, bool by_color,
                 int ept) {
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
  const std::int64_t slot_base = b.slot_offset;

  Policy probe(league, Kokkos::AUTO);
  const int team_size = std::min(
      ept * NGP, probe.team_size_max(KOKKOS_LAMBDA(const Member&){}, Kokkos::ParallelForTag()));
  Policy policy(league, std::max(team_size, 1));
  policy.set_scratch_size(0, Kokkos::PerTeam(static_cast<std::size_t>(ept) * PER_ELEMENT * sizeof(double)));

  Kokkos::parallel_for(
      "explicit_force_team", policy, KOKKOS_LAMBDA(const Member& m) {
        Scratch buf(m.team_scratch(0), static_cast<std::size_t>(ept) * PER_ELEMENT);
        const Index first = begin + m.league_rank() * ept;
        auto element = [&](int slot) -> Index {
          const Index i = first + slot;
          if (i >= end) return -1;
          return by_color ? colors(i) : i;
        };

        // 1. Gather: work item (slot, a) loads node a of element slot.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NNODE), [&](int t) {
          const int slot = t / NNODE, a = t % NNODE;
          const Index e = element(slot);
          if (e < 0) return;
          double* g = &buf(slot * PER_ELEMENT);
          const Index nd = node(a * P + e);
          for (int d = 0; d < 3; ++d) {
            if (!Cached) g[3 * a + d] = coords(d * N + nd);
            const Index q = eq((3 * a + d) * P + e);
            g[NDOF + 3 * a + d] = q >= 0 ? u(q) : 0.0;
            g[2 * NDOF + 3 * a + d] = q >= 0 ? v(q) : 0.0;
          }
        });
        m.team_barrier();

        // 2. Points: work item (slot, gp) updates point gp and writes its contribution to
        // every node.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NGP), [&](int t) {
          const int slot = t / NGP, gp = t % NGP;
          const Index e = element(slot);
          if (e < 0) return;
          double* g = &buf(slot * PER_ELEMENT);
          const double* ue = g + NDOF;
          const double* ve = g + 2 * NDOF;
          const DevParams& pr = params(param(e));
          Brick::PointShape s;
          if (Cached) {  // a constant (template argument); nvcc cannot capture inside if constexpr
            Brick::point_load_geometry(gp, cache_of(geo, e, P).point(gp), s);
          } else {
            double xl[3][8];
            Brick::local_coords(reinterpret_cast<const double(*)[3]>(g), xl);
            Brick::point_geometry(xl, gp, s);
          }
          double D0[6][6];
          if (pr.damp.betaKinit != 0.0) ElasticIsotropic::initial_tangent(pr.p.mat, D0);
          Brick::PointResult r;
          double dd[6][6];
          Brick::point_update_shape(pr.p, pr.damp, s, ue, ve, {}, {}, D0, r, dd);
          double* c = g + GATHER + gp * NNODE * 9;
          for (int j = 0; j < NNODE; ++j)
            Brick::point_node_contribution(pr.p, pr.damp, r, ve, j, c + 9 * j, c + 9 * j + 3,
                                           c + 9 * j + 6);
        });
        m.team_barrier();

        // 3. Nodes: work item (slot, j) adds node j's contributions in Gauss point order
        // (the order of StdBrick::compute).
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NNODE), [&](int t) {
          const int slot = t / NNODE, j = t % NNODE;
          if (element(slot) < 0) return;
          double* g = &buf(slot * PER_ELEMENT);
          const double* c = g + GATHER;
          for (int q3 = 0; q3 < 3; ++q3) {
            double resid = 0.0, fdamp = 0.0;
            for (int gp = 0; gp < NGP; ++gp) {
              const double* cj = c + gp * NNODE * 9 + 9 * j;
              resid += cj[q3];
              resid -= cj[3 + q3];
              fdamp += cj[6 + q3];
            }
            g[FE + 3 * j + q3] = resid + fdamp;
          }
        });
        m.team_barrier();

        // 4. Out: one work item per element, its 24 values in DOF order (as the flat kernel).
        // Nodes of one element can share an equation (tied nodes), so the element's own
        // writes must not run concurrently.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept), [&](int slot) {
          const Index e = element(slot);
          if (e < 0) return;
          const double* fe = &buf(slot * PER_ELEMENT) + FE;
          for (int k = 0; k < NDOF; ++k) emit(out, f, slots, slot_base, P, e, k, eq(k * P + e), fe[k]);
        });
      });
}

// Gather: f(q) = sum of q's slots, in list order.
void launch_gather(const Vec& f, const Vec& slots, const Kokkos::View<std::int64_t*, Space>& start,
                   const Kokkos::View<std::int64_t*, Space>& list, Index neq) {
  Kokkos::parallel_for(
      "explicit_force_gather", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq),
      KOKKOS_LAMBDA(const Index q) {
        double sum = 0.0;
        for (std::int64_t j = start(q); j < start(q + 1); ++j) sum += slots(list(j));
        f(q) = sum;
      });
}

}  // namespace

ExplicitForce::ExplicitForce(const compile::Compiled& c, GeometryCache cache)
    : u("u", static_cast<std::size_t>(c.dofs.num_equations)),
      v("v", static_cast<std::size_t>(c.dofs.num_equations)),
      f("f", static_cast<std::size_t>(c.dofs.num_equations)),
      neq_(c.dofs.num_equations),
      nnodes_(c.dofs.num_nodes()) {
  coords_ = upload("coords", c.coords);

  // Gather lists: each equation's slots in (batch, element, DOF) order, built on the host.
  std::vector<std::int64_t> count(static_cast<std::size_t>(neq_) + 1, 0);
  std::int64_t nslots = 0;
  for (const auto& hb : c.batches) {
    for (Index e = 0; e < hb.count; ++e)
      for (int k = 0; k < hb.ndof; ++k) {
        const Index q = hb.eq[static_cast<std::size_t>(k) * hb.padded + e];
        if (q >= 0) ++count[q + 1];
      }
    nslots += static_cast<std::int64_t>(hb.ndof) * hb.padded;
  }
  for (Index q = 0; q < neq_; ++q) count[q + 1] += count[q];
  std::vector<std::int64_t> list(static_cast<std::size_t>(count[neq_]));
  {
    std::vector<std::int64_t> fill(count.begin(), count.end() - 1);
    std::int64_t offset = 0;
    for (const auto& hb : c.batches) {
      for (Index e = 0; e < hb.count; ++e)
        for (int k = 0; k < hb.ndof; ++k) {
          const Index q = hb.eq[static_cast<std::size_t>(k) * hb.padded + e];
          if (q >= 0) list[fill[q]++] = offset + static_cast<std::int64_t>(k) * hb.padded + e;
        }
      offset += static_cast<std::int64_t>(hb.ndof) * hb.padded;
    }
  }

  // Geometry cache: on, off, or if everything fits in 80% of the free device memory.
  for (const auto& hb : c.batches) cache_bytes_ += static_cast<std::int64_t>(NGEOM) * NGP * hb.padded * 8;
  if (cache == GeometryCache::Auto) {
    const std::int64_t need = compile::memory_estimate_bytes(c) + gather_bytes() + cache_bytes_ +
                              nslots * 8 + static_cast<std::int64_t>(list.size()) * 8;
    const std::int64_t free_b = device_free_memory_bytes();
    has_cache_ = free_b > 0 && need <= free_b / 10 * 8;
  } else {
    has_cache_ = cache == GeometryCache::On;
  }

  std::int64_t offset = 0;
  for (const auto& hb : c.batches) {
    DevBatch b;
    b.count = hb.count;
    b.padded = hb.padded;
    b.node = upload("batch_node", hb.node);
    b.eq = upload("batch_eq", hb.eq);
    b.param = upload("batch_param", hb.param);
    b.color_elements = upload("batch_colors", hb.color_elements);
    b.color_start = hb.color_start;
    b.slot_offset = offset;
    offset += static_cast<std::int64_t>(hb.ndof) * hb.padded;
    std::vector<DevParams> ps;
    for (const auto& p : hb.params)
      ps.push_back({Brick::Params{{p.E, p.nu, p.rho}, {p.b[0], p.b[1], p.b[2]}, p.massType}, p.damping});
    b.params = upload("batch_params", ps);
    if (has_cache_) {
      b.geometry = Vec("batch_geometry", static_cast<std::size_t>(NGEOM) * NGP * b.padded);
      fill_geometry(b, coords_, nnodes_);
    }
    batches_.push_back(std::move(b));
  }
  slots_ = Vec("gather_slots", static_cast<std::size_t>(nslots));
  gather_start_ = upload("gather_start", count);
  gather_slot_ = upload("gather_slot", list);
  Kokkos::fence();
}

void ExplicitForce::fill(double u0, double v0) {
  Kokkos::deep_copy(u, u0);
  Kokkos::deep_copy(v, v0);
}

std::int64_t ExplicitForce::num_elements() const {
  std::int64_t n = 0;
  for (const auto& b : batches_) n += b.count;
  return n;
}

std::int64_t ExplicitForce::geometry_cache_bytes() const { return cache_bytes_; }

std::int64_t ExplicitForce::gather_bytes() const {
  return static_cast<std::int64_t>(slots_.extent(0) + gather_start_.extent(0) + gather_slot_.extent(0)) * 8;
}

int ExplicitForce::max_elements_per_team() {
  const auto max_bytes = Kokkos::TeamPolicy<Space>::scratch_size_max(0);
  return std::max(1, static_cast<int>(max_bytes / (PER_ELEMENT * sizeof(double))));
}

int ExplicitForce::default_elements_per_team() {
  return std::min(std::max(1, simd_width() / NGP), max_elements_per_team());
}

ForceSettings ExplicitForce::resolve(const ForceSettings& s) const {
  ForceSettings r = s;
  if (r.geometry == Geometry::Cached && !has_cache_) r.geometry = Geometry::Recompute;
  if (r.layout == Layout::Team) {
    const int ept = r.elements_per_team > 0 ? r.elements_per_team : default_elements_per_team();
    r.elements_per_team = std::min(ept, max_elements_per_team());
  } else {
    r.elements_per_team = 0;
  }
  return r;
}

void ExplicitForce::compute(const ForceSettings& settings) {
  const ForceSettings s = resolve(settings);
  const bool cached = s.geometry == Geometry::Cached;
  if (s.scatter != Scatter::Gather) Kokkos::deep_copy(f, 0.0);
  for (const auto& b : batches_) {
    auto run = [&](int out, Index begin, Index end, bool by_color) {
      if (s.layout == Layout::Flat) {
        if (cached) launch_flat<true>(b, coords_, nnodes_, u, v, f, slots_, out, begin, end, by_color);
        else launch_flat<false>(b, coords_, nnodes_, u, v, f, slots_, out, begin, end, by_color);
      } else {
        const int ept = s.elements_per_team;
        if (cached) launch_team<true>(b, coords_, nnodes_, u, v, f, slots_, out, begin, end, by_color, ept);
        else launch_team<false>(b, coords_, nnodes_, u, v, f, slots_, out, begin, end, by_color, ept);
      }
    };
    switch (s.scatter) {
      case Scatter::Atomic: run(ATOMIC, 0, b.count, false); break;
      case Scatter::Gather: run(SLOTS, 0, b.count, false); break;
      case Scatter::Colored:
        for (std::size_t col = 0; col + 1 < b.color_start.size(); ++col)
          run(PLAIN, b.color_start[col], b.color_start[col + 1], true);
        break;
    }
  }
  if (s.scatter == Scatter::Gather) launch_gather(f, slots_, gather_start_, gather_slot_, neq_);
}

}  // namespace femorax::exec
