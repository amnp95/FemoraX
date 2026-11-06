#include "exec/explicit_force.hpp"

#include "exec/runtime.hpp"

#include <string>

namespace femorax::exec {

const char* to_string(Layout l) { return l == Layout::Flat ? "flat" : "team"; }
const char* to_string(Scatter s) { return s == Scatter::Atomic ? "atomic" : "colored"; }

namespace {

template <class T>
Kokkos::View<T*, ExplicitForce::Space> upload(const char* label, const std::vector<T>& h) {
  Kokkos::View<T*, ExplicitForce::Space> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)),
                                           h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

constexpr int NNODE = 8, NDOF = 24, NGP = 8;

}  // namespace

ExplicitForce::ExplicitForce(const compile::Compiled& c)
    : u("u", static_cast<std::size_t>(c.dofs.num_equations)),
      v("v", static_cast<std::size_t>(c.dofs.num_equations)),
      f("f", static_cast<std::size_t>(c.dofs.num_equations)),
      neq_(c.dofs.num_equations),
      nnodes_(c.dofs.num_nodes()) {
  coords_ = upload("coords", c.coords);
  for (const auto& hb : c.batches) {
    DevBatch b;
    b.count = hb.count;
    b.padded = hb.padded;
    b.node = upload("batch_node", hb.node);
    b.eq = upload("batch_eq", hb.eq);
    b.param = upload("batch_param", hb.param);
    b.color_elements = upload("batch_colors", hb.color_elements);
    b.color_start = hb.color_start;
    std::vector<DevParams> ps;
    for (const auto& p : hb.params)
      ps.push_back({Brick::Params{{p.E, p.nu, p.rho}, {p.b[0], p.b[1], p.b[2]}, p.massType}, p.damping});
    b.params = upload("batch_params", ps);
    batches_.push_back(std::move(b));
  }
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

int ExplicitForce::default_elements_per_team() {
  return std::min(std::max(1, simd_width() / NGP), max_elements_per_team());
}

void ExplicitForce::compute(Layout layout, Scatter scatter, int elements_per_team) {
  Kokkos::deep_copy(f, 0.0);
  const int ept = std::min(elements_per_team > 0 ? elements_per_team : default_elements_per_team(),
                           max_elements_per_team());
  for (const auto& b : batches_) {
    if (scatter == Scatter::Atomic) {
      if (layout == Layout::Flat) launch_flat(b, true, 0, b.count, false);
      else launch_team(b, true, 0, b.count, false, ept);
    } else {
      for (std::size_t col = 0; col + 1 < b.color_start.size(); ++col) {
        if (layout == Layout::Flat) launch_flat(b, false, b.color_start[col], b.color_start[col + 1], true);
        else launch_team(b, false, b.color_start[col], b.color_start[col + 1], true, ept);
      }
    }
  }
}

// One thread per element.
void ExplicitForce::launch_flat(const DevBatch& b, bool atomic, std::int32_t begin,
                                std::int32_t end, bool by_color) {
  const auto coords = coords_;
  const auto node = b.node, eq = b.eq, param = b.param, colors = b.color_elements;
  const auto params = b.params;
  const auto uu = u, vv = v, ff = f;
  const std::int32_t P = b.padded, N = nnodes_;
  Kokkos::parallel_for(
      "explicit_force_flat", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int32_t>>(begin, end),
      KOKKOS_LAMBDA(const std::int32_t i) {
        const std::int32_t e = by_color ? colors(i) : i;
        double xe[NNODE][3], ue[NDOF], ve[NDOF], fe[NDOF];
        for (int a = 0; a < NNODE; ++a) {
          const std::int32_t n = node(a * P + e);
          for (int d = 0; d < 3; ++d) xe[a][d] = coords(d * N + n);
        }
        for (int k = 0; k < NDOF; ++k) {
          const std::int32_t q = eq(k * P + e);
          ue[k] = q >= 0 ? uu(q) : 0.0;
          ve[k] = q >= 0 ? vv(q) : 0.0;
        }
        const DevParams& pr = params(param(e));
        Brick::compute<false>(pr.p, pr.damp, xe, ue, ve, {}, {}, fe, nullptr);
        for (int k = 0; k < NDOF; ++k) {
          const std::int32_t q = eq(k * P + e);
          if (q < 0) continue;
          if (atomic) Kokkos::atomic_add(&ff(q), fe[k]);
          else ff(q) += fe[k];
        }
      });
}

namespace {
// Team-layout scratch per element, in doubles (see launch_team).
constexpr int TEAM_SCRATCH_PER_ELEMENT = 3 * 24 + 8 * 8 * 9 + 24;
}  // namespace

int ExplicitForce::max_elements_per_team() {
  const auto max_bytes = Kokkos::TeamPolicy<Space>::scratch_size_max(0);
  return std::max(1, static_cast<int>(max_bytes / (TEAM_SCRATCH_PER_ELEMENT * sizeof(double))));
}

// One thread per Gauss point; ept elements per team.
void ExplicitForce::launch_team(const DevBatch& b, bool atomic, std::int32_t begin,
                                std::int32_t end, bool by_color, int ept) {
  using Policy = Kokkos::TeamPolicy<Space>;
  using Member = Policy::member_type;
  using Scratch = Kokkos::View<double*, Space::scratch_memory_space, Kokkos::MemoryUnmanaged>;
  // Per element in scratch: xe (24), ue (24), ve (24), per (point, node) the three
  // contributions residJ, body, dampJ (8 x 8 x 9), and the element force fe (24).
  constexpr int GATHER = 3 * NDOF;
  constexpr int CONTRIB = NGP * NNODE * 9;
  constexpr int FE = GATHER + CONTRIB;
  constexpr int PER_ELEMENT = FE + NDOF;
  static_assert(PER_ELEMENT == TEAM_SCRATCH_PER_ELEMENT);

  const std::int32_t n = end - begin;
  if (n <= 0) return;
  const int league = (n + ept - 1) / ept;
  const auto coords = coords_;
  const auto node = b.node, eq = b.eq, param = b.param, colors = b.color_elements;
  const auto params = b.params;
  const auto uu = u, vv = v, ff = f;
  const std::int32_t P = b.padded, N = nnodes_;

  Policy probe(league, Kokkos::AUTO);
  const int team_size = std::min(ept * NGP, probe.team_size_max(
      KOKKOS_LAMBDA(const Member&) {}, Kokkos::ParallelForTag()));
  Policy policy(league, std::max(team_size, 1));
  policy.set_scratch_size(0, Kokkos::PerTeam(static_cast<std::size_t>(ept) * PER_ELEMENT * sizeof(double)));

  Kokkos::parallel_for(
      "explicit_force_team", policy, KOKKOS_LAMBDA(const Member& m) {
        Scratch buf(m.team_scratch(0), static_cast<std::size_t>(ept) * PER_ELEMENT);
        const std::int32_t first = begin + m.league_rank() * ept;
        auto element = [&](int slot) -> std::int32_t {
          const std::int32_t i = first + slot;
          if (i >= end) return -1;
          return by_color ? colors(i) : i;
        };

        // 1. Gather: work item (slot, a) loads node a of element slot.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NNODE), [&](int t) {
          const int slot = t / NNODE, a = t % NNODE;
          const std::int32_t e = element(slot);
          if (e < 0) return;
          double* g = &buf(slot * PER_ELEMENT);
          const std::int32_t nd = node(a * P + e);
          for (int d = 0; d < 3; ++d) {
            g[3 * a + d] = coords(d * N + nd);
            const std::int32_t q = eq((3 * a + d) * P + e);
            g[NDOF + 3 * a + d] = q >= 0 ? uu(q) : 0.0;
            g[2 * NDOF + 3 * a + d] = q >= 0 ? vv(q) : 0.0;
          }
        });
        m.team_barrier();

        // 2. Points: work item (slot, gp) updates point gp and writes its contribution to
        // every node.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept * NGP), [&](int t) {
          const int slot = t / NGP, gp = t % NGP;
          const std::int32_t e = element(slot);
          if (e < 0) return;
          double* g = &buf(slot * PER_ELEMENT);
          const auto* xe = reinterpret_cast<const double(*)[3]>(g);
          const double* ue = g + NDOF;
          const double* ve = g + 2 * NDOF;
          const DevParams& pr = params(param(e));
          double xl[3][8];
          Brick::local_coords(xe, xl);
          double D0[6][6];
          if (pr.damp.betaKinit != 0.0) ElasticIsotropic::initial_tangent(pr.p.mat, D0);
          Brick::PointResult r;
          double dd[6][6];
          Brick::point_update(pr.p, pr.damp, xl, ue, ve, gp, {}, {}, D0, r, dd);
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

        // 4. Scatter: one work item per element, its 24 values in DOF order (as the flat
        // kernel). Nodes of one element can share an equation (tied nodes), so the element's
        // own writes must not run concurrently.
        Kokkos::parallel_for(Kokkos::TeamThreadRange(m, ept), [&](int slot) {
          const std::int32_t e = element(slot);
          if (e < 0) return;
          const double* fe = &buf(slot * PER_ELEMENT) + FE;
          for (int k = 0; k < NDOF; ++k) {
            const std::int32_t q = eq(k * P + e);
            if (q < 0) continue;
            if (atomic) Kokkos::atomic_add(&ff(q), fe[k]);
            else ff(q) += fe[k];
          }
        });
      });
}

}  // namespace femorax::exec
