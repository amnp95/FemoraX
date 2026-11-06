#include "exec/internal_force.hpp"

#include <algorithm>
#include <sstream>
#include <string>

#include "compile/registry.hpp"
#include "exec/device_memory.hpp"
#include "exec/element_kernels.hpp"
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

bool parse_settings(const std::string& text, ForceSettings& out) {
  std::istringstream in(text);
  std::string part;
  ForceSettings s;
  int field = 0;
  while (std::getline(in, part, '_')) {
    if (field == 0) {
      if (part == "flat") s.layout = Layout::Flat;
      else if (part == "team") s.layout = Layout::Team;
      else return false;
    } else if (field == 1) {
      if (part == "gather") s.scatter = Scatter::Gather;
      else if (part == "atomic") s.scatter = Scatter::Atomic;
      else if (part == "colored") s.scatter = Scatter::Colored;
      else return false;
    } else if (part.rfind("ept", 0) == 0) {
      s.elements_per_team = std::atoi(part.c_str() + 3);
    } else if (part == "cached") {
      s.geometry = Geometry::Cached;
    } else {
      return false;
    }
    ++field;
  }
  if (field < 2) return false;
  out = s;
  return true;
}

using Index = std::int32_t;

namespace {
template <class T>
Kokkos::View<T*, Space> upload(const char* label, const std::vector<T>& h) {
  Kokkos::View<T*, Space> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)),
                            h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}
}  // namespace

// Type-erased batch: the typed kernels behind one interface (virtual dispatch on the host,
// once per batch per call; the launch layer may use it, physics may not).
struct InternalForce::Batch {
  virtual ~Batch() = default;
  virtual void run(const ForceSettings& s, Out out, const Vec& coords, Index N, const Vec& u,
                   const Vec& v, const Vec& f, const Vec& slots) = 0;
  virtual std::int64_t count() const = 0;
  virtual std::int64_t device_bytes() const = 0;
  virtual int max_elements_per_team() const = 0;
};

namespace {

template <class Elem>
struct TypedBatch final : InternalForce::Batch {
  kernels::DevBatch<Elem> b;

  TypedBatch(const compile::Batch& hb, std::int64_t slot_offset, bool cache, const Vec& coords,
             Index N) {
    b.count = hb.count;
    b.padded = hb.padded;
    b.node = upload("batch_node", hb.node);
    b.eq = upload("batch_eq", hb.eq);
    b.param = upload("batch_param", hb.param);
    b.color_elements = upload("batch_colors", hb.color_elements);
    b.color_start = hb.color_start;
    b.slot_offset = slot_offset;
    std::vector<kernels::DevParams<Elem>> ps;
    for (const auto& p : hb.params) ps.push_back({p.template params<Elem>(), p.damping});
    b.params = upload("batch_params", ps);
    if (cache) {
      b.geometry = Vec("batch_geometry", static_cast<std::size_t>(Elem::NGEOM) * Elem::NGP * b.padded);
      kernels::fill_geometry<Elem>(b, coords, N);
    }
  }

  void run(const ForceSettings& s, Out out, const Vec& coords, Index N, const Vec& u,
           const Vec& v, const Vec& f, const Vec& slots) override {
    const bool cached = s.geometry == Geometry::Cached;
    auto launch = [&](Index begin, Index end, bool by_color) {
      if (s.layout == Layout::Flat) {
        if (cached) kernels::launch_flat<Elem, true>(b, coords, N, u, v, f, slots, out, begin, end, by_color);
        else kernels::launch_flat<Elem, false>(b, coords, N, u, v, f, slots, out, begin, end, by_color);
      } else {
        if constexpr (PointwiseElement<Elem>) {
          const int ept = s.elements_per_team;
          if (cached) kernels::launch_team<Elem, true>(b, coords, N, u, v, f, slots, out, begin, end, by_color, ept);
          else kernels::launch_team<Elem, false>(b, coords, N, u, v, f, slots, out, begin, end, by_color, ept);
        } else {
          kernels::launch_flat<Elem, false>(b, coords, N, u, v, f, slots, out, begin, end, by_color);
        }
      }
    };
    if (s.scatter == Scatter::Colored) {
      for (std::size_t col = 0; col + 1 < b.color_start.size(); ++col)
        launch(b.color_start[col], b.color_start[col + 1], true);
    } else {
      launch(0, b.count, false);
    }
  }

  std::int64_t count() const override { return b.count; }
  std::int64_t device_bytes() const override {
    return static_cast<std::int64_t>(b.node.extent(0) + b.eq.extent(0) + b.param.extent(0) +
                                     b.color_elements.extent(0)) * 4 +
           static_cast<std::int64_t>(b.params.extent(0) * sizeof(kernels::DevParams<Elem>)) +
           static_cast<std::int64_t>(b.geometry.extent(0)) * 8;
  }
  int max_elements_per_team() const override {
    const auto max_bytes = Kokkos::TeamPolicy<Space>::scratch_size_max(0);
    return std::max(1, static_cast<int>(max_bytes / (kernels::TeamScratch<Elem>::PER_ELEMENT * sizeof(double))));
  }
};

}  // namespace

InternalForce::InternalForce(const compile::Compiled& c, GeometryCache cache)
    : u("u", static_cast<std::size_t>(c.dofs.num_equations)),
      v("v", static_cast<std::size_t>(c.dofs.num_equations)),
      f("f", static_cast<std::size_t>(c.dofs.num_equations)),
      neq_(c.dofs.num_equations),
      nnodes_(c.dofs.num_nodes()) {
  coords_ = upload("coords", c.coords);

  // Gather pairs: each equation's slots in (batch, element, DOF) order.
  std::vector<std::pair<Index, std::int64_t>> pairs;
  std::vector<std::int64_t> offsets;
  std::int64_t nslots = 0;
  for (const auto& hb : c.batches) {
    offsets.push_back(nslots);
    for (Index e = 0; e < hb.count; ++e)
      for (int k = 0; k < hb.ndof; ++k) {
        const Index q = hb.eq[static_cast<std::size_t>(k) * hb.padded + e];
        if (q >= 0) pairs.emplace_back(q, nslots + static_cast<std::int64_t>(k) * hb.padded + e);
      }
    nslots += static_cast<std::int64_t>(hb.ndof) * hb.padded;
    if (!hb.has_coloring()) has_coloring_ = false;
  }

  // Geometry cache size, and whether to build it.
  registry::for_each_pair([&]<class E, class M>(int index) {
    using Elem = typename E::template Element<typename M::Material>;
    for (const auto& hb : c.batches)
      if (hb.pair == index && Elem::GEOMETRY_CACHEABLE)
        cache_bytes_ += static_cast<std::int64_t>(Elem::NGEOM) * Elem::NGP * hb.padded * 8;
  });
  if (cache == GeometryCache::Auto) {
    const std::int64_t need = compile::memory_estimate_bytes(c) + cache_bytes_ +
                              (nslots + 2 * static_cast<std::int64_t>(pairs.size()) + neq_) * 8;
    const std::int64_t free_b = device_free_memory_bytes();
    has_cache_ = free_b > 0 && need <= free_b / 10 * 8;  // unknown free memory: no cache
  } else {
    has_cache_ = cache == GeometryCache::On && cache_bytes_ > 0;
  }

  for (std::size_t i = 0; i < c.batches.size(); ++i) {
    const auto& hb = c.batches[i];
    registry::for_each_pair([&]<class E, class M>(int index) {
      if (index != hb.pair) return;
      using Elem = typename E::template Element<typename M::Material>;
      batches_.push_back(std::make_unique<TypedBatch<Elem>>(
          hb, offsets[i], has_cache_ && Elem::GEOMETRY_CACHEABLE, coords_, nnodes_));
    });
  }
  gather_ = SlotGather(nslots, neq_, pairs);
  Kokkos::fence();
}

InternalForce::~InternalForce() = default;

void InternalForce::fill(double u0, double v0) {
  Kokkos::deep_copy(u, u0);
  Kokkos::deep_copy(v, v0);
}

std::int64_t InternalForce::num_elements() const {
  std::int64_t n = 0;
  for (const auto& b : batches_) n += b->count();
  return n;
}

std::int64_t InternalForce::device_bytes() const {
  std::int64_t n = static_cast<std::int64_t>(3 * neq_ + coords_.extent(0)) * 8 + gather_.bytes();
  for (const auto& b : batches_) n += b->device_bytes();
  return n;
}

int InternalForce::default_elements_per_team() { return std::max(1, simd_width() / 8); }

int InternalForce::max_elements_per_team() const {
  int m = 1 << 20;
  for (const auto& b : batches_) m = std::min(m, b->max_elements_per_team());
  return m;
}

ForceSettings InternalForce::resolve(const ForceSettings& s) const {
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

void InternalForce::compute(const ForceSettings& settings) {
  const ForceSettings s = resolve(settings);
  if (s.scatter == Scatter::Colored && !has_coloring_) {
    Kokkos::abort("InternalForce: colored scatter needs CompileOptions::build_coloring");
  }
  const Out out = s.scatter == Scatter::Gather ? Out::Slots
                  : s.scatter == Scatter::Atomic ? Out::Atomic
                                                 : Out::Plain;
  if (out != Out::Slots) Kokkos::deep_copy(f, 0.0);
  for (auto& b : batches_) b->run(s, out, coords_, nnodes_, u, v, f, gather_.slots());
  if (out == Out::Slots) gather_.gather(f);
}

}  // namespace femorax::exec
