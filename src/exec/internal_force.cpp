#include "exec/internal_force.hpp"

#include <algorithm>
#include <cstring>
#include <type_traits>
#include <unordered_map>
#include <sstream>
#include <string>

#include "compile/registry.hpp"
#include "io/h5.hpp"
#include "exec/device_memory.hpp"
#include "exec/compacted_kernels.hpp"
#include "exec/element_kernels.hpp"
#include "exec/operator_kernels.hpp"
#include "exec/runtime.hpp"

namespace femorax::exec {

const char* to_string(Layout l) { return l == Layout::Flat ? "flat" : "team"; }
const char* to_string(Scatter s) {
  return s == Scatter::Atomic ? "atomic" : s == Scatter::Colored ? "colored" : "gather";
}
const char* to_string(Geometry g) { return g == Geometry::Recompute ? "recompute" : "cached"; }
const char* to_string(MaterialPath m) { return m == MaterialPath::Fused ? "fused" : "compacted"; }

std::string to_string(const ForceSettings& s) {
  std::string r = std::string(to_string(s.layout)) + "_" + to_string(s.scatter);
  if (s.layout == Layout::Team) r += "_ept" + std::to_string(s.elements_per_team);
  if (s.geometry == Geometry::Cached) r += "_cached";
  if (s.material == MaterialPath::Compacted) r += "_compacted";
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
    } else if (part == "compacted") {
      s.material = MaterialPath::Compacted;
    } else {
      return false;
    }
    ++field;
  }
  if (field < 2) return false;
  out = s;
  return true;
}

const char* to_string(TangentSource t) {
  switch (t) {
    case TangentSource::Auto: return "auto";
    case TangentSource::Constant: return "constant";
    case TangentSource::Stored: return "stored";
    case TangentSource::Recomputed: break;
  }
  return "recomputed";
}

std::string to_string(const OperatorSettings& s) {
  return std::string(to_string(s.scatter)) + "_" + to_string(s.geometry) + "_" + to_string(s.tangent);
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
  // Returns the number of points that took the plastic corrector (compacted path; 0 else).
  virtual std::int64_t run(const ForceSettings& s, Out out, const Vec& coords, Index N, const Vec& u,
                           const Vec& v, const Vec& f, const Vec& slots) = 0;
  virtual bool split_material() const = 0;
  virtual std::int64_t count() const = 0;
  virtual std::int64_t device_bytes() const = 0;
  virtual int max_elements_per_team() const = 0;
  virtual double min_critical_dt(const Vec& coords, Index N, ExplicitScheme s) const = 0;
  // Matrix-free operator (resolved settings): y (or the diagonal) through `out`.
  virtual void run_operator(const OperatorSettings& s, bool diagonal, const OperatorCoefficients& k, Out out,
                            const Vec& coords, Index N, const Vec& u, const Vec& x, const Vec& y,
                            const Vec& slots, int unit_dir) = 0;
  virtual void store_tangents(const OperatorSettings& s, const Vec& coords, Index N, const Vec& u) = 0;
  virtual std::int64_t enable_stored_tangents() = 0;  // returns the bytes allocated
  virtual TangentSymmetry tangent_symmetry() const = 0;
  virtual bool constant_tangent() const = 0;
  // Material history state (none for materials without it).
  virtual bool has_state() const = 0;
  virtual void commit_state() = 0;
  // updateMaterialStage for the parameter sets of model material `material` (an index into
  // model.materials); returns whether any set of this batch uses it.
  virtual bool set_material_stage(std::int32_t material, int stage) = 0;
  virtual void element_output(int which, const Kokkos::View<Index*, Space>& elems,
                              const Kokkos::View<Index*, Space>& rows, const Vec& out, std::int64_t offset,
                              const Vec& coords, Index N, const Vec& u) = 0;
};

namespace {

template <class Elem>
struct TypedBatch final : InternalForce::Batch {
  kernels::DevBatch<Elem> b;
  kernels::DevTangents<Elem> t;
  static constexpr bool kCompact = kernels::kCompactable<Elem>;
  struct NoCompact {};
  std::conditional_t<kCompact, kernels::DevCompact<Elem>, NoCompact> compact;

  std::vector<kernels::DevParams<Elem>> host_params;
  std::vector<std::int32_t> param_material;

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
    std::vector<kernels::DevParams<Elem>>& ps = host_params;
    for (const auto& p : hb.params) {
      ps.push_back({p.template params<Elem>(), p.damping});
      param_material.push_back(p.material);
    }
    b.params = upload("batch_params", ps);
    kernels::init_state<Elem>(b);
    if constexpr (kCompact) {
      using R = kernels::Replay<typename Elem::Material>;
      const auto points = static_cast<std::size_t>(Elem::NGP) * b.padded;
      compact.results = Vec("batch_compact_results", static_cast<std::size_t>(R::NSTATE) * points);
      compact.flag = Kokkos::View<std::int32_t*, Space>("batch_compact_flag", points);
      compact.list = Kokkos::View<std::int32_t*, Space>("batch_compact_list", points);
      compact.params = Kokkos::View<kernels::DevParams<kernels::ReplayElement<Elem>>*, Space>(
          "batch_compact_params", ps.size());
      upload_replay_params();
    }
    using Mat = typename Elem::Material;
    if constexpr (Mat::CONSTANT_TANGENT) {
      constexpr int NS = Mat::NSTRAIN;
      std::vector<double> d(ps.size() * NS * NS);
      for (std::size_t i = 0; i < ps.size(); ++i) {
        double D[NS][NS];
        Mat::initial_tangent(ps[i].p.mat, D);
        for (int a = 0; a < NS; ++a)
          for (int c = 0; c < NS; ++c) d[(i * NS + a) * NS + c] = D[a][c];
      }
      t.constant = upload("batch_constant_tangent", d);
    }
    if (cache) {
      b.geometry = Vec("batch_geometry", static_cast<std::size_t>(Elem::NGEOM) * Elem::NGP * b.padded);
      kernels::fill_geometry<Elem>(b, coords, N);
      if constexpr (ElementTermElement<Elem>) {
        b.constants = Vec("batch_element_constants", static_cast<std::size_t>(Elem::NCONST) * b.padded);
        kernels::fill_constants<Elem>(b, coords, N);
      }
    }
  }

  // The parameter sets retyped for the Replay element (the same bytes: Replay's Params is
  // the material's).
  void upload_replay_params() {
    if constexpr (kCompact) {
      using RP = kernels::DevParams<kernels::ReplayElement<Elem>>;
      static_assert(sizeof(RP) == sizeof(kernels::DevParams<Elem>) && std::is_trivially_copyable_v<RP>);
      auto m = Kokkos::create_mirror_view(compact.params);
      for (std::size_t i = 0; i < host_params.size(); ++i)
        std::memcpy(static_cast<void*>(&m(i)), static_cast<const void*>(&host_params[i]), sizeof(RP));
      Kokkos::deep_copy(compact.params, m);
    }
  }

  bool split_material() const override { return kCompact; }

  std::int64_t run(const ForceSettings& s, Out out, const Vec& coords, Index N, const Vec& u,
                   const Vec& v, const Vec& f, const Vec& slots) override {
    if constexpr (kCompact) {
      if (s.material == MaterialPath::Compacted) return run_compacted(s, out, coords, N, u, v, f, slots);
    }
    run_kernels(b, s, out, coords, N, u, v, f, slots);
    return 0;
  }

  std::int64_t run_compacted(const ForceSettings& s, Out out, const Vec& coords, Index N, const Vec& u,
                             const Vec& v, const Vec& f, const Vec& slots) {
    if constexpr (kCompact) {
      if (s.geometry == Geometry::Cached) kernels::launch_predict<Elem, true>(b, compact, coords, N, u);
      else kernels::launch_predict<Elem, false>(b, compact, coords, N, u);
      const Index plastic = kernels::compact_plastic_points<Elem>(b, compact);
      kernels::launch_correct<Elem>(b, compact, plastic);
      run_kernels(kernels::replay_batch<Elem>(b, compact), s, out, coords, N, u, v, f, slots);
      return plastic;
    } else {
      (void)s, (void)out, (void)coords, (void)N, (void)u, (void)v, (void)f, (void)slots;
      return 0;
    }
  }

  template <class E>
  static void run_kernels(const kernels::DevBatch<E>& b, const ForceSettings& s, Out out, const Vec& coords, Index N,
                          const Vec& u, const Vec& v, const Vec& f, const Vec& slots) {
    const bool cached = s.geometry == Geometry::Cached;
    auto launch = [&](Index begin, Index end, bool by_color) {
      if (s.layout == Layout::Flat) {
        if (cached) kernels::launch_flat<E, true>(b, coords, N, u, v, f, slots, out, begin, end, by_color);
        else kernels::launch_flat<E, false>(b, coords, N, u, v, f, slots, out, begin, end, by_color);
      } else {
        if constexpr (PointwiseElement<E>) {
          const int ept = s.elements_per_team;
          if (cached) kernels::launch_team<E, true>(b, coords, N, u, v, f, slots, out, begin, end, by_color, ept);
          else kernels::launch_team<E, false>(b, coords, N, u, v, f, slots, out, begin, end, by_color, ept);
        } else {
          kernels::launch_flat<E, false>(b, coords, N, u, v, f, slots, out, begin, end, by_color);
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
           static_cast<std::int64_t>(b.geometry.extent(0) + b.constants.extent(0) + t.constant.extent(0) +
                                     t.stored.extent(0) + b.state[0].extent(0) + b.state[1].extent(0)) * 8 +
           compact_bytes();
  }
  std::int64_t compact_bytes() const {
    if constexpr (kCompact) {
      return static_cast<std::int64_t>(compact.results.extent(0)) * 8 +
             static_cast<std::int64_t>(compact.flag.extent(0) + compact.list.extent(0)) * 4 +
             static_cast<std::int64_t>(compact.params.extent(0) * sizeof(kernels::DevParams<kernels::ReplayElement<Elem>>));
    } else {
      return 0;
    }
  }
  double min_critical_dt(const Vec& coords, Index N, ExplicitScheme s) const override {
    return kernels::min_critical_dt<Elem>(b, coords, N, s);
  }
  void run_operator(const OperatorSettings& s, bool diagonal, const OperatorCoefficients& k, Out out,
                    const Vec& coords, Index N, const Vec& u, const Vec& x, const Vec& y,
                    const Vec& slots, int unit_dir) override {
    if constexpr (MatrixFreeElement<Elem>) {
      const bool cached = s.geometry == Geometry::Cached;
      const int src = s.tangent == TangentSource::Constant ? 0 : s.tangent == TangentSource::Stored ? 1 : 2;
      auto launch = [&](Index begin, Index end, bool by_color) {
        dispatch(cached, src, diagonal, k, coords, N, u, x, y, slots, out, begin, end, by_color, unit_dir);
      };
      if (s.scatter == Scatter::Colored) {
        for (std::size_t col = 0; col + 1 < b.color_start.size(); ++col)
          launch(b.color_start[col], b.color_start[col + 1], true);
      } else {
        launch(0, b.count, false);
      }
    } else {
      Kokkos::abort("InternalForce: this element has no matrix-free operator (MatrixFreeElement)");
    }
  }

  // The template instantiations: geometry x source x diagonal.
  template <bool C, int S, bool D>
  void go(const OperatorCoefficients& k, const Vec& coords, Index N, const Vec& u, const Vec& x, const Vec& y,
          const Vec& slots, Out out, Index begin, Index end, bool by_color, int unit_dir) {
    kernels::launch_operator<Elem, C, S, D>(b, t, coords, N, k, u, x, y, slots, out, begin, end, by_color, unit_dir);
  }
  template <bool C, int S>
  void go2(bool diagonal, const OperatorCoefficients& k, const Vec& coords, Index N, const Vec& u,
           const Vec& x, const Vec& y, const Vec& slots, Out out, Index begin, Index end, bool by_color,
           int unit_dir) {
    if (diagonal) go<C, S, true>(k, coords, N, u, x, y, slots, out, begin, end, by_color, unit_dir);
    else go<C, S, false>(k, coords, N, u, x, y, slots, out, begin, end, by_color, unit_dir);
  }
  template <bool C>
  void go1(int src, bool diagonal, const OperatorCoefficients& k, const Vec& coords, Index N, const Vec& u,
           const Vec& x, const Vec& y, const Vec& slots, Out out, Index begin, Index end, bool by_color,
           int unit_dir) {
    if (src == 0) go2<C, 0>(diagonal, k, coords, N, u, x, y, slots, out, begin, end, by_color, unit_dir);
    else if (src == 1) go2<C, 1>(diagonal, k, coords, N, u, x, y, slots, out, begin, end, by_color, unit_dir);
    else go2<C, 2>(diagonal, k, coords, N, u, x, y, slots, out, begin, end, by_color, unit_dir);
  }
  void dispatch(bool cached, int src, bool diagonal, const OperatorCoefficients& k, const Vec& coords, Index N,
                const Vec& u, const Vec& x, const Vec& y, const Vec& slots, Out out, Index begin, Index end,
                bool by_color, int unit_dir) {
    if (cached) go1<true>(src, diagonal, k, coords, N, u, x, y, slots, out, begin, end, by_color, unit_dir);
    else go1<false>(src, diagonal, k, coords, N, u, x, y, slots, out, begin, end, by_color, unit_dir);
  }

  void store_tangents(const OperatorSettings& s, const Vec& coords, Index N, const Vec& u) override {
    if constexpr (MatrixFreeElement<Elem>) {
      if (s.geometry == Geometry::Cached) kernels::launch_store_tangents<Elem, true>(b, t, coords, N, u);
      else kernels::launch_store_tangents<Elem, false>(b, t, coords, N, u);
    }
  }
  std::int64_t enable_stored_tangents() override {
    if (t.stored.extent(0) > 0) return 0;
    const auto n = static_cast<std::size_t>(kernels::Storage<Elem>::SIZE) * Elem::NGP * b.padded;
    t.stored = Vec("batch_stored_tangent", n);
    return static_cast<std::int64_t>(n) * 8;
  }
  TangentSymmetry tangent_symmetry() const override { return Elem::TANGENT; }
  bool has_state() const override { return Elem::Material::NSTATE > 0; }
  void element_output(int which, const Kokkos::View<Index*, Space>& elems, const Kokkos::View<Index*, Space>& rows,
                      const Vec& out, std::int64_t offset, const Vec& coords, Index N, const Vec& u) override {
    if (elems.extent(0) > 0) kernels::launch_element_output<Elem>(b, which, elems, rows, out, offset, coords, N, u);
  }
  void commit_state() override { b.current = 1 - b.current; }
  bool set_material_stage(std::int32_t material, int stage) override {
    using MP = typename Elem::Material::Params;
    if constexpr (requires(MP q) { q.loadStage = 0; }) {
      bool any = false;
      for (std::size_t i = 0; i < host_params.size(); ++i)
        if (param_material[i] == material) {
          host_params[i].p.mat.loadStage = stage;
          any = true;
        }
      if (any) {
        auto m = Kokkos::create_mirror_view(b.params);
        for (std::size_t i = 0; i < host_params.size(); ++i) m(i) = host_params[i];
        Kokkos::deep_copy(b.params, m);
        upload_replay_params();
      }
      return any;
    } else {
      (void)material;
      (void)stage;
      return false;
    }
  }
  bool constant_tangent() const override { return Elem::Material::CONSTANT_TANGENT; }

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
  eq_dof_ = upload("eq_dof", c.eq_dof.empty() ? std::vector<std::int8_t>(static_cast<std::size_t>(neq_), 0) : c.eq_dof);
  nodal_mass_ = upload("nodal_mass", c.nodal_mass.empty() ? std::vector<double>(static_cast<std::size_t>(neq_), 0.0)
                                                          : c.nodal_mass);

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
        cache_bytes_ += (static_cast<std::int64_t>(Elem::NGEOM) * Elem::NGP + element_constant_count<Elem>()) *
                        hb.padded * 8;
  });
  if (cache == GeometryCache::Auto) {
    const std::int64_t need = compile::memory_estimate_bytes(c) + cache_bytes_ +
                              (nslots + 2 * static_cast<std::int64_t>(pairs.size()) + neq_) * 8;
    const std::int64_t free_b = device_free_memory_bytes();
    has_cache_ = free_b > 0 && need <= free_b / 10 * 8;  // unknown free memory: no cache
  } else {
    has_cache_ = cache == GeometryCache::On && cache_bytes_ > 0;
  }

  for (const auto& hb : c.batches) batch_tags_.push_back(hb.tag);
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
  std::int64_t n = static_cast<std::int64_t>(4 * neq_ + coords_.extent(0)) * 8 + gather_.bytes();
  for (const auto& b : batches_) n += b->device_bytes();
  return n;
}

double InternalForce::min_critical_dt(ExplicitScheme scheme) const {
  double m = 1e300;
  for (const auto& b : batches_) m = std::min(m, b->min_critical_dt(coords_, nnodes_, scheme));
  return m;
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
  last_plastic_points_ = 0;
  for (auto& b : batches_) last_plastic_points_ += b->run(s, out, coords_, nnodes_, u, v, f, gather_.slots());
  if (out == Out::Slots) gather_.gather(f);
}

bool InternalForce::has_split_material() const {
  for (const auto& b : batches_)
    if (b->split_material()) return true;
  return false;
}

// --- Matrix-free operator ---------------------------------------------------------------

namespace {

void add_nodal_mass(double c, const Vec& m, const Kokkos::View<const double*, Space>& x, const Vec& y,
                    bool diagonal) {
  if (c == 0.0) return;
  Kokkos::parallel_for(
      "operator_nodal_mass", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, static_cast<Index>(y.extent(0))),
      KOKKOS_LAMBDA(const Index q) { y(q) += c * m(q) * (diagonal ? 1.0 : x(q)); });
}

}  // namespace

TangentSymmetry InternalForce::tangent_symmetry() const {
  TangentSymmetry t = TangentSymmetry::SymmetricPositiveDefinite;
  for (const auto& b : batches_) t = weakest(t, b->tangent_symmetry());
  return t;
}

bool InternalForce::constant_tangents() const {
  for (const auto& b : batches_)
    if (!b->constant_tangent()) return false;
  return true;
}

int InternalForce::select_elements(const std::vector<std::int32_t>& tags) {
  std::unordered_map<std::int32_t, std::pair<std::size_t, Index>> where;
  for (std::size_t bi = 0; bi < batch_tags_.size(); ++bi)
    for (std::size_t e = 0; e < batch_tags_[bi].size(); ++e)
      if (batch_tags_[bi][e] >= 0) where[batch_tags_[bi][e]] = {bi, static_cast<Index>(e)};
  std::vector<std::vector<Index>> elems(batches_.size()), rows(batches_.size());
  for (std::size_t r = 0; r < tags.size(); ++r) {
    const auto it = where.find(tags[r]);
    if (it == where.end()) throw io::ModelError("element output: element " + std::to_string(tags[r]) + " is not in the model");
    elems[it->second.first].push_back(it->second.second);
    rows[it->second.first].push_back(static_cast<Index>(r));
  }
  Selection s;
  for (std::size_t bi = 0; bi < batches_.size(); ++bi) {
    s.elems.push_back(upload("output_elements", elems[bi]));
    s.rows.push_back(upload("output_rows", rows[bi]));
  }
  selections_.push_back(std::move(s));
  return static_cast<int>(selections_.size()) - 1;
}

void InternalForce::element_output(int selection, int which, const Vec& out, std::int64_t offset) {
  const auto& s = selections_.at(static_cast<std::size_t>(selection));
  for (std::size_t bi = 0; bi < batches_.size(); ++bi)
    batches_[bi]->element_output(which, s.elems[bi], s.rows[bi], out, offset, coords_, nnodes_, u);
}

bool InternalForce::has_state() const {
  for (const auto& b : batches_)
    if (b->has_state()) return true;
  return false;
}

void InternalForce::commit_state() {
  for (auto& b : batches_) b->commit_state();
}

bool InternalForce::set_material_stage(std::int32_t material, int stage) {
  bool any = false;
  for (auto& b : batches_) any = b->set_material_stage(material, stage) || any;
  Kokkos::fence();
  return any;
}

OperatorSettings InternalForce::resolve(const OperatorSettings& s) const {
  OperatorSettings r = s;
  // The recomputed tangent would call the material without its history.
  if (r.tangent == TangentSource::Recomputed && has_state()) r.tangent = TangentSource::Stored;
  if (r.geometry == Geometry::Cached && !has_cache_) r.geometry = Geometry::Recompute;
  if (r.tangent == TangentSource::Auto) r.tangent = constant_tangents() ? TangentSource::Constant : TangentSource::Stored;
  if (r.tangent == TangentSource::Constant && !constant_tangents()) r.tangent = TangentSource::Stored;
  return r;
}

void InternalForce::enable_stored_tangents() {
  for (auto& b : batches_) stored_bytes_ += b->enable_stored_tangents();
}

void InternalForce::store_tangents(const Vec& u_at, const OperatorSettings& settings) {
  const OperatorSettings s = resolve(settings);
  if (!has_stored_tangents()) Kokkos::abort("InternalForce::store_tangents: call enable_stored_tangents() first");
  for (auto& b : batches_) b->store_tangents(s, coords_, nnodes_, u_at);
}

void InternalForce::apply_operator(const OperatorCoefficients& k, const OperatorSettings& settings, const Vec& u_at,
                                   const Kokkos::View<const double*, Space>& x, const Vec& y) {
  const OperatorSettings s = resolve(settings);
  if (s.tangent == TangentSource::Stored && !has_stored_tangents())
    Kokkos::abort("InternalForce: stored tangents requested without enable_stored_tangents()");
  if (s.scatter == Scatter::Colored && !has_coloring_)
    Kokkos::abort("InternalForce: colored scatter needs CompileOptions::build_coloring");
  const Out out = s.scatter == Scatter::Gather ? Out::Slots : s.scatter == Scatter::Atomic ? Out::Atomic : Out::Plain;
  if (out != Out::Slots) Kokkos::deep_copy(y, 0.0);
  // The kernels read x through a non-const view type; they never write it.
  const Vec xv(const_cast<double*>(x.data()), x.extent(0));
  for (auto& b : batches_) b->run_operator(s, false, k, out, coords_, nnodes_, u_at, xv, y, gather_.slots(), -1);
  if (out == Out::Slots) gather_.gather(y);
  add_nodal_mass(k.mass, nodal_mass_, x, y, false);
}

void InternalForce::operator_diagonal(const OperatorCoefficients& k, const OperatorSettings& settings,
                                      const Vec& u_at, const Vec& d) {
  const OperatorSettings s = resolve(settings);
  if (s.tangent == TangentSource::Stored && !has_stored_tangents())
    Kokkos::abort("InternalForce: stored tangents requested without enable_stored_tangents()");
  if (s.scatter == Scatter::Colored && !has_coloring_)
    Kokkos::abort("InternalForce: colored scatter needs CompileOptions::build_coloring");
  const Out out = s.scatter == Scatter::Gather ? Out::Slots : s.scatter == Scatter::Atomic ? Out::Atomic : Out::Plain;
  if (out != Out::Slots) Kokkos::deep_copy(d, 0.0);
  for (auto& b : batches_) b->run_operator(s, true, k, out, coords_, nnodes_, u_at, d, d, gather_.slots(), -1);
  if (out == Out::Slots) gather_.gather(d);
  add_nodal_mass(k.mass, nodal_mass_, d, d, true);
}

void InternalForce::mass_times_ground(int direction, const OperatorSettings& settings, const Vec& u_at,
                                      const Vec& y) {
  const OperatorSettings s = resolve(settings);
  const Out out = s.scatter == Scatter::Gather ? Out::Slots : s.scatter == Scatter::Atomic ? Out::Atomic : Out::Plain;
  if (s.scatter == Scatter::Colored && !has_coloring_)
    Kokkos::abort("InternalForce: colored scatter needs CompileOptions::build_coloring");
  if (out != Out::Slots) Kokkos::deep_copy(y, 0.0);
  const OperatorCoefficients m{0.0, 0.0, 1.0};
  for (auto& b : batches_) b->run_operator(s, false, m, out, coords_, nnodes_, u_at, y, y, gather_.slots(), direction);
  if (out == Out::Slots) gather_.gather(y);
  // Nodal masses: the free DOFs in this direction (a fixed node's mass carries no load).
  const Vec nm = nodal_mass_;
  const auto dof = eq_dof_;
  Kokkos::parallel_for(
      "ground_nodal_mass", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq_),
      KOKKOS_LAMBDA(const Index q) { if (dof(q) == direction) y(q) += nm(q); });
}

}  // namespace femorax::exec
