#include "exec/internal_force.hpp"
#include "exec/internal_force_batch.hpp"

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



// The typed batches are instantiated in internal_force_pairs_<k>.cpp (internal_force_batch.hpp).
namespace batch_impl {
extern template void register_batches<0>(BatchFactory*);
extern template void register_batches<1>(BatchFactory*);
extern template void register_batches<2>(BatchFactory*);
extern template void register_batches<3>(BatchFactory*);
extern template void register_batches<4>(BatchFactory*);
extern template void register_batches<5>(BatchFactory*);
extern template void register_batches<6>(BatchFactory*);
extern template void register_batches<7>(BatchFactory*);
extern template void register_batches<8>(BatchFactory*);
extern template void register_batches<9>(BatchFactory*);
extern template void register_batches<10>(BatchFactory*);
extern template void register_batches<11>(BatchFactory*);
extern template void register_batches<12>(BatchFactory*);
extern template void register_batches<13>(BatchFactory*);
extern template void register_batches<14>(BatchFactory*);
extern template void register_batches<15>(BatchFactory*);
}  // namespace batch_impl
using batch_impl::upload;

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
  for (const double am : c.nodal_alpha_mass) nodal_damping_ |= am != 0.0;
  if (nodal_damping_) nodal_alpha_mass_ = upload("nodal_alpha_mass", c.nodal_alpha_mass);

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
  std::vector<batch_impl::BatchFactory> factory(static_cast<std::size_t>(registry::num_elements * registry::num_materials),
                                                nullptr);
  [&]<int... K>(std::integer_sequence<int, K...>) {
    (batch_impl::register_batches<K>(factory.data()), ...);
  }(std::make_integer_sequence<int, batch_impl::kSplitUnits>{});
  for (std::size_t i = 0; i < c.batches.size(); ++i) {
    const auto& hb = c.batches[i];
    bool cacheable = false;
    registry::for_each_pair([&]<class E, class M>(int index) {
      if (index == hb.pair) cacheable = E::template Element<typename M::Material>::GEOMETRY_CACHEABLE;
    });
    batches_.push_back(factory[static_cast<std::size_t>(hb.pair)](hb, offsets[i], has_cache_ && cacheable, coords_,
                                                                  nnodes_));
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

std::vector<std::vector<double>> InternalForce::element_critical_dts(ExplicitScheme scheme) const {
  std::vector<std::vector<double>> out;
  for (const auto& b : batches_) out.push_back(b->element_critical_dts(coords_, nnodes_, scheme));
  return out;
}

double InternalForce::min_critical_dt(ExplicitScheme scheme, Batches which) const {
  double m = 1e300;
  for (const auto& b : batches_) {
    if (which == Batches::Changing && !b->stability_may_change()) continue;
    if (which == Batches::Fixed && b->stability_may_change()) continue;
    m = std::min(m, b->min_critical_dt(coords_, nnodes_, scheme));
  }
  return m;
}

bool InternalForce::stability_may_change() const {
  for (const auto& b : batches_)
    if (b->stability_may_change()) return true;
  return false;
}

void InternalForce::prepare_stability_bounds() {
  for (auto& b : batches_)
    if (b->stability_may_change()) b->prepare_stability_bound(coords_, nnodes_);
}

double InternalForce::min_critical_dt_bound(ExplicitScheme scheme) const {
  double m = 1e300;
  for (const auto& b : batches_)
    if (b->stability_may_change()) m = std::min(m, b->min_critical_dt_bound(coords_, nnodes_, scheme));
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
  for (auto& b : batches_) b->set_step(step);
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
  if (nodal_damping_) {
    // The nodal masses' Rayleigh damping force alphaM M v, with the velocity the elements see
    // (OpenSees: Node::getUnbalancedLoadIncInertia).
    const Vec am = nodal_alpha_mass_, ff = f, vv = v;
    Kokkos::parallel_for(
        "nodal_damping_force", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, neq_),
        KOKKOS_LAMBDA(const Index q) { ff(q) += am(q) * vv(q); });
  }
}

void InternalForce::drm_forces(int batch, const Kokkos::View<std::int32_t*, Space>& elems,
                               const Kokkos::View<std::uint32_t*, Space>& masks,
                               const Kokkos::View<std::int32_t*, Space>& node_slot, const Vec& du, const Vec& da,
                               const Vec& dv, bool damped, const Vec& sc, const Vec& st, const Vec& slots,
                               std::int64_t base) {
  batches_[static_cast<std::size_t>(batch)]->drm_forces(has_cache_, coords_, nnodes_, elems, masks, node_slot, du, da,
                                                        dv, damped, sc, st, slots, base);
}

bool InternalForce::drm_needs_state(int batch) const {
  return batches_[static_cast<std::size_t>(batch)]->drm_needs_state();
}

std::int64_t InternalForce::drm_state_doubles(int batch) const {
  return batches_[static_cast<std::size_t>(batch)]->drm_state_doubles();
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
  for (auto& b : batches_) b->set_step(step);
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
  for (auto& b : batches_) b->set_step(step);
  const OperatorSettings s = resolve(settings);
  if (!has_stored_tangents()) Kokkos::abort("InternalForce::store_tangents: call enable_stored_tangents() first");
  for (auto& b : batches_) b->store_tangents(s, coords_, nnodes_, u_at);
}

void InternalForce::apply_operator(const OperatorCoefficients& k, const OperatorSettings& settings, const Vec& u_at,
                                   const Kokkos::View<const double*, Space>& x, const Vec& y) {
  for (auto& b : batches_) b->set_step(step);
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
  if (nodal_damping_) add_nodal_mass(k.damping, nodal_alpha_mass_, x, y, false);
}

void InternalForce::operator_diagonal(const OperatorCoefficients& k, const OperatorSettings& settings,
                                      const Vec& u_at, const Vec& d) {
  for (auto& b : batches_) b->set_step(step);
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
  if (nodal_damping_) add_nodal_mass(k.damping, nodal_alpha_mass_, d, d, true);
}

void InternalForce::mass_times_ground(int direction, const OperatorSettings& settings, const Vec& u_at,
                                      const Vec& y) {
  for (auto& b : batches_) b->set_step(step);
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
