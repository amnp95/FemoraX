// Explicit internal-force kernels (stdBrick + ElasticIsotropic, Rayleigh damping) on a
// Femora-style soil column, renumbering on: every layout x scatter mode, and the team layout
// with several elements per team. Compulsory traffic per call: coordinates, u, v, the
// element maps (8 nodes + 24 equations + 1 parameter index, int32), and f (zeroed, then
// read-modify-write once per equation).

#include "compile/batches.hpp"
#include "exec/explicit_force.hpp"
#include "exec/layout_defaults.hpp"
#include "exec/runtime.hpp"
#include "flop_count.hpp"
#include "harness.hpp"
#include "../tests/common/synthetic_models.hpp"

#include <memory>
#include <string>

namespace femorax::bench {

void register_explicit(Registry& registry, const Options& options) {
  testing::ColumnSpec s;
  s.nx = s.ny = s.nz = options.quick ? 6 : 40;
  const auto model = testing::soil_column(s);
  compile::CompileOptions co;
  co.renumbering = compile::Renumbering::RCM;
  co.simd_width = simd_width();
  const auto compiled = std::make_shared<compile::Compiled>(compile::compile_model(model, co));
  auto k = std::make_shared<exec::ExplicitForce>(*compiled);
  k->fill(1e-4, 1e-3);

  const std::int64_t neq = k->num_equations(), nel = k->num_elements();
  const std::int64_t N = compiled->dofs.num_nodes();
  // Compulsory traffic: u, v read, f zeroed and read-modify-written, the element maps, and
  // either the coordinates (recompute) or the geometry cache (cached).
  const std::int64_t common = 2 * neq * 8 + 3 * neq * 8 + nel * 33 * 4;
  const std::int64_t bytes_recompute = common + 3 * N * 8;
  const std::int64_t bytes_cached = common + k->geometry_cache_bytes();

  // Exact FLOPs of one call: per element, from the physics source, by parameter set.
  auto flops_of = [&](bool team, bool cached) {
    double total = 0;
    for (const auto& b : compiled->batches)
      for (std::int32_t e = 0; e < b.count; ++e) {
        const auto& ps = b.params[b.param[e]];
        total += brick_flops_per_element({ps.massType, ps.damping.alphaM, ps.damping.betaK,
                                          ps.damping.betaKinit, ps.damping.betaKcomm},
                                         team, cached);
      }
    return total;
  };
  const double flops[2][2] = {{flops_of(false, false), flops_of(false, true)},
                              {flops_of(true, false), flops_of(true, true)}};

  struct Variant {
    exec::Layout layout;
    exec::Scatter scatter;
    int ept;
    exec::Geometry geometry;
  };
  std::vector<Variant> variants;
  for (auto g : {exec::Geometry::Recompute, exec::Geometry::Cached})
    for (auto sc : {exec::Scatter::Atomic, exec::Scatter::Colored}) {
      variants.push_back({exec::Layout::Flat, sc, 1, g});
      for (int ept : {1, 2, 4, 8})
        if (ept <= exec::ExplicitForce::max_elements_per_team())
          variants.push_back({exec::Layout::Team, sc, ept, g});
    }
  const std::string size = std::to_string(s.nx) + "^3";
  const exec::LayoutChoice production = exec::default_explicit_brick_layout();
  for (const auto& v : variants) {
    const bool team = v.layout == exec::Layout::Team, cached = v.geometry == exec::Geometry::Cached;
    std::string name = "explicit_force/" + size + "/" + exec::to_string(v.layout) + "_" +
                       exec::to_string(v.scatter);
    if (team) name += "_ept" + std::to_string(v.ept);
    if (cached) name += "_cached";
    registry.add({.name = name,
                  .unit = "call",
                  .bytes = cached ? bytes_cached : bytes_recompute,
                  .flops = flops[team][cached],
                  .items = nel,
                  .run = [k, v] { k->compute(v.layout, v.scatter, v.ept, v.geometry); },
                  // Gate the production default and the deterministic (colored) form of the
                  // same layout; the other variants inform layout_defaults.hpp.
                  .gated = v.layout == production.layout &&
                           (v.layout == exec::Layout::Flat ||
                            v.ept == exec::ExplicitForce::default_elements_per_team())});
  }
}

}  // namespace femorax::bench
