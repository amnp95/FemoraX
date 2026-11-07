// Explicit internal-force kernels (stdBrick + ElasticIsotropic) on a Femora-style soil column,
// renumbering on.
//   explicit_force/<n>^3/<variant>         Rayleigh damping (the soil column's two regions),
//                                          every layout x scatter x geometry variant
//   explicit_force_nodamp/<n>^3/<variant>  no damping, flat atomic only: the work of
//                                          OpenSees' update() + getResistingForce(), for the
//                                          comparison with tests/reference/drivers/brick_bench
// Compulsory traffic per call: u, v, the element maps (8 nodes + 24 equations + 1 parameter
// index, int32), f (zeroed, then read-modify-written once per equation), and either the
// coordinates (recompute) or the geometry cache (cached); the gather scatter adds its slots
// (written once, read once) and lists. FLOPs are exact, from the physics source
// (flop_count.cpp).

#include "compile/batches.hpp"
#include "exec/internal_force.hpp"
#include "exec/force_settings.hpp"
#include "exec/runtime.hpp"
#include "flop_count.hpp"
#include "harness.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
#include "../tests/common/synthetic_models.hpp"

#include <memory>
#include <string>

namespace femorax::bench {

namespace {

void register_model(Registry& registry, const Options& options, bool damped) {
  testing::ColumnSpec s;
  s.nx = s.ny = s.nz = options.side();
  s.damping = damped;
  const auto model = testing::soil_column(s);
  compile::CompileOptions co;
  co.renumbering = compile::Renumbering::RCM;
  co.simd_width = simd_width();
  co.build_coloring = damped;  // colored variants are measured on the damped model
  const auto compiled = std::make_shared<compile::Compiled>(compile::compile_model(model, co));
  auto k = std::make_shared<exec::InternalForce>(*compiled, exec::GeometryCache::On);
  k->fill(1e-4, 1e-3);

  const std::int64_t neq = k->num_equations(), nel = k->num_elements();
  const std::int64_t N = compiled->dofs.num_nodes();
  const std::int64_t common = 2 * neq * 8 + 3 * neq * 8 + nel * 33 * 4;
  auto bytes_of = [&](const exec::ForceSettings& v) {
    std::int64_t b = common + (v.geometry == exec::Geometry::Cached ? k->geometry_cache_bytes() : 3 * N * 8);
    if (v.scatter == exec::Scatter::Gather) b += 2 * nel * 24 * 8 + nel * 24 * 8 + neq * 8;
    return b;
  };
  auto flops_of = [&](bool team, bool cached) {
    double total = 0;
    for (const auto& b : compiled->batches)
      for (std::int32_t e = 0; e < b.count; ++e) {
        const auto& ps = b.params[b.param[e]];
        const int massType = ps.params<StdBrick<ElasticIsotropic>>().massType;
        total += brick_flops_per_element({massType, ps.damping.alphaM, ps.damping.betaK,
                                          ps.damping.betaKinit, ps.damping.betaKcomm},
                                         team, cached);
      }
    return total;
  };
  const double flops[2][2] = {{flops_of(false, false), flops_of(false, true)},
                              {flops_of(true, false), flops_of(true, true)}};

  std::vector<exec::ForceSettings> variants;
  if (damped) {
    variants = exec::all_variants(*k);
  } else {
    variants = {{exec::Layout::Flat, exec::Scatter::Atomic, 0, exec::Geometry::Recompute},
                {exec::Layout::Flat, exec::Scatter::Atomic, 0, exec::Geometry::Cached}};
  }
  const std::string def = exec::to_string(k->resolve(exec::default_force_settings()));
  const std::string group = std::string(damped ? "explicit_force/" : "explicit_force_nodamp/") +
                            std::to_string(s.nx) + "^3/";
  for (const auto& v0 : variants) {
    const auto v = k->resolve(v0);
    const std::string name = exec::to_string(v);
    registry.add({.name = group + name,
                  .unit = "call",
                  .bytes = bytes_of(v),
                  .flops = flops[v.layout == exec::Layout::Team][v.geometry == exec::Geometry::Cached],
                  .items = nel,
                  .run = [k, v] { k->compute(v); },
                  // Gate this device's default (and the undamped comparison runs); the other
                  // variants inform the defaults and the auto-tune.
                  .gated = !damped || name == def});
  }
}

}  // namespace

void register_explicit(Registry& registry, const Options& options) {
  register_model(registry, options, true);
  register_model(registry, options, false);
}

}  // namespace femorax::bench
