// Explicit internal-force kernels (stdBrick + ElasticIsotropic, Rayleigh damping) on a
// Femora-style soil column, renumbering on: every layout x scatter mode, and the team layout
// with several elements per team. Compulsory traffic per call: coordinates, u, v, the
// element maps (8 nodes + 24 equations + 1 parameter index, int32), and f (zeroed, then
// read-modify-write once per equation).

#include "compile/batches.hpp"
#include "exec/explicit_force.hpp"
#include "exec/layout_defaults.hpp"
#include "exec/runtime.hpp"
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
  const std::int64_t bytes = 3 * N * 8 + 2 * neq * 8 + nel * 33 * 4 + 3 * neq * 8;

  struct Variant {
    exec::Layout layout;
    exec::Scatter scatter;
    int ept;
  };
  std::vector<Variant> variants;
  for (auto sc : {exec::Scatter::Atomic, exec::Scatter::Colored}) {
    variants.push_back({exec::Layout::Flat, sc, 1});
    for (int ept : {1, 2, 4, 8})
      if (ept <= exec::ExplicitForce::max_elements_per_team())
        variants.push_back({exec::Layout::Team, sc, ept});
  }
  const std::string size = std::to_string(s.nx) + "^3";
  const exec::LayoutChoice production = exec::default_explicit_brick_layout();
  for (const auto& v : variants) {
    std::string name = "explicit_force/" + size + "/" + exec::to_string(v.layout) + "_" +
                       exec::to_string(v.scatter);
    if (v.layout == exec::Layout::Team) name += "_ept" + std::to_string(v.ept);
    registry.add({.name = name,
                  .unit = "call",
                  .bytes = bytes,
                  .flops = 0.0,
                  .items = nel,
                  .run = [k, v] { k->compute(v.layout, v.scatter, v.ept); },
                  // Gate the production default and the deterministic (colored) form of the
                  // same layout; the other variants inform layout_defaults.hpp.
                  .gated = v.layout == production.layout &&
                           (v.layout == exec::Layout::Flat ||
                            v.ept == exec::ExplicitForce::default_elements_per_team())});
  }
}

}  // namespace femorax::bench
