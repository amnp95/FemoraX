// Explicit internal-force kernels (stdBrick + ElasticIsotropic) on a Femora-style soil column,
// renumbering on.
//   explicit_force/<n>^3/<variant>         Rayleigh damping (the soil column's two regions),
//                                          every layout x scatter x geometry variant
//   explicit_force_ssp/<n>^3/<variant>     the same with SSPbrick (one point and the
//                                          hourglass term; the geometry cache adds its
//                                          element constants); FLOPs not counted yet
//   explicit_force_nodamp/<n>^3/<variant>  no damping, flat atomic only: the work of
//                                          OpenSees' update() + getResistingForce(), for the
//                                          comparison with tests/reference/drivers/brick_bench
//   explicit_force_pdmy[_ssp]/<n>^3/<mix>/<variant>
//                                          PressureDependMultiYield (dense below, loose above,
//                                          plastic stage) under a simple shear field of two
//                                          sizes: "elastic" (1e-7, every point stays elastic),
//                                          "mixed" (growing as h^8 to 2e-3 at the top: only
//                                          the upper part yields) and "plastic" (2e-3, every
//                                          point needs the corrector); the gather variants, fused and with the
//                                          compacted corrector (exec/compacted_kernels.hpp).
//                                          The fraction of plastic points is in the name's
//                                          report line. Traffic adds the material state
//                                          (committed read, trial written: NSTATE doubles per
//                                          point each) and, compacted, the per-point results
//                                          (48 doubles written and read) and flag and list.
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
#include "physics/materials/pressure_depend_multi_yield.hpp"
#include "../tests/common/synthetic_models.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

namespace femorax::bench {

namespace {

void register_model(Registry& registry, const Options& options, bool damped,
                    io::ElementType type = io::ElementType::StdBrick) {
  const bool ssp = type == io::ElementType::SSPbrick;
  testing::ColumnSpec s;
  s.nx = s.ny = s.nz = options.side();
  s.damping = damped;
  s.element = type;
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
    if (ssp) return total;
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
  const std::string group = std::string(ssp ? "explicit_force_ssp/" : damped ? "explicit_force/" : "explicit_force_nodamp/") +
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
                  .gated = !damped || name == def,
                  .bound = ssp ? "compute" : ""});
  }
}

io::PressureDependMultiYieldData soil(bool loose) {  // Femora's dense and loose Ottawa sand
  io::PressureDependMultiYieldData d;
  d.rho = loose ? 19.1 / 9.81 : 19.9 / 9.81;
  d.refShearModul = loose ? 75.0e3 : 145.0e3;
  d.refBulkModul = 2.0 * d.refShearModul * 1.3 / 1.2;
  d.frictionAng = loose ? 29.0 : 40.0;
  d.peakShearStra = 0.1;
  d.refPress = 80.0;
  d.pressDependCoe = 0.5;
  d.PTAng = loose ? 29.0 : 27.0;
  d.contrac = loose ? 0.21 : 0.03;
  d.dilat1 = loose ? 0.0 : 0.8;
  d.dilat2 = loose ? 0.0 : 5.0;
  d.liquefac1 = loose ? 10.0 : 0.0;
  d.liquefac2 = loose ? 0.02 : 0.0;
  d.liquefac3 = loose ? 1.0 : 0.0;
  d.noYieldSurf = 20;
  d.e = loose ? 0.85 : 0.45;
  return d;
}

void register_pdmy(Registry& registry, const Options& options, io::ElementType type) {
  const bool ssp = type == io::ElementType::SSPbrick;
  testing::ColumnSpec s;
  s.nx = s.ny = s.nz = options.side();
  s.element = type;
  auto model = testing::soil_column(s);
  model.materials[0].data = soil(false);
  model.materials[1].data = soil(true);
  compile::CompileOptions co;
  co.renumbering = compile::Renumbering::RCM;
  co.simd_width = simd_width();
  const auto compiled = std::make_shared<compile::Compiled>(compile::compile_model(model, co));
  const std::int64_t neq = compiled->dofs.num_equations;
  for (const char* mix : {"elastic", "mixed", "plastic"}) {
    auto k = std::make_shared<exec::InternalForce>(*compiled, exec::GeometryCache::On);
    k->set_material_stage(0, 1);
    k->set_material_stage(1, 1);
    // Shear u_x with strain gamma(h) (and a small compression), from the committed initial
    // state; no commit, so every call does the same work. elastic and plastic: uniform
    // strain; mixed: strain growing as h^8 to 2e-3 at the top, so only the upper part yields.
    const std::string m(mix);
    const double gamma = m == "elastic" ? 1e-7 : 2e-3;
    const int power = m == "mixed" ? 8 : 0;
    auto hu = Kokkos::create_mirror_view(k->u);
    const std::int32_t N = compiled->dofs.num_nodes();
    double zmin = 1e300, zmax = -1e300;
    for (std::int32_t i = 0; i < N; ++i) {
      zmin = std::min(zmin, compiled->coords[2 * N + i]);
      zmax = std::max(zmax, compiled->coords[2 * N + i]);
    }
    const double H = zmax - zmin;
    for (std::int32_t i = 0; i < N; ++i)
      for (int d = 0; d < 3; ++d) {
        const auto q = compiled->dofs.equation(i, d);
        if (q < 0) continue;
        const double h = compiled->coords[2 * N + i] - zmin;
        // u = integral of gamma (h/H)^power: gamma H (h/H)^(power+1) / (power+1)
        const double ux = gamma * H * std::pow(h / H, power + 1) / (power + 1);
        hu(q) = d == 0 ? ux : d == 2 ? -0.1 * ux : 0.0;
      }
    Kokkos::deep_copy(k->u, hu);
    Kokkos::deep_copy(k->v, 1e-3);
    exec::ForceSettings probe = exec::default_force_settings();
    probe.material = exec::MaterialPath::Compacted;
    k->compute(probe);
    const std::int64_t points = k->num_elements() * (ssp ? 1 : 8);
    std::printf("[ explicit_force_pdmy%s %s: %lld of %lld points need the corrector ]\n", ssp ? "_ssp" : "", mix,
                static_cast<long long>(k->last_plastic_points()), static_cast<long long>(points));
    std::int64_t nstate = 0;
    for (const auto& b : compiled->batches) nstate += static_cast<std::int64_t>(b.nstate) * b.count;
    const std::int64_t common = 2 * neq * 8 + 3 * neq * 8 + k->num_elements() * 33 * 4 + k->geometry_cache_bytes() +
                                2 * nstate * 8 + 2 * k->num_elements() * 24 * 8 * 2;
    const std::string group = std::string(ssp ? "explicit_force_pdmy_ssp/" : "explicit_force_pdmy/") +
                              std::to_string(s.nx) + "^3/" + mix + "/";
    for (const auto& v0 : exec::all_variants(*k)) {
      if (v0.scatter != exec::Scatter::Gather || v0.geometry != exec::Geometry::Cached) continue;
      const auto v = k->resolve(v0);
      const bool compacted = v.material == exec::MaterialPath::Compacted;
      registry.add({.name = group + exec::to_string(v),
                    .unit = "call",
                    .bytes = common + (compacted ? points * (2 * 48 * 8 + 3 * 4) : 0),
                    .flops = 0,
                    .items = k->num_elements(),
                    .run = [k, v] { k->compute(v); },
                    .gated = false,
                    .bound = "compute"});
    }
  }
}

}  // namespace

void register_explicit(Registry& registry, const Options& options) {
  register_model(registry, options, true);
  register_model(registry, options, false);
  register_model(registry, options, true, io::ElementType::SSPbrick);
  register_pdmy(registry, options, io::ElementType::StdBrick);
  register_pdmy(registry, options, io::ElementType::SSPbrick);
}

}  // namespace femorax::bench
