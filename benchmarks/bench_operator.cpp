// Matrix-free operator y = K_eff x (exec/matrix_free_operator.hpp) on the n x n x n soil
// column with Rayleigh damping and consistent mass, in every scatter x geometry x tangent
// source variant: the per-iteration cost of the implicit solvers. Compare with
// solvers/spmv/<n>^3, the assembled-matrix reference on the same model.
//
// Compulsory traffic: x read and y written once per equation, the element equation maps
// and parameter indices, the geometry (cache, or the node coordinates), the stored tangent
// when used, u when the tangent is recomputed, and the gather's slots (written and read)
// and pair lists.

#include <memory>
#include <string>

#include "../tests/common/synthetic_models.hpp"
#include "compile/batches.hpp"
#include "exec/internal_force.hpp"
#include "exec/runtime.hpp"
#include "harness.hpp"

namespace femorax::bench {

void register_operator(Registry& registry, const Options& options) {
  testing::ColumnSpec s;
  s.nx = s.ny = s.nz = options.quick ? 6 : 40;
  s.lumped = false;
  const auto model = testing::soil_column(s);
  compile::CompileOptions co;
  co.simd_width = simd_width();
  co.build_coloring = true;
  const auto compiled = std::make_shared<compile::Compiled>(compile::compile_model(model, co));
  auto k = std::make_shared<exec::InternalForce>(*compiled, exec::GeometryCache::On);
  k->enable_stored_tangents();
  const std::int64_t neq = k->num_equations(), nel = k->num_elements(), N = compiled->dofs.num_nodes();
  auto x = std::make_shared<exec::Vec>("x", static_cast<std::size_t>(neq));
  auto y = std::make_shared<exec::Vec>("y", static_cast<std::size_t>(neq));
  auto u = std::make_shared<exec::Vec>("u", static_cast<std::size_t>(neq));
  Kokkos::deep_copy(*x, 1e-3);
  k->store_tangents(*u, {});
  const OperatorCoefficients coef{1.0, 2e3, 1.6e7};

  const std::string def = exec::to_string(k->resolve(exec::OperatorSettings{}));
  for (auto sc : {exec::Scatter::Gather, exec::Scatter::Colored, exec::Scatter::Atomic})
    for (auto g : {exec::Geometry::Recompute, exec::Geometry::Cached})
      for (auto t : {exec::TangentSource::Constant, exec::TangentSource::Stored, exec::TangentSource::Recomputed}) {
        const exec::OperatorSettings v = k->resolve({sc, g, t});
        const std::string name = exec::to_string(v);
        std::int64_t bytes = 2 * neq * 8 + nel * 25 * 4;
        bytes += g == exec::Geometry::Cached ? k->geometry_cache_bytes() : 3 * N * 8;
        if (t == exec::TangentSource::Stored) bytes += k->stored_tangent_bytes();
        if (t == exec::TangentSource::Recomputed) bytes += neq * 8;
        if (sc == exec::Scatter::Gather) bytes += 2 * nel * 24 * 8 + nel * 24 * 8 + neq * 8;
        registry.add({.name = "operator/" + std::to_string(s.nx) + "^3/" + name,
                      .unit = "call",
                      .bytes = bytes,
                      .flops = 0.0,
                      .items = nel,
                      .run = [k, v, coef, x, y, u] { k->apply_operator(coef, v, *u, *x, *y); },
                      // The default (what the solvers run) is gated; the variants decide it.
                      .gated = name == def});
      }
}

}  // namespace femorax::bench
