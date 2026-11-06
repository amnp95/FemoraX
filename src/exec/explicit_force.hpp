#pragma once

// Internal plus Rayleigh damping force of all element batches, for explicit stages:
//   f = sum over elements of fe(u, v), scattered into equation space.
// One fused gather-compute-scatter kernel per batch (CLAUDE.md, Performance), with three
// independent choices (docs/design.md, "Element kernels"):
//   Layout::Flat        one thread per element, the whole element (StdBrick::compute)
//   Layout::Team        one thread per Gauss point, elements_per_team elements per team:
//                       nodal data gathered cooperatively into team scratch once, point
//                       updates in parallel, per-node reduction in Gauss point order
//   Scatter::Atomic     atomic adds (fast, summation order varies between runs)
//   Scatter::Colored    one launch per color, no two elements of a color share an equation
//                       (deterministic: bit-identical for any thread count)
//   Geometry::Recompute shape-function derivatives and det(J) recomputed every call
//   Geometry::Cached    computed once at construction (small strain), read every call
// Every combination gives bit-identical element forces on a given backend. All memory is
// allocated in the constructor; compute() allocates nothing.

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <vector>

#include "compile/batches.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"

namespace femorax::exec {

enum class Layout { Flat, Team };
enum class Scatter { Atomic, Colored };
enum class Geometry { Recompute, Cached };

const char* to_string(Layout l);
const char* to_string(Scatter s);
const char* to_string(Geometry g);

namespace detail {
// Device data of one batch. Public (not class members) because nvcc does not allow private
// types in device lambdas.
using Space = Kokkos::DefaultExecutionSpace;
using Brick = StdBrick<ElasticIsotropic>;

struct DevParams {
  Brick::Params p;
  Rayleigh damp;
};

struct DevBatch {
  std::int32_t count = 0, padded = 0;
  Kokkos::View<std::int32_t*, Space> node, eq, param, color_elements;
  std::vector<std::int32_t> color_start;  // host: launch bounds per color
  Kokkos::View<DevParams*, Space> params;
  // Geometry cache, per Gauss point: value[k * (NGP * padded) + e * NGP + gp], k < NGEOM.
  Kokkos::View<double*, Space> geometry;
};
}  // namespace detail

class ExplicitForce {
 public:
  using Space = detail::Space;
  using Vec = Kokkos::View<double*, Space>;
  using Brick = detail::Brick;

  // with_geometry_cache allocates and fills the geometry cache (NGEOM x NGP doubles per
  // element); Geometry::Cached needs it.
  explicit ExplicitForce(const compile::Compiled& c, bool with_geometry_cache = true);

  // f = internal + damping force at (u, v). Zeroes f first.
  void compute(Layout layout, Scatter scatter, int elements_per_team = 0,
               Geometry geometry = Geometry::Recompute);

  // Set every entry of u and v (benchmarks; tests copy full fields into u and v).
  void fill(double u0, double v0);

  // Equation-space vectors, device-resident. Set u and v before compute().
  Vec u, v, f;
  std::int32_t num_equations() const { return neq_; }
  std::int32_t num_nodes() const { return nnodes_; }
  std::int64_t num_elements() const;
  bool has_geometry_cache() const { return has_cache_; }
  std::int64_t geometry_cache_bytes() const;

  // Default elements per team: fill one SIMD width (warp/wavefront) with Gauss points,
  // within the backend's team scratch limit.
  static int default_elements_per_team();
  // Most elements per team that fit the backend's level-0 team scratch (requests above it
  // are clamped).
  static int max_elements_per_team();

 private:
  std::int32_t neq_ = 0, nnodes_ = 0;
  bool has_cache_ = false;
  Kokkos::View<double*, Space> coords_;
  std::vector<detail::DevBatch> batches_;
};

}  // namespace femorax::exec
