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
//   Scatter::Gather     each element writes its contributions to its own slots, in
//                       locality order; then each equation adds its slots in a fixed,
//                       precomputed order (deterministic, keeps the element order)
//   Geometry::Recompute shape-function derivatives and det(J) recomputed every call
//   Geometry::Cached    computed once at construction (small strain), read every call
// Every combination gives bit-identical element forces on a given backend. All memory is
// allocated in the constructor; compute() allocates nothing.

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "compile/batches.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"

namespace femorax::exec {

enum class Layout { Flat, Team };
enum class Scatter { Atomic, Colored, Gather };
enum class Geometry { Recompute, Cached };

const char* to_string(Layout l);
const char* to_string(Scatter s);
const char* to_string(Geometry g);

// How compute() runs. Everything above the kernels (time loop, future elements and
// materials) asks for forces through these settings; the per-device defaults and the
// auto-tune are in exec/force_settings.hpp.
struct ForceSettings {
  Layout layout = Layout::Flat;
  Scatter scatter = Scatter::Atomic;
  int elements_per_team = 0;  // team layout; 0 = the backend's default
  Geometry geometry = Geometry::Recompute;
  bool deterministic() const { return scatter != Scatter::Atomic; }
};

std::string to_string(const ForceSettings& s);

// Whether the geometry cache is built: On, Off, or Auto (built when the model, the cache
// and the work vectors fit in 80% of the device's free memory).
enum class GeometryCache { Auto, On, Off };

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
  // Gather scatter: this batch's slots start at slot_offset in the global slot array,
  // slot[slot_offset + k * padded + e] = element e's k-th force value.
  std::int64_t slot_offset = 0;
};
}  // namespace detail

class ExplicitForce {
 public:
  using Space = detail::Space;
  using Vec = Kokkos::View<double*, Space>;
  using Brick = detail::Brick;

  // The geometry cache holds NGEOM x NGP doubles per element (Geometry::Cached reads it).
  explicit ExplicitForce(const compile::Compiled& c, GeometryCache cache = GeometryCache::Auto);

  // f = internal + damping force at (u, v). Geometry::Cached without a cache falls back to
  // recomputing (same bits). Allocates nothing.
  void compute(const ForceSettings& settings);

  // The settings compute() actually runs for these (cache fallback, team size clamp).
  ForceSettings resolve(const ForceSettings& settings) const;

  // Set every entry of u and v (benchmarks; tests copy full fields into u and v).
  void fill(double u0, double v0);

  // Equation-space vectors, device-resident. Set u and v before compute().
  Vec u, v, f;
  std::int32_t num_equations() const { return neq_; }
  std::int32_t num_nodes() const { return nnodes_; }
  std::int64_t num_elements() const;
  bool has_geometry_cache() const { return has_cache_; }
  // Bytes of the cache (allocated or not) and of the gather slots and lists.
  std::int64_t geometry_cache_bytes() const;
  std::int64_t gather_bytes() const;

  // Default elements per team: fill one SIMD width (warp/wavefront) with Gauss points,
  // within the backend's team scratch limit.
  static int default_elements_per_team();
  // Most elements per team that fit the backend's level-0 team scratch (requests above it
  // are clamped).
  static int max_elements_per_team();

 private:
  std::int32_t neq_ = 0, nnodes_ = 0;
  bool has_cache_ = false;
  std::int64_t cache_bytes_ = 0;
  Kokkos::View<double*, Space> coords_;
  std::vector<detail::DevBatch> batches_;
  // Gather scatter: element slots, and for equation q its slots
  // gather_slot[gather_start[q] .. gather_start[q + 1]) in summation order.
  Kokkos::View<double*, Space> slots_;
  Kokkos::View<std::int64_t*, Space> gather_start_;
  Kokkos::View<std::int64_t*, Space> gather_slot_;
};

}  // namespace femorax::exec
