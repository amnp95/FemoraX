#pragma once

// Internal plus Rayleigh damping force of all element batches, for explicit stages:
//   f = sum over elements of fe(u, v), scattered into equation space.
// One fused gather-compute-scatter kernel per batch (CLAUDE.md, Performance), in two thread
// layouts and two scatter modes (docs/design.md, "Element kernels"):
//   Layout::Flat     one thread per element, the whole element (StdBrick::compute)
//   Layout::Team     one thread per Gauss point, elements_per_team elements per team:
//                    nodal data gathered cooperatively into team scratch once, point
//                    updates in parallel, per-node reduction in Gauss point order
//   Scatter::Atomic  atomic adds (fast, summation order varies between runs)
//   Scatter::Colored one launch per color, no two elements of a color share an equation
//                    (deterministic: bit-identical for any thread count)
// Flat and Team give bit-identical element forces. All memory is allocated in the
// constructor; compute() allocates nothing.

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <vector>

#include "compile/batches.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"

namespace femorax::exec {

enum class Layout { Flat, Team };
enum class Scatter { Atomic, Colored };

const char* to_string(Layout l);
const char* to_string(Scatter s);

class ExplicitForce {
 public:
  using Space = Kokkos::DefaultExecutionSpace;
  using Vec = Kokkos::View<double*, Space>;
  using Brick = StdBrick<ElasticIsotropic>;

  explicit ExplicitForce(const compile::Compiled& c);

  // f = internal + damping force at (u, v). Zeroes f first.
  void compute(Layout layout, Scatter scatter, int elements_per_team = 0);

  // Set every entry of u and v (benchmarks; tests copy full fields into u and v).
  void fill(double u0, double v0);

  // Equation-space vectors, device-resident. Set u and v before compute().
  Vec u, v, f;
  std::int32_t num_equations() const { return neq_; }
  std::int64_t num_elements() const;

  // Default elements per team: fill one SIMD width (warp/wavefront) with Gauss points,
  // within the backend's team scratch limit.
  static int default_elements_per_team();
  // Most elements per team that fit the backend's level-0 team scratch (requests above it
  // are clamped).
  static int max_elements_per_team();

 private:
  struct DevParams {
    Brick::Params p;
    Rayleigh damp;
  };
  struct DevBatch {
    std::int32_t count = 0, padded = 0;
    Kokkos::View<std::int32_t*, Space> node, eq, param, color_elements;
    std::vector<std::int32_t> color_start;  // host: launch bounds per color
    Kokkos::View<DevParams*, Space> params;
  };

  void launch_flat(const DevBatch& b, bool atomic, std::int32_t begin, std::int32_t end,
                   bool by_color);
  void launch_team(const DevBatch& b, bool atomic, std::int32_t begin, std::int32_t end,
                   bool by_color, int ept);

  std::int32_t neq_ = 0, nnodes_ = 0;
  Kokkos::View<double*, Space> coords_;
  std::vector<DevBatch> batches_;
};

}  // namespace femorax::exec
