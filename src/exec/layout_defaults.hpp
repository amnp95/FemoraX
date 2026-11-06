#pragma once

// Default kernel layout per backend and element-material pair, chosen by the benchmark
// (benchmarks/bench_explicit.cpp; docs/design.md, "Choosing the layout"). Update an entry only
// with the benchmark numbers in the commit message.

#include <Kokkos_Core.hpp>

#include "exec/explicit_force.hpp"

namespace femorax::exec {

struct LayoutChoice {
  Layout layout;
  Scatter scatter;  // production default; validation runs always use Scatter::Colored
};

// stdBrick + ElasticIsotropic, explicit force.
// - OpenMP (i7-11370H, 8 threads, 64k elements, 2026-11-05): flat atomic 19.5 ms, flat colored
//   21.6 ms, team 221-680 ms (ept 1-4). Flat wins by 10-30x on CPU.
// - CUDA, HIP: not measured yet (no GPU); the team layout is the expected winner and the
//   entry becomes measured when the GPU builds run.
inline LayoutChoice default_explicit_brick_layout() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
  if constexpr (!std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::DefaultHostExecutionSpace>)
    return {Layout::Team, Scatter::Atomic};
#endif
  return {Layout::Flat, Scatter::Atomic};
}

}  // namespace femorax::exec
