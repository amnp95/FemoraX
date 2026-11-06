#pragma once

// Inner loops of physics code (docs/design.md, "Memory and parallelism contract", rule 3).
//
// Work inside one element or point that can use several threads (node-pair tangent
// blocks, yield surfaces, fibers, network neurons) is written as
//
//   inner_for(ctx, n, [&](int i) { ... });             // independent iterations
//   T s = inner_reduce(ctx, n, [&](int i) { return term(i); });   // sum
//
// called unqualified, so argument-dependent lookup picks the overload for the context
// type. SerialInner (here) runs plain loops; the launch layer's context
// (src/exec/inner_kokkos.hpp) maps them to Kokkos::ThreadVectorRange lanes.
//
// Rules that make every mapping give bit-identical results:
// - Each iteration of inner_for writes only its own outputs, into memory the caller
//   provides (the launch layer puts it in team scratch): locals are private per lane on a
//   GPU, so a local array written in an inner_for is not shared between lanes.
// - After an inner_for, a lane reads only what the same iteration index wrote: combine
//   results with an inner_for over the same index space that produced them. There is no
//   lane barrier (Kokkos has no portable one below the team level).
// - inner_reduce adds the terms in index order (term 0 first), whatever the lane split.
// - Code outside inner loops runs redundantly on every lane of a point. State it writes
//   must be a pure function of its inputs (all lanes write the same value).

#include "physics/macros.hpp"

namespace femorax {

// Plain loops: CPU kernels, tests, and elements that do not split their inner work.
struct SerialInner {};

template <class F>
FEMORAX_HD inline void inner_for(const SerialInner&, int n, F&& f) {
  for (int i = 0; i < n; ++i) f(i);
}

template <class F>
FEMORAX_HD inline auto inner_reduce(const SerialInner&, int n, F&& f) -> decltype(f(0)) {
  decltype(f(0)) acc = 0;
  for (int i = 0; i < n; ++i) acc += f(i);
  return acc;
}

}  // namespace femorax
