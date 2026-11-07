#pragma once

// Vector kernels for the solvers, with deterministic reductions.
//
// Kokkos' parallel_reduce sums in an order that depends on the number of threads, so a dot
// product could change in the last bit between a 1-thread and an 8-thread run. These
// reductions fix the order instead:
//   - the vector is cut into chunks of kChunk entries, in index order;
//   - a team of T lanes (the caller passes exec::simd_width(): 1 on CPU backends, the warp
//     or wavefront on GPUs) reduces one chunk: lane l sums entries l, l + T, l + 2T, ... of the chunk in
//     index order, then lane 0 adds the T lane sums in lane order;
//   - the chunk sums are reduced the same way, level by level, until one value is left.
// The order depends only on n and T, never on how many threads run: bit-identical for any
// thread count on a backend (host_dot reproduces it exactly, for tests).
//
// Everything is allocated in the constructor (CLAUDE.md, memory contract); the reductions
// return the result to the host (one scalar).

#include <cstdint>
#include <vector>

#include "solvers/linear_solver.hpp"

namespace femorax::solvers {

class VectorOps {
 public:
  static constexpr std::int64_t kChunk = 2048;

  // For vectors of up to n entries; lanes = exec::simd_width() (src/solvers does not depend
  // on src/exec, which uses the solvers).
  VectorOps(std::int64_t n, int lanes);

  int lanes() const { return lanes_; }

  double dot(const ConstVector& x, const ConstVector& y) const;
  double norm2(const ConstVector& x) const;  // sqrt(dot(x, x))

  // Element-wise, no reductions (order-free, so trivially deterministic).
  static void fill(const Vector& y, double a);                                // y = a
  static void copy(const ConstVector& x, const Vector& y);                    // y = x
  static void axpy(double a, const ConstVector& x, const Vector& y);          // y += a x
  static void xpay(const ConstVector& x, double a, const Vector& y);          // y = x + a y
  static void multiply(const ConstVector& d, const ConstVector& x, const Vector& y);  // y = d .* x
  static void reciprocal(const ConstVector& d, const Vector& y);              // y = 1 / d

  // The reduction order above, on the host, for tests.
  static double host_dot(const std::vector<double>& x, const std::vector<double>& y, int lanes);

 private:
  double reduce_levels(std::int64_t n_partial) const;

  int lanes_;
  Vector partial_[2];  // chunk sums of the current and the next level
};

}  // namespace femorax::solvers
