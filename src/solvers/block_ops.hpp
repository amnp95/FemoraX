#pragma once

// Operations on a block of vectors (the Krylov basis of GMRES) with the same deterministic
// reduction order as VectorOps (chunks of VectorOps::kChunk entries, a fixed lane order,
// level by level), so results are bit-identical for any thread count on a backend:
//   dots(V, k, w, h):    h[c] = V(:, c) . w for c < k, all in one fused reduction (one
//                        device-to-host copy instead of k), as classical Gram-Schmidt needs
//   combine(V, k, y, a, w): w += a * sum_c y[c] V(:, c), each entry summed in column order
// All workspace is allocated in the constructor.

#include <cstdint>
#include <vector>

#include "solvers/linear_solver.hpp"

namespace femorax::solvers {

using Block = Kokkos::View<double**, Kokkos::LayoutLeft, Space>;  // (n, columns), columns contiguous

class BlockOps {
 public:
  // Vectors of up to n entries, up to max_columns columns; lanes = exec::simd_width().
  BlockOps(std::int64_t n, int max_columns, int lanes);

  void dots(const Block& V, int k, const ConstVector& w, std::vector<double>& h) const;
  void combine(const Block& V, int k, const std::vector<double>& y, double a, const Vector& w) const;

  // The reduction order of dots, on the host, for tests: column c of V (row-major n x k).
  static double host_dot(const std::vector<double>& v, const std::vector<double>& w, int lanes);

 private:
  int lanes_, max_columns_;
  Block partial_[2];
  Kokkos::View<double*, Space> coef_;
  Kokkos::View<double*, Kokkos::HostSpace> coef_host_;
};

}  // namespace femorax::solvers
