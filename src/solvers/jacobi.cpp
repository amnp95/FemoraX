#include "solvers/jacobi.hpp"

#include <stdexcept>
#include <string>

#include "solvers/vector_ops.hpp"

namespace femorax::solvers {

void Jacobi::setup(const LinearOperator& a) {
  const auto n = static_cast<std::size_t>(a.info().rows);
  if (diag_.extent(0) != n) {
    diag_ = Vector("jacobi_diagonal", n);
    inv_ = Vector("jacobi_inverse", n);
  }
  a.diagonal(diag_);
  const auto d = diag_;
  std::int64_t zeros = 0;
  Kokkos::parallel_reduce(
      "jacobi_zero_check", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, static_cast<std::int64_t>(n)),
      KOKKOS_LAMBDA(const std::int64_t i, std::int64_t& c) { c += d(i) == 0.0 ? 1 : 0; }, zeros);
  if (zeros > 0) throw std::runtime_error("Jacobi: " + std::to_string(zeros) + " zero diagonal entries");
  VectorOps::reciprocal(diag_, inv_);
}

void Jacobi::apply(const ConstVector& r, const Vector& z) const { VectorOps::multiply(inv_, r, z); }

}  // namespace femorax::solvers
