#pragma once

// An assembled operator: a CSR matrix in a fixed pattern on the device. apply() is y = A x
// with each row summed in column order, so the result is deterministic; diagonal() reads
// the diagonal entries. Used in tests and as the reference for the matrix-free operator
// (docs/design.md, "Linear solvers": the production implicit path is matrix-free; any
// future assembled preconditioner such as BoomerAMG would also use it).

#include <cstdint>
#include <vector>

#include "solvers/linear_solver.hpp"

namespace femorax::solvers {

class CsrOperator final : public LinearOperator {
 public:
  // The pattern is copied to the device once; values start at zero.
  CsrOperator(const std::vector<std::int64_t>& row_offsets, const std::vector<std::int32_t>& columns,
              Symmetry symmetry, std::vector<Field> fields = {});

  const OperatorInfo& info() const override { return info_; }
  void apply(const ConstVector& x, const Vector& y) const override;
  void diagonal(const Vector& d) const override;
  const CsrMatrix* assembled() const override { return &matrix_; }

  // The values, for assembly kernels to write into (same order as columns).
  const Kokkos::View<double*, Space>& values() const { return values_; }
  const Kokkos::View<std::int64_t*, Space>& row_offsets() const { return row_offsets_; }
  const Kokkos::View<std::int32_t*, Space>& columns() const { return columns_; }
  std::int64_t nnz() const { return static_cast<std::int64_t>(columns_.extent(0)); }

 private:
  OperatorInfo info_;
  Kokkos::View<std::int64_t*, Space> row_offsets_;
  Kokkos::View<std::int32_t*, Space> columns_;
  Kokkos::View<double*, Space> values_;
  CsrMatrix matrix_;
};

}  // namespace femorax::solvers
