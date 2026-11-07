#pragma once

// The sparsity pattern of an assembled operator: host code. FemoraX's implicit path is
// matrix-free (docs/design.md, "Linear solvers"); assembled matrices exist only in tests,
// built on the host from element matrices for small reference models (Eigen comparisons).
//
// Equation i couples with equation j when some element has both among its equations, after
// DOF elimination (tied DOFs share their retained DOF's equation, fixed DOFs are gone). The
// pattern is stored as CSR with both triangles and the diagonal, columns ascending within a
// row; row offsets are 64-bit (nonzero counts can pass 2^31), column indices 32-bit.
//
// No per-element position maps are stored: an entry (i, j) is found by a binary search in
// row i (rows hold at most a few hundred columns).

#include <cstdint>
#include <vector>

#include "compile/batches.hpp"

namespace femorax::compile {

struct SparsityPattern {
  std::int32_t rows = 0;
  std::vector<std::int64_t> row_offsets;  // [rows + 1]
  std::vector<std::int32_t> columns;      // [nnz], ascending within each row

  std::int64_t nnz() const { return row_offsets.empty() ? 0 : row_offsets.back(); }
  // Position of (i, j) in columns/values, -1 if (i, j) is not in the pattern.
  std::int64_t find(std::int32_t i, std::int32_t j) const;
};

SparsityPattern build_pattern(const Compiled& c);

// Bytes of the assembled operator on the device: row offsets, columns and values.
std::int64_t pattern_bytes(const SparsityPattern& p);

}  // namespace femorax::compile
