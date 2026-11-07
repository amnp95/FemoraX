#pragma once

// The linear-solver interface (docs/design.md, "Linear solvers"). Everything outside
// src/solvers/ talks to solvers only through these types (CLAUDE.md rule 6).
//
//   LinearOperator   y = A x, the diagonal, and a description (symmetry, fields). Either
//                    assembled (a CSR matrix in a fixed pattern, whose device arrays are
//                    exposed for preconditioners that need them) or matrix-free (element
//                    kernels; no global matrix).
//   Preconditioner   z = P^-1 r, set up from an operator.
//   LinearSolver     setup(operator, preconditioner) when the operator's values change;
//                    solve(b, x) with x holding the initial guess on entry.
//
// Vectors live on the device, FP64, in the compile step's equation numbering. Implemented
// first: symmetric positive definite assembled operators, Jacobi, CG. The interface also
// carries what non-symmetric operators (another Krylov method), coupled fields (block
// structure) and matrix-free operators need, so adding them changes no caller.

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace femorax::solvers {

using Space = Kokkos::DefaultExecutionSpace;
using Vector = Kokkos::View<double*, Space>;
using ConstVector = Kokkos::View<const double*, Space>;

enum class Symmetry { SymmetricPositiveDefinite, Symmetric, General };
const char* to_string(Symmetry s);

// A group of equations solved together (displacement, pore pressure, ...): equations
// [begin, end) of the operator. One field for a single-field problem.
struct Field {
  std::string name;
  std::int64_t begin = 0, end = 0;
};

struct OperatorInfo {
  std::int64_t rows = 0;  // square operators: rows == columns
  Symmetry symmetry = Symmetry::SymmetricPositiveDefinite;
  std::vector<Field> fields;
};

// Device arrays of an assembled operator: CSR with 64-bit row offsets (nonzero counts can
// pass 2^31) and 32-bit column indices (CLAUDE.md performance rule 4). Both triangles are
// stored, also for symmetric operators.
struct CsrMatrix {
  Kokkos::View<const std::int64_t*, Space> row_offsets;  // [rows + 1]
  Kokkos::View<const std::int32_t*, Space> columns;      // [nnz], ascending within a row
  Kokkos::View<const double*, Space> values;              // [nnz]
};

class LinearOperator {
 public:
  virtual ~LinearOperator() = default;
  virtual const OperatorInfo& info() const = 0;
  // y = A x. x and y never alias.
  virtual void apply(const ConstVector& x, const Vector& y) const = 0;
  // d = diag(A).
  virtual void diagonal(const Vector& d) const = 0;
  // The matrix, for an assembled operator; nullptr for a matrix-free one.
  virtual const CsrMatrix* assembled() const { return nullptr; }
};

class Preconditioner {
 public:
  virtual ~Preconditioner() = default;
  virtual std::string name() const = 0;
  // Called whenever the operator's values change.
  virtual void setup(const LinearOperator& a) = 0;
  // z = P^-1 r. r and z never alias.
  virtual void apply(const ConstVector& r, const Vector& z) const = 0;
};

struct SolveOptions {
  double relative_tolerance = 1e-10;  // stop when ||r|| <= max(rtol * ||b||, atol)
  double absolute_tolerance = 0.0;
  int max_iterations = 10000;
};

struct SolveResult {
  bool converged = false;
  int iterations = 0;
  double rhs_norm = 0.0;           // ||b||
  double initial_residual = 0.0;   // ||b - A x0||
  double final_residual = 0.0;     // ||b - A x|| as the method tracks it
  // CG met p^T A p <= 0: the operator is not positive definite (or a declaration was too
  // optimistic); the caller falls back to a non-symmetric solver.
  bool indefinite = false;
};

class LinearSolver {
 public:
  virtual ~LinearSolver() = default;
  virtual std::string name() const = 0;
  // The operator (and the preconditioner, already set up for it, or nullptr) stay owned by
  // the caller and must outlive the solves. Allocates the solver's work vectors.
  virtual void setup(const LinearOperator& a, const Preconditioner* p) = 0;
  // Solve A x = b; x holds the initial guess on entry (the previous solution, for example).
  virtual SolveResult solve(const ConstVector& b, const Vector& x) = 0;
};

}  // namespace femorax::solvers
