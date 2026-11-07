// Solver vector kernels (src/solvers/vector_ops): the deterministic dot product and axpy,
// the two kernels every CG iteration runs besides the operator. Both are bandwidth bound;
// judge them by %triad.

#include <memory>
#include <string>

#include "../tests/common/synthetic_models.hpp"
#include "compile/batches.hpp"
#include "compile/sparsity.hpp"
#include "exec/runtime.hpp"
#include "harness.hpp"
#include "solvers/csr_operator.hpp"
#include "solvers/vector_ops.hpp"

namespace femorax::bench {

namespace {

struct SolverVectors {
  explicit SolverVectors(std::int64_t n)
      : ops(n, simd_width()), x("x", static_cast<std::size_t>(n)), y("y", static_cast<std::size_t>(n)) {
    solvers::VectorOps::fill(x, 1.0);
    solvers::VectorOps::fill(y, 0.5);
  }
  solvers::VectorOps ops;
  solvers::Vector x, y;
  double sink = 0.0;
};

struct Spmv {
  explicit Spmv(int n) {
    testing::ColumnSpec s;
    s.nx = n;
    s.ny = n;
    s.nz = n;
    compile::CompileOptions co;
    co.simd_width = simd_width();
    const auto c = compile::compile_model(testing::soil_column(s), co);
    const auto p = compile::build_pattern(c);
    a = std::make_unique<solvers::CsrOperator>(p.row_offsets, p.columns, solvers::Symmetry::SymmetricPositiveDefinite);
    Kokkos::deep_copy(a->values(), 1.0);
    x = solvers::Vector("x", static_cast<std::size_t>(p.rows));
    y = solvers::Vector("y", static_cast<std::size_t>(p.rows));
    solvers::VectorOps::fill(x, 1.0);
  }
  std::unique_ptr<solvers::CsrOperator> a;
  solvers::Vector x, y;
};

}  // namespace

void register_solvers(Registry& registry, const Options& options) {
  // 2^25 doubles = 256 MiB per vector (the 40^3 column has 1.8e5 equations; this is the
  // bandwidth regime of large models). 2^20 for the quick run.
  const std::int64_t n = options.quick ? (1 << 20) : (1 << 25);
  // Allocated on first use, so filtered-out runs do not hold 512 MiB during other benchmarks.
  auto holder = std::make_shared<std::unique_ptr<SolverVectors>>();
  auto get = [holder, n]() -> SolverVectors& {
    if (!*holder) *holder = std::make_unique<SolverVectors>(n);
    return **holder;
  };
  // dot: reads x and y once; the chunk sums (n / 2048 values) are negligible.
  registry.add({.name = "solvers/dot",
                .unit = "call",
                .bytes = std::int64_t{16} * n,
                .flops = 2.0 * static_cast<double>(n),
                .items = n,
                .run = [get] {
                  auto& v = get();
                  v.sink += v.ops.dot(v.x, v.y);
                }});
  // axpy: reads x and y, writes y.
  registry.add({.name = "solvers/axpy",
                .unit = "call",
                .bytes = std::int64_t{24} * n,
                .flops = 2.0 * static_cast<double>(n),
                .items = n,
                .run = [get] {
                  auto& v = get();
                  solvers::VectorOps::axpy(1e-30, v.x, v.y);
                }});

  // SpMV on the 40^3 soil column's pattern (renumbering on): reads the row offsets
  // (8 bytes per row), columns and values (12 bytes per nonzero), x once and writes y.
  {
    const int side = options.side();
    auto spmv = std::make_shared<Spmv>(side);
    const std::int64_t rows = spmv->a->info().rows, nnz = spmv->a->nnz();
    registry.add({.name = "solvers/spmv/" + std::to_string(side) + "^3",
                  .unit = "call",
                  .bytes = 8 * (rows + 1) + 12 * nnz + 16 * rows,
                  .flops = 2.0 * static_cast<double>(nnz),
                  .items = nnz,
                  .run = [spmv] { spmv->a->apply(spmv->x, spmv->y); },
                  // Assembled operators exist only in tests (docs/design.md); the reference
                  // point for the matrix-free operator, not a production path.
                  .gated = false});
  }
}

}  // namespace femorax::bench
