// Sparsity pattern of the assembled operator, and the CSR operator's apply and diagonal.

#include "compile/sparsity.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "../common/synthetic_models.hpp"
#include "compile/batches.hpp"
#include "solvers/csr_operator.hpp"

namespace {

using femorax::compile::compile_model;
using femorax::compile::CompileOptions;
using femorax::compile::Compiled;
using femorax::compile::Renumbering;
using femorax::compile::SparsityPattern;

constexpr bool kHostBackend =
    std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;

// Every pair of equations that share an element, by brute force.
std::set<std::pair<std::int32_t, std::int32_t>> element_pairs(const Compiled& c) {
  std::set<std::pair<std::int32_t, std::int32_t>> s;
  for (const auto& b : c.batches)
    for (std::int32_t e = 0; e < b.count; ++e)
      for (int a = 0; a < b.ndof; ++a)
        for (int k = 0; k < b.ndof; ++k) {
          const auto qa = b.eq[static_cast<std::size_t>(a) * b.padded + e];
          const auto qk = b.eq[static_cast<std::size_t>(k) * b.padded + e];
          if (qa >= 0 && qk >= 0) s.insert({qa, qk});
        }
  return s;
}

void check_pattern(const Compiled& c) {
  const SparsityPattern p = femorax::compile::build_pattern(c);
  ASSERT_EQ(p.rows, c.dofs.num_equations);
  const auto want = element_pairs(c);
  ASSERT_EQ(p.nnz(), static_cast<std::int64_t>(want.size()));
  std::set<std::pair<std::int32_t, std::int32_t>> got;
  for (std::int32_t i = 0; i < p.rows; ++i) {
    for (std::int64_t k = p.row_offsets[i]; k < p.row_offsets[i + 1]; ++k) {
      if (k > p.row_offsets[i]) {
        ASSERT_LT(p.columns[k - 1], p.columns[k]);  // strictly ascending
      }
      got.insert({i, p.columns[k]});
      ASSERT_EQ(p.find(i, p.columns[k]), k);
    }
    ASSERT_GE(p.find(i, i), 0) << "diagonal of row " << i;
  }
  EXPECT_EQ(got, want);
  for (const auto& [i, j] : want) ASSERT_GE(p.find(j, i), 0) << "symmetric";
  EXPECT_EQ(p.find(0, p.rows), -1);
  EXPECT_EQ(femorax::compile::pattern_bytes(p), 8 * (p.rows + 1) + 12 * p.nnz());
}

TEST(Sparsity, PatternIsTheElementCouplingAfterDofElimination) {
  // Laminar ties and a fixed base (soil_column), renumbering off and on.
  for (auto r : {Renumbering::Off, Renumbering::RCM}) {
    femorax::testing::ColumnSpec spec;
    spec.nx = 3;
    spec.ny = 2;
    spec.nz = 6;
    CompileOptions o;
    o.renumbering = r;
    check_pattern(compile_model(femorax::testing::soil_column(spec), o));
  }
}

TEST(Sparsity, RegularMeshHasThe27PointStencil) {
  // No ties: drop the laminar boundary; a 3x3x3 interior node couples with 27 nodes.
  femorax::testing::ColumnSpec spec;
  spec.nx = 4;
  spec.ny = 4;
  spec.nz = 4;
  auto m = femorax::testing::soil_column(spec);
  m.equal_dof = {};
  CompileOptions o;
  o.renumbering = Renumbering::Off;
  const Compiled c = compile_model(m, o);
  const SparsityPattern p = femorax::compile::build_pattern(c);
  std::int64_t longest = 0;
  for (std::int32_t i = 0; i < p.rows; ++i) longest = std::max(longest, p.row_offsets[i + 1] - p.row_offsets[i]);
  EXPECT_EQ(longest, 81);
}

TEST(CsrOperator, ApplyAndDiagonalMatchTheHost) {
  femorax::testing::ColumnSpec spec;
  spec.nx = 3;
  spec.ny = 3;
  spec.nz = 8;
  CompileOptions o;
  const Compiled c = compile_model(femorax::testing::soil_column(spec), o);
  const SparsityPattern p = femorax::compile::build_pattern(c);
  femorax::solvers::CsrOperator a(p.row_offsets, p.columns, femorax::solvers::Symmetry::General);
  EXPECT_EQ(a.info().rows, p.rows);
  ASSERT_NE(a.assembled(), nullptr);

  std::mt19937_64 g(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  std::vector<double> hv(static_cast<std::size_t>(p.nnz())), hx(static_cast<std::size_t>(p.rows));
  for (auto& v : hv) v = u(g);
  for (auto& v : hx) v = u(g);
  auto mv = Kokkos::create_mirror_view(a.values());
  for (std::size_t k = 0; k < hv.size(); ++k) mv(k) = hv[k];
  Kokkos::deep_copy(a.values(), mv);
  femorax::solvers::Vector x("x", hx.size()), y("y", hx.size()), d("d", hx.size());
  auto mx = Kokkos::create_mirror_view(x);
  for (std::size_t i = 0; i < hx.size(); ++i) mx(i) = hx[i];
  Kokkos::deep_copy(x, mx);

  a.apply(x, y);
  a.diagonal(d);
  auto hy = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  for (std::int32_t i = 0; i < p.rows; ++i) {
    double s = 0.0, scale = 0.0;
    for (std::int64_t k = p.row_offsets[i]; k < p.row_offsets[i + 1]; ++k) {
      s += hv[k] * hx[p.columns[k]];
      scale += std::abs(hv[k] * hx[p.columns[k]]);
    }
    if (kHostBackend) {
      ASSERT_EQ(hy(i), s) << "row " << i;
    } else {
      ASSERT_NEAR(hy(i), s, 1e-15 * scale) << "row " << i;
    }
    ASSERT_EQ(hd(i), hv[p.find(i, i)]);
  }
}

}  // namespace
