// The matrix-free operator K_eff = cK K_t + cC C + cM M (exec/matrix_free_operator.hpp)
// against the host-assembled reference (tests/common/reference_assembly.hpp), for
// consistent and lumped mass, nodal masses, every damping term, every scatter, geometry
// and tangent source; bit-identity between the deterministic gather variants; the
// diagonal; and the symmetry the solver choice reads.

#include "exec/matrix_free_operator.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "../common/reference_assembly.hpp"
#include "../common/synthetic_models.hpp"

namespace {

using femorax::OperatorCoefficients;
using femorax::compile::compile_model;
using femorax::compile::CompileOptions;
using femorax::compile::Compiled;
using femorax::exec::Geometry;
using femorax::exec::GeometryCache;
using femorax::exec::InternalForce;
using femorax::exec::OperatorSettings;
using femorax::exec::Scatter;
using femorax::exec::TangentSource;
using femorax::exec::Vec;

femorax::io::Model column(bool lumped) {
  femorax::testing::ColumnSpec s;
  s.nx = 3;
  s.ny = 2;
  s.nz = 8;
  s.lumped = lumped;
  auto m = femorax::testing::soil_column(s);
  // Every Rayleigh term somewhere (the top region has betaKinit; add betaKcomm), and nodal
  // masses on two nodes.
  m.rayleigh[1].betaKcomm = 2e-5;
  for (int d = 1; d <= 3; ++d) {
    m.mass.node.insert(m.mass.node.end(), {20, 45});
    m.mass.dof.insert(m.mass.dof.end(), {d, d});
    m.mass.value.insert(m.mass.value.end(), {0.7 * d, 1.3});
  }
  return m;
}

Vec upload(const std::vector<double>& h) {
  Vec d("x", h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

std::vector<double> download(const Vec& d) {
  auto m = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d);
  return std::vector<double>(m.data(), m.data() + m.extent(0));
}

double max_abs(const std::vector<double>& a) {
  double m = 0.0;
  for (double x : a) m = std::max(m, std::abs(x));
  return m;
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

const OperatorCoefficients kCases[] = {
    {1.0, 0.0, 0.0},       // K
    {0.0, 1.0, 0.0},       // C
    {0.0, 0.0, 1.0},       // M
    {1.0, 2e3, 1.6e7},     // Newmark-like: K + c2 C + c3 M at dt = 2.5e-4
};

TEST(MatrixFreeOperator, MatchesTheAssembledReferenceInEveryVariant) {
  for (bool lumped : {false, true}) {
    SCOPED_TRACE(lumped ? "lumped" : "consistent");
    const auto m = column(lumped);
    CompileOptions o;
    o.build_coloring = true;
    const Compiled c = compile_model(m, o);
    const auto ref = femorax::testing::assemble_reference(m, c);
    InternalForce force(c, GeometryCache::On);
    force.enable_stored_tangents();
    const auto n = static_cast<std::size_t>(c.dofs.num_equations);
    std::vector<double> hx(n);
    for (std::size_t i = 0; i < n; ++i) hx[i] = std::cos(0.61 * static_cast<double>(i)) * 1e-3;
    const Vec x = upload(hx), u("u", n), y("y", n), d("d", n);
    force.store_tangents(u, {});

    for (const auto& k : kCases) {
      SCOPED_TRACE("coefficients " + std::to_string(k.stiffness) + " " + std::to_string(k.damping) + " " +
                   std::to_string(k.mass));
      const auto A = ref.effective(k.stiffness, k.damping, k.mass);
      const auto want = ref.multiply(A, hx);
      std::vector<double> want_diag(n);
      for (std::size_t i = 0; i < n; ++i)
        want_diag[i] = A[static_cast<std::size_t>(ref.pattern.find(static_cast<std::int32_t>(i), static_cast<std::int32_t>(i)))];
      std::vector<double> first, first_diag;
      for (auto sc : {Scatter::Gather, Scatter::Colored, Scatter::Atomic})
        for (auto g : {Geometry::Recompute, Geometry::Cached})
          for (auto t : {TangentSource::Constant, TangentSource::Stored, TangentSource::Recomputed}) {
            const OperatorSettings s{sc, g, t};
            SCOPED_TRACE(femorax::exec::to_string(s));
            force.apply_operator(k, s, u, x, y);
            force.operator_diagonal(k, s, u, d);
            const auto got = download(y), got_diag = download(d);
            EXPECT_LE(max_diff(got, want), 1e-12 * max_abs(want));
            EXPECT_LE(max_diff(got_diag, want_diag), 1e-12 * max_abs(want_diag));
            if (sc == Scatter::Gather) {
              // Every deterministic gather variant gives the same bits.
              if (first.empty()) {
                first = got;
                first_diag = got_diag;
              } else {
                EXPECT_EQ(got, first);
                EXPECT_EQ(got_diag, first_diag);
              }
            }
            if (sc == Scatter::Colored) {  // deterministic: run to run
              force.apply_operator(k, s, u, x, y);
              EXPECT_EQ(download(y), got);
            }
          }
    }
  }
}

TEST(MatrixFreeOperator, GroundLoadIncludesTheMovingBase) {
  // M r_d over every element DOF (the fixed base too) plus nodal masses: a consistent mass
  // loads the DOFs next to the base through their coupling to it.
  for (bool lumped : {false, true}) {
    SCOPED_TRACE(lumped ? "lumped" : "consistent");
    const auto m = column(lumped);
    const Compiled c = compile_model(m, CompileOptions{});
    const auto ref = femorax::testing::assemble_reference(m, c);
    InternalForce force(c, GeometryCache::On);
    const auto n = static_cast<std::size_t>(c.dofs.num_equations);
    const Vec u("u", n), y("y", n);
    for (int d = 0; d < 3; ++d) {
      force.mass_times_ground(d, {}, u, y);
      EXPECT_LE(max_diff(download(y), ref.Mr[d]), 1e-12 * max_abs(ref.Mr[d]));
    }
    if (!lumped) {
      // Not the same as M times the free-DOF indicator: the base coupling is there.
      std::vector<double> r(n);
      for (std::size_t q = 0; q < n; ++q) r[q] = c.eq_dof[q] == 0 ? 1.0 : 0.0;
      EXPECT_GT(max_diff(ref.multiply(ref.M, r), ref.Mr[0]), 1e-3 * max_abs(ref.Mr[0]));
    }
  }
}

TEST(MatrixFreeOperator, SolverInterfaceAndSymmetry) {
  const auto m = column(false);
  const Compiled c = compile_model(m, CompileOptions{});
  InternalForce force(c, GeometryCache::Off);
  const auto n = static_cast<std::size_t>(c.dofs.num_equations);
  const Vec u("u", n);
  femorax::exec::MatrixFreeOperator a(force, u, {1.0, 2e3, 1.6e7});
  EXPECT_EQ(a.info().rows, static_cast<std::int64_t>(n));
  EXPECT_EQ(a.assembled(), nullptr);
  EXPECT_EQ(force.tangent_symmetry(), femorax::TangentSymmetry::SymmetricPositiveDefinite);
  EXPECT_TRUE(force.constant_tangents());
  EXPECT_EQ(a.settings().tangent, TangentSource::Constant);  // Auto on an elastic model
  EXPECT_EQ(a.settings().geometry, Geometry::Recompute);     // no cache: falls back
  EXPECT_EQ(a.info().symmetry, femorax::solvers::Symmetry::SymmetricPositiveDefinite);
  a.set_coefficients({1.0, -1.0, 0.0});
  EXPECT_EQ(a.info().symmetry, femorax::solvers::Symmetry::Symmetric);
  EXPECT_EQ(femorax::exec::operator_symmetry(femorax::TangentSymmetry::General, {1.0, 0.0, 0.0}),
            femorax::solvers::Symmetry::General);

  // apply through the interface equals the reference.
  const auto ref = femorax::testing::assemble_reference(m, c);
  a.set_coefficients({1.0, 2e3, 1.6e7});
  std::vector<double> hx(n);
  for (std::size_t i = 0; i < n; ++i) hx[i] = std::sin(0.3 * static_cast<double>(i));
  const Vec x = upload(hx), y("y", n);
  a.apply(x, y);
  const auto want = ref.multiply(ref.effective(1.0, 2e3, 1.6e7), hx);
  EXPECT_LE(max_diff(download(y), want), 1e-12 * max_abs(want));
}

}  // namespace
