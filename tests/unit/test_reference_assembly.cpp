// The test-only reference assembly (tests/common/reference_assembly.hpp) against the
// verified explicit force kernels (K u, C v), the compile step's lumped mass, symmetry,
// and an exact solution: static gravity on the laminar soil column solved with Eigen is
// the one-dimensional oedometric column, which linear elements reproduce at the nodes.

#include "../common/reference_assembly.hpp"

#include <Eigen/SparseCholesky>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "../common/synthetic_models.hpp"
#include "exec/force_settings.hpp"
#include "exec/internal_force.hpp"

namespace {

using femorax::compile::compile_model;
using femorax::compile::CompileOptions;
using femorax::compile::Compiled;
using femorax::exec::InternalForce;

std::vector<double> force(InternalForce& k, const std::vector<double>& u, const std::vector<double>& v) {
  auto hu = Kokkos::create_mirror_view(k.u);
  auto hv = Kokkos::create_mirror_view(k.v);
  for (std::size_t i = 0; i < u.size(); ++i) {
    hu(i) = u[i];
    hv(i) = v[i];
  }
  Kokkos::deep_copy(k.u, hu);
  Kokkos::deep_copy(k.v, hv);
  k.compute(femorax::exec::default_force_settings());
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k.f);
  return std::vector<double>(hf.data(), hf.data() + hf.extent(0));
}

double max_abs(const std::vector<double>& a) {
  double m = 0.0;
  for (double x : a) m = std::max(m, std::abs(x));
  return m;
}

std::vector<double> wave(std::size_t n, double phase) {
  std::vector<double> x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = std::sin(0.37 * static_cast<double>(i) + phase) * 1e-3;
  return x;
}

femorax::io::Model column(bool lumped, bool damping) {
  femorax::testing::ColumnSpec s;
  s.nx = 3;
  s.ny = 2;
  s.nz = 8;
  s.lumped = lumped;
  s.damping = damping;
  return femorax::testing::soil_column(s);
}

TEST(ReferenceAssembly, StiffnessAndDampingMatchTheForceKernels) {
  for (bool lumped : {false, true}) {
    SCOPED_TRACE(lumped ? "lumped" : "consistent");
    const auto m = column(lumped, true);
    const Compiled c = compile_model(m, CompileOptions{});
    const auto ref = femorax::testing::assemble_reference(m, c);
    InternalForce k(c, femorax::exec::GeometryCache::Off);
    const auto n = static_cast<std::size_t>(c.dofs.num_equations);
    const std::vector<double> zero(n, 0.0), u = wave(n, 0.1), v = wave(n, 0.7);
    const auto f0 = force(k, zero, zero);  // minus the body force
    // Elastic: f(u, 0) - f(0, 0) = K u, and f(0, v) - f(0, 0) = C v.
    auto fu = force(k, u, zero), fv = force(k, zero, v);
    for (std::size_t i = 0; i < n; ++i) {
      fu[i] -= f0[i];
      fv[i] -= f0[i];
    }
    const auto Ku = ref.multiply(ref.K, u), Cv = ref.multiply(ref.C, v);
    double eu = 0.0, ev = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      eu = std::max(eu, std::abs(Ku[i] - fu[i]));
      ev = std::max(ev, std::abs(Cv[i] - fv[i]));
    }
    EXPECT_LE(eu, 1e-12 * max_abs(Ku));
    EXPECT_LE(ev, 1e-12 * max_abs(Cv));
    // Elastic: K_0 = K_c = K_t.
    for (std::size_t k2 = 0; k2 < ref.K.size(); ++k2) {
      ASSERT_EQ(ref.K0[k2], ref.Kc[k2]);
      ASSERT_NEAR(ref.K0[k2], ref.K[k2], 1e-12 * max_abs(ref.K));
    }
  }
}

TEST(ReferenceAssembly, MassAndSymmetry) {
  for (bool lumped : {false, true}) {
    SCOPED_TRACE(lumped ? "lumped" : "consistent");
    // No fixed base here, so every row keeps all its couplings (row sums are complete).
    auto m = column(lumped, true);
    m.fix = {};
    const Compiled c = compile_model(m, CompileOptions{});
    const auto ref = femorax::testing::assemble_reference(m, c);
    const auto& p = ref.pattern;
    for (std::int32_t i = 0; i < p.rows; ++i) {
      double row = 0.0;
      for (std::int64_t k = p.row_offsets[i]; k < p.row_offsets[i + 1]; ++k) {
        const std::int32_t j = p.columns[k];
        const std::int64_t t = p.find(j, i);
        ASSERT_NEAR(ref.M[k], ref.M[t], 1e-15 * max_abs(ref.M));  // OpenSees order: rounding only
        ASSERT_NEAR(ref.K[k], ref.K[t], 1e-12 * max_abs(ref.K));
        ASSERT_NEAR(ref.C[k], ref.C[t], 1e-12 * max_abs(ref.C));
        if (lumped && j != i) {
          ASSERT_EQ(ref.M[k], 0.0);
        }
        row += ref.M[k];
      }
      // Lumped: the diagonal is the compile step's lumped mass; consistent: the same row sum.
      if (lumped) {
        ASSERT_EQ(ref.M[p.find(i, i)], c.lumped_mass[i]);
      } else {
        ASSERT_NEAR(row, c.lumped_mass[i], 1e-12 * c.lumped_mass[i]);
      }
    }
  }
}

TEST(ReferenceAssembly, StaticGravityIsTheOedometricColumn) {
  // Laminar boundary (side nodes tied at each level), fixed base, two layers with their
  // body forces: a one-dimensional constrained column. sigma_zz(z) = integral of b3 from z
  // to the top, eps = sigma / Mc with Mc = E (1 - nu) / ((1 + nu)(1 - 2 nu)), u_z(z) the
  // integral of eps from the base; exact at the nodes for linear elements.
  femorax::testing::ColumnSpec s;
  s.nx = 3;
  s.ny = 3;
  s.nz = 12;
  s.damping = false;
  const auto m = femorax::testing::soil_column(s);
  const Compiled c = compile_model(m, CompileOptions{});
  const auto ref = femorax::testing::assemble_reference(m, c);
  InternalForce k(c, femorax::exec::GeometryCache::Off);
  const auto n = static_cast<std::size_t>(c.dofs.num_equations);
  const auto f0 = force(k, std::vector<double>(n, 0.0), std::vector<double>(n, 0.0));
  Eigen::VectorXd F(static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i) F[static_cast<Eigen::Index>(i)] = -f0[i];  // the body force
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(ref.to_eigen(ref.K));
  ASSERT_EQ(ldlt.info(), Eigen::Success);
  const Eigen::VectorXd u = ldlt.solve(F);

  // The exact solution at each level k (z = -H + k dz).
  const double H = s.nz * s.dz;
  auto Mc = [](double E, double nu) { return E * (1 - nu) / ((1 + nu) * (1 - 2 * nu)); };
  std::vector<double> b3(static_cast<std::size_t>(s.nz)), mc(b3.size());
  for (int e = 0; e < s.nz; ++e) {
    const bool lower = e < s.nz / 2;
    b3[e] = lower ? -19.9 : -19.8;
    mc[e] = lower ? Mc(377000.0, 0.3) : Mc(109200.0, 0.3);
  }
  std::vector<double> sigma(static_cast<std::size_t>(s.nz) + 1, 0.0), uz(sigma.size(), 0.0);
  for (int k2 = s.nz - 1; k2 >= 0; --k2) sigma[k2] = sigma[k2 + 1] + b3[k2] * s.dz;
  for (int e = 0; e < s.nz; ++e) uz[e + 1] = uz[e] + 0.5 * (sigma[e] + sigma[e + 1]) / mc[e] * s.dz;

  double err = 0.0, scale = std::abs(uz.back()), lateral = 0.0;
  for (std::int32_t i = 0; i < c.dofs.num_nodes(); ++i) {
    const double z = c.coords[2 * static_cast<std::size_t>(c.dofs.num_nodes()) + i];
    const int level = static_cast<int>(std::lround((z + H) / s.dz));
    const std::int32_t qz = c.dofs.equation(i, 2);
    if (qz < 0) continue;
    err = std::max(err, std::abs(u[qz] - uz[level]));
    for (int d = 0; d < 2; ++d) {
      const std::int32_t q = c.dofs.equation(i, d);
      if (q >= 0) lateral = std::max(lateral, std::abs(u[q]));
    }
  }
  EXPECT_GT(scale, 1e-4);
  EXPECT_LE(err, 1e-10 * scale);
  EXPECT_LE(lateral, 1e-10 * scale);
}

}  // namespace
