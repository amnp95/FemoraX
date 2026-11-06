// ElasticIsotropic (3D) against closed-form linear elasticity. The comparison with the
// OpenSees class along strain paths is tests/unit/test_elastic_isotropic_reference.cpp,
// which reads the reference driver's data.

#include "physics/materials/elastic_isotropic.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <ostream>
#include <string>

namespace {

using femorax::ElasticIsotropic;
using Params = ElasticIsotropic::Params;

constexpr int N = ElasticIsotropic::NSTRAIN;

struct Result {
  double sig[N];
  double D[N][N];
};

Result run(const Params& p, const double (&eps)[N]) {
  Result r{};
  ElasticIsotropic::update(p, nullptr, nullptr, eps, r.sig, r.D);
  return r;
}

// Relative closeness with a floor of 1 for the scale.
void expect_close(double got, double want, double rel, const char* what) {
  EXPECT_NEAR(got, want, rel * std::max(1.0, std::abs(want))) << what;
}

const Params soils[] = {
    {377000.0, 0.3, 2.0285},     // dense Ottawa sand (kPa, Mg/m^3), Femora soil column
    {195000.0, 0.3, 1.947},      // loose Ottawa sand
    {2.0e11, 0.29, 7850.0},      // steel in SI
    {1.0, 0.0, 1.0},             // nu = 0
    {3.0e4, 0.499, 2.4},         // nearly incompressible
};

class ElasticIsotropicClosedForm : public ::testing::TestWithParam<Params> {};

TEST_P(ElasticIsotropicClosedForm, LameConstantsInTangent) {
  const Params p = GetParam();
  const double lambda = p.E * p.v / ((1.0 + p.v) * (1.0 - 2.0 * p.v));
  const double mu = p.E / (2.0 * (1.0 + p.v));
  const auto r = run(p, {0, 0, 0, 0, 0, 0});
  for (int i = 0; i < 3; ++i) {
    expect_close(r.D[i][i], lambda + 2.0 * mu, 1e-14, "D_ii");
    for (int j = 0; j < 3; ++j)
      if (i != j) expect_close(r.D[i][j], lambda, 1e-14, "D_ij");
    expect_close(r.D[3 + i][3 + i], mu, 1e-14, "shear");
  }
  // Everything else is exactly zero, and D is exactly symmetric.
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      EXPECT_EQ(r.D[i][j], r.D[j][i]);
      if ((i < 3) != (j < 3) || (i >= 3 && j >= 3 && i != j)) {
        EXPECT_EQ(r.D[i][j], 0.0);
      }
    }
}

TEST_P(ElasticIsotropicClosedForm, UniaxialStress) {
  const Params p = GetParam();
  const double e = 1e-3;
  const auto r = run(p, {e, -p.v * e, -p.v * e, 0, 0, 0});
  expect_close(r.sig[0], p.E * e, 1e-12, "sigma_11 = E*eps");
  for (int i = 1; i < N; ++i) EXPECT_NEAR(r.sig[i], 0.0, 1e-12 * p.E * e) << i;
}

TEST_P(ElasticIsotropicClosedForm, VolumetricAndShear) {
  const Params p = GetParam();
  const double e = 2e-4, g = 3e-4;
  const double K = p.E / (3.0 * (1.0 - 2.0 * p.v));
  const double mu = p.E / (2.0 * (1.0 + p.v));
  const auto vol = run(p, {e, e, e, 0, 0, 0});
  for (int i = 0; i < 3; ++i) expect_close(vol.sig[i], 3.0 * K * e, 1e-12, "mean stress");
  const auto shear = run(p, {0, 0, 0, g, 2 * g, 3 * g});  // engineering shear strains
  for (int i = 0; i < 3; ++i) {
    expect_close(shear.sig[3 + i], mu * (i + 1) * g, 1e-14, "tau = mu*gamma");
    EXPECT_EQ(shear.sig[i], 0.0);
  }
}

TEST_P(ElasticIsotropicClosedForm, StressEqualsTangentTimesStrain) {
  const Params p = GetParam();
  const double eps[N] = {1.1e-3, -0.4e-3, 0.25e-3, 0.7e-3, -0.3e-3, 0.05e-3};
  const auto r = run(p, eps);
  for (int i = 0; i < N; ++i) {
    double s = 0.0, scale = 0.0;
    for (int j = 0; j < N; ++j) {
      s += r.D[i][j] * eps[j];
      scale += std::abs(r.D[i][j] * eps[j]);
    }
    EXPECT_NEAR(r.sig[i], s, 4e-16 * scale) << i;  // rounding only
  }
}

TEST_P(ElasticIsotropicClosedForm, InitialTangentEqualsTangent) {
  const Params p = GetParam();
  const auto r = run(p, {1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3});
  double D0[N][N];
  ElasticIsotropic::initial_tangent(p, D0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) EXPECT_EQ(D0[i][j], r.D[i][j]);
}

TEST_P(ElasticIsotropicClosedForm, CommittedTangentEqualsTangent) {
  const Params p = GetParam();
  double D0[N][N], Dc[N][N];
  ElasticIsotropic::initial_tangent(p, D0);
  ElasticIsotropic::committed_tangent(p, nullptr, Dc);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) EXPECT_EQ(Dc[i][j], D0[i][j]);
}

TEST_P(ElasticIsotropicClosedForm, DensityIsRho) {
  EXPECT_EQ(ElasticIsotropic::density(GetParam()), GetParam().rho);
}

const char* const soil_names[] = {"dense_ottawa", "loose_ottawa", "steel", "nu_zero",
                                  "nearly_incompressible"};

std::string soil_name(const ::testing::TestParamInfo<Params>& info_) {
  return soil_names[info_.index];
}

INSTANTIATE_TEST_SUITE_P(Materials, ElasticIsotropicClosedForm, ::testing::ValuesIn(soils),
                         soil_name);

TEST(ElasticIsotropic, NoHistory) {
  // The same strain gives the same stress whatever came before (NSTATE = 0).
  const Params p{377000.0, 0.3, 2.0};
  const double a[N] = {1e-3, 0, 0, 0, 0, 0}, b[N] = {-5e-3, 2e-3, 0, 1e-3, 0, 0};
  const auto first = run(p, a);
  run(p, b);
  const auto again = run(p, a);
  for (int i = 0; i < N; ++i) EXPECT_EQ(first.sig[i], again.sig[i]);
}

}  // namespace

// Parameter printing for readable test names in failures.
namespace femorax {
inline void PrintTo(const ElasticIsotropic::Params& p, std::ostream* os) {
  *os << "E=" << p.E << " nu=" << p.v << " rho=" << p.rho;
}
}  // namespace femorax
