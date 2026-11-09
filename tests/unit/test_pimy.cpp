// PressureIndependMultiYield properties that need no reference data: the contract, the state
// size per point, setup errors, the elastic stage, the switch to the plastic stage (the
// point's moduli scaled by the confinement), update as a pure function of the committed
// state, and the stable step's tangent. The comparison with OpenSees is
// test_pimy_reference.cpp.

#include "physics/materials/pressure_independ_multi_yield.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using P20 = femorax::PressureIndependMultiYield<20>;
using P40 = femorax::PressureIndependMultiYield<40>;

static_assert(femorax::Material<P20>);
static_assert(femorax::Material<P40>);
static_assert(femorax::SplitMaterial<P20>);
static_assert(P20::NSTATE == 30 + 8 * 20);
static_assert(P20::TANGENT == femorax::TangentSymmetry::SymmetricPositiveSemidefinite);
static_assert(std::is_trivially_copyable_v<P20::Params>);

// Femora's basin clay (examples/domain_reduction_method/basin.py).
P20::Input basin_clay() {
  P20::Input in{};
  in.rho = 1.8;
  in.refShearModul = 1.8 * 150.0 * 150.0;
  in.refBulkModul = 1.8 * (300.0 * 300.0 - 4.0 * 150.0 * 150.0 / 3.0);
  in.cohesi = 8.0;
  in.peakShearStra = 0.1;
  in.frictionAng = 0.0;
  in.refPress = 100.0;
  in.pressDependCoe = 0.0;
  in.numberOfYieldSurf = 20;
  return in;
}

P20::Input frictional() {
  P20::Input in = basin_clay();
  in.rho = 2.0;
  in.refShearModul = 60.0e3;
  in.refBulkModul = 150.0e3;
  in.cohesi = 5.0;
  in.frictionAng = 30.0;
  in.refPress = 80.0;
  in.pressDependCoe = 0.5;
  return in;
}

TEST(PressureIndependMultiYield, StateSizePerPoint) {
  const double kb20 = 2.0 * P20::NSTATE * 8 / 1024.0, kb40 = 2.0 * P40::NSTATE * 8 / 1024.0;
  std::printf("[ PIMY state per Gauss point: NS 20: %d doubles x 2 = %.2f KiB; NS 40: %d x 2 = %.2f KiB ]\n",
              P20::NSTATE, kb20, P40::NSTATE, kb40);
  std::printf("[ Params: NS 20 %zu bytes, NS 40 %zu bytes ]\n", sizeof(P20::Params), sizeof(P40::Params));
  EXPECT_EQ(P20::NSTATE, 190);
  EXPECT_EQ(P40::NSTATE, 350);
}

TEST(PressureIndependMultiYield, SetupChecksTheBucketAndTheBackbone) {
  P20::Params p{};
  auto in = basin_clay();
  EXPECT_EQ(P20::setup(in, p), P20::Ok);
  EXPECT_EQ(p.numOfSurfaces, 20);
  EXPECT_LT(p.refPressure, 0.0);
  EXPECT_EQ(p.size[0], 0.0);
  EXPECT_EQ(p.modulus[20], 0.0);  // the outermost surface is perfectly plastic
  for (int i = 1; i < 20; ++i) {
    EXPECT_GT(p.size[i + 1], p.size[i]);
    EXPECT_GT(p.modulus[i], 0.0);
  }
  // friction 0: the outermost size is the peak shear strength sqrt(3/2) * 2 sqrt(2) c / 3 ...
  EXPECT_NEAR(p.size[20], 3.0 * (2.0 * std::sqrt(2.0) * 8.0 / 3.0) / std::sqrt(2.0), 1e-12);
  in.numberOfYieldSurf = 30;
  EXPECT_EQ(P20::setup(in, p), P20::TooManySurfaces);
  P40::Params q{};
  EXPECT_EQ(P40::setup(in, q), P40::Ok);
  in = basin_clay();
  in.cohesi = 0.0;  // no strength at all
  EXPECT_EQ(P20::setup(in, p), P20::BadInput);
  const double one[] = {1e-4, 0.9};
  in = basin_clay();
  in.numberOfYieldSurf = 1;
  in.gredu = one;
  EXPECT_EQ(P20::setup(in, p), P20::TooFewPairs);
  // pressDependCoe is reset to 0 without friction, as OpenSees does.
  in = basin_clay();
  in.pressDependCoe = 0.5;
  EXPECT_EQ(P20::setup(in, p), P20::Ok);
  EXPECT_EQ(p.pressDependCoeff, 0.0);
}

TEST(PressureIndependMultiYield, ElasticStageIsLinearWithTheReferenceModuli) {
  P20::Params p{};
  ASSERT_EQ(P20::setup(basin_clay(), p), P20::Ok);
  std::vector<double> C(P20::NSTATE), T(P20::NSTATE);
  P20::initial_state(p, {C.data(), 1});
  const double eps[6] = {1e-4, -2e-4, 3e-4, 5e-4, -1e-4, 2e-4};
  double sig[6], D[6][6];
  P20::update(p, {C.data(), 1}, {T.data(), 1}, eps, sig, D);
  for (int i = 0; i < 6; ++i) {
    double want = 0.0;
    for (int j = 0; j < 6; ++j) want += D[i][j] * eps[j];
    EXPECT_NEAR(sig[i], want, 1e-12 * std::abs(want) + 1e-12);
  }
  EXPECT_DOUBLE_EQ(D[3][3], p.refShearModulus);
}

TEST(PressureIndependMultiYield, SwitchScalesThePointsModuliByTheConfinement) {
  // Friction angle 30, pressDependCoe 0.5: at the switch each point's G and K become the
  // reference ones times ((p' + residual) / (pref + residual))^0.5 (paramScaling), and the
  // stable step's tangent anticipates it while the switch is pending.
  P20::Params p{};
  ASSERT_EQ(P20::setup(frictional(), p), P20::Ok);
  std::vector<double> C(P20::NSTATE), T(P20::NSTATE);
  P20::initial_state(p, {C.data(), 1});
  const double ev = -2e-3;
  for (int k = 1; k <= 10; ++k) {
    const double eps[6] = {0, 0, ev * k / 10, 0, 0, 0};
    double sig[6], D[6][6];
    P20::update(p, {C.data(), 1}, {T.data(), 1}, eps, sig, D);
    std::swap(C, T);
  }
  femorax::soil::T2 stress;
  stress.load(femorax::Strided<const double>{C.data(), 1}, P20::STATE::STRESS);
  const double scale =
      std::pow((stress.volume() - p.residualPress) / (p.refPressure - p.residualPress), p.pressDependCoeff);
  p.loadStage = 1;
  double Ds[6][6];
  P20::stability_tangent(p, {C.data(), 1}, Ds);
  const double eps[6] = {0, 0, ev, 0, 0, 0};
  double sig[6], D[6][6];
  P20::update(p, {C.data(), 1}, {T.data(), 1}, eps, sig, D);
  EXPECT_EQ(T[P20::STATE::E2P], 1.0);
  EXPECT_DOUBLE_EQ(T[P20::STATE::G], p.refShearModulus * scale);
  EXPECT_DOUBLE_EQ(Ds[3][3], p.refShearModulus * scale);
  EXPECT_DOUBLE_EQ(D[3][3], p.refShearModulus * scale);  // no shear: still elastic
  EXPECT_GT(scale, 1.0);
}

TEST(PressureIndependMultiYield, UpdateIsAPureFunctionOfTheCommittedState) {
  for (const auto& in : {basin_clay(), frictional()}) {
    P20::Params p{};
    ASSERT_EQ(P20::setup(in, p), P20::Ok);
    std::vector<double> C(P20::NSTATE), T1(P20::NSTATE), T2(P20::NSTATE);
    P20::initial_state(p, {C.data(), 1});
    for (int k = 1; k <= 60; ++k) {
      p.loadStage = k <= 10 ? 0 : 1;
      const double ev = -1.0e-3 * std::min(k, 10) / 10.0;
      const double eps[6] = {0, 0, ev, 0, 0, k > 10 ? 4e-3 * std::sin(0.2 * (k - 10)) : 0.0};
      for (auto& v : T2) v = 1e300;  // garbage
      const auto before = C;
      double s1[6], D1[6][6], s2[6], D2[6][6];
      P20::update(p, {C.data(), 1}, {T1.data(), 1}, eps, s1, D1);
      P20::update(p, {C.data(), 1}, {T2.data(), 1}, eps, s2, D2);
      ASSERT_EQ(C, before) << "step " << k;
      ASSERT_EQ(T1, T2) << "step " << k;
      ASSERT_EQ(std::memcmp(s1, s2, sizeof s1), 0);
      ASSERT_EQ(std::memcmp(D1, D2, sizeof D1), 0);
      for (double v : s1) ASSERT_TRUE(std::isfinite(v));
      std::swap(C, T1);  // commit
    }
    EXPECT_EQ(C[P20::STATE::E2P], 1.0);
    EXPECT_GT(C[P20::STATE::ACTIVE], 0.0);
  }
}

}  // namespace
