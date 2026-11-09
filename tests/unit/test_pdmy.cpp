// PressureDependMultiYield properties that need no reference data: the contract, the state
// size per point, setup errors, the elastic stage, and update as a pure function of the
// committed state (what makes commit a buffer swap and revert free). The comparison with
// OpenSees is test_pdmy_reference.cpp.

#include "physics/materials/pressure_depend_multi_yield.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using P20 = femorax::PressureDependMultiYield<20>;
using P40 = femorax::PressureDependMultiYield<40>;

static_assert(femorax::Material<P20>);
static_assert(femorax::Material<P40>);
static_assert(P20::NSTATE == 89 + 6 * 20);
static_assert(P40::NSTATE == 89 + 6 * 40);
static_assert(P20::TANGENT == femorax::TangentSymmetry::SymmetricPositiveSemidefinite);
static_assert(std::is_trivially_copyable_v<P20::Params>);

// Femora's dense Ottawa sand (examples/site_response/nonlinear_layered_soil_column.py).
P20::Input dense_ottawa() {
  P20::Input in{};
  in.rho = 19.9 / 9.81;
  in.refShearModul = 145.0e3;
  in.refBulkModul = 2.0 * 145.0e3 * 1.3 / 1.2;
  in.frictionAng = 40.0;
  in.peakShearStra = 0.1;
  in.refPress = 80.0;
  in.pressDependCoe = 0.5;
  in.phaseTransformAng = 27.0;
  in.contractionParam1 = 0.03;
  in.dilationParam1 = 0.8;
  in.dilationParam2 = 5.0;
  in.numberOfYieldSurf = 20;
  in.e = 0.45;
  in.volLimit1 = 0.9;
  in.volLimit2 = 0.02;
  in.volLimit3 = 0.7;
  in.atm = 101.0;
  in.cohesi = 0.3;
  in.pv = 1.0;
  return in;
}

TEST(PressureDependMultiYield, StateSizePerPoint) {
  // Committed and trial buffers: 2 x (89 + 6 NS) doubles per Gauss point.
  const double kb20 = 2.0 * P20::NSTATE * 8 / 1024.0, kb40 = 2.0 * P40::NSTATE * 8 / 1024.0;
  std::printf("[ PDMY state per Gauss point: NS 20: %d doubles x 2 = %.2f KiB; NS 40: %d x 2 = %.2f KiB ]\n",
              P20::NSTATE, kb20, P40::NSTATE, kb40);
  std::printf("[ Params: NS 20 %zu bytes, NS 40 %zu bytes (one per material and damping region) ]\n",
              sizeof(P20::Params), sizeof(P40::Params));
  EXPECT_EQ(P20::NSTATE, 209);
  EXPECT_EQ(P40::NSTATE, 329);
}

TEST(PressureDependMultiYield, SetupChecksTheBucketAndTheBackbone) {
  P20::Params p{};
  auto in = dense_ottawa();
  EXPECT_EQ(P20::setup(in, p), P20::Ok);
  EXPECT_EQ(p.numOfSurfaces, 20);
  EXPECT_LT(p.refPressure, 0.0);  // compression negative, as OpenSees stores it
  EXPECT_EQ(p.size[0], 0.0);      // OpenSees' unused surface 0
  EXPECT_EQ(p.modulus[20], 0.0);  // the outermost surface is perfectly plastic
  for (int i = 1; i < 20; ++i) {
    EXPECT_GT(p.size[i + 1], p.size[i]);
    EXPECT_GT(p.modulus[i], 0.0);
  }
  in.numberOfYieldSurf = 30;
  EXPECT_EQ(P20::setup(in, p), P20::TooManySurfaces);
  P40::Params q{};
  EXPECT_EQ(P40::setup(in, q), P40::Ok);
  // A backbone whose strength implies sin(phi) > 1 (OpenSees exits).
  const double too_strong[] = {1e-4, 0.9, 1e-2, 0.5};
  in = dense_ottawa();
  in.numberOfYieldSurf = 2;
  in.gredu = too_strong;
  EXPECT_EQ(P20::setup(in, p), P20::InvalidFrictionAngle);
  const double one[] = {1e-4, 0.9};
  in.numberOfYieldSurf = 1;
  in.gredu = one;
  EXPECT_EQ(P20::setup(in, p), P20::TooFewPairs);
  // PTAng above the friction angle is reset to it, as OpenSees does.
  in = dense_ottawa();
  in.phaseTransformAng = 45.0;
  EXPECT_EQ(P20::setup(in, p), P20::Ok);
  EXPECT_EQ(p.phaseTransfAngle, 40.0);
}

TEST(PressureDependMultiYield, ElasticStageIsLinearWithTheReferenceModuli) {
  P20::Params p{};
  ASSERT_EQ(P20::setup(dense_ottawa(), p), P20::Ok);
  std::vector<double> C(P20::NSTATE), T(P20::NSTATE);
  P20::initial_state(p, {C.data(), 1});
  const double eps[6] = {1e-4, -2e-4, 3e-4, 5e-4, -1e-4, 2e-4};
  double sig[6], D[6][6];
  P20::update(p, {C.data(), 1}, {T.data(), 1}, eps, sig, D);
  const double G = 145.0e3, K = 2.0 * 145.0e3 * 1.3 / 1.2;
  for (int i = 0; i < 6; ++i) {
    double want = 0.0;
    for (int j = 0; j < 6; ++j) want += D[i][j] * eps[j];
    EXPECT_NEAR(sig[i], want, 1e-12 * std::abs(want) + 1e-12);
  }
  EXPECT_DOUBLE_EQ(D[3][3], G);
  EXPECT_DOUBLE_EQ(D[0][0], K + 4.0 * G / 3.0);
}

TEST(PressureDependMultiYield, UpdateIsAPureFunctionOfTheCommittedState) {
  // Consolidate in stage 0, switch to plastic, shear: at every step two updates from the
  // same committed state give the same bits, leave the committed buffer untouched and write
  // every trial value (a trial buffer full of garbage before the update changes nothing).
  P20::Params p{};
  ASSERT_EQ(P20::setup(dense_ottawa(), p), P20::Ok);
  std::vector<double> C(P20::NSTATE), T1(P20::NSTATE), T2(P20::NSTATE);
  P20::initial_state(p, {C.data(), 1});
  for (int k = 1; k <= 60; ++k) {
    p.loadStage = k <= 10 ? 0 : 1;
    const double ev = -3.4e-4 * std::min(k, 10) / 10.0;
    const double eps[6] = {0, 0, ev, 0, 0, k > 10 ? 2e-3 * std::sin(0.2 * (k - 10)) : 0.0};
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
  EXPECT_EQ(C[P20::STATE::E2P], 1.0);  // the switch to the plastic stage happened
  EXPECT_GT(C[P20::STATE::ACTIVE], 0.0);
}

TEST(PressureDependMultiYield, StabilityTangentIsTheElasticTangentAtTheConfinement) {
  // The stable step's tangent: stage 0, the reference moduli; stage 1, the moduli scaled by
  // getModulusFactor of the committed stress, at least as stiff as the committed
  // (elastoplastic) tangent, and stiffer than the reference below the column's depth.
  P20::Params p{};
  ASSERT_EQ(P20::setup(dense_ottawa(), p), P20::Ok);
  std::vector<double> C(P20::NSTATE), T(P20::NSTATE);
  P20::initial_state(p, {C.data(), 1});
  double D[6][6], Dc[6][6], Dref[6][6];
  P20::stability_tangent(p, {C.data(), 1}, Dref);
  EXPECT_EQ(Dref[3][3], 145.0e3);
  for (int k = 1; k <= 40; ++k) {
    p.loadStage = k <= 10 ? 0 : 1;
    const double ev = -1.2e-3 * std::min(k, 10) / 10.0;  // confinement above refPress
    const double eps[6] = {0, 0, ev, 0, 0, k > 10 ? 1.5e-3 * std::sin(0.3 * (k - 10)) : 0.0};
    double sig[6];
    P20::update(p, {C.data(), 1}, {T.data(), 1}, eps, sig, D);
    std::swap(C, T);
  }
  P20::stability_tangent(p, {C.data(), 1}, D);
  P20::committed_tangent(p, {C.data(), 1}, Dc);
  femorax::soil::T2 stress;
  stress.load(femorax::Strided<const double>{C.data(), 1}, P20::STATE::STRESS);
  const double factor =
      std::pow((stress.volume() - p.residualPress) / (p.refPressure - p.residualPress), p.pressDependCoeff);
  EXPECT_GT(factor, 1.0);
  EXPECT_DOUBLE_EQ(D[3][3], 145.0e3 * factor);
  EXPECT_DOUBLE_EQ(D[0][0], (2.0 * 145.0e3 * 1.3 / 1.2 + 4.0 * 145.0e3 / 3.0) * factor);
  for (int i = 0; i < 6; ++i) EXPECT_GE(D[i][i], Dc[i][i] * (1.0 - 1e-12)) << i;
  EXPECT_GT(D[3][3], Dref[3][3]);
}

}  // namespace
