// The uniaxial materials of fiber sections (Elastic, Steel01, Concrete02) against the OpenSees
// classes along the strain paths of tests/reference/drivers/uniaxial_material.cpp (data in
// tests/reference/data/uniaxial), through the run-time dispatcher a fiber uses
// (physics/materials/uniaxial/uniaxial.hpp), and closed forms.

#include "physics/materials/uniaxial/uniaxial.hpp"
#include "reference_data.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <string>

namespace {

using namespace femorax;
using namespace femorax::test;

static_assert(UniaxialMaterial<ElasticUniaxial>);
static_assert(UniaxialMaterial<Steel01>);
static_assert(UniaxialMaterial<Concrete02>);

UniaxialParams from_reference(const ReferenceFile& ref) {
  const std::string m = ref.header.at("material");
  if (m == "Elastic")
    return uniaxial_params<ElasticUniaxial>(UniaxialType::Elastic, {ref.param("Epos"), ref.param("Eneg")});
  if (m == "Steel01")
    return uniaxial_params<Steel01>(UniaxialType::Steel01,
                                    {ref.param("fy"), ref.param("E0"), ref.param("b"), ref.param("a1"),
                                     ref.param("a2"), ref.param("a3"), ref.param("a4")});
  return uniaxial_params<Concrete02>(UniaxialType::Concrete02,
                                     {ref.param("fc"), ref.param("epsc0"), ref.param("fcu"), ref.param("epscu"),
                                      ref.param("rat"), ref.param("ft"), ref.param("Ets")});
}

// The driver's protocol: a thrown-away trial every 7th step, the same iterate twice every 11th,
// then the recorded evaluation and the commit. State in two buffers, with a stride of 3 so the
// materials are shown to go through the strided views.
TEST(UniaxialReference, MatchesOpenSeesAlongStrainPaths) {
  long values = 0, bitwise = 0;
  for (const std::string material : {"Elastic", "Steel01", "Concrete02"}) {
    const auto files = reference_files("uniaxial", material + "_");
    const std::size_t cases = material == "Elastic" ? 2u : 3u;
    ASSERT_EQ(files.size(), cases * 6u) << "reference data missing in " << reference_data_dir()
                                        << "; see tests/reference/README.md";
    long mvalues = 0, mbitwise = 0;
    for (const auto& path : files) {
      SCOPED_TRACE(path.filename().string());
      const ReferenceFile ref = read_reference(path);
      ASSERT_EQ(ref.header.at("opensees_commit"), opensees_pinned_commit);
      const UniaxialParams u = from_reference(ref);
      const double scale = ref.values("scale")[0];
      const int c_eps = ref.column("eps"), c_sig = ref.column("sig"), c_E = ref.column("E");
      constexpr int kStride = 3;
      double a[kUniaxialState * kStride] = {}, b[kUniaxialState * kStride] = {};
      double* committed = a;
      double* trial = b;
      double s_scale = 0.0;
      for (const auto& row : ref.rows) s_scale = std::max(s_scale, std::fabs(row[c_sig]));
      const double E0 = std::fabs(uniaxial_initial_tangent(u));
      for (std::size_t k = 0; k < ref.rows.size(); ++k) {
        const auto& row = ref.rows[k];
        const double strain = row[c_eps];
        double sig = 0.0, E = 0.0;
        if (k % 7 == 6) uniaxial_update(u, {committed, kStride}, {trial, kStride}, -2.5 * strain + scale, sig, E);
        if (k % 11 == 10) uniaxial_update(u, {committed, kStride}, {trial, kStride}, strain, sig, E);
        uniaxial_update(u, {committed, kStride}, {trial, kStride}, strain, sig, E);
        EXPECT_NEAR(sig, row[c_sig], 1e-14 * s_scale) << "step " << k;
        EXPECT_NEAR(E, row[c_E], 1e-14 * E0) << "step " << k;
        mbitwise += (sig == row[c_sig]) + (E == row[c_E]);
        mvalues += 2;
        std::swap(committed, trial);
      }
    }
    std::printf("%s: %ld of %ld values bit-identical to OpenSees\n", material.c_str(), mbitwise, mvalues);
    values += mvalues;
    bitwise += mbitwise;
  }
  ::testing::Test::RecordProperty("values", std::to_string(values));
  ::testing::Test::RecordProperty("bitwise_equal", std::to_string(bitwise));
}

// Steel01 without isotropic hardening is the bilinear law: elastic to fy, then fy + b E0 (eps -
// eps_y); unloading is elastic with E0, and reverse yielding starts 2 fy below (kinematic).
TEST(Uniaxial, Steel01IsTheBilinearKinematicLaw) {
  const Steel01::Params p{400.0, 2.0e5, 0.02, 0.0, 55.0, 0.0, 55.0};
  const double ey = p.fy / p.E0;
  double c[Steel01::NSTATE] = {}, t[Steel01::NSTATE] = {};
  double sig, E;
  Steel01::update(p, {c, 1}, {t, 1}, 0.5 * ey, sig, E);
  EXPECT_DOUBLE_EQ(sig, 0.5 * p.fy);
  EXPECT_DOUBLE_EQ(E, p.E0);
  Steel01::update(p, {c, 1}, {t, 1}, 3.0 * ey, sig, E);
  EXPECT_NEAR(sig, p.fy + p.b * p.E0 * 2.0 * ey, 1e-12 * p.fy);
  EXPECT_DOUBLE_EQ(E, p.b * p.E0);
  const double s3 = sig;
  for (int i = 0; i < Steel01::NSTATE; ++i) c[i] = t[i];  // commit at 3 ey
  Steel01::update(p, {c, 1}, {t, 1}, 2.0 * ey, sig, E);   // elastic unloading
  EXPECT_NEAR(sig, s3 - p.fy, 1e-12 * p.fy);
  EXPECT_DOUBLE_EQ(E, p.E0);
  Steel01::update(p, {c, 1}, {t, 1}, -1.0 * ey, sig, E);  // reverse yielding on the lower line
  EXPECT_NEAR(sig, p.b * p.E0 * (-ey) - p.fy * (1.0 - p.b), 1e-12 * p.fy);
  EXPECT_DOUBLE_EQ(E, p.b * p.E0);
}

// Concrete02's compression envelope: the parabola to (epsc0, fc), the line to (epscu, fcu),
// then fcu; in tension, linear to ft and softening with Ets to zero.
TEST(Uniaxial, Concrete02FollowsItsEnvelopes) {
  const Concrete02::Params p{-30.0, -0.002, -6.0, -0.006, 0.1, 3.0, 1500.0};
  const double Ec = 2.0 * p.fc / p.epsc0;
  auto at = [&](double eps, double& sig, double& E) {
    double c[Concrete02::NSTATE] = {}, t[Concrete02::NSTATE] = {};
    Concrete02::update(p, {c, 1}, {t, 1}, eps, sig, E);
  };
  double sig, E;
  at(-0.001, sig, E);
  EXPECT_NEAR(sig, p.fc * 0.5 * 1.5, 1e-13 * 30.0);
  EXPECT_NEAR(E, Ec * 0.5, 1e-13 * Ec);
  at(p.epsc0, sig, E);
  EXPECT_NEAR(sig, p.fc, 1e-13 * 30.0);
  at(-0.004, sig, E);
  EXPECT_NEAR(sig, 0.5 * (p.fc + p.fcu), 1e-13 * 30.0);
  at(-0.01, sig, E);
  EXPECT_DOUBLE_EQ(sig, p.fcu);
  const double et = p.ft / Ec;
  at(0.5 * et, sig, E);
  EXPECT_NEAR(sig, 0.5 * p.ft, 1e-13 * 3.0);
  at(et + 0.001, sig, E);
  EXPECT_NEAR(sig, p.ft - p.Ets * 0.001, 1e-13 * 3.0);
  EXPECT_DOUBLE_EQ(E, -p.Ets);
  at(et + p.ft / p.Ets + 0.001, sig, E);
  EXPECT_DOUBLE_EQ(sig, 0.0);
}

// A state of zeros is the initial state: the first evaluation from a fresh buffer gives the
// initial tangent for a tiny strain, for every material.
TEST(Uniaxial, AFreshStateIsTheInitialState) {
  const UniaxialParams all[] = {
      uniaxial_params<ElasticUniaxial>(UniaxialType::Elastic, {2.0e8, 2.0e8}),
      uniaxial_params<Steel01>(UniaxialType::Steel01, {4.2e5, 2.0e8, 0.01, 0.0, 55.0, 0.0, 55.0}),
      uniaxial_params<Concrete02>(UniaxialType::Concrete02, {-28000.0, -0.002, -5600.0, -0.006, 0.1, 2800.0, 1.4e6}),
  };
  for (const auto& u : all) {
    double c[kUniaxialState] = {}, t[kUniaxialState] = {};
    double sig, E;
    uniaxial_update(u, {c, 1}, {t, 1}, 0.0, sig, E);  // no strain change: the committed state
    EXPECT_EQ(sig, 0.0);
    EXPECT_EQ(E, uniaxial_initial_tangent(u));
    uniaxial_update(u, {c, 1}, {t, 1}, 1e-9, sig, E);
    EXPECT_NEAR(sig, 1e-9 * uniaxial_initial_tangent(u), 1e-6 * std::fabs(sig));
  }
}

}  // namespace
