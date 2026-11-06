// ElasticIsotropic (3D) against OpenSees ElasticIsotropicThreeDimensional along the
// strain paths of tests/reference/drivers/nd_material.cpp (data in
// tests/reference/data/nd). Tolerance: the default for material tests, relative 1e-14
// (docs/deviations.md). The port keeps OpenSees' operation order, so the test also
// reports how many values match bit for bit.

#include "physics/materials/elastic_isotropic.hpp"
#include "reference_data.hpp"

#include <gtest/gtest.h>

namespace {

using femorax::ElasticIsotropic;
using namespace femorax::test;

constexpr int N = ElasticIsotropic::NSTRAIN;
constexpr double kRelTol = 1e-14;

TEST(ElasticIsotropicReference, MatchesOpenSeesAlongStrainPaths) {
  const auto files = reference_files("nd", "ElasticIsotropic_");
  // 5 parameter sets x 4 strain paths, as written by the driver.
  ASSERT_EQ(files.size(), 20u) << "reference data missing in " << reference_data_dir()
                               << "; see tests/reference/README.md";

  long values = 0, bitwise = 0;
  for (const auto& path : files) {
    SCOPED_TRACE(path.filename().string());
    const ReferenceFile ref = read_reference(path);
    ASSERT_EQ(ref.header.at("opensees_commit"), opensees_pinned_commit);
    const ElasticIsotropic::Params p{ref.param("E"), ref.param("nu"), ref.param("rho")};
    const int c_eps = ref.column("eps0"), c_sig = ref.column("sig0"), c_D = ref.column("D00");
    ASSERT_FALSE(ref.rows.empty());

    for (const auto& row : ref.rows) {
      double eps[N], sig[N], D[N][N];
      for (int i = 0; i < N; ++i) eps[i] = row[c_eps + i];
      ElasticIsotropic::update(p, nullptr, nullptr, eps, sig, D);

      const double* want_sig = &row[c_sig];
      const double* want_D = &row[c_D];
      const double s_scale = max_abs(want_sig, N), d_scale = max_abs(want_D, N * N);
      for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(sig[i], want_sig[i], kRelTol * s_scale) << "step " << row[0] << " sig" << i;
        bitwise += (sig[i] == want_sig[i]);
        ++values;
        for (int j = 0; j < N; ++j) {
          EXPECT_NEAR(D[i][j], want_D[i * N + j], kRelTol * d_scale)
              << "step " << row[0] << " D" << i << j;
          bitwise += (D[i][j] == want_D[i * N + j]);
          ++values;
        }
      }
    }
  }
  ::testing::Test::RecordProperty("values", std::to_string(values));
  ::testing::Test::RecordProperty("bitwise_equal", std::to_string(bitwise));
  std::printf("ElasticIsotropic: %ld of %ld values bit-identical to OpenSees\n", bitwise, values);
}

}  // namespace
