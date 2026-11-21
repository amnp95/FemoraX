// FiberSection3d against OpenSees' class along section-deformation paths (reference driver
// tests/reference/drivers/fiber_section.cpp, data in tests/reference/data/section), closed forms,
// and the lane rules of its fiber loop.

#include "physics/sections/fiber_section_3d.hpp"
#include "reference_data.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

using namespace femorax;
using namespace femorax::test;

using Sec = FiberSection3d<128>;
static_assert(Section<Sec>);
static_assert(Section<FiberSection3d<32>>);
constexpr int N = Sec::NSTRAIN;

std::unique_ptr<Sec::Params> from_reference(const ReferenceFile& ref) {
  auto p = std::make_unique<Sec::Params>();
  const auto mv = ref.values("materials"), fv = ref.values("fibers");
  EXPECT_EQ(mv.size() % 8, 0u);
  EXPECT_LE(mv.size() / 8, static_cast<std::size_t>(kFiberMaterials));
  for (std::size_t m = 0; m < mv.size() / 8; ++m) {
    p->materials[m].type = static_cast<int>(mv[8 * m]);
    for (int k = 0; k < 7; ++k) p->materials[m].v[k] = mv[8 * m + 1 + static_cast<std::size_t>(k)];
  }
  p->numFibers = static_cast<int>(fv.size() / 4);
  EXPECT_LE(p->numFibers, Sec::MAX_FIBERS);
  for (int i = 0; i < p->numFibers; ++i) {
    const std::size_t o = 4 * static_cast<std::size_t>(i);
    p->matData[3 * i] = fv[o];
    p->matData[3 * i + 1] = fv[o + 1];
    p->matData[3 * i + 2] = fv[o + 2];
    p->material[i] = static_cast<int>(fv[o + 3]);
  }
  p->GJ = ref.values("GJ")[0];
  if (ref.values("centroid")[0] != 0.0) Sec::compute_centroid(*p);
  return p;
}

// A lane-parallel context emulated on the host: inner_for runs the indices in reverse and
// inner_reduce still adds in index order, as the launch layer's lanes do. Each lane has private
// locals, which the section respects by writing its fibers' results to the work array.
struct ReversedInner {};
template <class F>
void inner_for(const ReversedInner&, int n, F&& f) {
  for (int i = n - 1; i >= 0; --i) f(i);
}
template <class F>
auto inner_reduce(const ReversedInner&, int n, F&& f) -> decltype(f(0)) {
  decltype(f(0)) acc = 0;
  for (int i = 0; i < n; ++i) acc += f(i);
  return acc;
}

TEST(FiberSection3dReference, MatchesOpenSeesAlongDeformationPaths) {
  const auto files = reference_files("section", "FiberSection3d_");
  ASSERT_EQ(files.size(), 12u) << "reference data missing in " << reference_data_dir()
                               << "; see tests/reference/README.md";
  long values = 0, bitwise = 0;
  double worst_s = 0.0, worst_k = 0.0;
  for (const auto& path : files) {
    SCOPED_TRACE(path.filename().string());
    const ReferenceFile ref = read_reference(path);
    ASSERT_EQ(ref.header.at("opensees_commit"), opensees_pinned_commit);
    const auto p = from_reference(ref);
    const int c_e = ref.column("e0"), c_s = ref.column("s0"), c_k = ref.column("ks00");
    // The scales of the driver's thrown-away trial, recovered from its paths' definition.
    const std::string name = ref.header.at("case");
    const double scale[4] = {name == "steel_i" ? 2e-3 : 1e-3, name == "rc" ? 0.012 : name == "steel_i" ? 0.03 : 0.02,
                             name == "rc" ? 0.018 : name == "steel_i" ? 0.06 : 0.02, 0.01};
    constexpr int kStride = 2;
    std::vector<double> a(static_cast<std::size_t>(Sec::NSTATE) * kStride, 0.0), b(a), work(Sec::NWORK);
    double* committed = a.data();
    double* trial = b.data();
    double s_scale[N] = {}, k_scale = 0.0;
    for (const auto& row : ref.rows) {
      for (int i = 0; i < N; ++i) s_scale[i] = std::max(s_scale[i], std::fabs(row[c_s + i]));
      k_scale = std::max(k_scale, max_abs(&row[c_k], N * N));
    }
    const double s_all = max_abs(s_scale, N);
    auto compare = [&](const double* s, const double (*ks)[N], const std::vector<double>& row, bool forces) {
      for (int i = 0; i < N; ++i) {
        if (forces) {
          EXPECT_NEAR(s[i], row[c_s + i], 2e-13 * s_all) << "step " << row[0] << " s" << i;
          worst_s = std::max(worst_s, std::fabs(s[i] - row[c_s + i]) / s_all);
          bitwise += (s[i] == row[c_s + i]);
          ++values;
        }
        for (int j = 0; j < N; ++j) {
          EXPECT_NEAR(ks[i][j], row[c_k + i * N + j], 2e-13 * k_scale) << "step " << row[0] << " ks" << i << j;
          worst_k = std::max(worst_k, std::fabs(ks[i][j] - row[c_k + i * N + j]) / k_scale);
          bitwise += (ks[i][j] == row[c_k + i * N + j]);
          ++values;
        }
      }
    };
    double e[N], s[N], ks[N][N];
    ASSERT_EQ(ref.rows[0][0], -1.0);
    Sec::initial_tangent(*p, ks);
    compare(nullptr, ks, ref.rows[0], false);
    for (std::size_t r = 1; r < ref.rows.size(); ++r) {
      const auto& row = ref.rows[r];
      const std::size_t k = r - 1;
      if (k % 7 == 6) {
        for (int i = 0; i < N; ++i) e[i] = -1.7 * row[c_e + i] + 0.3 * scale[i];
        Sec::update(SerialInner{}, *p, {committed, kStride}, {trial, kStride}, e, s, ks, work.data());
      }
      for (int i = 0; i < N; ++i) e[i] = row[c_e + i];
      Sec::update(SerialInner{}, *p, {committed, kStride}, {trial, kStride}, e, s, ks, work.data());
      compare(s, ks, row, true);
      // Another lane order gives the same bits.
      double s2[N], ks2[N][N];
      std::vector<double> t2(a.size()), w2(Sec::NWORK);
      Sec::update(ReversedInner{}, *p, {committed, kStride}, {t2.data(), kStride}, e, s2, ks2, w2.data());
      for (int i = 0; i < N; ++i) {
        ASSERT_EQ(s2[i], s[i]);
        for (int j = 0; j < N; ++j) ASSERT_EQ(ks2[i][j], ks[i][j]);
      }
      for (int i = 0; i < p->numFibers * kUniaxialState; ++i)
        ASSERT_EQ(t2[static_cast<std::size_t>(i) * kStride], trial[static_cast<std::size_t>(i) * kStride]);
      // The committed tangent of the new state is the tangent just computed.
      std::swap(committed, trial);
      double kc[N][N];
      Sec::committed_tangent(*p, {committed, kStride}, kc);
      for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) ASSERT_EQ(kc[i][j], ks[i][j]) << "step " << k;
    }
  }
  ::testing::Test::RecordProperty("values", std::to_string(values));
  ::testing::Test::RecordProperty("bitwise_equal", std::to_string(bitwise));
  std::printf("FiberSection3d: %ld of %ld values bit-identical to OpenSees; worst s %.1e, ks %.1e\n", bitwise, values,
              worst_s, worst_k);
}

// Elastic fibers on a fine grid of a rectangle are the elastic section: EA, EIz, EIy (the
// midpoint rule's second moments: b h^3 / 12 (1 - 1 / n^2)), no coupling about the centroid.
TEST(FiberSection3d, ElasticFibersGiveTheElasticSection) {
  auto p = std::make_unique<Sec::Params>();
  const double E = 2.0e8, h = 0.6, b = 0.3, y0 = 0.4, z0 = -0.7;
  const int ny = 12, nz = 8;
  p->materials[0] = uniaxial_params<ElasticUniaxial>(UniaxialType::Elastic, {E, E});
  for (int i = 0; i < ny; ++i)
    for (int j = 0; j < nz; ++j) {
      const int f = p->numFibers++;
      p->matData[3 * f] = y0 + (i + 0.5) * h / ny;
      p->matData[3 * f + 1] = z0 + (j + 0.5) * b / nz;
      p->matData[3 * f + 2] = (h / ny) * (b / nz);
    }
  p->GJ = 123.0;
  Sec::compute_centroid(*p);
  EXPECT_NEAR(p->yBar, y0 + 0.5 * h, 1e-14);
  EXPECT_NEAR(p->zBar, z0 + 0.5 * b, 1e-14);
  const double EA = E * b * h, EIz = E * b * h * h * h / 12.0 * (1.0 - 1.0 / (ny * ny)),
               EIy = E * h * b * b * b / 12.0 * (1.0 - 1.0 / (nz * nz));
  std::vector<double> c(Sec::NSTATE, 0.0), t(Sec::NSTATE, 0.0);
  const double e[N] = {1e-4, 2e-3, -3e-3, 5e-3};
  double s[N], ks[N][N];
  Sec::update(*p, {c.data(), 1}, {t.data(), 1}, e, s, ks);
  EXPECT_NEAR(s[0], EA * e[0], 1e-12 * EA * 1e-3);
  EXPECT_NEAR(s[1], EIz * e[1], 1e-12 * EIz * 1e-3);
  EXPECT_NEAR(s[2], EIy * e[2], 1e-12 * EIy * 1e-3);
  EXPECT_DOUBLE_EQ(s[3], p->GJ * e[3]);
  EXPECT_NEAR(ks[0][0], EA, 1e-12 * EA);
  EXPECT_NEAR(ks[1][1], EIz, 1e-12 * EIz);
  EXPECT_NEAR(ks[2][2], EIy, 1e-12 * EIy);
  EXPECT_NEAR(ks[0][1], 0.0, 1e-12 * EA);
  EXPECT_NEAR(ks[0][2], 0.0, 1e-12 * EA);
  EXPECT_NEAR(ks[1][2], 0.0, 1e-12 * EA);
  const SectionProperties g = Sec::properties(*p);
  EXPECT_NEAR(g.A, b * h, 1e-14);
  EXPECT_NEAR(g.Iz, EIz / E, 1e-14);
  EXPECT_NEAR(g.Iy, EIy / E, 1e-14);
}

// The tangent is the derivative of the resultants (central differences on a yielding RC-like
// section, away from the materials' kinks).
TEST(FiberSection3d, TheTangentIsTheDerivativeOfTheResultants) {
  auto p = std::make_unique<Sec::Params>();
  p->materials[0] = uniaxial_params<Concrete02>(UniaxialType::Concrete02,
                                                {-30000.0, -0.002, -6000.0, -0.006, 0.1, 3000.0, 1.5e6});
  p->materials[1] = uniaxial_params<Steel01>(UniaxialType::Steel01, {4.2e5, 2.0e8, 0.01, 0.0, 55.0, 0.0, 55.0});
  for (int i = 0; i < 7; ++i)
    for (int j = 0; j < 5; ++j) {
      const int f = p->numFibers++;
      p->matData[3 * f] = -0.3 + (i + 0.5) * 0.6 / 7;
      p->matData[3 * f + 1] = -0.2 + (j + 0.5) * 0.4 / 5;
      p->matData[3 * f + 2] = (0.6 / 7) * (0.4 / 5);
    }
  for (const double y : {-0.25, 0.25})
    for (const double z : {-0.15, 0.15}) {
      const int f = p->numFibers++;
      p->matData[3 * f] = y;
      p->matData[3 * f + 1] = z;
      p->matData[3 * f + 2] = 5e-4;
      p->material[f] = 1;
    }
  p->GJ = 1.0e4;
  Sec::compute_centroid(*p);
  std::vector<double> c(Sec::NSTATE, 0.0), t(Sec::NSTATE, 0.0);
  const double e[N] = {-4.1e-4, 7.3e-3, -3.7e-3, 1e-3};
  double s[N], ks[N][N];
  Sec::update(*p, {c.data(), 1}, {t.data(), 1}, e, s, ks);
  for (int j = 0; j < N; ++j) {
    const double h = 1e-7 * (j == 0 ? 1e-3 : 1e-2);
    double ep[N], em[N], sp[N], sm[N], dummy[N][N];
    for (int i = 0; i < N; ++i) ep[i] = em[i] = e[i];
    ep[j] += h;
    em[j] -= h;
    Sec::update(*p, {c.data(), 1}, {t.data(), 1}, ep, sp, dummy);
    Sec::update(*p, {c.data(), 1}, {t.data(), 1}, em, sm, dummy);
    for (int i = 0; i < N; ++i)
      EXPECT_NEAR((sp[i] - sm[i]) / (2 * h), ks[i][j], 1e-5 * std::fabs(ks[j][j]) + 1e-5 * std::fabs(ks[i][i]))
          << i << j;
  }
}

}  // namespace
