// ElasticSection3d (physics/sections/elastic_section_3d.hpp) against OpenSees'
// ElasticSection3d.cpp formulas, commit 955ceb0: s = (EA e0, EIz e1, EIy e2, GJ e3), the
// tangent diag(EA, EIz, EIy, GJ) for the trial, initial and committed state alike.

#include <gtest/gtest.h>

#include "physics/sections/elastic_section_3d.hpp"

namespace {

using femorax::ElasticSection3d;
using femorax::SectionResponse;

TEST(ElasticSection3d, ResultantsAndTangentsAreOpenSees) {
  const ElasticSection3d::Params p{2.1e8, 0.03, 4.5e-4, 2.2e-4, 8.1e7, 1.1e-4};
  const double e[4] = {1.3e-4, -2.1e-3, 7.0e-4, 3.3e-3};
  double s[4], ks[4][4], k0[4][4], kc[4][4];
  femorax::Strided<const double> none_c{nullptr, 0};
  femorax::Strided<double> none_t{nullptr, 0};
  ElasticSection3d::update(p, none_c, none_t, e, s, ks);
  ElasticSection3d::initial_tangent(p, k0);
  ElasticSection3d::committed_tangent(p, none_c, kc);
  EXPECT_EQ(s[0], p.E * p.A * e[0]);
  EXPECT_EQ(s[1], p.E * p.Iz * e[1]);
  EXPECT_EQ(s[2], p.E * p.Iy * e[2]);
  EXPECT_EQ(s[3], p.G * p.J * e[3]);
  const double diag[4] = {p.E * p.A, p.E * p.Iz, p.E * p.Iy, p.G * p.J};
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) {
      const double want = i == j ? diag[i] : 0.0;
      EXPECT_EQ(ks[i][j], want);
      EXPECT_EQ(k0[i][j], want);
      EXPECT_EQ(kc[i][j], want);
    }
  // The resultant is the tangent times the deformation (linear section).
  for (int i = 0; i < 4; ++i) {
    double r = 0.0;
    for (int j = 0; j < 4; ++j) r += ks[i][j] * e[j];
    EXPECT_EQ(r, s[i]);
  }
}

TEST(ElasticSection3d, OrderAndPropertiesAreOpenSees) {
  // getType(): P, MZ, MY, T, with OpenSees' code numbers.
  EXPECT_EQ(static_cast<int>(ElasticSection3d::CODE[0]), 2);
  EXPECT_EQ(static_cast<int>(ElasticSection3d::CODE[1]), 1);
  EXPECT_EQ(static_cast<int>(ElasticSection3d::CODE[2]), 4);
  EXPECT_EQ(static_cast<int>(ElasticSection3d::CODE[3]), 6);
  EXPECT_EQ(ElasticSection3d::CODE[3], SectionResponse::T);
  const ElasticSection3d::Params p{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const auto g = ElasticSection3d::properties(p);
  EXPECT_EQ(g.A, 2.0);
  EXPECT_EQ(g.Iz, 3.0);
  EXPECT_EQ(g.Iy, 4.0);
  EXPECT_EQ(g.J, 6.0);
}

}  // namespace
