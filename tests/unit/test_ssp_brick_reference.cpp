// SSPbrick against OpenSees SSPbrick (tests/reference/drivers/brick_element.cpp, data in
// tests/reference/data/element): internal force with body force, tangent, initial
// tangent (OpenSees returns the tangent), mass and the Rayleigh damping force, on four
// geometries x two materials x four displacement/velocity fields, through every geometry
// source (recomputed, cached). Tolerance: the default for element tests, relative 1e-12 of
// the largest entry (docs/deviations.md).
//
// FemoraX's SSPbrick uses the exact lumped mass (docs/deviations.md); OpenSees' formula is
// exact only for parallelepipeds. So every geometry is compared through the test-only
// variant with OpenSees' mass formula, and the production element is compared on the
// parallelepiped geometries (cube, soil column), where the two formulas agree.

#include "physics/elements/ssp_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
#include "reference_data.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace {

using femorax::ElasticIsotropic;
using femorax::Rayleigh;
using Ssp = femorax::SSPbrick<ElasticIsotropic, femorax::SSPbrickMass::OpenSees>;
using SspExact = femorax::SSPbrick<ElasticIsotropic>;
using namespace femorax::test;
constexpr int NDOF = Ssp::NDOF;
constexpr double kRelTol = 1e-12;

struct Stat {
  long values = 0, bitwise = 0;
  double max_rel = 0.0;
};

struct Counter {
  long values = 0, bitwise = 0;
  std::map<std::string, Stat> by_quantity;
};

void compare(const double* got, const double* want, int n, const char* what, Counter& c) {
  const double scale = max_abs(want, n);
  Stat& st = c.by_quantity[what];
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(got[i], want[i], kRelTol * scale) << what << "[" << i << "]";
    const bool same = got[i] == want[i];
    c.bitwise += same;
    st.bitwise += same;
    ++c.values;
    ++st.values;
    if (scale > 0) st.max_rel = std::max(st.max_rel, std::abs(got[i] - want[i]) / scale);
  }
}

template <class E>
void check_files(bool parallelepipeds_only) {
  const auto files = reference_files("element", "SSPbrick_");
  ASSERT_EQ(files.size(), 16u) << "reference data missing in " << reference_data_dir()
                               << "; see tests/reference/README.md";
  Counter c;
  int states = 0;
  for (const auto& path : files) {
    SCOPED_TRACE(path.filename().string());
    const ReferenceFile ref = read_reference(path);
    const std::string geometry = ref.header.at("geometry");
    if (parallelepipeds_only && geometry != "cube" && geometry != "soil_column") continue;
    ASSERT_EQ(ref.header.at("opensees_commit"), opensees_pinned_commit);
    const auto x = ref.values("coords");
    const auto b = ref.values("body_force");
    const auto r = ref.values("rayleigh");
    double xe[8][3];
    for (int n = 0; n < 8; ++n)
      for (int i = 0; i < 3; ++i) xe[n][i] = x[3 * n + i];
    const ElasticIsotropic::Params m{ref.param("E"), ref.param("nu"), ref.param("rho")};
    const typename E::Params p{m, {b[0], b[1], b[2]}};
    const typename E::Params nobody{m, {0, 0, 0}};
    const Rayleigh damp{r[0], r[1], r[2], r[3]};

    if (path.filename().string().find("_mass.txt") != std::string::npos) {
      ASSERT_EQ(ref.rows.size(), 1u);
      std::vector<double> M(NDOF * NDOF);
      E::mass(p, xe, M.data());
      compare(M.data(), &ref.rows[0][ref.column("M0")], NDOF * NDOF, "M", c);
      continue;
    }

    // The geometry cache and the element constants, built as the launch layer builds them.
    double xl[3][8];
    E::local_coords(xe, xl);
    double geom[E::NGEOM], consts[E::NCONST];
    E::point_store_geometry(xl, 0, {geom, 1});
    E::element_constants(p, xl, {consts, 1});
    const femorax::PointArray<const double> g{geom, 1, E::NGEOM};

    ASSERT_EQ(ref.rows.size(), 4u);
    const int cu = ref.column("u0"), cv = ref.column("v0"), cf = ref.column("f0");
    const int cK = ref.column("K0"), cK0 = ref.column("K0_0"), cd = ref.column("d0");
    for (const auto& row : ref.rows) {
      SCOPED_TRACE("field " + std::to_string(static_cast<int>(row[0])));
      ++states;
      const double* u = &row[cu];
      const double* v = &row[cv];
      for (bool cached : {false, true}) {
        SCOPED_TRACE(cached ? "cached" : "recomputed");
        double fe[NDOF], Ke[NDOF][NDOF];
        if (cached) E::template compute_cached<true>(p, Rayleigh{}, g, {consts, 1}, u, v, {}, {}, fe, Ke);
        else E::template compute<true>(p, Rayleigh{}, xe, u, v, {}, {}, fe, Ke);
        compare(fe, &row[cf], NDOF, "f", c);
        compare(&Ke[0][0], &row[cK], NDOF * NDOF, "K", c);
        compare(&Ke[0][0], &row[cK0], NDOF * NDOF, "K0 (= K in OpenSees)", c);

        // Rayleigh damping alone: zero displacement and no body force leave C v.
        const std::vector<double> zero(NDOF, 0.0);
        double fd[NDOF];
        if (cached) E::template compute_cached<false>(nobody, damp, g, {consts, 1}, zero.data(), v, {}, {}, fd, nullptr);
        else E::template compute<false>(nobody, damp, xe, zero.data(), v, {}, {}, fd, nullptr);
        compare(fd, &row[cd], NDOF, "damping", c);
      }
    }
  }
  EXPECT_EQ(states, parallelepipeds_only ? 16 : 32);
  std::printf("SSPbrick (%s): %ld of %ld values bit-identical to OpenSees SSPbrick\n",
              parallelepipeds_only ? "exact mass, parallelepipeds" : "OpenSees mass formula", c.bitwise, c.values);
  for (const auto& [name, st] : c.by_quantity) {
    std::printf("  %-22s %6ld/%6ld bit-identical, max error %.2e of the largest entry\n", name.c_str(),
                st.bitwise, st.values, st.max_rel);
  }
}

TEST(SSPbrickReference, MatchesOpenSeesSSPbrickWithItsMassFormula) { check_files<Ssp>(false); }

TEST(SSPbrickReference, ExactMassMatchesOpenSeesOnParallelepipeds) { check_files<SspExact>(true); }

}  // namespace
