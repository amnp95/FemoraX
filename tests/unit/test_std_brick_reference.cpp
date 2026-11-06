// stdBrick against OpenSees Brick (tests/reference/drivers/brick_element.cpp, data in
// tests/reference/data/element): internal force with body force, tangent, initial
// tangent, consistent and lumped mass, and the Rayleigh damping forces of both mass
// types, on four geometries x two materials x four displacement/velocity fields.
// Tolerance: the default for element tests, relative 1e-12 of the largest entry
// (docs/deviations.md).

#include "physics/elements/std_brick.hpp"
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
using Brick = femorax::StdBrick<ElasticIsotropic>;
using namespace femorax::test;
constexpr int NDOF = Brick::NDOF;
constexpr double kRelTol = 1e-12;

struct Stat {
  long values = 0, bitwise = 0;
  double max_rel = 0.0;  // largest |got - want| / scale
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

struct BrickCase {
  double xe[8][3];
  Brick::Params cons, lump, cons_nobody, lump_nobody;
  Rayleigh damp;
};

BrickCase make_case(const ReferenceFile& ref) {
  BrickCase s{};
  const auto x = ref.values("coords");
  const auto b = ref.values("body_force");
  const auto r = ref.values("rayleigh");
  EXPECT_EQ(x.size(), 24u);
  for (int n = 0; n < 8; ++n)
    for (int i = 0; i < 3; ++i) s.xe[n][i] = x[3 * n + i];
  const ElasticIsotropic::Params m{ref.param("E"), ref.param("nu"), ref.param("rho")};
  s.cons = {m, {b[0], b[1], b[2]}, 0};
  s.lump = {m, {b[0], b[1], b[2]}, 1};
  s.cons_nobody = {m, {0, 0, 0}, 0};
  s.lump_nobody = {m, {0, 0, 0}, 1};
  s.damp = {r[0], r[1], r[2], r[3]};
  return s;
}

TEST(StdBrickReference, MatchesOpenSeesBrick) {
  const auto state_files = reference_files("element", "Brick_");
  // 4 geometries x 2 materials, a _state and a _mass file each.
  ASSERT_EQ(state_files.size(), 16u) << "reference data missing in " << reference_data_dir()
                                     << "; see tests/reference/README.md";
  Counter c;
  int states = 0;
  for (const auto& path : state_files) {
    SCOPED_TRACE(path.filename().string());
    const ReferenceFile ref = read_reference(path);
    ASSERT_EQ(ref.header.at("opensees_commit"), opensees_pinned_commit);
    const BrickCase s = make_case(ref);
    const bool is_mass = path.filename().string().find("_mass.txt") != std::string::npos;

    if (is_mass) {
      ASSERT_EQ(ref.rows.size(), 1u);
      std::vector<double> M(NDOF * NDOF);
      Brick::mass(s.cons, s.xe, M.data());
      compare(M.data(), &ref.rows[0][ref.column("Mc0")], NDOF * NDOF, "Mc", c);
      Brick::mass(s.lump, s.xe, M.data());
      compare(M.data(), &ref.rows[0][ref.column("Ml0")], NDOF * NDOF, "Ml", c);
      continue;
    }

    ASSERT_EQ(ref.rows.size(), 4u);
    const int cu = ref.column("u0"), cv = ref.column("v0"), cf = ref.column("f0");
    const int cK = ref.column("K0"), cK0 = ref.column("K0_0");
    const int cdc = ref.column("dc0"), cdl = ref.column("dl0");
    for (const auto& row : ref.rows) {
      SCOPED_TRACE("field " + std::to_string(static_cast<int>(row[0])));
      ++states;
      const double* u = &row[cu];
      const double* v = &row[cv];

      // Internal force with body force, and tangent (no damping).
      double fe[NDOF], Ke[NDOF][NDOF];
      Brick::compute<true>(s.cons, Rayleigh{}, s.xe, u, v, {}, {}, fe, Ke);
      compare(fe, &row[cf], NDOF, "f", c);
      compare(&Ke[0][0], &row[cK], NDOF * NDOF, "K", c);
      compare(&Ke[0][0], &row[cK0], NDOF * NDOF, "K0 (elastic: equals K)", c);

      // Rayleigh damping forces alone: zero displacement and no body force leave C v.
      const std::vector<double> zero(NDOF, 0.0);
      double fd[NDOF];
      Brick::compute<false>(s.cons_nobody, s.damp, s.xe, zero.data(), v, {}, {}, fd,
                            nullptr);
      compare(fd, &row[cdc], NDOF, "damping, consistent mass", c);
      Brick::compute<false>(s.lump_nobody, s.damp, s.xe, zero.data(), v, {}, {}, fd,
                            nullptr);
      compare(fd, &row[cdl], NDOF, "damping, lumped mass", c);
    }
  }
  EXPECT_EQ(states, 32);
  std::printf("stdBrick: %ld of %ld values bit-identical to OpenSees Brick\n", c.bitwise, c.values);
  for (const auto& [name, st] : c.by_quantity) {
    std::printf("  %-28s %6ld/%6ld bit-identical, max error %.2e of the largest entry\n",
                name.c_str(), st.bitwise, st.values, st.max_rel);
  }
}

}  // namespace
