// Local time stepping levels (compile/lts_levels.hpp; docs/design.md, "Local time stepping",
// step 2): each element's level from its own stable step, each equation the finest level of
// its elements, each element evaluated up to the finest level of its equations (the ring), the
// work estimate; and the per-element stable steps from the device (exec::InternalForce).

#include "compile/lts_levels.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "../common/synthetic_models.hpp"
#include "exec/internal_force.hpp"

namespace {

using femorax::compile::assign_lts_levels;
using femorax::compile::Compiled;
using femorax::compile::compile_model;
using femorax::compile::CompileOptions;
using femorax::compile::LtsOptions;

Compiled column() {
  femorax::testing::ColumnSpec s;
  s.nx = 3;
  s.ny = 3;
  s.nz = 6;
  return compile_model(femorax::testing::soil_column(s), CompileOptions{});
}

std::vector<std::vector<double>> uniform_steps(const Compiled& c, double v) {
  std::vector<std::vector<double>> d;
  for (const auto& b : c.batches) d.emplace_back(static_cast<std::size_t>(b.count), v);
  return d;
}

TEST(LtsLevels, OneLevelWhenEveryElementAllowsTheStep) {
  const Compiled c = column();
  const auto L = assign_lts_levels(c, uniform_steps(c, 1.0), 0.9);
  EXPECT_EQ(L.levels, 1);
  EXPECT_EQ(L.too_fine, 0);
  EXPECT_DOUBLE_EQ(L.speedup(), 1.0);
  for (const auto l : L.equation_level) EXPECT_EQ(l, 0);
}

TEST(LtsLevels, ASmallElementIsFineAndItsNeighboursFormTheRing) {
  // Every element allows 1.0 except one (0.3): with dt = 0.9 and margin 0.9 it needs
  // 0.9 / 4 = 0.225 <= 0.27, level 2. Its equations are level 2; every element sharing a node
  // with it (the ring) is evaluated in the level-2 substeps.
  const Compiled c = column();
  ASSERT_EQ(c.batches.size(), 1u);
  const auto& b = c.batches[0];
  auto dts = uniform_steps(c, 1.0);
  const std::int32_t small = b.count / 2;
  dts[0][static_cast<std::size_t>(small)] = 0.3;
  const auto L = assign_lts_levels(c, dts, 0.9);
  EXPECT_EQ(L.levels, 3);
  EXPECT_EQ(L.element_level[0][static_cast<std::size_t>(small)], 2);
  EXPECT_EQ(L.elements_at_level[0], b.count - 1);
  EXPECT_EQ(L.elements_at_level[2], 1);
  // The ring, independently: elements sharing a free equation with the small one. The column's
  // laminar ties merge each level's side nodes into one equation, so the ring reaches across
  // the column (a constraint group sits in one level); the node-sharing elements are among it.
  const auto P = static_cast<std::size_t>(b.padded);
  std::set<std::int32_t> nodes;
  for (int a = 0; a < b.nnode; ++a) nodes.insert(b.node[static_cast<std::size_t>(a) * P + static_cast<std::size_t>(small)]);
  std::int64_t ring = 0;
  for (std::int32_t e = 0; e < b.count; ++e) {
    bool shares = false;
    for (int a = 0; a < b.nnode; ++a) shares |= nodes.count(b.node[static_cast<std::size_t>(a) * P + static_cast<std::size_t>(e)]) > 0;
    // An element touching only fixed equations of the small element's nodes is not in the ring.
    bool free_shared = false;
    for (int a = 0; a < b.ndof; ++a) {
      const std::int32_t q = b.eq[static_cast<std::size_t>(a) * P + static_cast<std::size_t>(e)];
      if (q >= 0 && L.equation_level[static_cast<std::size_t>(q)] == 2) free_shared = true;
    }
    if (free_shared) ++ring;
    if (shares) {
      EXPECT_TRUE(free_shared || L.element_level[0][static_cast<std::size_t>(e)] == 0) << "element " << e;
    }
    EXPECT_EQ(L.touched_level[0][static_cast<std::size_t>(e)], free_shared ? 2 : 0) << "element " << e;
  }
  EXPECT_GE(ring, 2);
  // Work: every element at the finest step (4 runs each) against 4 runs for the ring, 1 for the rest.
  EXPECT_DOUBLE_EQ(L.global_work, 4.0 * b.count);
  EXPECT_DOUBLE_EQ(L.lts_work, 4.0 * ring + (b.count - ring));
  EXPECT_NEAR(L.speedup(), 4.0 * b.count / (4.0 * ring + (b.count - ring)), 1e-12);
}

TEST(LtsLevels, ElementsBeyondTheLastLevelAreReported) {
  const Compiled c = column();
  auto dts = uniform_steps(c, 1.0);
  dts[0][0] = 1e-6;
  LtsOptions opt;
  opt.max_levels = 4;
  const auto L = assign_lts_levels(c, dts, 0.9, opt);
  EXPECT_EQ(L.levels, 4);
  EXPECT_EQ(L.too_fine, 1);
  EXPECT_DOUBLE_EQ(L.min_element_dt, 1e-6);
}

TEST(LtsLevels, DeviceElementStepsAgreeWithTheStableStep) {
  // The per-element steps from the device: their minimum is the stage's stable step, bit for
  // bit, and with dt at that step every element is level 0.
  const Compiled c = column();
  femorax::exec::InternalForce force(c, femorax::exec::GeometryCache::Auto);
  for (const auto scheme : {femorax::ExplicitScheme::ExplicitDifference, femorax::ExplicitScheme::CentralDifference}) {
    const auto dts = force.element_critical_dts(scheme);
    ASSERT_EQ(dts.size(), c.batches.size());
    double m = 1e300;
    for (const auto& v : dts)
      for (const double d : v) m = std::min(m, d);
    EXPECT_EQ(m, force.min_critical_dt(scheme));
    const auto L = assign_lts_levels(c, dts, 0.9 * m);
    EXPECT_EQ(L.levels, 1);
  }
}

}  // namespace
