// Node and element renumbering: valid permutations, deterministic, off = file order, a
// scrambled grid gets back a small bandwidth, components and equalDOF ties handled.

#include "compile/dof_numbering.hpp"
#include "compile/renumber.hpp"
#include "../reference/drivers/driver_common.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace {

using femorax::compile::compute_ordering;
using femorax::compile::node_bandwidth;
using femorax::compile::Renumbering;
using femorax::io::ElementGroup;
using femorax::io::ElementType;
using femorax::io::Model;

// nx x ny x nz bricks; node tags permuted by `scramble` so file order is bad.
Model grid(int nx, int ny, int nz, bool scramble) {
  Model m;
  const int n = (nx + 1) * (ny + 1) * (nz + 1);
  std::vector<int> tag_of(n);
  std::iota(tag_of.begin(), tag_of.end(), 1);
  if (scramble) {
    refdrv::Lcg rng(9);
    for (int i = n - 1; i > 0; --i)
      std::swap(tag_of[i], tag_of[static_cast<int>((rng.symmetric() + 1) * 0.5 * (i + 1)) % (i + 1)]);
  }
  auto id = [&](int i, int j, int k) { return (k * (ny + 1) + j) * (nx + 1) + i; };
  // File order = ascending tag.
  std::vector<int> by_tag(n);
  for (int v = 0; v < n; ++v) by_tag[tag_of[v] - 1] = v;
  for (int t = 0; t < n; ++t) {
    const int v = by_tag[t];
    m.nodes.tags.push_back(t + 1);
    m.nodes.ndf.push_back(3);
    const int i = v % (nx + 1), j = (v / (nx + 1)) % (ny + 1), k = v / ((nx + 1) * (ny + 1));
    m.nodes.coords.insert(m.nodes.coords.end(), {double(i), double(j), double(k)});
  }
  ElementGroup g;
  g.id = "000000";
  g.type = ElementType::StdBrick;
  int etag = 1;
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        const int c[8] = {id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k),
                          id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1), id(i, j + 1, k + 1)};
        g.tags.push_back(etag++);
        for (int a : c) g.connectivity.push_back(tag_of[a]);
      }
  m.elements.push_back(g);
  return m;
}

bool is_permutation(const std::vector<std::int32_t>& p) {
  std::vector<std::int32_t> s(p);
  std::sort(s.begin(), s.end());
  for (std::size_t i = 0; i < s.size(); ++i)
    if (s[i] != static_cast<std::int32_t>(i)) return false;
  return true;
}

TEST(Renumber, OffKeepsFileOrder) {
  const Model m = grid(3, 3, 4, true);
  const auto o = compute_ordering(m, Renumbering::Off);
  for (std::size_t i = 0; i < o.node_order.size(); ++i) EXPECT_EQ(o.node_order[i], static_cast<int>(i));
  for (std::size_t i = 0; i < o.element_order.size(); ++i) EXPECT_EQ(o.element_order[i], static_cast<int>(i));
}

TEST(Renumber, RcmIsAValidDeterministicPermutation) {
  const Model m = grid(4, 3, 6, true);
  const auto a = compute_ordering(m, Renumbering::RCM);
  const auto b = compute_ordering(m, Renumbering::RCM);
  EXPECT_TRUE(is_permutation(a.node_order));
  EXPECT_TRUE(is_permutation(a.element_order));
  EXPECT_EQ(a.node_order, b.node_order);
  EXPECT_EQ(a.element_order, b.element_order);
  for (std::size_t i = 0; i < a.node_order.size(); ++i) EXPECT_EQ(a.node_rank[a.node_order[i]], static_cast<int>(i));
}

TEST(Renumber, RcmRecoversASmallBandwidthFromAScrambledGrid) {
  const Model scrambled = grid(6, 5, 20, true);
  const auto off = compute_ordering(scrambled, Renumbering::Off);
  const auto rcm = compute_ordering(scrambled, Renumbering::RCM);
  const int bw_off = node_bandwidth(scrambled, off.node_rank);
  const int bw_rcm = node_bandwidth(scrambled, rcm.node_rank);
  // RCM orders by BFS levels; on a brick grid (26-neighbour node graph) a level is a whole
  // cross-section of (nx+1)(ny+1) = 42 nodes and neighbours span two levels, so the
  // bandwidth is about 2 to 2.5 cross-sections. (A lexicographic order of a regular grid
  // reaches 50, but real meshes do not come in that order.)
  const int section = 7 * 6;
  EXPECT_GT(bw_off, 500);
  EXPECT_LE(bw_rcm, 5 * section / 2);
  EXPECT_LT(bw_rcm * 5, bw_off);
}

TEST(Renumber, ElementsFollowTheirNodes) {
  const Model m = grid(5, 4, 8, true);
  const auto o = compute_ordering(m, Renumbering::RCM);
  // Smallest node rank of consecutive elements never decreases.
  std::unordered_map<int, int> idx;
  for (std::size_t i = 0; i < m.nodes.size(); ++i) idx[m.nodes.tags[i]] = static_cast<int>(i);
  int prev = -1;
  for (const auto e : o.element_order) {
    int k = INT32_MAX;
    for (int a = 0; a < 8; ++a) k = std::min(k, o.node_rank[idx[m.elements[0].connectivity[e * 8 + a]]]);
    EXPECT_GE(k, prev);
    prev = k;
  }
}

TEST(Renumber, HandlesComponentsAndTies) {
  // Two separate grids in one model, plus a tie from the first to the second.
  Model m = grid(2, 2, 2, false);
  Model second = grid(2, 2, 2, false);
  const int offset = static_cast<int>(m.nodes.size());
  for (std::size_t i = 0; i < second.nodes.size(); ++i) {
    m.nodes.tags.push_back(second.nodes.tags[i] + offset);
    m.nodes.ndf.push_back(3);
    m.nodes.coords.insert(m.nodes.coords.end(), {0.0, 0.0, 10.0});
  }
  ElementGroup g = second.elements[0];
  g.id = "000001";
  for (auto& c : g.connectivity) c += offset;
  for (auto& t : g.tags) t += 100;
  m.elements.push_back(g);
  const auto apart = compute_ordering(m, Renumbering::RCM);
  EXPECT_TRUE(is_permutation(apart.node_order));
  EXPECT_TRUE(is_permutation(apart.element_order));
  // The renumbering gives the same DOF classes (here: none tied) and a valid numbering.
  const auto n = femorax::compile::number_dofs(m, apart.node_order);
  EXPECT_EQ(n.num_equations, 3 * static_cast<int>(m.nodes.size()));

  m.equal_dof = {{1}, {1 + offset}, {1}, {}};
  const auto tied = compute_ordering(m, Renumbering::RCM);
  EXPECT_TRUE(is_permutation(tied.node_order));
}

}  // namespace
