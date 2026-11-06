// DOF numbering against OpenSees' TransformationConstraintHandler (driver
// tests/reference/drivers/dof_numbering.cpp, cases from make_constraint_cases.py), plus
// general-ndf unit cases.

#include "compile/dof_numbering.hpp"
#include "io/h5.hpp"
#include "reference_data.hpp"
#include "../reference/drivers/driver_common.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>

namespace {

using femorax::compile::DofNumbering;
using femorax::compile::number_dofs;
using femorax::io::Model;
using femorax::io::ModelError;
using namespace femorax::test;

struct Case {
  Model model;
  std::string expect;  // valid | invalid | schema_gap
  std::string text;    // for invalid: part of the message
};

Case read_case(const std::filesystem::path& p) {
  Case c;
  std::ifstream in(p);
  for (std::string line; std::getline(in, line);) {
    std::istringstream w(line);
    std::string kind;
    w >> kind;
    if (kind == "node") {
      int tag, ndf;
      double x, y, z;
      w >> tag >> ndf >> x >> y >> z;
      c.model.nodes.tags.push_back(tag);
      c.model.nodes.ndf.push_back(ndf);
      c.model.nodes.coords.insert(c.model.nodes.coords.end(), {x, y, z});
    } else if (kind == "fix") {
      int tag, dof;
      w >> tag >> dof;
      c.model.fix.node.push_back(tag);
      c.model.fix.dof.push_back(dof);
    } else if (kind == "equal") {
      int r, k;
      w >> r >> k;
      for (int d; w >> d;) {
        c.model.equal_dof.retained.push_back(r);
        c.model.equal_dof.constrained.push_back(k);
        c.model.equal_dof.dof.push_back(d);
      }
    } else if (kind == "expect") {
      w >> c.expect;
      std::getline(w, c.text);
      c.text.erase(0, c.text.find_first_not_of(' '));
    }
  }
  return c;
}

// OpenSees equation per (tag, dof), -1 fixed.
std::map<std::pair<int, int>, int> opensees_equations(const ReferenceFile& ref) {
  std::map<std::pair<int, int>, int> eq;
  for (const auto& row : ref.rows) {
    const int tag = static_cast<int>(row[0]), ndf = static_cast<int>(row[1]);
    for (int d = 0; d < ndf; ++d) eq[{tag, d}] = static_cast<int>(row[2 + d]);
  }
  return eq;
}

// Number of (tag, dof) where the two numberings disagree: fixed vs free, or a pairing of
// equations that is not one-to-one.
int disagreements(const DofNumbering& n, const std::map<std::pair<int, int>, int>& os) {
  std::map<int, int> a, b;
  int bad = 0;
  for (int i = 0; i < n.num_nodes(); ++i)
    for (int d = 0; d < n.ndf[i]; ++d) {
      const int e = n.equation(i, d), o = os.at({n.node_tag[i], d});
      if ((e < 0) != (o < 0)) {
        ++bad;
        continue;
      }
      if (e < 0) continue;
      const auto [x, fx] = a.emplace(e, o);
      const auto [y, fy] = b.emplace(o, e);
      if (x->second != o || y->second != e) ++bad;
    }
  return bad;
}

// The emulation reproduces OpenSees exactly, also where OpenSees is wrong.
TEST(DofNumberingReference, EmulationReproducesOpenSeesInEveryCase) {
  const auto dir = reference_data_dir() / "constraints";
  int checked = 0;
  for (const auto& e : std::filesystem::directory_iterator(dir / "cases")) {
    const std::string name = e.path().filename().string();
    if (name == "two_retained_nodes.txt" || name == "split_equal_dof_commands.txt") continue;
    SCOPED_TRACE(name);
    const Case c = read_case(e.path());
    const auto os = opensees_equations(read_reference(dir / name));
    const DofNumbering n = number_dofs(c.model, femorax::compile::file_order(c.model), false);
    const auto v = femorax::compile::emulate_transformation_handler(c.model, n);
    std::map<std::int64_t, int> a;
    std::map<int, std::int64_t> b;
    for (int i = 0; i < n.num_nodes(); ++i)
      for (int d = 0; d < n.ndf[i]; ++d) {
        const std::int64_t x = v[n.dof_offset[i] + d];
        const int o = os.at({n.node_tag[i], d});
        ASSERT_EQ(x < 0, o < 0) << "node " << n.node_tag[i] << " dof " << d + 1;
        if (o < 0) continue;
        const auto [p, fp] = a.emplace(x, o);
        const auto [q, fq] = b.emplace(o, x);
        ASSERT_TRUE(p->second == o && q->second == x) << "node " << n.node_tag[i] << " dof " << d + 1;
      }
    ++checked;
  }
  EXPECT_EQ(checked, 9);
}

TEST(DofNumberingReference, MatchesOpenSeesTransformationHandler) {
  const auto dir = reference_data_dir() / "constraints";
  std::vector<std::filesystem::path> cases;
  for (const auto& e : std::filesystem::directory_iterator(dir / "cases")) cases.push_back(e.path());
  std::sort(cases.begin(), cases.end());
  ASSERT_EQ(cases.size(), 11u) << "constraint cases missing; see tests/reference/README.md";

  int valid = 0, invalid = 0;
  for (const auto& path : cases) {
    const std::string name = path.filename().string();
    SCOPED_TRACE(name);
    const Case c = read_case(path);
    const ReferenceFile ref = read_reference(dir / name);
    ASSERT_EQ(ref.header.at("opensees_commit"), opensees_pinned_commit);
    const auto os = opensees_equations(ref);

    if (c.expect == "valid") {
      ++valid;
      const DofNumbering n = number_dofs(c.model, femorax::compile::file_order(c.model));
      EXPECT_EQ(disagreements(n, os), 0);
      // Same equation count as OpenSees.
      int os_neq = 0;
      for (const auto& [k, v] : os) os_neq = std::max(os_neq, v + 1);
      EXPECT_EQ(n.num_equations, os_neq);
      // Renumbering changes equation numbers, never the classes.
      std::vector<std::int32_t> order(c.model.nodes.size());
      std::iota(order.begin(), order.end(), 0);
      refdrv::Lcg rng(3);
      for (std::size_t i = order.size(); i > 1; --i)
        std::swap(order[i - 1], order[static_cast<std::size_t>((rng.symmetric() + 1) * 0.5 * i) % i]);
      EXPECT_EQ(disagreements(number_dofs(c.model, order), os), 0) << "renumbered";
    } else if (c.expect == "invalid") {
      ++invalid;
      try {
        number_dofs(c.model, femorax::compile::file_order(c.model));
        ADD_FAILURE() << "expected ModelError containing: " << c.text;
      } catch (const ModelError& e) {
        EXPECT_NE(std::string(e.what()).find(c.text), std::string::npos) << e.what();
      }
    } else {
      ASSERT_EQ(c.expect, "schema_gap");
      // Separate one-DOF equalDOF commands: OpenSees ties DOF 1 only (first MP per node);
      // schema v0 rows cannot show the command boundaries, so FemoraX ties all three.
      const DofNumbering n = number_dofs(c.model, femorax::compile::file_order(c.model));
      int tied_by_opensees = 0;
      for (std::size_t r = 0; r < c.model.equal_dof.retained.size(); ++r) {
        const int rt = c.model.equal_dof.retained[r], ct = c.model.equal_dof.constrained[r];
        const int d = c.model.equal_dof.dof[r] - 1;
        tied_by_opensees += os.at({rt, d}) == os.at({ct, d});
      }
      EXPECT_EQ(tied_by_opensees * 3, static_cast<int>(c.model.equal_dof.retained.size()));
      EXPECT_GT(disagreements(n, os), 0) << "the known schema gap no longer shows";
    }
  }
  EXPECT_EQ(valid, 6);
  EXPECT_EQ(invalid, 4);
}

// ---------------------------------------------------------------------------
Model nodes_model(std::vector<int> tags, std::vector<int> ndf) {
  Model m;
  m.nodes.tags = std::move(tags);
  m.nodes.ndf = std::move(ndf);
  m.nodes.coords.assign(m.nodes.tags.size() * 3, 0.0);
  return m;
}

TEST(DofNumbering, AnyNumberOfDofsPerNode) {
  // A 6-DOF beam node, a 3-DOF solid node, a 4-DOF u-p node.
  Model m = nodes_model({10, 20, 30}, {6, 3, 4});
  m.fix.node = {10, 30};
  m.fix.dof = {6, 4};
  const DofNumbering n = number_dofs(m, femorax::compile::file_order(m));
  EXPECT_EQ(n.dof_offset, (std::vector<std::int32_t>{0, 6, 9, 13}));
  EXPECT_EQ(n.num_equations, 11);
  EXPECT_EQ(n.equation(0, 5), -1);
  EXPECT_EQ(n.equation(2, 3), -1);
  EXPECT_EQ(n.equation(1, 0), 5);  // equations follow DOF order, skipping fixed DOFs
}

TEST(DofNumbering, TiedGroupsShareAnEquationAndFixedGroupsAreFixed) {
  Model m = nodes_model({1, 2, 3}, {6, 6, 3});
  m.equal_dof = {{1, 1, 1}, {2, 2, 3}, {4, 5, 1}};  // 2.4 = 1.4, 2.5 = 1.5, 3.1 = 1.1
  m.fix.node = {1};
  m.fix.dof = {5};
  const DofNumbering n = number_dofs(m, femorax::compile::file_order(m));
  EXPECT_EQ(n.equation(1, 3), n.equation(0, 3));
  EXPECT_EQ(n.equation(2, 0), n.equation(0, 0));
  EXPECT_EQ(n.equation(0, 4), -1);
  EXPECT_EQ(n.equation(1, 4), -1);  // tied to a fixed DOF
  EXPECT_EQ(n.num_equations, 6 + 6 + 3 - 3 - 1);
}

TEST(DofNumbering, RejectsTwoRetainedNodes) {
  Model m = nodes_model({1, 2, 3}, {3, 3, 3});
  m.equal_dof = {{1, 2}, {3, 3}, {1, 2}};
  try {
    number_dofs(m, femorax::compile::file_order(m));
    FAIL();
  } catch (const ModelError& e) {
    EXPECT_NE(std::string(e.what()).find("would apply only the first"), std::string::npos);
  }
}

TEST(DofNumbering, RejectsBadOrder) {
  Model m = nodes_model({1, 2}, {3, 3});
  EXPECT_THROW(number_dofs(m, {0, 0}), ModelError);
  EXPECT_THROW(number_dofs(m, {0}), ModelError);
}

}  // namespace
