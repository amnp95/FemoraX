// Model reader (schema v0): reads the example soil column exactly, and handles every
// fixture in tests/io/make_fixtures.py the way its manifest says.

#include "io/model_reader.hpp"
#include "io/support.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace femorax::io;

fs::path fixture_dir() {
  const char* d = std::getenv("FEMORAX_IO_FIXTURES");
  return d ? fs::path(d) : fs::path();
}

struct Case {
  std::string name, expect, text, file;
};

std::vector<Case> manifest() {
  std::vector<Case> cases;
  std::ifstream in(fixture_dir() / "manifest.tsv");
  for (std::string line; std::getline(in, line);) {
    if (line.empty()) continue;
    std::istringstream row(line);
    Case c;
    std::getline(row, c.name, '\t');
    std::getline(row, c.expect, '\t');
    std::getline(row, c.text, '\t');
    std::getline(row, c.file, '\t');
    cases.push_back(c);
  }
  return cases;
}

class ReaderFixtures : public ::testing::Test {
 protected:
  void SetUp() override {
    if (fixture_dir().empty() || !fs::exists(fixture_dir() / "manifest.tsv")) {
      FAIL() << "FEMORAX_IO_FIXTURES is not set or has no manifest.tsv; run through ctest";
    }
  }
  static std::string path(const std::string& file) { return (fixture_dir() / file).string(); }
};

// ---------------------------------------------------------------------------
// The example column, field by field (schema/v0/make_example.py)
// ---------------------------------------------------------------------------
TEST_F(ReaderFixtures, ReadsExplicitSoilColumn) {
  const Model m = read_model(path("explicit.femorax.h5"));
  EXPECT_EQ(m.version.major, 0);
  EXPECT_EQ(m.version.minor, 1);
  EXPECT_EQ(m.ndm, 3);
  EXPECT_EQ(m.model_name, "layered_elastic_soil_column_explicit");
  EXPECT_TRUE(m.warnings.empty());

  // 26 levels of 4 nodes (layers of 2, 2, 5, 12 and 4 elements).
  ASSERT_EQ(m.nodes.size(), 104u);
  EXPECT_EQ(m.nodes.tags.front(), 1);
  EXPECT_EQ(m.nodes.coords[0 * 3 + 2], -18.0);
  EXPECT_EQ(m.nodes.coords[103 * 3 + 2], 0.0);
  EXPECT_EQ(m.nodes.coords[1 * 3 + 0], 1.0);
  for (auto n : m.nodes.ndf) EXPECT_EQ(n, 3);

  ASSERT_EQ(m.materials.size(), 5u);
  const auto& mat = std::get<ElasticIsotropicData>(m.materials[0].data);
  EXPECT_DOUBLE_EQ(mat.E, 2.0 * 145.0e6 * 1.3 / 1000.0);
  EXPECT_DOUBLE_EQ(mat.nu, 0.3);
  EXPECT_DOUBLE_EQ(mat.rho, 19.9 * 1000.0 / 9.81 / 1000.0);
  EXPECT_EQ(m.materials[0].meta.femora_name.value_or(""), "dense_ottawa_lower_material");

  ASSERT_EQ(m.elements.size(), 5u);
  EXPECT_EQ(m.element_count(), 25u);
  const std::vector<std::size_t> per_layer = {2, 2, 5, 12, 4};
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(m.elements[i].size(), per_layer[i]);
    EXPECT_EQ(m.elements[i].type, ElementType::StdBrick);
    EXPECT_TRUE(m.elements[i].lumped);
    EXPECT_EQ(m.elements[i].material, static_cast<std::int32_t>(i + 1));
  }
  // First brick: bottom face nodes 1-4, top face 5-8.
  const std::vector<std::int32_t> first(m.elements[0].connectivity.begin(),
                                        m.elements[0].connectivity.begin() + 8);
  EXPECT_EQ(first, (std::vector<std::int32_t>{1, 2, 3, 4, 5, 6, 7, 8}));
  EXPECT_DOUBLE_EQ(m.elements[0].body_force[2], -9.81 * mat.rho);

  EXPECT_EQ(m.fix.node.size(), 12u);           // 4 base nodes x 3 DOFs
  EXPECT_EQ(m.equal_dof.retained.size(), 225u);  // 25 levels x 3 slaves x 3 DOFs
  EXPECT_EQ(m.equal_dof.retained[0], 5);
  EXPECT_EQ(m.equal_dof.constrained[0], 6);

  ASSERT_EQ(m.rayleigh.size(), 1u);
  EXPECT_EQ(m.rayleigh[0].elements.size(), 25u);
  EXPECT_GT(m.rayleigh[0].alphaM, 0.0);
  EXPECT_GT(m.rayleigh[0].betaK, 0.0);
  EXPECT_EQ(m.rayleigh[0].meta.femora_tag.value_or(0), 1);

  ASSERT_EQ(m.time_series.size(), 1u);
  EXPECT_FALSE(m.time_series[0].dt.has_value());
  EXPECT_EQ(m.time_series[0].time.size(), m.time_series[0].values.size());
  EXPECT_DOUBLE_EQ(m.time_series[0].factor, 9.81);

  ASSERT_EQ(m.patterns.size(), 1u);
  EXPECT_EQ(m.patterns[0].tag, 2);
  EXPECT_EQ(m.patterns[0].dof, 1);

  ASSERT_EQ(m.recorders.size(), 2u);
  ASSERT_EQ(m.process.size(), 5u);
  EXPECT_EQ(std::get<PatternStep>(m.process[0].step).tag, 2);
  EXPECT_EQ(std::get<SetTimeStep>(m.process[3].step).time, 0.0);
  const auto& a = std::get<AnalysisStep>(m.process[4].step);
  EXPECT_EQ(a.integrator, Integrator::ExplicitDifference);
  EXPECT_EQ(a.algorithm, Algorithm::Linear);
  EXPECT_DOUBLE_EQ(a.dt, 1e-4);
  EXPECT_DOUBLE_EQ(a.final_time.value(), 10.0);
  EXPECT_FALSE(a.num_steps.has_value());
  EXPECT_EQ(m.process[4].label, "Run the site-response analysis");

  EXPECT_NO_THROW(check_supported(m, Capabilities{}));
}

TEST_F(ReaderFixtures, ImplicitModelNeedsMilestone1b) {
  const Model m = read_model(path("implicit_newmark.femorax.h5"));
  EXPECT_FALSE(m.elements[0].lumped);
  const auto& gravity = std::get<AnalysisStep>(m.process[0].step);
  EXPECT_EQ(gravity.integrator, Integrator::Newmark);
  EXPECT_DOUBLE_EQ(gravity.gamma, 0.6);
  EXPECT_DOUBLE_EQ(gravity.beta, 1.21 / 4.0);
  EXPECT_EQ(gravity.num_steps.value(), 30);
  EXPECT_EQ(gravity.test_type.value(), TestType::NormUnbalance);

  EXPECT_THROW(check_supported(m, Capabilities{}), UnsupportedComponent);
  Capabilities with_implicit;
  with_implicit.implicit = true;
  EXPECT_NO_THROW(check_supported(m, with_implicit));
}

TEST_F(ReaderFixtures, ReadsOlderMinorVersion) {
  const Model m = read_model(path("schema_0_0.femorax.h5"));
  EXPECT_EQ(m.version.minor, 0);
  EXPECT_EQ(m.nodes.size(), 104u);
}

TEST_F(ReaderFixtures, VtkhdfElementList) {
  const Model m = read_model(path("vtkhdf_elements.femorax.h5"));
  const auto& v = std::get<VtkhdfRecorder>(m.recorders[0].data);  // "site" sorts first
  EXPECT_EQ(v.elements, (std::vector<std::int32_t>{1, 2, 25}));
  EXPECT_EQ(v.responses, (std::vector<std::string>{"accel", "disp", "vel"}));
}

TEST_F(ReaderFixtures, UnsupportedIsAlsoAModelError) {
  // Callers that only distinguish "can run" from "cannot" may catch ModelError alone.
  EXPECT_THROW(read_model(path("element_type.femorax.h5")), ModelError);
}

// ---------------------------------------------------------------------------
// Every fixture in the manifest
// ---------------------------------------------------------------------------
TEST_F(ReaderFixtures, EveryFixtureBehavesAsItsManifestSays) {
  const auto cases = manifest();
  ASSERT_GE(cases.size(), 50u);
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    std::string got = "no exception";
    std::string kind = "valid";
    std::vector<std::string> warnings;
    try {
      const Model m = read_model(path(c.file));
      warnings = m.warnings;
      check_supported(m, Capabilities{});
    } catch (const UnsupportedComponent& e) {
      kind = "unsupported";
      got = e.what();
    } catch (const ModelError& e) {
      kind = "invalid";
      got = e.what();
    }
    if (c.expect == "valid" || c.expect == "warning") {
      EXPECT_EQ(kind, "valid") << got;
      if (c.expect == "valid") {
        EXPECT_TRUE(warnings.empty()) << (warnings.empty() ? "" : warnings[0]);
      } else {
        ASSERT_EQ(warnings.size(), 1u);
        EXPECT_NE(warnings[0].find(c.text), std::string::npos) << warnings[0];
      }
    } else {
      EXPECT_EQ(kind, c.expect) << got;
      EXPECT_NE(got.find(c.text), std::string::npos)
          << "message: " << got << "\nexpected to contain: " << c.text;
      if (kind == "unsupported") {
        EXPECT_NE(got.find("OpenSees backend"), std::string::npos) << got;
      }
    }
  }
}

}  // namespace
