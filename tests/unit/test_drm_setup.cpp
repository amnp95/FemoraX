// The compile step of an H5DRM pattern (compile/drm.hpp, io/drm_file.hpp): a DRM layer of
// one element around a 4 x 4 x 4 column, its stations written in a DRM file.
//   - every station matches its node; the DRM elements are the ring's, each with the right
//     boundary nodes;
//   - the same with the stations in another frame, brought back by OpenSees' transformation;
//   - a tolerance smaller than the stations' offset matches nothing (a clear error);
//   - of two stations at one node the first is taken, as OpenSees' brute-force search does;
//   - a DRM element with a history material is refused: the DRM layer must stay linear.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

#include "../common/drm_writer.hpp"
#include "../common/synthetic_models.hpp"
#include "compile/drm.hpp"

namespace {

using namespace femorax;
namespace fs = std::filesystem;
constexpr int kN = 4, kNz = 4;

std::string temp_file(const std::string& name) {
  const fs::path d = fs::temp_directory_path() / ("femorax_test_drm_" + std::to_string(::getpid()));
  fs::create_directories(d);
  return (d / name).string();
}

io::Model column() {
  femorax::testing::ColumnSpec s;
  s.nx = s.ny = kN;
  s.nz = kNz;
  return femorax::testing::soil_column(s);
}

bool in_ring(int i, int j) { return i == 0 || j == 0 || i == kN - 1 || j == kN - 1; }

// The ring's nodes as stations: boundary = on the inner box [1, N-1]^2.
femorax::testing::DrmData ring_stations(const io::Model& m) {
  femorax::testing::DrmData d;
  for (std::size_t n = 0; n < m.nodes.tags.size(); ++n) {
    const double x = m.nodes.coords[3 * n], y = m.nodes.coords[3 * n + 1], z = m.nodes.coords[3 * n + 2];
    const int i = static_cast<int>(std::lround(x)), j = static_cast<int>(std::lround(y));
    const bool ring_node = i <= 1 || j <= 1 || i >= kN - 1 || j >= kN - 1;
    if (!ring_node) continue;
    d.xyz.insert(d.xyz.end(), {x, y, z});
    const bool inner = i >= 1 && i <= kN - 1 && j >= 1 && j <= kN - 1;
    d.internal.push_back(inner ? 1 : 0);
  }
  return d;
}

compile::Compiled compiled(const io::Model& m) {
  compile::CompileOptions co;
  co.renumbering = compile::Renumbering::Off;
  return compile::compile_model(m, co);
}

io::H5DrmPattern pattern(const std::string& path, double tol = 1e-6) {
  io::H5DrmPattern p;
  p.tag = 7;
  p.file = p.path = path;
  p.distance_tolerance = tol;
  return p;
}

void expect_ring(const compile::DrmSetup& s, const compile::Compiled& c, const io::DrmFile& f) {
  EXPECT_EQ(s.stations_matched, f.stations());
  EXPECT_EQ(s.elements.size(), static_cast<std::size_t>((kN * kN - (kN - 2) * (kN - 2)) * kNz));
  const std::int32_t N = c.dofs.num_nodes();
  for (const auto& e : s.elements) {
    const auto& b = c.batches[static_cast<std::size_t>(e.batch)];
    const auto P = static_cast<std::size_t>(b.padded);
    double cx = 0, cy = 0;
    for (int a = 0; a < 8; ++a) {
      const auto n = static_cast<std::size_t>(b.node[static_cast<std::size_t>(a) * P + static_cast<std::size_t>(e.element)]);
      const double x = c.coords[n], y = c.coords[static_cast<std::size_t>(N) + n];
      cx += x / 8;
      cy += y / 8;
      const bool inner = x >= 0.5 && x <= kN - 0.5 && y >= 0.5 && y <= kN - 0.5;
      EXPECT_EQ(((e.boundary_mask >> a) & 1u) != 0, inner) << "element " << b.tag[static_cast<std::size_t>(e.element)];
    }
    EXPECT_TRUE(in_ring(static_cast<int>(std::floor(cx)), static_cast<int>(std::floor(cy))));
  }
}

TEST(DrmSetup, MatchesTheStationsAndFindsTheLayer) {
  const io::Model m = column();
  const auto c = compiled(m);
  const std::string path = temp_file("ring.h5drm");
  femorax::testing::write_drm_file(path, ring_stations(m));
  const io::DrmFile f = io::read_drm_header(path);
  const auto s = compile::setup_drm(pattern(path), f, c);
  EXPECT_EQ(s.max_match_distance, 0.0);
  expect_ring(s, c, f);
}

TEST(DrmSetup, TheTransformationBringsTheStationsBack) {
  // Stations in a frame rotated 90 degrees about z, scaled by 2 and moved: OpenSees places
  // each at T (crd_scale (x - drmbox_x0)) + x0.
  const io::Model m = column();
  const auto c = compiled(m);
  auto d = ring_stations(m);
  d.x0[0] = 10.0;
  d.x0[1] = -5.0;
  d.x0[2] = 3.0;
  for (std::size_t s = 0; s < d.internal.size(); ++s) {
    const double x = d.xyz[3 * s], yy = d.xyz[3 * s + 1], z = d.xyz[3 * s + 2];
    // model = R (2 (file - x0f)) + x0m with R = [[0,-1,0],[1,0,0],[0,0,1]], x0m = (1, 2, 0):
    // file = x0f + R^T (model - x0m) / 2.
    const double mx = x - 1.0, my = yy - 2.0, mz = z;
    d.xyz[3 * s] = d.x0[0] + my / 2.0;
    d.xyz[3 * s + 1] = d.x0[1] - mx / 2.0;
    d.xyz[3 * s + 2] = d.x0[2] + mz / 2.0;
  }
  const std::string path = temp_file("ring_frame.h5drm");
  femorax::testing::write_drm_file(path, d);
  const io::DrmFile f = io::read_drm_header(path);
  auto p = pattern(path, 1e-9);
  p.do_coordinate_transformation = true;
  p.crd_scale = 2.0;
  const double R[3][3] = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) p.T[i][j] = R[i][j];
  p.x0[0] = 1.0;
  p.x0[1] = 2.0;
  const auto s = compile::setup_drm(p, f, c);
  EXPECT_LT(s.max_match_distance, 1e-12);
  expect_ring(s, c, f);
}

TEST(DrmSetup, StationsOutsideTheToleranceMatchNothing) {
  const io::Model m = column();
  const auto c = compiled(m);
  auto d = ring_stations(m);
  for (std::size_t s = 0; s < d.internal.size(); ++s) d.xyz[3 * s] += 2e-3;
  const std::string path = temp_file("ring_off.h5drm");
  femorax::testing::write_drm_file(path, d);
  const io::DrmFile f = io::read_drm_header(path);
  try {
    compile::setup_drm(pattern(path, 1e-3), f, c);
    FAIL() << "expected an error";
  } catch (const io::ModelError& e) {
    EXPECT_NE(std::string(e.what()).find("0 nodes matched 0 of"), std::string::npos) << e.what();
  }
  EXPECT_NO_THROW(compile::setup_drm(pattern(path, 3e-3), f, c));
}

TEST(DrmSetup, TheFirstOfTwoEqualStationsIsTaken) {
  const io::Model m = column();
  const auto c = compiled(m);
  auto d = ring_stations(m);
  // A second copy of station 0 at the end, flagged the other way.
  d.xyz.insert(d.xyz.end(), {d.xyz[0], d.xyz[1], d.xyz[2]});
  d.internal.push_back(1 - d.internal[0]);
  const std::string path = temp_file("ring_twice.h5drm");
  femorax::testing::write_drm_file(path, d);
  const auto s = compile::setup_drm(pattern(path), io::read_drm_header(path), c);
  for (std::size_t k = 0; k < s.node.size(); ++k) EXPECT_NE(s.station[k], static_cast<std::int32_t>(d.internal.size() - 1));
}

TEST(DrmSetup, UniformDampingInTheLayerIsUnsupported) {
  // Damped elastic soil stays linear, but the DRM forces do not include the damping yet:
  // an unsupported component with a message that names it, not an invalid model.
  io::Model m = column();
  io::UniformDampingRegion r;
  r.id = "soil";
  r.ratio = 0.05;
  r.freq1 = 0.1;
  r.freq2 = 10.0;
  for (const auto& g : m.elements) r.elements.insert(r.elements.end(), g.tags.begin(), g.tags.end());
  m.uniform.push_back(r);
  const auto c = compiled(m);
  const std::string path = temp_file("ring_damped.h5drm");
  femorax::testing::write_drm_file(path, ring_stations(m));
  try {
    compile::setup_drm(pattern(path), io::read_drm_header(path), c);
    FAIL() << "expected the refusal";
  } catch (const io::UnsupportedComponent& e) {
    EXPECT_NE(std::string(e.what()).find("has Uniform damping"), std::string::npos) << e.what();
  }
}

TEST(DrmSetup, AHistoryMaterialInTheLayerIsRefused) {
  io::Model m = column();
  io::PressureDependMultiYieldData soil;
  soil.rho = 2.0;
  soil.refShearModul = 1e5;
  soil.refBulkModul = 2e5;
  soil.frictionAng = 30;
  soil.peakShearStra = 0.1;
  soil.refPress = 80;
  soil.pressDependCoe = 0.5;
  soil.PTAng = 25;
  m.materials[1].data = soil;  // the upper layer
  const auto c = compiled(m);
  const std::string path = temp_file("ring_soil.h5drm");
  femorax::testing::write_drm_file(path, ring_stations(m));
  try {
    compile::setup_drm(pattern(path), io::read_drm_header(path), c);
    FAIL() << "expected the refusal";
  } catch (const io::ModelError& e) {
    EXPECT_NE(std::string(e.what()).find("valid only if the DRM layer stays linear"), std::string::npos) << e.what();
  }
}

}  // namespace
