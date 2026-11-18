// H5DRM end to end (exec/drm_load.hpp): the Domain Reduction Method reproduces a free field
// it is given. The free field is the response of a 1 x 1 x 8 laminar column (two layers) to a
// base acceleration pulse, run with ExplicitDifference; its total motion (the recorded
// relative motion plus the ground motion, integrated by the same recurrence) is written to a
// DRM file for the nodes of a one-element DRM layer around a 6 x 6 x 8 box of the same
// layers, whose outer faces are fixed and which has no base excitation. With the same
// element equations, the same integrator and the same step, the DRM forces make every node
// inside the layer move exactly as the free field (linear superposition of the discrete
// recurrence), to rounding: checked for every node inside and on the DRM boundary, and
// against the OpenSees convention of a downward z axis in the file.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "../common/drm_writer.hpp"
#include "../common/synthetic_models.hpp"
#include "exec/explicit_analysis.hpp"
#include "io/support.hpp"

namespace {

using namespace femorax;
namespace fs = std::filesystem;
constexpr int kNz = 8, kBox = 6, kSteps = 400;
constexpr double kDz = 0.5, kDt = 2e-4;

std::string temp_file(const std::string& name) {
  const fs::path d = fs::temp_directory_path() / ("femorax_test_drm_ff_" + std::to_string(::getpid()));
  fs::create_directories(d);
  return (d / name).string();
}

std::vector<std::vector<double>> read_rows(const std::string& file) {
  std::vector<std::vector<double>> rows;
  std::ifstream in(file);
  for (std::string line; std::getline(in, line);) {
    std::istringstream s(line);
    std::vector<double> r;
    for (double x; s >> x;) r.push_back(x);
    if (!r.empty()) rows.push_back(r);
  }
  return rows;
}

// A windowed sine, zero with zero slope at t = 0 and after kPulse.
constexpr double kPulse = 0.04;
double pulse(double t) {
  if (t <= 0.0 || t >= kPulse) return 0.0;
  const double w = std::sin(M_PI * t / kPulse);
  return 4.0 * std::sin(2 * M_PI * 50.0 * t) * w * w;
}

// GroundMotions' evaluation of the Path series (PathSeries::getFactor), at the integrator's
// accumulated time.
double ag_at(const io::PathSeries& s, double t) {
  if (t < 0.0) return 0.0;
  const double incr = t / *s.dt;
  const auto i1 = static_cast<long long>(std::floor(incr));
  if (i1 + 1 >= static_cast<long long>(s.values.size())) return 0.0;
  return s.factor * (s.values[i1] + (s.values[i1 + 1] - s.values[i1]) * (incr - static_cast<double>(i1)));
}

io::AnalysisStep explicit_stage() {
  io::AnalysisStep a;
  a.name = "shake";
  a.dt = kDt;
  a.num_steps = kSteps;
  a.integrator = io::Integrator::ExplicitDifference;
  return a;
}

void add_node_recorder(io::Model& m, const std::string& name, const std::string& file, io::NodeResponse r,
                       const std::vector<std::int32_t>& nodes) {
  io::NodeRecorder nr;
  nr.file = file;
  nr.response = r;
  nr.time = true;
  nr.precision = 17;
  nr.nodes = nodes;
  nr.dofs = {1, 2, 3};
  m.recorders.push_back({name, nr, {}});
  m.process.push_back({"0" + name, "", io::RecorderStep{name}});
}

// damped: Rayleigh damping (the synthetic column's coefficients) in the free-field column
// and in the whole box, DRM layer included: the DRM forces then carry the layer's damping
// (docs/design.md, "Damped DRM layer"), with the free field's velocity from central
// differences of its displacement samples. Returns the largest difference from the free
// field over the box's inside, relative to the largest displacement.
enum class Damping { None, Rayleigh, Uniform };

// All elements of a model in one Uniform damping region (5 % over 0.1-10 Hz).
void damp_uniform(io::Model& m) {
  io::UniformDampingRegion r;
  r.id = "soil";
  r.ratio = 0.05;
  r.freq1 = 0.1;
  r.freq2 = 10.0;
  for (const auto& g : m.elements) r.elements.insert(r.elements.end(), g.tags.begin(), g.tags.end());
  m.uniform.push_back(r);
}

double free_field_box(Damping damping) {

  // 1. The free field: a 1 x 1 x 8 laminar column, base pulse in x.
  femorax::testing::ColumnSpec cs;
  cs.nx = cs.ny = 1;
  cs.nz = kNz;
  cs.dz = kDz;
  cs.element = io::ElementType::SSPbrick;
  cs.damping = damping == Damping::Rayleigh;
  io::Model col = femorax::testing::soil_column(cs);
  if (damping == Damping::Uniform) damp_uniform(col);
  io::PathSeries ps;
  ps.tag = 1;
  ps.factor = 1.0;
  ps.dt = kDt;
  for (int k = 0; k <= kSteps + 2; ++k) ps.values.push_back(pulse(k * kDt));
  col.time_series.push_back(ps);
  col.patterns.push_back({2, 1, 1, 0.0, 1.0, {}});
  std::vector<std::int32_t> line;  // the corner line, base to top: node tag(0, 0, k)
  for (int k = 0; k <= kNz; ++k) line.push_back(1 + 4 * k);
  const std::string cd = temp_file("col_disp.txt"), ca = temp_file("col_accel.txt");
  add_node_recorder(col, "d", cd, io::NodeResponse::Disp, line);
  add_node_recorder(col, "a", ca, io::NodeResponse::Accel, line);
  col.process.push_back({"1", "", io::PatternStep{2}});
  col.process.push_back({"2", "", explicit_stage()});
  exec::RunOptions o;
  o.verbose = false;
  o.renumbering = compile::Renumbering::Off;
  exec::run_model(col, o);
  const auto rd = read_rows(cd), ra = read_rows(ca);
  EXPECT_EQ(rd.size(), static_cast<std::size_t>(kSteps));
  if (rd.size() != static_cast<std::size_t>(kSteps)) return 1.0;

  // Total motion: relative + ground, the ground displacement by ExplicitDifference's own
  // recurrence (v += dt a_n, u += dt v, a at the new time) from rest.
  std::vector<double> ug(kSteps + 1, 0.0), agv(kSteps + 1, 0.0);
  {
    double v = 0.0, u = 0.0, t = 0.0, a = 0.0;
    for (int n = 0; n < kSteps; ++n) {
      v += kDt * a;
      u += kDt * v;
      t += kDt;
      a = ag_at(ps, t);
      ug[n + 1] = u;
      agv[n + 1] = a;
    }
  }
  // free[k][n][c]: the total free-field displacement (and acceleration) of level k after
  // step n + 1, component c.
  auto total = [&](const std::vector<std::vector<double>>& rows, const std::vector<double>& g, int n, int k, int c) {
    const double rel = rows[static_cast<std::size_t>(n)][static_cast<std::size_t>(1 + 3 * k + c)];
    return c == 0 ? rel + g[static_cast<std::size_t>(n + 1)] : rel;
  };

  // 2. The box: 6 x 6 x 8, the same layers, outer faces fixed, no laminar ties, no base
  // excitation. DRM layer: the outer ring of elements and the bottom layer.
  femorax::testing::ColumnSpec bs = cs;
  bs.nx = bs.ny = kBox;
  io::Model box = femorax::testing::soil_column(bs);
  if (damping == Damping::Uniform) damp_uniform(box);
  box.equal_dof = {};
  const int NX = kBox + 1;
  auto tag = [&](int i, int j, int k) { return 1 + i + NX * (j + NX * k); };
  for (int k = 1; k <= kNz; ++k)
    for (int j = 0; j <= kBox; ++j)
      for (int i = 0; i <= kBox; ++i)
        if (i == 0 || j == 0 || i == kBox || j == kBox)
          for (int d = 1; d <= 3; ++d) {
            box.fix.node.push_back(tag(i, j, k));
            box.fix.dof.push_back(d);
          }
  // Stations: the nodes of the layer's elements; boundary: on the surface of the inside
  // region (1 <= i, j <= 5, k >= 1).
  femorax::testing::DrmData d;
  std::vector<std::int32_t> inside;
  std::map<std::int32_t, int> level;
  for (int k = 0; k <= kNz; ++k)
    for (int j = 0; j <= kBox; ++j)
      for (int i = 0; i <= kBox; ++i) {
        const bool in_closure = i >= 1 && i <= kBox - 1 && j >= 1 && j <= kBox - 1 && k >= 1;
        const bool on_layer = !(i >= 2 && i <= kBox - 2 && j >= 2 && j <= kBox - 2 && k >= 2);
        level[tag(i, j, k)] = k;
        if (in_closure) inside.push_back(tag(i, j, k));
        if (!on_layer) continue;
        d.xyz.insert(d.xyz.end(), {static_cast<double>(i), static_cast<double>(j), -kNz * kDz + k * kDz});
        d.internal.push_back(in_closure ? 1 : 0);
      }
  const auto S = d.internal.size();
  d.samples = kSteps + 1;
  d.dt = kDt;
  d.tstart = 0.0;
  d.tend = kSteps * kDt;
  d.disp.assign(3 * S * static_cast<std::size_t>(d.samples), 0.0);
  d.acc = d.disp;
  for (std::size_t st = 0; st < S; ++st) {
    const int k = static_cast<int>(std::lround((d.xyz[3 * st + 2] + kNz * kDz) / kDz));
    for (int n = 0; n < kSteps; ++n)
      for (int c = 0; c < 3; ++c) {
        const double sign = c == 2 ? -1.0 : 1.0;  // the file's z axis points down
        const std::size_t r = (3 * st + static_cast<std::size_t>(c)) * static_cast<std::size_t>(d.samples) +
                              static_cast<std::size_t>(n + 1);
        d.disp[r] = sign * total(rd, ug, n, k, c);
        d.acc[r] = sign * total(ra, agv, n, k, c);
      }
  }
  const std::string drm = temp_file("free_field.h5drm");
  femorax::testing::write_drm_file(drm, d);
  io::H5DrmPattern hp;
  hp.tag = 3;
  hp.file = hp.path = drm;
  hp.distance_tolerance = 1e-6;
  box.drm_patterns.push_back(hp);
  const std::string bd = temp_file("box_disp.txt");
  add_node_recorder(box, "d", bd, io::NodeResponse::Disp, inside);
  box.process.push_back({"1", "", io::PatternStep{3}});
  box.process.push_back({"2", "", explicit_stage()});
  io::check_supported(box, io::Capabilities{});
  exec::run_model(box, o);
  const auto rb = read_rows(bd);
  EXPECT_EQ(rb.size(), static_cast<std::size_t>(kSteps));
  if (rb.size() != static_cast<std::size_t>(kSteps)) return 1.0;

  double worst = 0.0, scale = 0.0;
  for (int n = 0; n < kSteps; ++n)
    for (std::size_t q = 0; q < inside.size(); ++q) {
      const int k = level[inside[q]];
      for (int c = 0; c < 3; ++c) {
        const double want = total(rd, ug, n, k, c);
        scale = std::max(scale, std::abs(want));
        worst = std::max(worst, std::abs(rb[static_cast<std::size_t>(n)][1 + 3 * q + static_cast<std::size_t>(c)] - want));
      }
    }
  std::printf("[ DRM box%s: %zu stations, %zu nodes inside the layer, %d steps: largest difference from the free "
              "field %.2e of its largest displacement %.3e ]\n",
              damping == Damping::Rayleigh ? " (Rayleigh-damped)" : damping == Damping::Uniform ? " (Uniform-damped)" : "",
              S, inside.size(), kSteps, worst / scale, scale);
  EXPECT_GT(scale, 1e-6);

  // Streaming (drm_load.hpp): history windows of 2 samples (a new window every step) and of 7
  // (boundaries at odd steps) give the same bits as the whole history on the device.
  for (const std::int64_t w : {2, 7}) {
    SCOPED_TRACE("window " + std::to_string(w));
    exec::RunOptions ow = o;
    ow.drm_window_samples = w;
    exec::run_model(box, ow);
    const auto rw = read_rows(bd);
    EXPECT_EQ(rw.size(), rb.size());
    if (rw.size() != rb.size()) return 1.0;
    for (std::size_t n = 0; n < rb.size(); ++n) EXPECT_EQ(rw[n], rb[n]) << "step " << n;
  }
  return worst / scale;
}

TEST(DrmFreeField, TheBoxInsideTheLayerMovesAsTheFreeField) { EXPECT_LE(free_field_box(Damping::None), 1e-10); }

// With the layer's damping in the DRM forces: 8.5e-4 (the velocity's central differences
// and the half-step damping of ExplicitDifference); with OpenSees' forces (mass and stiffness
// only) the same box is off by 4.1e-2 (measured 2026-11-17).
TEST(DrmFreeField, ADampedBoxInsideADampedLayerMovesAsTheFreeField) {
  EXPECT_LE(free_field_box(Damping::Rayleigh), 2e-3);
}

// Uniform damping in the column and in the whole box, DRM layer included: the DRM forces come
// from the layer elements' own force with a state of their own for each side, so the damping
// stress of the free field is the same as in the column (no velocity estimate is needed).
TEST(DrmFreeField, AUniformDampedBoxInsideAUniformDampedLayerMovesAsTheFreeField) {
  EXPECT_LE(free_field_box(Damping::Uniform), 1e-10);
}

}  // namespace
