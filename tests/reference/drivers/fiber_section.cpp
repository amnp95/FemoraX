// Reference driver for OpenSees' FiberSection3d with Elastic, Steel01 and Concrete02 fibers and
// elastic torsion (ElasticMaterial(0, GJ), as `section Fiber tag -GJ GJ { ... }` builds it).
//
//   fiber_section <out_dir>
//
// Sections: "rc" (a 0.4 x 0.6 reinforced-concrete rectangle: 6 x 8 confined core fibers, 28
// cover fibers, 8 bars; off the origin, so the centroid matters), "steel_i" (an I shape of
// Steel01 fibers), "mixed_nocentroid" (elastic, steel and concrete fibers with -noCentroid).
// Paths of section deformations e = (eps0, kappa_z, kappa_y, twist): bending about z under
// compression, bending about y, biaxial growing cycles with varying axial strain, a random walk.
// Every step: setTrialSectionDeformation, getStressResultant, getSectionTangent, commitState;
// every 7th step first a thrown-away trial with revertToLastCommit. The first row (step -1) is
// getInitialTangent (e and s zero).
// Output: <out_dir>/FiberSection3d_<section>_<path>.txt (format: driver_common.hpp); the header
// holds the materials (type and the 7 packed parameters of FemoraX's adapters: 0 Elastic, 1
// Steel01, 2 Concrete02), the fibers (y z A material-index, in the section's order), GJ and
// whether the centroid is computed.

#include "driver_common.hpp"

#include <Concrete02.h>
#include <ElasticMaterial.h>
#include <FiberSection3d.h>
#include <Matrix.h>
#include <Steel01.h>
#include <UniaxialFiber3d.h>
#include <UniaxialMaterial.h>
#include <Vector.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

const double pi = 3.14159265358979323846;

struct Mat {
  int type;  // 0 Elastic, 1 Steel01, 2 Concrete02
  std::array<double, 7> v;
  UniaxialMaterial* make() const {
    if (type == 0) return new ElasticMaterial(1, v[0], 0.0, v[1]);
    if (type == 1) return new Steel01(1, v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    return new Concrete02(1, v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
  }
};
struct Fib {
  double y, z, A;
  int mat;
};
struct Sec {
  std::string name;
  std::vector<Mat> mats;
  std::vector<Fib> fibers;
  double GJ;
  bool centroid;
  double eps_scale, kz_scale, ky_scale;  // deformation scales of the paths
};

void grid(std::vector<Fib>& f, double y0, double y1, double z0, double z1, int ny, int nz, int mat) {
  const double dy = (y1 - y0) / ny, dz = (z1 - z0) / nz;
  for (int i = 0; i < ny; ++i)
    for (int j = 0; j < nz; ++j) f.push_back({y0 + (i + 0.5) * dy, z0 + (j + 0.5) * dz, dy * dz, mat});
}

std::vector<Sec> sections() {
  std::vector<Sec> out;
  {  // reinforced concrete, 0.6 (y) x 0.4 (z), its corner at (0.1, -0.05): the centroid is not the origin
    Sec s{"rc", {}, {}, 2.4e5, true, 1e-3, 0.012, 0.018};
    s.mats.push_back({2, {-35000.0, -0.004, -28000.0, -0.014, 0.25, 2800.0, 1.4e6}});          // core
    s.mats.push_back({2, {-28000.0, -0.002, -5600.0, -0.006, 0.1, 2800.0, 1.4e6}});            // cover
    s.mats.push_back({1, {4.2e5, 2.0e8, 0.01, 0.0, 55.0, 0.0, 55.0}});                         // bars
    const double y0 = 0.1, z0 = -0.05, h = 0.6, b = 0.4, c = 0.04;
    grid(s.fibers, y0 + c, y0 + h - c, z0 + c, z0 + b - c, 8, 6, 0);
    grid(s.fibers, y0, y0 + c, z0, z0 + b, 1, 6, 1);
    grid(s.fibers, y0 + h - c, y0 + h, z0, z0 + b, 1, 6, 1);
    grid(s.fibers, y0 + c, y0 + h - c, z0, z0 + c, 8, 1, 1);
    grid(s.fibers, y0 + c, y0 + h - c, z0 + b - c, z0 + b, 8, 1, 1);
    const double As = 5.1e-4;
    for (const double y : {y0 + c, y0 + 0.5 * h, y0 + h - c})
      for (const double z : {z0 + c, z0 + 0.5 * b, z0 + b - c})
        if (!(y == y0 + 0.5 * h && z == z0 + 0.5 * b)) s.fibers.push_back({y, z, As, 2});
    out.push_back(s);
  }
  {  // steel I shape: flanges 0.3 x 0.02, web 0.36 x 0.012
    Sec s{"steel_i", {}, {}, 8.0e2, true, 2e-3, 0.03, 0.06};
    s.mats.push_back({1, {3.45e5, 2.0e8, 0.02, 0.05, 30.0, 0.08, 20.0}});
    grid(s.fibers, 0.18, 0.2, -0.15, 0.15, 2, 12, 0);
    grid(s.fibers, -0.2, -0.18, -0.15, 0.15, 2, 12, 0);
    grid(s.fibers, -0.18, 0.18, -0.006, 0.006, 16, 1, 0);
    out.push_back(s);
  }
  {  // three materials, the centroid not computed (-noCentroid)
    Sec s{"mixed_nocentroid", {}, {}, 5.0e3, false, 1e-3, 0.02, 0.02};
    s.mats.push_back({0, {3.0e7, 2.0e7, 0, 0, 0, 0, 0}});
    s.mats.push_back({1, {2.5e5, 2.1e8, 0.0, 0.0, 55.0, 0.0, 55.0}});
    s.mats.push_back({2, {-30000.0, -0.002, 0.0, -0.008, 0.1, 0.0, 1.0e6}});
    grid(s.fibers, -0.1, 0.3, -0.2, 0.1, 5, 4, 2);
    grid(s.fibers, 0.3, 0.35, -0.2, 0.1, 1, 3, 0);
    for (int k = 0; k < 7; ++k)
      s.fibers.push_back({0.1 + 0.18 * std::cos(2 * pi * k / 7.0), -0.05 + 0.12 * std::sin(2 * pi * k / 7.0), 3.0e-4, 1});
    out.push_back(s);
  }
  return out;
}

using Path = std::vector<std::array<double, 4>>;  // in units of (eps, kz, ky, twist) scales

Path bending_z() {  // axial compression held, curvature about z to 1 in 60 steps and back
  Path p;
  for (int k = 1; k <= 10; ++k) p.push_back({-0.3 * k / 10.0, 0, 0, 0});
  for (int k = 1; k <= 60; ++k) p.push_back({-0.3, k / 60.0, 0, 0.1 * k / 60.0});
  for (int k = 59; k >= -20; --k) p.push_back({-0.3, k / 60.0, 0, 0.1});
  return p;
}
Path bending_y() {
  Path p;
  for (int k = 1; k <= 60; ++k) p.push_back({0.05 * k / 60.0, 0, k / 60.0, 0});
  for (int k = 59; k >= -60; --k) p.push_back({0.05, 0, k / 60.0, -0.2});
  return p;
}
Path biaxial_cycles() {
  Path p;
  for (const double a : {0.2, 0.5, 1.0})
    for (int k = 1; k <= 48; ++k) {
      const double t = 2 * pi * k / 48.0;
      p.push_back({-0.2 * a * (1.0 - std::cos(t)), a * std::sin(t), 0.7 * a * std::sin(t + 0.9), 0.3 * a * std::cos(t)});
    }
  return p;
}
Path random_walk() {
  Path p;
  refdrv::Lcg rng(2026112102);
  std::array<double, 4> e{}, d{};
  for (int k = 0; k < 240; ++k) {
    if (k % 10 == 0)
      for (auto& x : d) x = 0.06 * rng.symmetric();
    if (k % 20 == 10)
      for (auto& x : d) x = -1.5 * x;
    for (int i = 0; i < 4; ++i) e[i] += d[i] * (1.0 + 0.2 * rng.symmetric());
    p.push_back(e);
  }
  return p;
}

const std::vector<std::pair<std::string, std::function<Path()>>> paths = {
    {"bending_z", bending_z}, {"bending_y", bending_y}, {"biaxial_cycles", biaxial_cycles}, {"random_walk", random_walk}};

std::string line(const std::string& key, const std::vector<double>& v) {
  std::string s = key;
  char buf[40];
  for (const double x : v) {
    std::snprintf(buf, sizeof(buf), " %.17g", x);
    s += buf;
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: fiber_section <out_dir>\n");
    return 1;
  }
  const std::string out = argv[1];
  int files = 0;
  for (const Sec& sec : sections()) {
    std::vector<double> mv, fv;
    for (const Mat& m : sec.mats) {
      mv.push_back(m.type);
      mv.insert(mv.end(), m.v.begin(), m.v.end());
    }
    for (const Fib& f : sec.fibers) fv.insert(fv.end(), {f.y, f.z, f.A, static_cast<double>(f.mat)});
    for (const auto& [pname, make] : paths) {
      ElasticMaterial torsion(0, sec.GJ);
      FiberSection3d section(1, static_cast<int>(sec.fibers.size()), torsion, sec.centroid);
      for (std::size_t i = 0; i < sec.fibers.size(); ++i) {
        const Fib& f = sec.fibers[i];
        std::unique_ptr<UniaxialMaterial> mat(sec.mats[static_cast<std::size_t>(f.mat)].make());
        Vector pos(2);
        pos(0) = f.y;
        pos(1) = f.z;
        UniaxialFiber3d fiber(static_cast<int>(i) + 1, *mat, f.A, pos);
        section.addFiber(fiber);
      }
      std::vector<std::string> cols = {"step"};
      for (int i = 0; i < 4; ++i) cols.push_back("e" + std::to_string(i));
      for (int i = 0; i < 4; ++i) cols.push_back("s" + std::to_string(i));
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) cols.push_back("ks" + std::to_string(i) + std::to_string(j));
      refdrv::Writer w(out + "/FiberSection3d_" + sec.name + "_" + pname + ".txt",
                       {"section FiberSection3d", "case " + sec.name, "path " + pname, line("materials", mv),
                        line("fibers", fv), line("GJ", {sec.GJ}), line("centroid", {sec.centroid ? 1.0 : 0.0})},
                       cols);
      auto record = [&](double step, const Vector& e, const Vector& s, const Matrix& ks) {
        std::vector<double> row = {step};
        for (int i = 0; i < 4; ++i) row.push_back(e(i));
        for (int i = 0; i < 4; ++i) row.push_back(s(i));
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j) row.push_back(ks(i, j));
        w.row(row);
      };
      Vector zero(4);
      record(-1.0, zero, zero, section.getInitialTangent());
      const Path path = make();
      const double scale[4] = {sec.eps_scale, sec.kz_scale, sec.ky_scale, 0.01};
      Vector e(4);
      for (std::size_t k = 0; k < path.size(); ++k) {
        if (k % 7 == 6) {  // a trial that is thrown away
          for (int i = 0; i < 4; ++i) e(i) = -1.7 * scale[i] * path[k][static_cast<std::size_t>(i)] + 0.3 * scale[i];
          section.setTrialSectionDeformation(e);
          section.getStressResultant();
          section.revertToLastCommit();
        }
        for (int i = 0; i < 4; ++i) e(i) = scale[i] * path[k][static_cast<std::size_t>(i)];
        if (section.setTrialSectionDeformation(e) != 0) {
          std::fprintf(stderr, "setTrialSectionDeformation failed at step %zu\n", k);
          return 1;
        }
        record(static_cast<double>(k), e, section.getStressResultant(), section.getSectionTangent());
        section.commitState();
      }
      ++files;
    }
  }
  std::printf("wrote %d files to %s\n", files, out.c_str());
  return 0;
}
