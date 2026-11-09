// Reference driver for the multi-yield soil materials (3D): runs the OpenSees class along
// strain paths at the Gauss-point level (no domain) and records strain, stress and tangent
// at every step.
//
//   soil_material <out_dir>
//
// writes <out_dir>/PressureDependMultiYield_<case>_<path>.txt and
// <out_dir>/PressureIndependMultiYield_<case>_<path>.txt (format: driver_common.hpp).
//
// Each path starts in the elastic stage (stage 0) with an oedometric consolidation to a
// vertical stress, switches to the plastic stage (updateParameter "updateMaterialStage",
// what OpenSees' updateMaterialStage command calls), then follows its strain history. Every
// step calls setTrialStrain, getStress and getTangent, then commitState; every 7th step
// first tries another trial strain (three times the increment) and discards it
// (revertToLastCommit), as Newton iterations do. The columns record the stage, the
// discarded trial (if any), the strain, the stress and the full tangent, so a test can
// replay the same calls.
//
// Materials: PressureDependMultiYield with Femora's three soils of
// examples/site_response/nonlinear_layered_soil_column.py (dense Ottawa, loose Ottawa with
// the liquefaction parameters, dense Monterey) and a user-defined G/Gmax backbone (negative
// noYieldSurf); PressureIndependMultiYield with the clay of Femora's basin example
// (examples/domain_reduction_method/basin.py: friction angle 0), a frictional material
// (friction angle 30, pressure-dependent moduli: paramScaling at the switch), and user
// backbones without and with friction.

#include "driver_common.hpp"

#include <Information.h>
#include <Matrix.h>
#include <NDMaterial.h>
#include <PressureDependMultiYield.h>
#include <PressureIndependMultiYield.h>
#include <Vector.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

using Strain = std::array<double, 6>;  // OpenSees order 11 22 33 12 23 31 (engineering)

struct Step {
  int stage;
  Strain eps;
};
using Path = std::vector<Step>;

const double pi = 3.14159265358979323846;

// Oedometric consolidation (eps33 only) to eps_v in n steps, stage 0.
void consolidate(Path& p, double eps_v, int n = 10) {
  for (int k = 1; k <= n; ++k) p.push_back({0, {0, 0, eps_v * k / n, 0, 0, 0}});
}

Path monotonic_shear(double ev) {  // constant-volume simple shear gamma31 to 5%
  Path p;
  consolidate(p, ev);
  for (int k = 1; k <= 200; ++k) p.push_back({1, {0, 0, ev, 0, 0, 0.05 * k / 200.0}});
  return p;
}

Path cyclic(double ev, double amp, int cycles, int per_cycle) {
  Path p;
  consolidate(p, ev);
  for (int k = 1; k <= cycles * per_cycle; ++k)
    p.push_back({1, {0, 0, ev, 0, 0, amp * std::sin(2 * pi * k / per_cycle)}});
  return p;
}

Path random_reversals(double ev) {  // deviatoric random walk with reversals, 200 steps
  Path p;
  consolidate(p, ev);
  refdrv::Lcg rng(20261107);
  Strain eps = {0, 0, ev, 0, 0, 0}, dir{};
  for (int k = 0; k < 200; ++k) {
    if (k % 15 == 0)
      for (auto& d : dir) d = 2e-4 * rng.symmetric();
    if (k % 30 == 15)
      for (auto& d : dir) d = -d;
    const double tr = (dir[0] + dir[1] + dir[2]) / 3.0;  // keep the volume
    for (int i = 0; i < 3; ++i) eps[i] += (dir[i] - tr) * (1.0 + 0.1 * rng.symmetric());
    for (int i = 3; i < 6; ++i) eps[i] += dir[i] * (1.0 + 0.1 * rng.symmetric());
    p.push_back({1, eps});
  }
  return p;
}

Path confinement_change(double ev) {  // shear while the vertical strain doubles, then unloads
  Path p;
  consolidate(p, ev);
  for (int k = 1; k <= 150; ++k) {
    const double t = k / 150.0;
    const double ezz = t < 0.5 ? ev * (1.0 + 2.0 * t) : ev * (2.0 - 2.6 * (t - 0.5));
    p.push_back({1, {0, 0, ezz, 0, 0, 3e-3 * std::sin(2 * pi * 2 * t)}});
  }
  return p;
}

Path tension(double ev) {  // shear, then extension past zero volume
  Path p;
  consolidate(p, ev);
  for (int k = 1; k <= 20; ++k) p.push_back({1, {0, 0, ev, 0, 0, 1e-3 * k / 20.0}});
  for (int k = 1; k <= 40; ++k) {
    const double e = ev * (1.0 - 1.6 * k / 40.0);
    p.push_back({1, {-0.2 * (e - ev), -0.2 * (e - ev), e, 0, 0, 1e-3}});
  }
  return p;
}

Path stage_switch_sheared(double ev) {  // elastic shear beyond the surfaces, then plastic
  Path p;
  consolidate(p, ev);
  for (int k = 1; k <= 10; ++k) p.push_back({0, {0, 0, ev, 0, 0, 4e-3 * k / 10.0}});
  for (int k = 1; k <= 120; ++k)
    p.push_back({1, {0, 0, ev, 0, 0, 4e-3 + 1e-3 * std::sin(2 * pi * k / 40.0)}});
  return p;
}

Path stage_back_and_forth(double ev) {  // plastic, elastic again, plastic again
  Path p;
  consolidate(p, ev);
  for (int k = 1; k <= 40; ++k) p.push_back({1, {0, 0, ev, 0, 0, 2e-3 * std::sin(2 * pi * k / 40.0)}});
  for (int k = 1; k <= 20; ++k) p.push_back({0, {0, 0, ev, 0, 0, 1e-3 * k / 20.0}});
  for (int k = 1; k <= 40; ++k) p.push_back({1, {0, 0, ev, 0, 0, 1e-3 + 2e-3 * std::sin(2 * pi * k / 40.0)}});
  return p;
}

Path switch_at_rest(double ev) {  // the switch with no strain increment (a static step's predictor), then shear
  Path p;
  consolidate(p, ev);
  for (int k = 1; k <= 3; ++k) p.push_back({1, {0, 0, ev, 0, 0, 0}});
  for (int k = 1; k <= 40; ++k) p.push_back({1, {0, 0, ev, 0, 0, 2e-3 * std::sin(2 * pi * k / 40.0)}});
  return p;
}

struct Soil {
  std::string name;
  double rho, G, B, phi, gmax, pref, n, pt, contrac, dil1, dil2, liq1, liq2, liq3;
  int surfaces;
  std::vector<double> pairs;
  double e, cs1, cs2, cs3, pa, c;
};

// Runs one path on `mat` (the copy an element gets) and writes the file.
void drive(NDMaterial& m, const Path& path, refdrv::Writer& out);

std::vector<std::string> columns() {
  std::vector<std::string> cols = {"step", "stage", "pre"};
  for (int i = 0; i < 6; ++i) cols.push_back("pre" + std::to_string(i));
  for (int i = 0; i < 6; ++i) cols.push_back("eps" + std::to_string(i));
  for (int i = 0; i < 6; ++i) cols.push_back("sig" + std::to_string(i));
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) cols.push_back("D" + std::to_string(i) + std::to_string(j));
  return cols;
}

std::string pairs_line(const std::vector<double>& pairs) {
  std::string line = "pairs";
  for (double v : pairs) {
    char b[40];
    std::snprintf(b, sizeof(b), " %.17g", v);
    line += b;
  }
  return line;
}

void run(const Soil& s, const Path& path, const std::string& file, const std::string& pname) {
  // The prototype (nDMaterial command) and the copy an element asks for.
  PressureDependMultiYield proto(1, 3, s.rho, s.G, s.B, s.phi, s.gmax, s.pref, s.n, s.pt, s.contrac, s.dil1,
                                 s.dil2, s.liq1, s.liq2, s.liq3, s.surfaces,
                                 s.pairs.empty() ? nullptr : const_cast<double*>(s.pairs.data()), s.e, s.cs1,
                                 s.cs2, s.cs3, s.pa, s.c);
  std::unique_ptr<NDMaterial> mat(proto.getCopy("ThreeDimensional"));

  char params[1024];
  std::snprintf(params, sizeof(params),
                "params rho %.17g refShearModul %.17g refBulkModul %.17g frictionAng %.17g peakShearStra %.17g "
                "refPress %.17g pressDependCoe %.17g PTAng %.17g contrac %.17g dilat1 %.17g dilat2 %.17g "
                "liquefac1 %.17g liquefac2 %.17g liquefac3 %.17g noYieldSurf %d e %.17g cs1 %.17g cs2 %.17g "
                "cs3 %.17g pa %.17g c %.17g",
                s.rho, s.G, s.B, s.phi, s.gmax, s.pref, s.n, s.pt, s.contrac, s.dil1, s.dil2, s.liq1, s.liq2,
                s.liq3, s.surfaces, s.e, s.cs1, s.cs2, s.cs3, s.pa, s.c);
  refdrv::Writer out(file, {"material PressureDependMultiYield ThreeDimensional", params, pairs_line(s.pairs),
                            "case " + s.name, "path " + pname},
                     columns());
  drive(*mat, path, out);
}

struct Clay {  // PressureIndependMultiYield
  std::string name;
  double rho, G, B, c, gmax, phi, pref, n;
  int surfaces;
  std::vector<double> pairs;
};

void run_pimy(const Clay& s, const Path& path, const std::string& file, const std::string& pname) {
  PressureIndependMultiYield proto(1, 3, s.rho, s.G, s.B, s.c, s.gmax, s.phi, s.pref, s.n, s.surfaces,
                                   s.pairs.empty() ? nullptr : const_cast<double*>(s.pairs.data()));
  std::unique_ptr<NDMaterial> mat(proto.getCopy("ThreeDimensional"));
  char params[512];
  std::snprintf(params, sizeof(params),
                "params rho %.17g refShearModul %.17g refBulkModul %.17g cohesi %.17g peakShearStra %.17g "
                "frictionAng %.17g refPress %.17g pressDependCoe %.17g noYieldSurf %d",
                s.rho, s.G, s.B, s.c, s.gmax, s.phi, s.pref, s.n, s.surfaces);
  refdrv::Writer out(file, {"material PressureIndependMultiYield ThreeDimensional", params, pairs_line(s.pairs),
                            "case " + s.name, "path " + pname},
                     columns());
  drive(*mat, path, out);
}

void drive(NDMaterial& m, const Path& path, refdrv::Writer& out) {
  NDMaterial* mat = &m;
  Vector strain(6);
  Strain prev{};
  int stage = 0;
  for (std::size_t k = 0; k < path.size(); ++k) {
    const Step& st = path[k];
    if (st.stage != stage) {
      Information info(st.stage);
      mat->updateParameter(1, info);
      stage = st.stage;
    }
    const bool pre = k % 7 == 6;
    Strain pre_eps{};
    if (pre) {  // a trial that is thrown away
      for (int i = 0; i < 6; ++i) {
        pre_eps[i] = prev[i] + 3.0 * (st.eps[i] - prev[i]);
        strain(i) = pre_eps[i];
      }
      mat->setTrialStrain(strain);
      mat->getStress();
      mat->getTangent();
      mat->revertToLastCommit();
    }
    for (int i = 0; i < 6; ++i) strain(i) = st.eps[i];
    mat->setTrialStrain(strain);
    const Vector& sig = mat->getStress();
    std::vector<double> row = {static_cast<double>(k), static_cast<double>(stage), pre ? 1.0 : 0.0};
    for (double v : pre_eps) row.push_back(v);
    for (double v : st.eps) row.push_back(v);
    for (int i = 0; i < 6; ++i) row.push_back(sig(i));
    const Matrix& D = mat->getTangent();
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) row.push_back(D(i, j));
    out.row(row);
    mat->commitState();
    prev = st.eps;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: soil_material <out_dir>\n");
    return 1;
  }
  const std::string out = argv[1];
  // Femora's nonlinear layered soil column (kN, m, s): G and B from the example's
  // shear modulus and Poisson ratio 0.3, rho = unit weight / g.
  auto bulk = [](double G) { return 2.0 * G * 1.3 / (3.0 * 0.4); };
  const double g = 9.81;
  const std::vector<Soil> soils = {
      {"dense_ottawa", 19.9 / g, 145.0e3, bulk(145.0e3), 40.0, 0.1, 80.0, 0.5, 27.0, 0.03, 0.8, 5.0, 0.0, 0.0, 0.0,
       20, {}, 0.45, 0.9, 0.02, 0.7, 101.0, 0.3},
      {"loose_ottawa", 19.1 / g, 75.0e3, bulk(75.0e3), 29.0, 0.1, 80.0, 0.5, 29.0, 0.21, 0.0, 0.0, 10.0, 0.02, 1.0,
       20, {}, 0.85, 0.9, 0.02, 0.7, 101.0, 0.3},
      {"dense_monterey", 19.8 / g, 42.0e3, bulk(42.0e3), 40.0, 0.1, 80.0, 0.5, 27.0, 0.03, 0.8, 5.0, 0.0, 0.0, 0.0,
       20, {}, 0.45, 0.9, 0.02, 0.7, 101.0, 0.3},
      {"backbone", 19.9 / g, 145.0e3, bulk(145.0e3), 40.0, 0.1, 80.0, 0.5, 27.0, 0.03, 0.8, 5.0, 0.0, 0.0, 0.0,
       8,
       {1e-5, 0.98, 3e-5, 0.95, 1e-4, 0.85, 3e-4, 0.65, 1e-3, 0.35, 3e-3, 0.15, 1e-2, 0.05, 3e-2, 0.018},
       0.45, 0.9, 0.02, 0.7, 101.0, 0.3},
  };
  int files = 0;
  for (const auto& s : soils) {
    // Oedometric vertical strain for a vertical stress of 50, 150 and 300 kPa (elastic).
    auto ev = [&](double sv) { return -sv / (s.B + 4.0 * s.G / 3.0); };
    std::vector<std::pair<std::string, Path>> paths = {
        {"monotonic_shear_150", monotonic_shear(ev(150))},
        {"cyclic_1e-3_150", cyclic(ev(150), 1e-3, 3, 40)},
        {"cyclic_1e-2_150", cyclic(ev(150), 1e-2, 3, 40)},
        {"random_reversals_150", random_reversals(ev(150))},
        {"stage_switch_sheared_150", stage_switch_sheared(ev(150))},
    };
    if (s.name == "dense_ottawa") {
      for (double sv : {50.0, 300.0}) {
        const std::string tag = "_" + std::to_string(static_cast<int>(sv));
        paths.push_back({"monotonic_shear" + tag, monotonic_shear(ev(sv))});
        paths.push_back({"cyclic_1e-3" + tag, cyclic(ev(sv), 1e-3, 3, 40)});
      }
      paths.push_back({"cyclic_1e-4_150", cyclic(ev(150), 1e-4, 3, 40)});
      paths.push_back({"cyclic_1e-5_150", cyclic(ev(150), 1e-5, 2, 40)});
      paths.push_back({"cyclic_large_steps_150", cyclic(ev(150), 1e-2, 2, 8)});
      paths.push_back({"confinement_change_150", confinement_change(ev(150))});
      paths.push_back({"tension_50", tension(ev(50))});
      paths.push_back({"stage_back_and_forth_150", stage_back_and_forth(ev(150))});
    }
    if (s.name == "loose_ottawa") {
      paths.push_back({"cyclic_3e-3_150", cyclic(ev(150), 3e-3, 6, 40)});
      paths.push_back({"confinement_change_150", confinement_change(ev(150))});
    }
    for (const auto& [pname, path] : paths) {
      run(s, path, out + "/PressureDependMultiYield_" + s.name + "_" + pname + ".txt", pname);
      ++files;
    }
  }
  // PressureIndependMultiYield (kN, m, s). Femora's basin clay: rho 1.8, Vs 150, Vp 300,
  // cohesion 8, friction 0 (OpenSees' defaults refPress 100, pressDependCoe 0).
  const std::vector<Clay> clays = {
      {"basin_clay", 1.8, 1.8 * 150.0 * 150.0, 1.8 * (300.0 * 300.0 - 4.0 * 150.0 * 150.0 / 3.0), 8.0, 0.1, 0.0, 100.0,
       0.0, 20, {}},
      {"frictional", 2.0, 60.0e3, 150.0e3, 5.0, 0.1, 30.0, 80.0, 0.5, 20, {}},
      {"backbone_clay", 1.8, 40.5e3, 108.0e3, 10.0, 0.1, 0.0, 100.0, 0.0, 8,
       {1e-5, 0.98, 3e-5, 0.95, 1e-4, 0.85, 3e-4, 0.65, 1e-3, 0.35, 3e-3, 0.15, 1e-2, 0.05, 3e-2, 0.018}},
      {"backbone_frictional", 2.0, 60.0e3, 150.0e3, 5.0, 0.1, 30.0, 80.0, 0.5, 8,
       {1e-5, 0.98, 3e-5, 0.95, 1e-4, 0.85, 3e-4, 0.65, 1e-3, 0.35, 3e-3, 0.15, 1e-2, 0.05, 3e-2, 0.018}},
  };
  for (const auto& s : clays) {
    auto ev = [&](double sv) { return -sv / (s.B + 4.0 * s.G / 3.0); };
    std::vector<std::pair<std::string, Path>> paths = {
        {"monotonic_shear_150", monotonic_shear(ev(150))},
        {"cyclic_1e-3_150", cyclic(ev(150), 1e-3, 3, 40)},
        {"cyclic_1e-2_150", cyclic(ev(150), 1e-2, 3, 40)},
        {"random_reversals_150", random_reversals(ev(150))},
        {"stage_switch_sheared_150", stage_switch_sheared(ev(150))},
    };
    if (s.name == "basin_clay" || s.name == "frictional") {
      paths.push_back({"cyclic_1e-4_150", cyclic(ev(150), 1e-4, 3, 40)});
      paths.push_back({"cyclic_large_steps_150", cyclic(ev(150), 1e-2, 2, 8)});
      paths.push_back({"confinement_change_150", confinement_change(ev(150))});
      paths.push_back({"tension_50", tension(ev(50))});
      paths.push_back({"stage_back_and_forth_150", stage_back_and_forth(ev(150))});
      paths.push_back({"switch_at_rest_150", switch_at_rest(ev(150))});
    }
    if (s.name == "frictional") {
      paths.push_back({"monotonic_shear_50", monotonic_shear(ev(50))});
      paths.push_back({"monotonic_shear_300", monotonic_shear(ev(300))});
    }
    for (const auto& [pname, path] : paths) {
      run_pimy(s, path, out + "/PressureIndependMultiYield_" + s.name + "_" + pname + ".txt", pname);
      ++files;
    }
  }
  std::printf("wrote %d files to %s\n", files, out.c_str());
  return 0;
}
