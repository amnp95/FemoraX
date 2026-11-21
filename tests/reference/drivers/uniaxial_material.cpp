// Reference driver for uniaxial materials: runs the OpenSees class along strain paths and
// records strain, stress and tangent at every step.
//
//   uniaxial_material <out_dir>
//
// writes <out_dir>/<Material>_<case>_<path>.txt (format: driver_common.hpp). Every step calls
// setTrialStrain, getStress and getTangent, then commitState; every 7th step first tries a
// different trial strain and calls revertToLastCommit, and every 11th step evaluates the same
// trial strain twice before the commit (a Newton iterate repeated), so the commit / revert
// semantics are exercised. The header's params line holds the constructor's arguments, in the
// order of FemoraX's packed parameters after the adapter's rules (Concrete02: signs made
// negative, the short form's rat, ft and Ets filled in).
//
// Materials: Elastic (ElasticMaterial), Steel01, Concrete02.

#include "driver_common.hpp"

#include <Concrete02.h>
#include <ElasticMaterial.h>
#include <Steel01.h>
#include <UniaxialMaterial.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

using Path = std::vector<double>;
const double pi = 3.14159265358979323846;

// --- strain paths (scaled by the case's reference strain) ---------------------------
Path monotonic_tension() {  // 0 to 8 in 80 steps
  Path p;
  for (int k = 1; k <= 80; ++k) p.push_back(8.0 * k / 80.0);
  return p;
}
Path monotonic_compression() {  // 0 to -8 in 80 steps
  Path p;
  for (int k = 1; k <= 80; ++k) p.push_back(-8.0 * k / 80.0);
  return p;
}
Path growing_cycles() {  // sine cycles of amplitude 0.5, 1.5, 3, 5, 40 steps per cycle
  Path p;
  for (const double a : {0.5, 1.5, 3.0, 5.0})
    for (int k = 1; k <= 40; ++k) p.push_back(a * std::sin(2 * pi * k / 40.0));
  return p;
}
Path compression_cycles() {  // compression with partial unloadings and excursions into tension
  Path p;
  double lo = 0.0;
  for (const double a : {0.6, 1.2, 2.5, 4.0, 7.0}) {
    for (int k = 1; k <= 20; ++k) p.push_back(lo + (-a - lo) * k / 20.0);
    const double up = -a + 0.9 * a + 0.3;  // past zero stress into tension
    for (int k = 1; k <= 20; ++k) p.push_back(-a + (up + a) * k / 20.0);
    lo = up;
  }
  return p;
}
Path random_reversals() {  // random walk with reversals, 300 steps
  Path p;
  refdrv::Lcg rng(20261121);
  double eps = 0.0, dir = 0.0;
  for (int k = 0; k < 300; ++k) {
    if (k % 12 == 0) dir = 0.35 * rng.symmetric();
    if (k % 24 == 12) dir = -dir;
    eps += dir * (1.0 + 0.2 * rng.symmetric());
    p.push_back(eps);
  }
  return p;
}
Path tiny_steps() {  // increments around DBL_EPSILON, between ordinary steps
  Path p;
  double eps = 0.0;
  for (int k = 0; k < 60; ++k) {
    eps += (k % 3 == 0) ? 0.2 * ((k / 15) % 2 ? -1.0 : 1.0) : ((k % 3 == 1) ? 1e-13 : -5e-14);
    p.push_back(eps);
  }
  return p;
}

const std::vector<std::pair<std::string, std::function<Path()>>> paths = {
    {"monotonic_tension", monotonic_tension},   {"monotonic_compression", monotonic_compression},
    {"growing_cycles", growing_cycles},         {"compression_cycles", compression_cycles},
    {"random_reversals", random_reversals},     {"tiny_steps", tiny_steps},
};

// --- runner -------------------------------------------------------------------------
void run(UniaxialMaterial& mat, const Path& path, double scale, const std::string& file,
         const std::vector<std::string>& header) {
  refdrv::Writer out(file, header, {"step", "eps", "sig", "E"});
  for (std::size_t k = 0; k < path.size(); ++k) {
    const double strain = scale * path[k];
    if (k % 7 == 6) {  // a trial that is thrown away
      mat.setTrialStrain(-2.5 * strain + scale);
      mat.getStress();
      mat.revertToLastCommit();
    }
    if (k % 11 == 10) mat.setTrialStrain(strain);  // the same iterate twice
    if (mat.setTrialStrain(strain) != 0) {
      std::fprintf(stderr, "setTrialStrain failed at step %zu\n", k);
      std::exit(1);
    }
    out.row({static_cast<double>(k), strain, mat.getStress(), mat.getTangent()});
    mat.commitState();
  }
}

std::string params(const std::vector<std::pair<std::string, double>>& v) {
  std::string s = "params";
  char buf[64];
  for (const auto& [k, x] : v) {
    std::snprintf(buf, sizeof(buf), " %s %.17g", k.c_str(), x);
    s += buf;
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: uniaxial_material <out_dir>\n");
    return 1;
  }
  const std::string out = argv[1];
  int files = 0;
  auto all_paths = [&](const std::string& material, const std::string& name, double scale,
                       const std::string& par, const std::function<UniaxialMaterial*()>& make) {
    for (const auto& [pname, path] : paths) {
      std::unique_ptr<UniaxialMaterial> proto(make());
      std::unique_ptr<UniaxialMaterial> mat(proto->getCopy());  // as a fiber section copies it
      char sc[64];
      std::snprintf(sc, sizeof(sc), "scale %.17g", scale);
      run(*mat, path(), scale, out + "/" + material + "_" + name + "_" + pname + ".txt",
          {"material " + material, par, "case " + name, "path " + pname, sc});
      ++files;
    }
  };

  // Elastic: E (Eneg = E), and different moduli in tension and compression.
  all_paths("Elastic", "steel", 1e-3, params({{"Epos", 2.0e8}, {"Eneg", 2.0e8}}),
            [] { return new ElasticMaterial(1, 2.0e8); });
  all_paths("Elastic", "bimodular", 1e-3, params({{"Epos", 3.0e6}, {"Eneg", 2.5e7}}),
            [] { return new ElasticMaterial(1, 3.0e6, 0.0, 2.5e7); });

  // Steel01: kinematic hardening only (the defaults), and with isotropic hardening.
  struct S {
    std::string name;
    double fy, E0, b, a1, a2, a3, a4;
  };
  for (const S& c : {S{"kinematic", 4.2e5, 2.0e8, 0.01, 0.0, 55.0, 0.0, 55.0},
                     S{"perfectly_plastic", 2.5e5, 2.1e8, 0.0, 0.0, 55.0, 0.0, 55.0},
                     S{"isotropic", 3.45e5, 2.0e8, 0.02, 0.05, 30.0, 0.08, 20.0}})
    all_paths("Steel01", c.name, c.fy / c.E0,
              params({{"fy", c.fy}, {"E0", c.E0}, {"b", c.b}, {"a1", c.a1}, {"a2", c.a2}, {"a3", c.a3}, {"a4", c.a4}}),
              [c] { return new Steel01(1, c.fy, c.E0, c.b, c.a1, c.a2, c.a3, c.a4); });

  // Concrete02: the short form (given with positive signs, as users write it) and the full form.
  {
    const double fc = 28000.0, epsc0 = 0.002, fcu = 5600.0, epscu = 0.006;  // unconfined
    all_paths("Concrete02", "short_form", 1e-3,
              params({{"fc", -fc}, {"epsc0", -epsc0}, {"fcu", -fcu}, {"epscu", -epscu}, {"rat", 0.1},
                      {"ft", 0.1 * fc}, {"Ets", 0.1 * fc / epsc0}}),
              [=] { return new Concrete02(1, fc, epsc0, fcu, epscu); });
  }
  {
    const double fc = -35000.0, epsc0 = -0.004, fcu = -28000.0, epscu = -0.014, rat = 0.25, ft = 2800.0,
                 Ets = 1.4e6;  // confined core
    all_paths("Concrete02", "confined", 2e-3,
              params({{"fc", fc}, {"epsc0", epsc0}, {"fcu", fcu}, {"epscu", epscu}, {"rat", rat}, {"ft", ft},
                      {"Ets", Ets}}),
              [=] { return new Concrete02(1, fc, epsc0, fcu, epscu, rat, ft, Ets); });
    all_paths("Concrete02", "no_tension", 1e-3,
              params({{"fc", -30000.0}, {"epsc0", -0.002}, {"fcu", 0.0}, {"epscu", -0.008}, {"rat", 0.1},
                      {"ft", 0.0}, {"Ets", 1.0e6}}),
              [] { return new Concrete02(1, -30000.0, -0.002, 0.0, -0.008, 0.1, 0.0, 1.0e6); });
  }
  std::printf("wrote %d files to %s\n", files, out.c_str());
  return 0;
}
