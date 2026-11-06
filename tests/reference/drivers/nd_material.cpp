// Reference driver for nD materials (3D): runs the OpenSees class along strain paths and
// records strain, stress and tangent at every step.
//
//   nd_material <out_dir>
//
// writes <out_dir>/<Material>_<case>_<path>.txt (format: driver_common.hpp). Every step
// calls setTrialStrain, getStress and getTangent; most steps then commitState, and every
// 7th step first tries a different trial strain and calls revertToLastCommit, so the
// commit/revert semantics are exercised too (they matter for materials with history).
//
// Materials: ElasticIsotropic (nDMaterial ElasticIsotropic, "ThreeDimensional" copy).

#include "driver_common.hpp"

#include <ElasticIsotropicMaterial.h>
#include <Matrix.h>
#include <NDMaterial.h>
#include <Vector.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace {

using Strain = std::vector<double>;  // 6 components, OpenSees order 11 22 33 12 23 31
using Path = std::vector<Strain>;

// --- strain paths -----------------------------------------------------------------
Path monotonic_uniaxial() {  // eps11 from 0 to 2e-3 in 20 steps
  Path p;
  for (int k = 1; k <= 20; ++k) p.push_back({2e-3 * k / 20.0, 0, 0, 0, 0, 0});
  return p;
}

Path cyclic_shear() {  // gamma12 = 1e-3 sin, 3 cycles, 40 steps per cycle
  Path p;
  const double pi = 3.14159265358979323846;
  for (int k = 1; k <= 120; ++k) p.push_back({0, 0, 0, 1e-3 * std::sin(2 * pi * k / 40.0), 0, 0});
  return p;
}

Path volumetric_load_unload() {  // isotropic compression to -1e-3 and back past zero
  Path p;
  for (int k = 1; k <= 10; ++k) p.push_back({-1e-4 * k, -1e-4 * k, -1e-4 * k, 0, 0, 0});
  for (int k = 9; k >= -5; --k) p.push_back({-1e-4 * k, -1e-4 * k, -1e-4 * k, 0, 0, 0});
  return p;
}

Path random_reversals() {  // 6D random walk with a reversal every 15 steps, 200 steps
  Path p;
  refdrv::Lcg rng(20261105);
  Strain eps(6, 0.0), dir(6, 0.0);
  for (int k = 0; k < 200; ++k) {
    if (k % 15 == 0)
      for (auto& d : dir) d = 1e-4 * rng.symmetric();
    if (k % 30 == 15)
      for (auto& d : dir) d = -d;  // reversal
    for (int i = 0; i < 6; ++i) eps[i] += dir[i] * (1.0 + 0.1 * rng.symmetric());
    p.push_back(eps);
  }
  return p;
}

const std::vector<std::pair<std::string, std::function<Path()>>> paths = {
    {"monotonic_uniaxial", monotonic_uniaxial},
    {"cyclic_shear", cyclic_shear},
    {"volumetric_load_unload", volumetric_load_unload},
    {"random_reversals", random_reversals},
};

// --- runner -------------------------------------------------------------------------
void run(NDMaterial& mat, const Path& path, const std::string& file,
         const std::vector<std::string>& header) {
  std::vector<std::string> cols = {"step"};
  for (int i = 0; i < 6; ++i) cols.push_back("eps" + std::to_string(i));
  for (int i = 0; i < 6; ++i) cols.push_back("sig" + std::to_string(i));
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < 6; ++j) cols.push_back("D" + std::to_string(i) + std::to_string(j));
  refdrv::Writer out(file, header, cols);

  Vector strain(6);
  for (std::size_t k = 0; k < path.size(); ++k) {
    if (k % 7 == 6) {  // a trial that is thrown away
      for (int i = 0; i < 6; ++i) strain(i) = 3.0 * path[k][i];
      mat.setTrialStrain(strain);
      mat.getStress();
      mat.revertToLastCommit();
    }
    for (int i = 0; i < 6; ++i) strain(i) = path[k][i];
    if (mat.setTrialStrain(strain) != 0) {
      std::fprintf(stderr, "setTrialStrain failed at step %zu\n", k);
      std::exit(1);
    }
    const Vector& sig = mat.getStress();
    const Matrix& D = mat.getTangent();
    std::vector<double> row = {static_cast<double>(k)};
    for (int i = 0; i < 6; ++i) row.push_back(strain(i));
    for (int i = 0; i < 6; ++i) row.push_back(sig(i));
    for (int i = 0; i < 6; ++i)
      for (int j = 0; j < 6; ++j) row.push_back(D(i, j));
    out.row(row);
    mat.commitState();
  }
}

struct ElasticCase {
  std::string name;
  double E, nu, rho;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: nd_material <out_dir>\n");
    return 1;
  }
  const std::string out = argv[1];

  // The same parameter sets as tests/unit/test_elastic_isotropic.cpp.
  const std::vector<ElasticCase> elastic = {
      {"dense_ottawa", 377000.0, 0.3, 2.0285},
      {"loose_ottawa", 195000.0, 0.3, 1.947},
      {"steel", 2.0e11, 0.29, 7850.0},
      {"nu_zero", 1.0, 0.0, 1.0},
      {"nearly_incompressible", 3.0e4, 0.499, 2.4},
  };
  int files = 0;
  for (const auto& c : elastic) {
    ElasticIsotropicMaterial proto(1, c.E, c.nu, c.rho);
    for (const auto& [pname, make] : paths) {
      // The copy stdBrick and SSPbrick ask for.
      std::unique_ptr<NDMaterial> mat(proto.getCopy("ThreeDimensional"));
      char params[256];
      std::snprintf(params, sizeof(params), "params E %.17g nu %.17g rho %.17g", c.E, c.nu,
                    c.rho);
      run(*mat, make(),
          out + "/ElasticIsotropic_" + c.name + "_" + pname + ".txt",
          {"material ElasticIsotropic ThreeDimensional", params, "case " + c.name,
           "path " + pname});
      ++files;
    }
  }
  std::printf("wrote %d files to %s\n", files, out.c_str());
  return 0;
}
