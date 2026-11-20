// Reference driver for OpenSees' 3D coordinate transformations: LinearCrdTransf3d and
// PDeltaCrdTransf3d, called directly on two 6-DOF nodes.
//
//   crd_transf <out_dir>
//
// For each type (Linear, PDelta), geometry (with and without rigid joint offsets) and three
// pseudo-random states (trial nodal displacements ug, basic forces pb, element-load end forces
// p0, a symmetric basic stiffness kb) it records
//   ub  getBasicTrialDisp()                (after update())
//   pg  getGlobalResistingForce(pb, p0)
//   kg  getGlobalStiffMatrix(kb, pb)
//   k0  getInitialGlobalStiffMatrix(kb)
// Output: <out_dir>/CrdTransf3d_<type>.txt, one row per (geometry, state); the inputs are
// columns too (x_i, x_j, vecxz, offsets, ug, pb, p0, kb), so each row is self-contained
// (format: driver_common.hpp; matrices row-major).

#include "driver_common.hpp"

#include <LinearCrdTransf3d.h>
#include <Matrix.h>
#include <Node.h>
#include <PDeltaCrdTransf3d.h>
#include <Vector.h>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Geometry {
  std::string name;
  std::array<double, 3> xi, xj, vecxz, offI, offJ;
};

std::vector<Geometry> geometries() {
  return {
      {"beam_x", {0, 0, 0}, {3, 0, 0}, {0, 0, 1}, {0, 0, 0}, {0, 0, 0}},
      {"column_z", {1, 2, -4}, {1, 2, 0.5}, {1, 0, 0}, {0, 0, 0}, {0, 0, 0}},
      {"skew", {0.3, -1.2, 0.7}, {2.9, 0.4, 2.2}, {-0.2, 0.3, 1.0}, {0, 0, 0}, {0, 0, 0}},
      {"skew_offset_i", {0.3, -1.2, 0.7}, {2.9, 0.4, 2.2}, {-0.2, 0.3, 1.0}, {0.25, -0.1, 0.05}, {0, 0, 0}},
      {"skew_offsets", {0.3, -1.2, 0.7}, {2.9, 0.4, 2.2}, {-0.2, 0.3, 1.0}, {0.25, -0.1, 0.05}, {-0.3, 0.2, -0.15}},
      {"column_offsets", {1, 2, -4}, {1, 2, 0.5}, {0, 1, 0}, {0, 0, 0.4}, {0.1, 0, -0.35}},
  };
}

Vector vec(const std::array<double, 3>& a) {
  Vector v(3);
  for (int i = 0; i < 3; ++i) v(i) = a[static_cast<std::size_t>(i)];
  return v;
}

std::vector<std::string> columns() {
  std::vector<std::string> c;
  auto add = [&](const std::string& n, int k) {
    for (int i = 0; i < k; ++i) c.push_back(n + std::to_string(i));
  };
  add("xi", 3);
  add("xj", 3);
  add("vecxz", 3);
  add("offi", 3);
  add("offj", 3);
  add("ug", 12);
  add("pb", 6);
  add("p0", 5);
  add("kb", 36);
  add("ub", 6);
  add("pg", 12);
  add("kg", 144);
  add("k0", 144);
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: crd_transf <out_dir>\n");
    return 2;
  }
  const std::string out = argv[1];
  for (const std::string type : {"Linear", "PDelta"}) {
    refdrv::Writer w(out + "/CrdTransf3d_" + type + ".txt",
                     {"component " + type + "CrdTransf3d", "states 3 per geometry"}, columns());
    int tag = 1;
    for (const auto& g : geometries()) {
      Node ni(tag, 6, g.xi[0], g.xi[1], g.xi[2]);
      Node nj(tag + 1, 6, g.xj[0], g.xj[1], g.xj[2]);
      std::unique_ptr<CrdTransf> t;
      if (type == "Linear") t = std::make_unique<LinearCrdTransf3d>(tag, vec(g.vecxz), vec(g.offI), vec(g.offJ));
      else t = std::make_unique<PDeltaCrdTransf3d>(tag, vec(g.vecxz), vec(g.offI), vec(g.offJ));
      if (t->initialize(&ni, &nj) != 0) {
        std::fprintf(stderr, "initialize failed: %s\n", g.name.c_str());
        return 1;
      }
      refdrv::Lcg rng(static_cast<std::uint64_t>(1000 + tag));
      for (int s = 0; s < 3; ++s) {
        Vector di(6), dj(6), pb(6), p0(5);
        for (int k = 0; k < 6; ++k) di(k) = 1e-2 * rng.symmetric();
        for (int k = 0; k < 6; ++k) dj(k) = 1e-2 * rng.symmetric();
        for (int k = 0; k < 6; ++k) pb(k) = 1e3 * rng.symmetric();
        for (int k = 0; k < 5; ++k) p0(k) = 1e2 * rng.symmetric();
        Matrix kb(6, 6);
        for (int a = 0; a < 6; ++a)
          for (int b = 0; b <= a; ++b) kb(a, b) = kb(b, a) = 1e5 * rng.symmetric() + (a == b ? 1e6 : 0.0);
        ni.setTrialDisp(di);
        nj.setTrialDisp(dj);
        t->update();
        std::vector<double> row;
        for (const auto& a : {g.xi, g.xj, g.vecxz, g.offI, g.offJ})
          for (const double x : a) row.push_back(x);
        for (int k = 0; k < 6; ++k) row.push_back(di(k));
        for (int k = 0; k < 6; ++k) row.push_back(dj(k));
        for (int k = 0; k < 6; ++k) row.push_back(pb(k));
        for (int k = 0; k < 5; ++k) row.push_back(p0(k));
        for (int a = 0; a < 6; ++a)
          for (int b = 0; b < 6; ++b) row.push_back(kb(a, b));
        const Vector& ub = t->getBasicTrialDisp();
        for (int k = 0; k < 6; ++k) row.push_back(ub(k));
        const Vector& pg = t->getGlobalResistingForce(pb, p0);
        for (int k = 0; k < 12; ++k) row.push_back(pg(k));
        const Matrix& kg = t->getGlobalStiffMatrix(kb, pb);
        for (int a = 0; a < 12; ++a)
          for (int b = 0; b < 12; ++b) row.push_back(kg(a, b));
        const Matrix& k0 = t->getInitialGlobalStiffMatrix(kb);
        for (int a = 0; a < 12; ++a)
          for (int b = 0; b < 12; ++b) row.push_back(k0(a, b));
        w.row(row);
      }
      tag += 2;
    }
  }
  return 0;
}
