// Reference driver for OpenSees DispBeamColumn3d with ElasticSection3d sections,
// LegendreBeamIntegration and the Linear or PDelta transformation (the legacy
// `element dispBeamColumn tag i j nIP secTag transfTag -mass rhoA` form): one element in a real
// Domain.
//
//   disp_beam_column <out_dir>
//
// For each type (Linear, PDelta), geometry (with and without rigid joint offsets), number of
// points (2, 3, 5) and two pseudo-random trial displacement fields it records
//   f   getResistingForce()   (after update())
//   K   getTangentStiff()
//   K0  getInitialStiff()
//   M   getMass()             (lumped: rhoA L / 2 on the translations only)
// Output: <out_dir>/DispBeamColumn3d_<type>.txt, one row per case, the inputs as columns
// (format: driver_common.hpp; matrices row-major).

#include "driver_common.hpp"

#include <DispBeamColumn3d.h>
#include <Domain.h>
#include <ElasticSection3d.h>
#include <LegendreBeamIntegration.h>
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
      {"skew_offsets", {0.3, -1.2, 0.7}, {2.9, 0.4, 2.2}, {-0.2, 0.3, 1.0}, {0.25, -0.1, 0.05}, {-0.3, 0.2, -0.15}},
  };
}

constexpr double E = 2.0e8, A = 0.05, Iz = 4.0e-4, Iy = 2.5e-4, G = 7.7e7, J = 3.1e-4, rhoA = 0.12;

Vector vec(const std::array<double, 3>& a) {
  Vector v(3);
  for (int i = 0; i < 3; ++i) v(i) = a[static_cast<std::size_t>(i)];
  return v;
}

std::vector<std::string> columns() {
  std::vector<std::string> c = {"nip"};
  auto add = [&](const std::string& n, int k) {
    for (int i = 0; i < k; ++i) c.push_back(n + std::to_string(i));
  };
  add("xi", 3);
  add("xj", 3);
  add("vecxz", 3);
  add("offi", 3);
  add("offj", 3);
  c.insert(c.end(), {"E", "A", "Iz", "Iy", "G", "J", "rhoA"});
  add("ug", 12);
  add("f", 12);
  add("K", 144);
  add("K0", 144);
  add("M", 144);
  return c;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: disp_beam_column <out_dir>\n");
    return 2;
  }
  const std::string out = argv[1];
  for (const std::string type : {"Linear", "PDelta"}) {
    refdrv::Writer w(out + "/DispBeamColumn3d_" + type + ".txt",
                     {"component DispBeamColumn3d ElasticSection3d LegendreBeamIntegration " + type + "CrdTransf3d",
                      "states 2 per geometry and number of points"},
                     columns());
    int tag = 1;
    for (const auto& g : geometries()) {
      for (const int nip : {2, 3, 5}) {
        Domain domain;
        auto* ni = new Node(tag, 6, g.xi[0], g.xi[1], g.xi[2]);
        auto* nj = new Node(tag + 1, 6, g.xj[0], g.xj[1], g.xj[2]);
        domain.addNode(ni);
        domain.addNode(nj);
        std::unique_ptr<CrdTransf> t;
        if (type == "Linear") t = std::make_unique<LinearCrdTransf3d>(tag, vec(g.vecxz), vec(g.offI), vec(g.offJ));
        else t = std::make_unique<PDeltaCrdTransf3d>(tag, vec(g.vecxz), vec(g.offI), vec(g.offJ));
        ElasticSection3d sec(tag, E, A, Iz, Iy, G, J);
        std::vector<SectionForceDeformation*> secs(static_cast<std::size_t>(nip), &sec);
        LegendreBeamIntegration bi;
        auto* ele = new DispBeamColumn3d(tag, tag, tag + 1, nip, secs.data(), bi, *t, rhoA, 0);
        domain.addElement(ele);
        refdrv::Lcg rng(static_cast<std::uint64_t>(500 + tag));
        for (int s = 0; s < 2; ++s) {
          Vector di(6), dj(6);
          for (int k = 0; k < 6; ++k) di(k) = 1e-2 * rng.symmetric();
          for (int k = 0; k < 6; ++k) dj(k) = 1e-2 * rng.symmetric();
          ni->setTrialDisp(di);
          nj->setTrialDisp(dj);
          ele->update();
          std::vector<double> row = {static_cast<double>(nip)};
          for (const auto& a : {g.xi, g.xj, g.vecxz, g.offI, g.offJ})
            for (const double x : a) row.push_back(x);
          row.insert(row.end(), {E, A, Iz, Iy, G, J, rhoA});
          for (int k = 0; k < 6; ++k) row.push_back(di(k));
          for (int k = 0; k < 6; ++k) row.push_back(dj(k));
          const Vector& f = ele->getResistingForce();
          for (int k = 0; k < 12; ++k) row.push_back(f(k));
          // One getter at a time: they share the element's matrix K.
          for (int which = 0; which < 3; ++which) {
            const Matrix copy = which == 0 ? ele->getTangentStiff() : which == 1 ? ele->getInitialStiff() : ele->getMass();
            for (int a = 0; a < 12; ++a)
              for (int b = 0; b < 12; ++b) row.push_back(copy(a, b));
          }
          w.row(row);
        }
        tag += 2;
      }
    }
  }
  return 0;
}
