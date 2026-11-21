// Reference driver for OpenSees' beams with fiber sections: ForceBeamColumn3d (Lobatto, 5
// points, -iter 10 1e-12) and DispBeamColumn3d (Gauss-Legendre, 3 points) with a reinforced-
// concrete FiberSection3d (Concrete02 core and cover, Steel01 bars, elastic torsion), Linear and
// PDelta transformations, one element in a real Domain.
//
//   fiber_beam_column <out_dir>
//
// A column along z, node i held, node j driven along a displacement history (axial shortening,
// growing lateral cycles in x and y, end rotations), one update per step and a commit after
// each (so OpenSees' update from its last trial state is an update from the committed state,
// as FemoraX's). Every step records the nodal displacement of node j, getResistingForce() and
// getTangentStiff(); the first row (step -1) is getInitialStiff() (f zero).
// Output: <out_dir>/FiberBeam3d_<element>_<type>.txt (format: driver_common.hpp; matrices
// row-major); the header holds the geometry and the section (materials, fibers, GJ) in the
// format of fiber_section.cpp.

#include "driver_common.hpp"

#include <Concrete02.h>
#include <DispBeamColumn3d.h>
#include <Domain.h>
#include <ElasticMaterial.h>
#include <FiberSection3d.h>
#include <ForceBeamColumn3d.h>
#include <LegendreBeamIntegration.h>
#include <LinearCrdTransf3d.h>
#include <LobattoBeamIntegration.h>
#include <Matrix.h>
#include <Node.h>
#include <PDeltaCrdTransf3d.h>
#include <Steel01.h>
#include <UniaxialFiber3d.h>
#include <Vector.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

const double pi = 3.14159265358979323846;

struct Mat {
  int type;  // 1 Steel01, 2 Concrete02
  std::array<double, 7> v;
  UniaxialMaterial* make() const {
    if (type == 1) return new Steel01(1, v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
    return new Concrete02(1, v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
  }
};
struct Fib {
  double y, z, A;
  int mat;
};

void grid(std::vector<Fib>& f, double y0, double y1, double z0, double z1, int ny, int nz, int mat) {
  const double dy = (y1 - y0) / ny, dz = (z1 - z0) / nz;
  for (int i = 0; i < ny; ++i)
    for (int j = 0; j < nz; ++j) f.push_back({y0 + (i + 0.5) * dy, z0 + (j + 0.5) * dz, dy * dz, mat});
}

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
    std::fprintf(stderr, "usage: fiber_beam_column <out_dir>\n");
    return 1;
  }
  const std::string out = argv[1];
  // The section: 0.5 (y) x 0.4 (z), 4 x 4 core fibers, 12 cover fibers, 4 bars: 32 fibers.
  const std::vector<Mat> mats = {{2, {-35000.0, -0.004, -28000.0, -0.014, 0.25, 2800.0, 1.4e6}},
                                 {2, {-28000.0, -0.002, -5600.0, -0.006, 0.1, 2800.0, 1.4e6}},
                                 {1, {4.2e5, 2.0e8, 0.01, 0.0, 55.0, 0.0, 55.0}}};
  std::vector<Fib> fibers;
  const double h = 0.5, b = 0.4, c = 0.05, GJ = 1.5e5;
  grid(fibers, -0.5 * h + c, 0.5 * h - c, -0.5 * b + c, 0.5 * b - c, 4, 4, 0);
  grid(fibers, -0.5 * h, -0.5 * h + c, -0.5 * b, 0.5 * b, 1, 4, 1);
  grid(fibers, 0.5 * h - c, 0.5 * h, -0.5 * b, 0.5 * b, 1, 4, 1);
  grid(fibers, -0.5 * h + c, 0.5 * h - c, -0.5 * b, -0.5 * b + c, 2, 1, 1);
  grid(fibers, -0.5 * h + c, 0.5 * h - c, 0.5 * b - c, 0.5 * b, 2, 1, 1);
  for (const double y : {-0.5 * h + c, 0.5 * h - c})
    for (const double z : {-0.5 * b + c, 0.5 * b - c}) fibers.push_back({y, z, 8.0e-4, 2});
  std::vector<double> mv, fv;
  for (const Mat& m : mats) {
    mv.push_back(m.type);
    mv.insert(mv.end(), m.v.begin(), m.v.end());
  }
  for (const Fib& f : fibers) fv.insert(fv.end(), {f.y, f.z, f.A, static_cast<double>(f.mat)});

  const std::array<double, 3> xi = {1.0, 2.0, -3.0}, xj = {1.0, 2.0, 0.0}, vecxz = {1.0, 0.0, 0.0};
  const double L = 3.0;
  int files = 0;
  for (const std::string element : {"force", "disp"})
    for (const std::string type : {"Linear", "PDelta"}) {
      Domain domain;
      auto* ni = new Node(1, 6, xi[0], xi[1], xi[2]);
      auto* nj = new Node(2, 6, xj[0], xj[1], xj[2]);
      domain.addNode(ni);
      domain.addNode(nj);
      Vector vz(3), zero3(3);
      for (int i = 0; i < 3; ++i) vz(i) = vecxz[static_cast<std::size_t>(i)];
      std::unique_ptr<CrdTransf> t;
      if (type == "Linear") t = std::make_unique<LinearCrdTransf3d>(1, vz, zero3, zero3);
      else t = std::make_unique<PDeltaCrdTransf3d>(1, vz, zero3, zero3);
      ElasticMaterial torsion(0, GJ);
      FiberSection3d sec(1, static_cast<int>(fibers.size()), torsion, true);
      for (std::size_t i = 0; i < fibers.size(); ++i) {
        std::unique_ptr<UniaxialMaterial> mat(mats[static_cast<std::size_t>(fibers[i].mat)].make());
        Vector pos(2);
        pos(0) = fibers[i].y;
        pos(1) = fibers[i].z;
        UniaxialFiber3d fiber(static_cast<int>(i) + 1, *mat, fibers[i].A, pos);
        sec.addFiber(fiber);
      }
      const int nip = element == "force" ? 5 : 3;
      std::vector<SectionForceDeformation*> secs(static_cast<std::size_t>(nip), &sec);
      LobattoBeamIntegration lobatto;
      LegendreBeamIntegration legendre;
      Element* ele = nullptr;
      if (element == "force") ele = new ForceBeamColumn3d(1, 1, 2, nip, secs.data(), lobatto, *t, 0.0, 10, 1.0e-12);
      else ele = new DispBeamColumn3d(1, 1, 2, nip, secs.data(), legendre, *t, 0.0, 0);
      domain.addElement(ele);

      std::vector<std::string> cols = {"step"};
      for (int k = 0; k < 6; ++k) cols.push_back("dj" + std::to_string(k));
      for (int k = 0; k < 12; ++k) cols.push_back("f" + std::to_string(k));
      for (int a = 0; a < 12; ++a)
        for (int bb = 0; bb < 12; ++bb) cols.push_back("K" + std::to_string(a) + "_" + std::to_string(bb));
      refdrv::Writer w(out + "/FiberBeam3d_" + element + "_" + type + ".txt",
                       {"component " + std::string(element == "force" ? "ForceBeamColumn3d" : "DispBeamColumn3d") +
                            " FiberSection3d " + type + "CrdTransf3d",
                        line("nip", {static_cast<double>(nip)}), line("xi", {xi[0], xi[1], xi[2]}),
                        line("xj", {xj[0], xj[1], xj[2]}), line("vecxz", {vecxz[0], vecxz[1], vecxz[2]}),
                        line("materials", mv), line("fibers", fv), line("GJ", {GJ}), line("centroid", {1.0})},
                       cols);
      auto record = [&](double step, const Vector& dj, const Vector* f, const Matrix& K) {
        std::vector<double> row = {step};
        for (int k = 0; k < 6; ++k) row.push_back(dj(k));
        for (int k = 0; k < 12; ++k) row.push_back(f ? (*f)(k) : 0.0);
        for (int a = 0; a < 12; ++a)
          for (int bb = 0; bb < 12; ++bb) row.push_back(K(a, bb));
        w.row(row);
      };
      Vector dj(6);
      {
        const Matrix K0 = ele->getInitialStiff();
        record(-1.0, dj, nullptr, K0);
      }
      // 120 steps: three lateral cycles of drift 0.3 %, 1 %, 2.5 % in x (out of phase in y at 40 %),
      // axial shortening growing to 0.05 % of L, small end rotations and twist.
      const int nstep = 120;
      for (int k = 1; k <= nstep; ++k) {
        const int cyc = (k - 1) / 40;
        const double amp = (cyc == 0 ? 0.003 : cyc == 1 ? 0.01 : 0.025) * L;
        const double th = 2 * pi * k / 40.0;
        dj(0) = amp * std::sin(th);
        dj(1) = 0.4 * amp * std::sin(th + 0.7);
        dj(2) = -5.0e-4 * L * k / nstep;
        dj(3) = -0.2 * dj(1) / L;
        dj(4) = 0.2 * dj(0) / L;
        dj(5) = 2.0e-3 * std::sin(0.5 * th);
        nj->setTrialDisp(dj);
        if (ele->update() != 0) {
          std::fprintf(stderr, "%s %s: update failed at step %d\n", element.c_str(), type.c_str(), k);
          return 1;
        }
        const Vector f = ele->getResistingForce();
        const Matrix K = ele->getTangentStiff();
        record(static_cast<double>(k), dj, &f, K);
        nj->commitState();
        ni->commitState();
        ele->commitState();
      }
      ++files;
    }
  std::printf("wrote %d files to %s\n", files, out.c_str());
  return 0;
}
