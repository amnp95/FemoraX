// Reference driver for OpenSees Brick (stdBrick): single elements in a real Domain.
//
//   brick_element <out_dir>
//
// For each geometry x material it builds two Bricks on the same eight nodes, one with
// consistent mass (massType 0) and one lumped (-lumped, massType 1), and for each
// displacement/velocity field records
//   f   getResistingForce()          (internal force with the body force)
//   K   getTangentStiff()
//   K0  getInitialStiff()
//   dc  getRayleighDampingForces()   consistent-mass element
//   dl  getRayleighDampingForces()   lumped-mass element
// and, once per geometry x material, both mass matrices (getMass()).
//
// Output: <out_dir>/Brick_<geometry>_<material>_state.txt and ..._mass.txt
// (format: driver_common.hpp; vectors and matrices flattened row-major).

#include "driver_common.hpp"

#include <Brick.h>
#include <Domain.h>
#include <ElasticIsotropicMaterial.h>
#include <Matrix.h>
#include <Node.h>
#include <Vector.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// Brick with Element::getRayleighDampingForces (protected in OpenSees) made callable.
// Nothing else changes: same constructor, same element.
class BrickProbe : public Brick {
 public:
  using Brick::Brick;
  using Element::getRayleighDampingForces;
};

using Coords = std::array<std::array<double, 3>, 8>;

Coords box(double a, double b, double c, double z0 = 0.0) {
  return {{{0, 0, z0}, {a, 0, z0}, {a, b, z0}, {0, b, z0},
           {0, 0, z0 + c}, {a, 0, z0 + c}, {a, b, z0 + c}, {0, b, z0 + c}}};
}

Coords distorted() {
  Coords x = box(1.3, 0.8, 1.1);
  refdrv::Lcg rng(101);
  for (auto& n : x)
    for (auto& v : n) v += 0.15 * rng.symmetric();
  return x;
}

Coords rotated() {  // stretched box rotated about two axes
  Coords x = box(2.0, 0.6, 0.9);
  const double a = 0.4, b = -0.7;
  for (auto& n : x) {
    const double X = n[0], Y = n[1], Z = n[2];
    const double y1 = std::cos(a) * Y - std::sin(a) * Z, z1 = std::sin(a) * Y + std::cos(a) * Z;
    const double x2 = std::cos(b) * X + std::sin(b) * z1, z2 = -std::sin(b) * X + std::cos(b) * z1;
    n = {x2 + 5.0, y1 - 2.0, z2 + 1.0};
  }
  return x;
}

struct Geometry {
  std::string name;
  Coords x;
};

struct Mat {
  std::string name;
  double E, nu, rho;
};

using Field = std::array<double, 24>;

std::vector<std::pair<std::string, std::array<Field, 2>>> fields(const Coords& x) {
  std::vector<std::pair<std::string, std::array<Field, 2>>> out;
  refdrv::Lcg rng(202);
  auto random = [&](double amp) {
    Field f;
    for (auto& v : f) v = amp * rng.symmetric();
    return f;
  };
  Field zero{};
  out.push_back({"zero", {zero, random(1e-2)}});
  // Constant strain: u = A x.
  const double A[3][3] = {{1e-3, 2e-4, -3e-4}, {5e-4, -7e-4, 1e-4}, {-2e-4, 3e-4, 4e-4}};
  Field lin;
  for (int n = 0; n < 8; ++n)
    for (int i = 0; i < 3; ++i) lin[3 * n + i] = A[i][0] * x[n][0] + A[i][1] * x[n][1] + A[i][2] * x[n][2];
  out.push_back({"constant_strain", {lin, random(1e-2)}});
  out.push_back({"random_small", {random(1e-4), random(1e-3)}});
  out.push_back({"random_large", {random(1e-2), random(1e-1)}});
  return out;
}

void append(std::vector<double>& row, const Vector& v) {
  for (int i = 0; i < v.Size(); ++i) row.push_back(v(i));
}

void append(std::vector<double>& row, const Matrix& m) {
  for (int i = 0; i < m.noRows(); ++i)
    for (int j = 0; j < m.noCols(); ++j) row.push_back(m(i, j));
}

std::vector<std::string> names(const std::string& prefix, int n) {
  std::vector<std::string> v;
  for (int i = 0; i < n; ++i) v.push_back(prefix + std::to_string(i));
  return v;
}

std::string fmt(const char* key, const double* v, int n) {
  std::string s = key;
  char buf[40];
  for (int i = 0; i < n; ++i) {
    std::snprintf(buf, sizeof(buf), " %.17g", v[i]);
    s += buf;
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: brick_element <out_dir>\n");
    return 1;
  }
  const std::string out = argv[1];
  const std::vector<Geometry> geometries = {
      {"cube", box(1, 1, 1)},
      {"soil_column", box(1.0, 1.0, 0.5, -3.5)},
      {"distorted", distorted()},
      {"rotated", rotated()},
  };
  const std::vector<Mat> materials = {
      {"dense_ottawa", 377000.0, 0.3, 2.0285},
      {"nearly_incompressible", 3.0e4, 0.499, 2.4},
  };
  const double b[3] = {0.5, -1.0, -19.9};
  const double damping[4] = {0.7, 2e-3, 5e-4, 1e-4};  // alphaM betaK betaK0 betaKc

  int files = 0;
  for (const auto& g : geometries) {
    for (const auto& m : materials) {
      Domain domain;
      for (int n = 0; n < 8; ++n) domain.addNode(new Node(n + 1, 3, g.x[n][0], g.x[n][1], g.x[n][2]));
      ElasticIsotropicMaterial mat(1, m.E, m.nu, m.rho);
      auto* cons = new BrickProbe(1, 1, 2, 3, 4, 5, 6, 7, 8, mat, b[0], b[1], b[2], nullptr, 0);
      auto* lump = new BrickProbe(2, 1, 2, 3, 4, 5, 6, 7, 8, mat, b[0], b[1], b[2], nullptr, 1);
      domain.addElement(cons);
      domain.addElement(lump);
      cons->setRayleighDampingFactors(damping[0], damping[1], damping[2], damping[3]);
      lump->setRayleighDampingFactors(damping[0], damping[1], damping[2], damping[3]);

      double coords[24];
      for (int n = 0; n < 8; ++n)
        for (int i = 0; i < 3; ++i) coords[3 * n + i] = g.x[n][i];
      char params[256];
      std::snprintf(params, sizeof(params), "params E %.17g nu %.17g rho %.17g", m.E, m.nu, m.rho);
      const std::vector<std::string> header = {
          "element Brick (stdBrick)", params, fmt("coords", coords, 24), fmt("body_force", b, 3),
          fmt("rayleigh", damping, 4), "geometry " + g.name, "material " + m.name};
      const std::string base = out + "/Brick_" + g.name + "_" + m.name;

      {  // mass matrices
        std::vector<std::string> cols = names("Mc", 576);
        for (const auto& c : names("Ml", 576)) cols.push_back(c);
        refdrv::Writer w(base + "_mass.txt", header, cols);
        std::vector<double> row;
        append(row, cons->getMass());
        append(row, lump->getMass());
        w.row(row);
      }

      std::vector<std::string> cols = {"field"};
      for (const char* p : {"u", "v", "f"})
        for (const auto& c : names(p, 24)) cols.push_back(c);
      for (const auto& c : names("K", 576)) cols.push_back(c);
      for (const auto& c : names("K0_", 576)) cols.push_back(c);
      for (const char* p : {"dc", "dl"})
        for (const auto& c : names(p, 24)) cols.push_back(c);
      refdrv::Writer w(base + "_state.txt", header, cols);

      int k = 0;
      for (const auto& [fname, uv] : fields(g.x)) {
        for (int n = 0; n < 8; ++n) {
          Vector u(3), v(3);
          for (int i = 0; i < 3; ++i) {
            u(i) = uv[0][3 * n + i];
            v(i) = uv[1][3 * n + i];
          }
          Node* node = domain.getNode(n + 1);
          node->setTrialDisp(u);
          node->setTrialVel(v);
        }
        cons->update();
        lump->update();
        std::vector<double> row = {static_cast<double>(k++)};
        for (double x : uv[0]) row.push_back(x);
        for (double x : uv[1]) row.push_back(x);
        append(row, cons->getResistingForce());
        append(row, cons->getTangentStiff());
        append(row, cons->getInitialStiff());
        append(row, cons->getRayleighDampingForces());
        append(row, lump->getRayleighDampingForces());
        w.row(row);
      }
      files += 2;
    }
  }
  std::printf("wrote %d files to %s\n", files, out.c_str());
  return 0;
}
