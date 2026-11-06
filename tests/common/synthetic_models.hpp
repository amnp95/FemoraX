#pragma once

// Synthetic models built in memory, the way Femora builds a layered soil column: a grid of
// stdBrick elements, two materials (lower and upper half), body force, fixed base, laminar
// boundary as equalDOF stars at each elevation above the base (side nodes tied to the first
// one in DOFs 1 2 3), one Rayleigh region over everything and a second over the top layer
// (later region wins, as in OpenSees). Used by tests and benchmarks.

#include <cstdint>
#include <string>

#include "io/model.hpp"

namespace femorax::testing {

struct ColumnSpec {
  int nx = 2, ny = 2, nz = 8;
  double dx = 1.0, dy = 1.0, dz = 0.5;
  bool lumped = true;
  bool damping = true;
};

inline io::Model soil_column(const ColumnSpec& s) {
  io::Model m;
  const int NX = s.nx + 1, NY = s.ny + 1, NZ = s.nz + 1;
  auto tag = [&](int i, int j, int k) { return 1 + i + NX * (j + NY * k); };
  for (int k = 0; k < NZ; ++k)
    for (int j = 0; j < NY; ++j)
      for (int i = 0; i < NX; ++i) {
        m.nodes.tags.push_back(tag(i, j, k));
        m.nodes.ndf.push_back(3);
        m.nodes.coords.insert(m.nodes.coords.end(),
                              {i * s.dx, j * s.dy, -s.nz * s.dz + k * s.dz});
      }
  m.materials.push_back({1, io::ElasticIsotropicData{377000.0, 0.3, 2.0285}, {}});
  m.materials.push_back({2, io::ElasticIsotropicData{109200.0, 0.3, 2.0183}, {}});
  int etag = 1;
  for (int layer = 0; layer < 2; ++layer) {
    io::ElementGroup g;
    g.id = std::to_string(layer);
    g.type = io::ElementType::StdBrick;
    g.material = layer + 1;
    g.lumped = s.lumped;
    g.body_force = {0.0, 0.0, layer == 0 ? -19.9 : -19.8};
    const int k0 = layer == 0 ? 0 : s.nz / 2, k1 = layer == 0 ? s.nz / 2 : s.nz;
    for (int k = k0; k < k1; ++k)
      for (int j = 0; j < s.ny; ++j)
        for (int i = 0; i < s.nx; ++i) {
          g.tags.push_back(etag++);
          for (int c : {tag(i, j, k), tag(i + 1, j, k), tag(i + 1, j + 1, k), tag(i, j + 1, k),
                        tag(i, j, k + 1), tag(i + 1, j, k + 1), tag(i + 1, j + 1, k + 1),
                        tag(i, j + 1, k + 1)})
            g.connectivity.push_back(c);
        }
    m.elements.push_back(g);
  }
  for (int j = 0; j < NY; ++j)
    for (int i = 0; i < NX; ++i)
      for (int d = 1; d <= 3; ++d) {
        m.fix.node.push_back(tag(i, j, 0));
        m.fix.dof.push_back(d);
      }
  for (int k = 1; k < NZ; ++k) {
    int master = -1;
    for (int j = 0; j < NY; ++j)
      for (int i = 0; i < NX; ++i) {
        if (!(i == 0 || i == s.nx || j == 0 || j == s.ny)) continue;
        if (master < 0) {
          master = tag(i, j, k);
          continue;
        }
        for (int d = 1; d <= 3; ++d) {
          m.equal_dof.retained.push_back(master);
          m.equal_dof.constrained.push_back(tag(i, j, k));
          m.equal_dof.dof.push_back(d);
        }
      }
  }
  if (s.damping) {
    io::RayleighRegion all, top;
    all.id = "000000";
    all.alphaM = 0.6;
    all.betaK = 5.8e-4;
    for (int e = 1; e < etag; ++e) all.elements.push_back(e);
    top.id = "000001";
    top.alphaM = 0.9;
    top.betaK = 3e-4;
    top.betaKinit = 1e-4;
    for (int e = etag - s.nx * s.ny; e < etag; ++e) top.elements.push_back(e);
    m.rayleigh = {all, top};
  }
  return m;
}

}  // namespace femorax::testing
