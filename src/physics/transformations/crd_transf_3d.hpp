#pragma once

// Linear and PDelta coordinate transformations of a 3D two-node beam.
//
// Port of OpenSees SRC/coordTransformation/LinearCrdTransf3d.cpp and PDeltaCrdTransf3d.cpp,
// OpenSees commit 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea (`geomTransf Linear|PDelta tag
// vecxz_x vecxz_y vecxz_z [-jntOffset dXi dYi dZi dXj dYj dZj]`). Variable names follow the
// OpenSees members (R, L, nodeIOffset, ...) and locals (ug, ul, ub, pl, pg, kb, kl, tmp, kg,
// RWI, RWJ), and the operation order is OpenSees', so results match to rounding.
//
// R rows: the local x axis (node I to node J, offsets included), y = v x x / |v x x|,
// z = x x y (getLocalAxes). Basic system (getBasicTrialDisp): ub = (axial elongation,
// rotations about z at I and J, rotations about y at I and J, twist), with the chord
// rotation removed. Rigid joint offsets are given in global coordinates; an offset is used
// when it is not zero (OpenSees: Norm() > 0).
//
// PDelta (leaning-column effect): the local transverse chord displacements ul17 = ul1 - ul7,
// ul28 = ul2 - ul8 of the trial displacements (PDeltaCrdTransf3d::update) add q0 ul / L to the
// local end shears, and the geometric stiffness N / L to the local transverse DOFs (not to the
// initial stiffness).
//
// Not ported: nodal displacements that exist when the element is created
// (nodeIInitialDisp, staged construction): FemoraX creates every element at the start.
//
// global_stiff forms the full 12 x 12 matrix: host references, tests and the stable step's
// element eigenvalue; the implicit kernels use node-pair blocks (performance rule 3a).

#include <cmath>

#include "physics/macros.hpp"

namespace femorax {

enum class CrdTransfType : int { Linear = 0, PDelta = 1 };

struct CrdTransf3dParams {
  CrdTransfType type = CrdTransfType::Linear;
  double vecxz[3] = {0.0, 0.0, 0.0};
  double nodeIOffset[3] = {0.0, 0.0, 0.0};  // rigid joint offsets, global coordinates
  double nodeJOffset[3] = {0.0, 0.0, 0.0};
};

// The transformation of one element: computed from its nodes once (initialize()).
struct CrdTransf3dGeometry {
  double R[3][3] = {};
  double L = 0.0;
  bool hasI = false, hasJ = false;  // offsets in use
};

// initialize(): 0 on success, -2 for a zero length, -3 when vecxz is parallel to the axis
// (OpenSees' return codes).
FEMORAX_HD inline int crd_transf_3d_geometry(const CrdTransf3dParams& p, const double xi[3], const double xj[3],
                                             CrdTransf3dGeometry& g) {
  const double* oI = p.nodeIOffset;
  const double* oJ = p.nodeJOffset;
  g.hasI = oI[0] != 0.0 || oI[1] != 0.0 || oI[2] != 0.0;
  g.hasJ = oJ[0] != 0.0 || oJ[1] != 0.0 || oJ[2] != 0.0;

  // computeElemtLengthAndOrient()
  double dx[3] = {xj[0] - xi[0], xj[1] - xi[1], xj[2] - xi[2]};
  if (g.hasJ) {
    dx[0] += oJ[0];
    dx[1] += oJ[1];
    dx[2] += oJ[2];
  }
  if (g.hasI) {
    dx[0] -= oI[0];
    dx[1] -= oI[1];
    dx[2] -= oI[2];
  }
  g.L = std::sqrt(dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]);  // Vector::Norm
  if (g.L == 0.0) return -2;
  g.R[0][0] = dx[0] / g.L;
  g.R[0][1] = dx[1] / g.L;
  g.R[0][2] = dx[2] / g.L;

  // getLocalAxes(): y = v x x, z = x x y
  const double v[3] = {p.vecxz[0], p.vecxz[1], p.vecxz[2]};
  const double x[3] = {g.R[0][0], g.R[0][1], g.R[0][2]};
  double y[3] = {v[1] * x[2] - v[2] * x[1], v[2] * x[0] - v[0] * x[2], v[0] * x[1] - v[1] * x[0]};
  const double ynorm = std::sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
  if (ynorm == 0.0) return -3;
  y[0] /= ynorm;
  y[1] /= ynorm;
  y[2] /= ynorm;
  const double z[3] = {x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
  for (int i = 0; i < 3; ++i) {
    g.R[1][i] = y[i];
    g.R[2][i] = z[i];
  }
  return 0;
}

namespace crd_transf_detail {

// ul = local displacements of the 12 global ones, the offsets' rigid-arm terms included
// (getBasicTrialDisp); PDelta's update() uses the same ul1, ul2, ul7, ul8.
FEMORAX_HD inline void local_disp(const CrdTransf3dParams& p, const CrdTransf3dGeometry& g, const double ug[12],
                                  double ul[12]) {
  const auto& R = g.R;
  for (int n = 0; n < 4; ++n) {
    const int o = 3 * n;
    ul[o + 0] = R[0][0] * ug[o] + R[0][1] * ug[o + 1] + R[0][2] * ug[o + 2];
    ul[o + 1] = R[1][0] * ug[o] + R[1][1] * ug[o + 1] + R[1][2] * ug[o + 2];
    ul[o + 2] = R[2][0] * ug[o] + R[2][1] * ug[o + 1] + R[2][2] * ug[o + 2];
  }
  double Wu[3];
  if (g.hasI) {
    const double* nodeIOffset = p.nodeIOffset;
    Wu[0] = nodeIOffset[2] * ug[4] - nodeIOffset[1] * ug[5];
    Wu[1] = -nodeIOffset[2] * ug[3] + nodeIOffset[0] * ug[5];
    Wu[2] = nodeIOffset[1] * ug[3] - nodeIOffset[0] * ug[4];
    ul[0] += R[0][0] * Wu[0] + R[0][1] * Wu[1] + R[0][2] * Wu[2];
    ul[1] += R[1][0] * Wu[0] + R[1][1] * Wu[1] + R[1][2] * Wu[2];
    ul[2] += R[2][0] * Wu[0] + R[2][1] * Wu[1] + R[2][2] * Wu[2];
  }
  if (g.hasJ) {
    const double* nodeJOffset = p.nodeJOffset;
    Wu[0] = nodeJOffset[2] * ug[10] - nodeJOffset[1] * ug[11];
    Wu[1] = -nodeJOffset[2] * ug[9] + nodeJOffset[0] * ug[11];
    Wu[2] = nodeJOffset[1] * ug[9] - nodeJOffset[0] * ug[10];
    ul[6] += R[0][0] * Wu[0] + R[0][1] * Wu[1] + R[0][2] * Wu[2];
    ul[7] += R[1][0] * Wu[0] + R[1][1] * Wu[1] + R[1][2] * Wu[2];
    ul[8] += R[2][0] * Wu[0] + R[2][1] * Wu[1] + R[2][2] * Wu[2];
  }
}

// RWI / RWJ of getGlobalStiffMatrix: R times the offset's cross-product matrix.
FEMORAX_HD inline void rw(const double R[3][3], const double* o, double RW[3][3]) {
  for (int a = 0; a < 3; ++a) {
    RW[a][0] = -R[a][1] * o[2] + R[a][2] * o[1];
    RW[a][1] = R[a][0] * o[2] - R[a][2] * o[0];
    RW[a][2] = -R[a][0] * o[1] + R[a][1] * o[0];
  }
}

}  // namespace crd_transf_detail

// getBasicTrialDisp() (the same for Linear and PDelta); also the basic velocity from the nodal
// velocities (getBasicTrialVel uses the same map).
FEMORAX_HD inline void crd_transf_3d_basic_disp(const CrdTransf3dParams& p, const CrdTransf3dGeometry& g,
                                                const double ug[12], double ub[6]) {
  double ul[12];
  crd_transf_detail::local_disp(p, g, ug, ul);
  const double oneOverL = 1.0 / g.L;
  ub[0] = ul[6] - ul[0];
  double tmp = oneOverL * (ul[1] - ul[7]);
  ub[1] = ul[5] + tmp;
  ub[2] = ul[11] + tmp;
  tmp = oneOverL * (ul[8] - ul[2]);
  ub[3] = ul[4] + tmp;
  ub[4] = ul[10] + tmp;
  ub[5] = ul[9] - ul[3];
}

// getGlobalResistingForce(pb, p0): pb the basic forces (6), p0 the element-load end forces
// (5: OpenSees' q0 order, axial at I, shears y at I and J, shears z at I and J). ug: the
// trial displacements (PDelta's ul17 and ul28; unused by Linear).
FEMORAX_HD inline void crd_transf_3d_global_force(const CrdTransf3dParams& p, const CrdTransf3dGeometry& g,
                                                  const double ug[12], const double pb[6], const double p0[5],
                                                  double pg[12]) {
  const double q0 = pb[0], q1 = pb[1], q2 = pb[2], q3 = pb[3], q4 = pb[4], q5 = pb[5];
  const double oneOverL = 1.0 / g.L;
  double pl[12];
  pl[0] = -q0;
  pl[1] = oneOverL * (q1 + q2);
  pl[2] = -oneOverL * (q3 + q4);
  pl[3] = -q5;
  pl[4] = q3;
  pl[5] = q1;
  pl[6] = q0;
  pl[7] = -pl[1];
  pl[8] = -pl[2];
  pl[9] = q5;
  pl[10] = q4;
  pl[11] = q2;

  pl[0] += p0[0];
  pl[1] += p0[1];
  pl[7] += p0[2];
  pl[2] += p0[3];
  pl[8] += p0[4];

  if (p.type == CrdTransfType::PDelta) {  // leaning column effects
    double ul[12];
    crd_transf_detail::local_disp(p, g, ug, ul);
    const double ul17 = ul[1] - ul[7];
    const double ul28 = ul[2] - ul[8];
    double NoverL = ul17 * q0 * oneOverL;
    pl[1] += NoverL;
    pl[7] -= NoverL;
    NoverL = ul28 * q0 * oneOverL;
    pl[2] += NoverL;
    pl[8] -= NoverL;
  }

  const auto& R = g.R;
  for (int n = 0; n < 4; ++n) {
    const int o = 3 * n;
    pg[o + 0] = R[0][0] * pl[o] + R[1][0] * pl[o + 1] + R[2][0] * pl[o + 2];
    pg[o + 1] = R[0][1] * pl[o] + R[1][1] * pl[o + 1] + R[2][1] * pl[o + 2];
    pg[o + 2] = R[0][2] * pl[o] + R[1][2] * pl[o + 1] + R[2][2] * pl[o + 2];
  }
  if (g.hasI) {
    const double* nodeIOffset = p.nodeIOffset;
    pg[3] += -nodeIOffset[2] * pg[1] + nodeIOffset[1] * pg[2];
    pg[4] += nodeIOffset[2] * pg[0] - nodeIOffset[0] * pg[2];
    pg[5] += -nodeIOffset[1] * pg[0] + nodeIOffset[0] * pg[1];
  }
  if (g.hasJ) {
    const double* nodeJOffset = p.nodeJOffset;
    pg[9] += -nodeJOffset[2] * pg[7] + nodeJOffset[1] * pg[8];
    pg[10] += nodeJOffset[2] * pg[6] - nodeJOffset[0] * pg[8];
    pg[11] += -nodeJOffset[1] * pg[6] + nodeJOffset[0] * pg[7];
  }
}

// getGlobalStiffMatrix(kb, pb) (geometric = true; PDelta's N / L terms) and
// getInitialGlobalStiffMatrix(kb) (geometric = false). Linear has no geometric term.
FEMORAX_HD inline void crd_transf_3d_global_stiff(const CrdTransf3dParams& p, const CrdTransf3dGeometry& g,
                                                  const double kb[6][6], const double pb[6], bool geometric,
                                                  double kg[12][12]) {
  const double oneOverL = 1.0 / g.L;
  double kl[12][12];
  double tmp[12][12];
  // kb * T_bl
  for (int i = 0; i < 6; i++) {
    tmp[i][0] = -kb[i][0];
    tmp[i][1] = oneOverL * (kb[i][1] + kb[i][2]);
    tmp[i][2] = -oneOverL * (kb[i][3] + kb[i][4]);
    tmp[i][3] = -kb[i][5];
    tmp[i][4] = kb[i][3];
    tmp[i][5] = kb[i][1];
    tmp[i][6] = kb[i][0];
    tmp[i][7] = -tmp[i][1];
    tmp[i][8] = -tmp[i][2];
    tmp[i][9] = kb[i][5];
    tmp[i][10] = kb[i][4];
    tmp[i][11] = kb[i][2];
  }
  // T_bl' * (kb * T_bl)
  for (int i = 0; i < 12; i++) {
    kl[0][i] = -tmp[0][i];
    kl[1][i] = oneOverL * (tmp[1][i] + tmp[2][i]);
    kl[2][i] = -oneOverL * (tmp[3][i] + tmp[4][i]);
    kl[3][i] = -tmp[5][i];
    kl[4][i] = tmp[3][i];
    kl[5][i] = tmp[1][i];
    kl[6][i] = tmp[0][i];
    kl[7][i] = -kl[1][i];
    kl[8][i] = -kl[2][i];
    kl[9][i] = tmp[5][i];
    kl[10][i] = tmp[4][i];
    kl[11][i] = tmp[2][i];
  }
  if (geometric && p.type == CrdTransfType::PDelta) {  // geometric stiffness, local system
    const double NoverL = pb[0] * oneOverL;
    kl[1][1] += NoverL;
    kl[2][2] += NoverL;
    kl[7][7] += NoverL;
    kl[8][8] += NoverL;
    kl[1][7] -= NoverL;
    kl[7][1] -= NoverL;
    kl[2][8] -= NoverL;
    kl[8][2] -= NoverL;
  }

  const auto& R = g.R;
  double RWI[3][3], RWJ[3][3];
  if (g.hasI) crd_transf_detail::rw(R, p.nodeIOffset, RWI);
  if (g.hasJ) crd_transf_detail::rw(R, p.nodeJOffset, RWJ);

  // kl * T_lg
  for (int m = 0; m < 12; m++) {
    tmp[m][0] = kl[m][0] * R[0][0] + kl[m][1] * R[1][0] + kl[m][2] * R[2][0];
    tmp[m][1] = kl[m][0] * R[0][1] + kl[m][1] * R[1][1] + kl[m][2] * R[2][1];
    tmp[m][2] = kl[m][0] * R[0][2] + kl[m][1] * R[1][2] + kl[m][2] * R[2][2];

    tmp[m][3] = kl[m][3] * R[0][0] + kl[m][4] * R[1][0] + kl[m][5] * R[2][0];
    tmp[m][4] = kl[m][3] * R[0][1] + kl[m][4] * R[1][1] + kl[m][5] * R[2][1];
    tmp[m][5] = kl[m][3] * R[0][2] + kl[m][4] * R[1][2] + kl[m][5] * R[2][2];

    if (g.hasI) {
      tmp[m][3] += kl[m][0] * RWI[0][0] + kl[m][1] * RWI[1][0] + kl[m][2] * RWI[2][0];
      tmp[m][4] += kl[m][0] * RWI[0][1] + kl[m][1] * RWI[1][1] + kl[m][2] * RWI[2][1];
      tmp[m][5] += kl[m][0] * RWI[0][2] + kl[m][1] * RWI[1][2] + kl[m][2] * RWI[2][2];
    }

    tmp[m][6] = kl[m][6] * R[0][0] + kl[m][7] * R[1][0] + kl[m][8] * R[2][0];
    tmp[m][7] = kl[m][6] * R[0][1] + kl[m][7] * R[1][1] + kl[m][8] * R[2][1];
    tmp[m][8] = kl[m][6] * R[0][2] + kl[m][7] * R[1][2] + kl[m][8] * R[2][2];

    tmp[m][9] = kl[m][9] * R[0][0] + kl[m][10] * R[1][0] + kl[m][11] * R[2][0];
    tmp[m][10] = kl[m][9] * R[0][1] + kl[m][10] * R[1][1] + kl[m][11] * R[2][1];
    tmp[m][11] = kl[m][9] * R[0][2] + kl[m][10] * R[1][2] + kl[m][11] * R[2][2];

    if (g.hasJ) {
      tmp[m][9] += kl[m][6] * RWJ[0][0] + kl[m][7] * RWJ[1][0] + kl[m][8] * RWJ[2][0];
      tmp[m][10] += kl[m][6] * RWJ[0][1] + kl[m][7] * RWJ[1][1] + kl[m][8] * RWJ[2][1];
      tmp[m][11] += kl[m][6] * RWJ[0][2] + kl[m][7] * RWJ[1][2] + kl[m][8] * RWJ[2][2];
    }
  }

  // T_lg' * (kl * T_lg)
  for (int m = 0; m < 12; m++) {
    kg[0][m] = R[0][0] * tmp[0][m] + R[1][0] * tmp[1][m] + R[2][0] * tmp[2][m];
    kg[1][m] = R[0][1] * tmp[0][m] + R[1][1] * tmp[1][m] + R[2][1] * tmp[2][m];
    kg[2][m] = R[0][2] * tmp[0][m] + R[1][2] * tmp[1][m] + R[2][2] * tmp[2][m];

    kg[3][m] = R[0][0] * tmp[3][m] + R[1][0] * tmp[4][m] + R[2][0] * tmp[5][m];
    kg[4][m] = R[0][1] * tmp[3][m] + R[1][1] * tmp[4][m] + R[2][1] * tmp[5][m];
    kg[5][m] = R[0][2] * tmp[3][m] + R[1][2] * tmp[4][m] + R[2][2] * tmp[5][m];

    if (g.hasI) {
      kg[3][m] += RWI[0][0] * tmp[0][m] + RWI[1][0] * tmp[1][m] + RWI[2][0] * tmp[2][m];
      kg[4][m] += RWI[0][1] * tmp[0][m] + RWI[1][1] * tmp[1][m] + RWI[2][1] * tmp[2][m];
      kg[5][m] += RWI[0][2] * tmp[0][m] + RWI[1][2] * tmp[1][m] + RWI[2][2] * tmp[2][m];
    }

    kg[6][m] = R[0][0] * tmp[6][m] + R[1][0] * tmp[7][m] + R[2][0] * tmp[8][m];
    kg[7][m] = R[0][1] * tmp[6][m] + R[1][1] * tmp[7][m] + R[2][1] * tmp[8][m];
    kg[8][m] = R[0][2] * tmp[6][m] + R[1][2] * tmp[7][m] + R[2][2] * tmp[8][m];

    kg[9][m] = R[0][0] * tmp[9][m] + R[1][0] * tmp[10][m] + R[2][0] * tmp[11][m];
    kg[10][m] = R[0][1] * tmp[9][m] + R[1][1] * tmp[10][m] + R[2][1] * tmp[11][m];
    kg[11][m] = R[0][2] * tmp[9][m] + R[1][2] * tmp[10][m] + R[2][2] * tmp[11][m];

    if (g.hasJ) {
      kg[9][m] += RWJ[0][0] * tmp[6][m] + RWJ[1][0] * tmp[7][m] + RWJ[2][0] * tmp[8][m];
      kg[10][m] += RWJ[0][1] * tmp[6][m] + RWJ[1][1] * tmp[7][m] + RWJ[2][1] * tmp[8][m];
      kg[11][m] += RWJ[0][2] * tmp[6][m] + RWJ[1][2] * tmp[7][m] + RWJ[2][2] * tmp[8][m];
    }
  }
}

}  // namespace femorax
