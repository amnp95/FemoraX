#pragma once

// SSPbrick: 8-node brick with one integration point and physical stabilization
// (stabilized single-point, enhanced assumed strain; McGann, Arduino and Mackenzie-Helnwein).
//
// Port of OpenSees SRC/element/UWelements/SSPbrick.cpp, OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea. Names follow the original (xi, et, ze, hut, hus,
// hst, hstu, J[], dNmod, Bnot, Mben, FCF, FeCFe, FeCFhg, mVol, ...) so the two can be read
// side by side.
//
// What is ported, and the OpenSees function it matches:
//   point_geometry       GetStab's geometry: J[0..19], mVol, dNmod (the rows of Bnot), and
//                        the bracket of getMass' massTerm / getResistingForce's polyJac
//   point_strain         update: strain = Bnot u
//   point_node_force     getResistingForce: mVol * Bnot^T stress
//   point_body_force     getResistingForce, applyLoad == 0: b * polyJac at every node
//   point_block_tangent  getTangentStiff's mVol * Bnot^T C Bnot, by node-pair block
//   element_constants    GetStab's hourglass vectors (gst gut gus gstu: the rows of Mben)
//                        and the 12 x 12 matrix interior = FCF - KuT FeCFe^-1 FeCFhg, from
//                        the material's initial tangent (C1, C2, C3)
//   element_term_*       Kstab = Mben^T interior Mben, applied as Mben^T (interior (Mben v))
//                        without forming the 24 x 24 matrix (CLAUDE.md, performance rule 3a)
//   mass, lumped_mass    getMass (always diagonal)
//   compute              getResistingForce + getRayleighDampingForces (+ getTangentStiff)
//
// OpenSees semantics kept:
// - Kstab is computed once (setDomain) from the material's getInitialTangent at that time;
//   here from Mat::initial_tangent when the element constants are built (or recomputed:
//   both give the same bits). For a material whose initial tangent changes during a run
//   (stage-dependent soil materials, milestone 2) this must be the tangent at
//   construction; the material contract will say so when such a material arrives.
// - getInitialStiff returns getTangentStiff, so betaKinit damping uses the current tangent
//   (INITIAL_STIFFNESS_IS_TANGENT). Kstab is part of the current, initial and committed
//   stiffness alike, so every stiffness-proportional damping term includes it.
//
// Deliberate differences (docs/deviations.md, "SSPbrick"):
// - Rayleigh damping forces are summed term by term (matrix-free) instead of forming
//   C = alphaM M + betaK K + betaK0 K0 + betaKc Kc and multiplying by v; Kstab v is applied
//   as Mben^T (interior (Mben v)) instead of the assembled 24 x 24 product; the 3 x 3 and
//   9 x 9 inverses are an LU with partial pivoting (OpenSees: LAPACK). Same quantities,
//   different rounding order.
// - interior is stored as its upper triangle (78 values), so Kstab is exactly symmetric;
//   OpenSees' product KuT FeCFe^-1 FeCFhg is symmetric to rounding only.
// - Only the body-force path with no eleLoad (applyLoad == 0) exists; schema v0 has no
//   eleLoad.

#include <array>
#include <cmath>

#include "physics/common/element_term.hpp"
#include "physics/common/inner.hpp"
#include "physics/common/small_dense.hpp"
#include "physics/common/strided.hpp"
#include "physics/common/sym_eigen.hpp"
#include "physics/contracts.hpp"
#include "physics/explicit_stability.hpp"
#include "physics/macros.hpp"

namespace femorax {

namespace ssp_detail {

// The natural-coordinate and hourglass vectors of SSPbrick::setDomain.
struct ShapeVectors {
  double xi[8], et[8], ze[8], hut[8], hus[8], hst[8], hstu[8];
  FEMORAX_HD ShapeVectors() {
    const double s[8][7] = {
        // xi      et      ze      hst     hut     hus     hstu
        {-0.125, -0.125, -0.125, 0.125, 0.125, 0.125, -0.125},
        {0.125, -0.125, -0.125, -0.125, 0.125, -0.125, 0.125},
        {0.125, 0.125, -0.125, 0.125, -0.125, -0.125, -0.125},
        {-0.125, 0.125, -0.125, -0.125, -0.125, 0.125, 0.125},
        {-0.125, -0.125, 0.125, 0.125, -0.125, -0.125, 0.125},
        {0.125, -0.125, 0.125, -0.125, -0.125, 0.125, -0.125},
        {0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125},
        {-0.125, 0.125, 0.125, -0.125, 0.125, -0.125, -0.125},
    };
    for (int i = 0; i < 8; ++i) {
      xi[i] = s[i][0];
      et[i] = s[i][1];
      ze[i] = s[i][2];
      hst[i] = s[i][3];
      hut[i] = s[i][4];
      hus[i] = s[i][5];
      hstu[i] = s[i][6];
    }
  }
};

FEMORAX_HD inline double dot8(const double a[8], const double b[8]) {
  double result = 0.0;
  for (int i = 0; i < 8; i++) result += a[i] * b[i];
  return result;
}

FEMORAX_HD inline double dot3(const double a[3], const double b[3]) {
  double result = 0.0;
  for (int i = 0; i < 3; i++) result += a[i] * b[i];
  return result;
}

// SSPbrick::CrossProduct
FEMORAX_HD inline void cross(const double v1[3], const double v2[3], double r[3]) {
  r[0] = v1[1] * v2[2] - v1[2] * v2[1];
  r[1] = v1[2] * v2[0] - v1[0] * v2[2];
  r[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

// e_a . (e_b x e_c)
FEMORAX_HD inline double triple(const double a[3], const double b[3], const double c[3]) {
  double t[3];
  cross(b, c, t);
  return dot3(a, t);
}

FEMORAX_HD inline double triple_neg(const double a[3], const double b[3], const double c[3]) {
  const double na[3] = {-1.0 * a[0], -1.0 * a[1], -1.0 * a[2]};
  return triple(na, b, c);
}

// The geometric part of SSPbrick::GetStab (no material): Jacobian determinant terms,
// volume, the modified shape-function derivatives dNmod (bx, by, bz), Jinv, and the
// hourglass vectors gst, gut, gus, gstu. mNodeCrd(d, i) = xl[d][i].
struct Geometry {
  double J[20];
  double mVol;
  double dNmod[8][3];
  double Jinv[3][3];
  double gst[8], gut[8], gus[8], gstu[8];
};

FEMORAX_HD inline void geometry(const double xl[3][8], Geometry& g, bool hourglass) {
  const ShapeVectors h;
  const double* x = xl[0];
  const double* y = xl[1];
  const double* z = xl[2];

  // define coefficient terms for jacobian determinant
  const double a1 = dot8(x, h.xi), a2 = dot8(x, h.et), a3 = dot8(x, h.ze), a4 = dot8(x, h.hut),
               a5 = dot8(x, h.hus), a6 = dot8(x, h.hst), a7 = dot8(x, h.hstu);
  const double b1 = dot8(y, h.xi), b2 = dot8(y, h.et), b3 = dot8(y, h.ze), b4 = dot8(y, h.hut),
               b5 = dot8(y, h.hus), b6 = dot8(y, h.hst), b7 = dot8(y, h.hstu);
  const double c1 = dot8(z, h.xi), c2 = dot8(z, h.et), c3 = dot8(z, h.ze), c4 = dot8(z, h.hut),
               c5 = dot8(z, h.hus), c6 = dot8(z, h.hst), c7 = dot8(z, h.hstu);

  const double e1[3] = {a1, b1, c1}, e2[3] = {a2, b2, c2}, e3[3] = {a3, b3, c3}, e4[3] = {a4, b4, c4},
               e5[3] = {a5, b5, c5}, e6[3] = {a6, b6, c6}, e7[3] = {a7, b7, c7};

  // jacobian determinant terms (J[1..3] are set to zero in OpenSees)
  double* J = g.J;
  J[0] = triple(e1, e2, e3);
  J[1] = 0.0;
  J[2] = 0.0;
  J[3] = 0.0;
  J[4] = triple(e7, e2, e3) + triple(e4, e5, e2) + triple(e4, e3, e6);
  J[5] = triple(e1, e7, e3) + triple(e4, e5, e1) + triple(e3, e5, e6);
  J[6] = triple(e1, e2, e7) + triple(e4, e1, e6) + triple(e2, e5, e6);
  J[7] = triple_neg(e1, e5, e6);
  J[8] = triple_neg(e4, e2, e6);
  J[9] = triple_neg(e4, e5, e3);
  J[10] = triple(e2, e4, e7);
  J[11] = triple_neg(e3, e4, e7);
  J[12] = triple(e3, e5, e7);
  J[13] = triple_neg(e1, e5, e7);
  J[14] = triple(e1, e6, e7);
  J[15] = triple_neg(e2, e6, e7);
  {
    const double e4x2[3] = {2.0 * e4[0], 2.0 * e4[1], 2.0 * e4[2]};
    J[16] = triple(e4x2, e5, e6);
  }
  J[17] = triple(e7, e5, e6);
  J[18] = triple(e4, e7, e6);
  J[19] = triple(e4, e5, e7);

  // compute element volume
  const double mVol = 8.0 * (J[0] + (J[7] + J[8] + J[9]) / 3.0);
  g.mVol = mVol;

  // kinematic vectors
  for (int i = 0; i < 8; i++) {
    const double bx = (8.0 * ((b2 * c3 - c2 * b3) * h.xi[i] + (b3 * c1 - c3 * b1) * h.et[i] + (b1 * c2 - c1 * b2) * h.ze[i]) +
                       (8.0 / 3.0) * ((b6 * c5 - c6 * b5) * h.xi[i] + (b4 * c6 - c4 * b6) * h.et[i] + (b5 * c4 - c5 * b4) * h.ze[i] +
                                      (b5 * c1 - c5 * b1 + b2 * c4 - c2 * b4) * h.hst[i] +
                                      (b6 * c2 - c6 * b2 + b3 * c5 - c3 * b5) * h.hut[i] +
                                      (b1 * c6 - c1 * b6 + b4 * c3 - c4 * b3) * h.hus[i])) /
                      mVol;
    const double by = (8.0 * ((c2 * a3 - a2 * c3) * h.xi[i] + (c3 * a1 - a3 * c1) * h.et[i] + (c1 * a2 - a1 * c2) * h.ze[i]) +
                       (8.0 / 3.0) * ((c6 * a5 - a6 * c5) * h.xi[i] + (c4 * a6 - a4 * c6) * h.et[i] + (c5 * a4 - a5 * c4) * h.ze[i] +
                                      (c5 * a1 - a5 * c1 + c2 * a4 - a2 * c4) * h.hst[i] +
                                      (c6 * a2 - a6 * c2 + c3 * a5 - a3 * c5) * h.hut[i] +
                                      (c1 * a6 - a1 * c6 + c4 * a3 - a4 * c3) * h.hus[i])) /
                      mVol;
    const double bz = (8.0 * ((a2 * b3 - b2 * a3) * h.xi[i] + (a3 * b1 - b3 * a1) * h.et[i] + (a1 * b2 - b1 * a2) * h.ze[i]) +
                       (8.0 / 3.0) * ((a6 * b5 - b6 * a5) * h.xi[i] + (a4 * b6 - b4 * a6) * h.et[i] + (a5 * b4 - b5 * a4) * h.ze[i] +
                                      (a5 * b1 - b5 * a1 + a2 * b4 - b2 * a4) * h.hst[i] +
                                      (a6 * b2 - b6 * a2 + a3 * b5 - b3 * a5) * h.hut[i] +
                                      (a1 * b6 - b1 * a6 + a4 * b3 - b4 * a3) * h.hus[i])) /
                      mVol;
    g.dNmod[i][0] = bx;
    g.dNmod[i][1] = by;
    g.dNmod[i][2] = bz;
  }
  (void)b7;
  (void)c7;
  (void)a7;
  if (!hourglass) return;

  // jacobian matrix Jmat = mNodeCrd * dNloc (dNloc columns: xi, et, ze) and its inverse
  double Jmat[3][3];
  for (int d = 0; d < 3; ++d) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (int i = 0; i < 8; ++i) {
      s0 += xl[d][i] * h.xi[i];
      s1 += xl[d][i] * h.et[i];
      s2 += xl[d][i] * h.ze[i];
    }
    Jmat[d][0] = s0;
    Jmat[d][1] = s1;
    Jmat[d][2] = s2;
  }
  if (!invert_small<3>(Jmat, g.Jinv)) {
    for (int a = 0; a < 3; ++a)
      for (int c = 0; c < 3; ++c) g.Jinv[a][c] = 0.0;
  }

  // hourglass transformation matrix G = I8 - dNmod * mNodeCrd, and the gamma vectors
  double G[8][8];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += g.dNmod[i][k] * xl[k][j];
      G[i][j] = (i == j ? 1.0 : 0.0) - s;
    }
  for (int i = 0; i < 8; ++i) {
    double st = 0.0, ut = 0.0, us = 0.0, stu = 0.0;
    for (int j = 0; j < 8; ++j) {
      st += G[i][j] * h.hst[j];
      ut += G[i][j] * h.hut[j];
      us += G[i][j] * h.hus[j];
      stu += G[i][j] * h.hstu[j];
    }
    g.gst[i] = st;
    g.gut[i] = ut;
    g.gus[i] = us;
    g.gstu[i] = stu;
  }
}

// The bracket of getMass' massTerm and getResistingForce's polyJac at node i (both are
// J[0] times it): 1 + (...)/3 + (...)/9 + (...)/27.
FEMORAX_HD inline double poly_bracket(const double J[20], const ShapeVectors& h, int i) {
  return 1.0 + (J[1] * h.xi[i] + J[2] * h.et[i] + J[3] * h.ze[i] + J[7] + J[8] + J[9]) / 3.0 +
         (J[4] * h.hut[i] + J[5] * h.hus[i] + J[6] * h.hst[i] + J[10] * h.ze[i] + J[11] * h.et[i] +
          J[12] * h.xi[i] + J[13] * h.ze[i] + J[14] * h.et[i] + J[15] * h.xi[i]) /
             9.0 +
         (J[16] * h.hstu[i] + J[17] * h.hut[i] + J[18] * h.hus[i] + J[19] * h.hst[i]) / 27.0;
}

// The rest of GetStab: interior = FCF - KuT * FeCFe^-1 * FeCFhg from the geometry and the
// initial tangent's C1 = C(0,0), C2 = C(0,1), C3 = C(3,3).
FEMORAX_HD inline void interior_matrix(const Geometry& g, double C1, double C2, double C3,
                                       double interior[12][12]) {
  const double* J = g.J;
  const double (&Jinv)[3][3] = g.Jinv;
  const double C4 = C2 + C3;

  // combined jacobian terms
  const double J0789 = 8.0 * (J[0] / 3.0 + J[7] / 5.0 + J[8] / 9.0 + J[9] / 9.0);
  const double J0879 = 8.0 * (J[0] / 3.0 + J[8] / 5.0 + J[7] / 9.0 + J[9] / 9.0);
  const double J0978 = 8.0 * (J[0] / 3.0 + J[9] / 5.0 + J[7] / 9.0 + J[8] / 9.0);
  const double J417 = 8.0 * (J[4] / 9.0 + J[17] / 27.0);
  const double J518 = 8.0 * (J[5] / 9.0 + J[18] / 27.0);
  const double J619 = 8.0 * (J[6] / 9.0 + J[19] / 27.0);
  const double J11215 = 8.0 * (J[1] / 9.0 + J[12] / 15.0 + J[15] / 27.0);
  const double J11512 = 8.0 * (J[1] / 9.0 + J[15] / 15.0 + J[12] / 27.0);
  const double J21114 = 8.0 * (J[2] / 9.0 + J[11] / 15.0 + J[14] / 27.0);
  const double J21411 = 8.0 * (J[2] / 9.0 + J[14] / 15.0 + J[11] / 27.0);
  const double J31013 = 8.0 * (J[3] / 9.0 + J[10] / 15.0 + J[13] / 27.0);
  const double J31310 = 8.0 * (J[3] / 9.0 + J[13] / 15.0 + J[10] / 27.0);
  const double J789 = 8.0 * (J[0] / 9.0 + J[7] / 15.0 + J[8] / 15.0 + J[9] / 27.0);
  const double J897 = 8.0 * (J[0] / 9.0 + J[8] / 15.0 + J[9] / 15.0 + J[7] / 27.0);
  const double J798 = 8.0 * (J[0] / 9.0 + J[7] / 15.0 + J[9] / 15.0 + J[8] / 27.0);
  const double J174 = 8.0 * (J[4] / 27.0 + 64.0 * J[17] / 45.0);
  const double J185 = 8.0 * (J[5] / 27.0 + 64.0 * J[18] / 45.0);
  const double J196 = 8.0 * (J[6] / 27.0 + 64.0 * J[19] / 45.0);
  const double J16 = 8.0 * J[16] / 27.0;

  // define terms for FCF matrix (GetStab, as written there)
  const double HstXX = J0879*Jinv[0][0]*Jinv[0][0] + J0789*Jinv[1][0]*Jinv[1][0] + J619*(Jinv[0][0]*Jinv[1][0] + Jinv[1][0]*Jinv[0][0]);
  const double HstXY = J0879*Jinv[0][0]*Jinv[0][1] + J0789*Jinv[1][0]*Jinv[1][1] + J619*(Jinv[0][0]*Jinv[1][1] + Jinv[1][0]*Jinv[0][1]);
  const double HstXZ = J0879*Jinv[0][0]*Jinv[0][2] + J0789*Jinv[1][0]*Jinv[1][2] + J619*(Jinv[0][0]*Jinv[1][2] + Jinv[1][0]*Jinv[0][2]);
  const double HstYY = J0879*Jinv[0][1]*Jinv[0][1] + J0789*Jinv[1][1]*Jinv[1][1] + J619*(Jinv[0][1]*Jinv[1][1] + Jinv[1][1]*Jinv[0][1]);
  const double HstYZ = J0879*Jinv[0][1]*Jinv[0][2] + J0789*Jinv[1][1]*Jinv[1][2] + J619*(Jinv[0][1]*Jinv[1][2] + Jinv[1][1]*Jinv[0][2]);
  const double HstZZ = J0879*Jinv[0][2]*Jinv[0][2] + J0789*Jinv[1][2]*Jinv[1][2] + J619*(Jinv[0][2]*Jinv[1][2] + Jinv[1][2]*Jinv[0][2]);
  const double HutXX = J0978*Jinv[1][0]*Jinv[1][0] + J0879*Jinv[2][0]*Jinv[2][0] + J417*(Jinv[1][0]*Jinv[2][0] + Jinv[2][0]*Jinv[1][0]);
  const double HutXY = J0978*Jinv[1][0]*Jinv[1][1] + J0879*Jinv[2][0]*Jinv[2][1] + J417*(Jinv[1][0]*Jinv[2][1] + Jinv[2][0]*Jinv[1][1]);
  const double HutXZ = J0978*Jinv[1][0]*Jinv[1][2] + J0879*Jinv[2][0]*Jinv[2][2] + J417*(Jinv[1][0]*Jinv[2][2] + Jinv[2][0]*Jinv[1][2]);
  const double HutYY = J0978*Jinv[1][1]*Jinv[1][1] + J0879*Jinv[2][1]*Jinv[2][1] + J417*(Jinv[1][1]*Jinv[2][1] + Jinv[2][1]*Jinv[1][1]);
  const double HutYZ = J0978*Jinv[1][1]*Jinv[1][2] + J0879*Jinv[2][1]*Jinv[2][2] + J417*(Jinv[1][1]*Jinv[2][2] + Jinv[2][1]*Jinv[1][2]);
  const double HutZZ = J0978*Jinv[1][2]*Jinv[1][2] + J0879*Jinv[2][2]*Jinv[2][2] + J417*(Jinv[1][2]*Jinv[2][2] + Jinv[2][2]*Jinv[1][2]);
  const double HusXX = J0978*Jinv[0][0]*Jinv[0][0] + J0789*Jinv[2][0]*Jinv[2][0] + J518*(Jinv[0][0]*Jinv[2][0] + Jinv[2][0]*Jinv[0][0]);
  const double HusXY = J0978*Jinv[0][0]*Jinv[0][1] + J0789*Jinv[2][0]*Jinv[2][1] + J518*(Jinv[0][0]*Jinv[2][1] + Jinv[2][0]*Jinv[0][1]);
  const double HusXZ = J0978*Jinv[0][0]*Jinv[0][2] + J0789*Jinv[2][0]*Jinv[2][2] + J518*(Jinv[0][0]*Jinv[2][2] + Jinv[2][0]*Jinv[0][2]);
  const double HusYY = J0978*Jinv[0][1]*Jinv[0][1] + J0789*Jinv[2][1]*Jinv[2][1] + J518*(Jinv[0][1]*Jinv[2][1] + Jinv[2][1]*Jinv[0][1]);
  const double HusYZ = J0978*Jinv[0][1]*Jinv[0][2] + J0789*Jinv[2][1]*Jinv[2][2] + J518*(Jinv[0][1]*Jinv[2][2] + Jinv[2][1]*Jinv[0][2]);
  const double HusZZ = J0978*Jinv[0][2]*Jinv[0][2] + J0789*Jinv[2][2]*Jinv[2][2] + J518*(Jinv[0][2]*Jinv[2][2] + Jinv[2][2]*Jinv[0][2]);
  const double HstuXX = J897*Jinv[0][0]*Jinv[0][0] + J798*Jinv[1][0]*Jinv[1][0] + J789*Jinv[2][0]*Jinv[2][0] + J185*(Jinv[0][0]*Jinv[2][0] + Jinv[2][0]*Jinv[0][0])
  + J196*(Jinv[0][0]*Jinv[1][0] + Jinv[1][0]*Jinv[0][0]) + J174*(Jinv[1][0]*Jinv[2][0] + Jinv[2][0]*Jinv[1][0]);
  const double HstuXY = J897*Jinv[0][0]*Jinv[0][1] + J798*Jinv[1][0]*Jinv[1][1] + J789*Jinv[2][0]*Jinv[2][1] + J185*(Jinv[0][0]*Jinv[2][1] + Jinv[2][0]*Jinv[0][1])
  + J196*(Jinv[0][0]*Jinv[1][1] + Jinv[1][0]*Jinv[0][1]) + J174*(Jinv[1][0]*Jinv[2][1] + Jinv[2][0]*Jinv[1][1]);
  const double HstuXZ = J897*Jinv[0][0]*Jinv[0][2] + J798*Jinv[1][0]*Jinv[1][2] + J789*Jinv[2][0]*Jinv[2][2] + J185*(Jinv[0][0]*Jinv[2][2] + Jinv[2][0]*Jinv[0][2])
  + J196*(Jinv[0][0]*Jinv[1][2] + Jinv[1][0]*Jinv[0][2]) + J174*(Jinv[1][0]*Jinv[2][2] + Jinv[2][0]*Jinv[1][2]);
  const double HstuYY = J897*Jinv[0][1]*Jinv[0][1] + J798*Jinv[1][1]*Jinv[1][1] + J789*Jinv[2][1]*Jinv[2][1] + J185*(Jinv[0][1]*Jinv[2][1] + Jinv[2][1]*Jinv[0][1])
  + J196*(Jinv[0][1]*Jinv[1][1] + Jinv[1][1]*Jinv[0][1]) + J174*(Jinv[1][1]*Jinv[2][1] + Jinv[2][1]*Jinv[1][1]);
  const double HstuYZ = J897*Jinv[0][1]*Jinv[0][2] + J798*Jinv[1][1]*Jinv[1][2] + J789*Jinv[2][1]*Jinv[2][2] + J185*(Jinv[0][1]*Jinv[2][2] + Jinv[2][1]*Jinv[0][2])
  + J196*(Jinv[0][1]*Jinv[1][2] + Jinv[1][1]*Jinv[0][2]) + J174*(Jinv[1][1]*Jinv[2][2] + Jinv[2][1]*Jinv[1][2]);
  const double HstuZZ = J897*Jinv[0][2]*Jinv[0][2] + J798*Jinv[1][2]*Jinv[1][2] + J789*Jinv[2][2]*Jinv[2][2] + J185*(Jinv[0][2]*Jinv[2][2] + Jinv[2][2]*Jinv[0][2])
  + J196*(Jinv[0][2]*Jinv[1][2] + Jinv[1][2]*Jinv[0][2]) + J174*(Jinv[1][2]*Jinv[2][2] + Jinv[2][2]*Jinv[1][2]);
  const double IttXX = J0879*Jinv[0][0]*Jinv[2][0] + J417*Jinv[0][0]*Jinv[1][0] + J518*Jinv[1][0]*Jinv[1][0] + J619*Jinv[1][0]*Jinv[2][0];
  const double IttXY = J0879*Jinv[0][0]*Jinv[2][1] + J417*Jinv[0][0]*Jinv[1][1] + J518*Jinv[1][0]*Jinv[1][1] + J619*Jinv[1][0]*Jinv[2][1];
  const double IttXZ = J0879*Jinv[0][0]*Jinv[2][2] + J417*Jinv[0][0]*Jinv[1][2] + J518*Jinv[1][0]*Jinv[1][2] + J619*Jinv[1][0]*Jinv[2][2];
  const double IttYX = J0879*Jinv[0][1]*Jinv[2][0] + J417*Jinv[0][1]*Jinv[1][0] + J518*Jinv[1][1]*Jinv[1][0] + J619*Jinv[1][1]*Jinv[2][0];
  const double IttYY = J0879*Jinv[0][1]*Jinv[2][1] + J417*Jinv[0][1]*Jinv[1][1] + J518*Jinv[1][1]*Jinv[1][1] + J619*Jinv[1][1]*Jinv[2][1];
  const double IttYZ = J0879*Jinv[0][1]*Jinv[2][2] + J417*Jinv[0][1]*Jinv[1][2] + J518*Jinv[1][1]*Jinv[1][2] + J619*Jinv[1][1]*Jinv[2][2];
  const double IttZX = J0879*Jinv[0][2]*Jinv[2][0] + J417*Jinv[0][2]*Jinv[1][0] + J518*Jinv[1][2]*Jinv[1][0] + J619*Jinv[1][2]*Jinv[2][0];
  const double IttZY = J0879*Jinv[0][2]*Jinv[2][1] + J417*Jinv[0][2]*Jinv[1][1] + J518*Jinv[1][2]*Jinv[1][1] + J619*Jinv[1][2]*Jinv[2][1];
  const double IttZZ = J0879*Jinv[0][2]*Jinv[2][2] + J417*Jinv[0][2]*Jinv[1][2] + J518*Jinv[1][2]*Jinv[1][2] + J619*Jinv[1][2]*Jinv[2][2];
  const double IssXX = J0789*Jinv[1][0]*Jinv[2][0] + J417*Jinv[0][0]*Jinv[0][0] + J518*Jinv[1][0]*Jinv[0][0] + J619*Jinv[0][0]*Jinv[2][0];
  const double IssXY = J0789*Jinv[1][0]*Jinv[2][1] + J417*Jinv[0][0]*Jinv[0][1] + J518*Jinv[1][0]*Jinv[0][1] + J619*Jinv[0][0]*Jinv[2][1];
  const double IssXZ = J0789*Jinv[1][0]*Jinv[2][2] + J417*Jinv[0][0]*Jinv[0][2] + J518*Jinv[1][0]*Jinv[0][2] + J619*Jinv[0][0]*Jinv[2][2];
  const double IssYX = J0789*Jinv[1][1]*Jinv[2][0] + J417*Jinv[0][1]*Jinv[0][0] + J518*Jinv[1][1]*Jinv[0][0] + J619*Jinv[0][1]*Jinv[2][0];
  const double IssYY = J0789*Jinv[1][1]*Jinv[2][1] + J417*Jinv[0][1]*Jinv[0][1] + J518*Jinv[1][1]*Jinv[0][1] + J619*Jinv[0][1]*Jinv[2][1];
  const double IssYZ = J0789*Jinv[1][1]*Jinv[2][2] + J417*Jinv[0][1]*Jinv[0][2] + J518*Jinv[1][1]*Jinv[0][2] + J619*Jinv[0][1]*Jinv[2][2];
  const double IssZX = J0789*Jinv[1][2]*Jinv[2][0] + J417*Jinv[0][2]*Jinv[0][0] + J518*Jinv[1][2]*Jinv[0][0] + J619*Jinv[0][2]*Jinv[2][0];
  const double IssZY = J0789*Jinv[1][2]*Jinv[2][1] + J417*Jinv[0][2]*Jinv[0][1] + J518*Jinv[1][2]*Jinv[0][1] + J619*Jinv[0][2]*Jinv[2][1];
  const double IssZZ = J0789*Jinv[1][2]*Jinv[2][2] + J417*Jinv[0][2]*Jinv[0][2] + J518*Jinv[1][2]*Jinv[0][2] + J619*Jinv[0][2]*Jinv[2][2];
  const double IuuXX = J0978*Jinv[1][0]*Jinv[0][0] + J417*Jinv[2][0]*Jinv[0][0] + J518*Jinv[1][0]*Jinv[2][0] + J619*Jinv[2][0]*Jinv[2][0];
  const double IuuXY = J0978*Jinv[1][0]*Jinv[0][1] + J417*Jinv[2][0]*Jinv[0][1] + J518*Jinv[1][0]*Jinv[2][1] + J619*Jinv[2][0]*Jinv[2][1];
  const double IuuXZ = J0978*Jinv[1][0]*Jinv[0][2] + J417*Jinv[2][0]*Jinv[0][2] + J518*Jinv[1][0]*Jinv[2][2] + J619*Jinv[2][0]*Jinv[2][2];
  const double IuuYX = J0978*Jinv[1][1]*Jinv[0][0] + J417*Jinv[2][1]*Jinv[0][0] + J518*Jinv[1][1]*Jinv[2][0] + J619*Jinv[2][1]*Jinv[2][0];
  const double IuuYY = J0978*Jinv[1][1]*Jinv[0][1] + J417*Jinv[2][1]*Jinv[0][1] + J518*Jinv[1][1]*Jinv[2][1] + J619*Jinv[2][1]*Jinv[2][1];
  const double IuuYZ = J0978*Jinv[1][1]*Jinv[0][2] + J417*Jinv[2][1]*Jinv[0][2] + J518*Jinv[1][1]*Jinv[2][2] + J619*Jinv[2][1]*Jinv[2][2];
  const double IuuZX = J0978*Jinv[1][2]*Jinv[0][0] + J417*Jinv[2][2]*Jinv[0][0] + J518*Jinv[1][2]*Jinv[2][0] + J619*Jinv[2][2]*Jinv[2][0];
  const double IuuZY = J0978*Jinv[1][2]*Jinv[0][1] + J417*Jinv[2][2]*Jinv[0][1] + J518*Jinv[1][2]*Jinv[2][1] + J619*Jinv[2][2]*Jinv[2][1];
  const double IuuZZ = J0978*Jinv[1][2]*Jinv[0][2] + J417*Jinv[2][2]*Jinv[0][2] + J518*Jinv[1][2]*Jinv[2][2] + J619*Jinv[2][2]*Jinv[2][2];
  const double IstXX = J31013*Jinv[0][0]*Jinv[0][0] + J31310*Jinv[1][0]*Jinv[1][0] + J21411*Jinv[2][0]*Jinv[1][0] + J11512*Jinv[2][0]*Jinv[0][0] + J16*(Jinv[0][0]*Jinv[1][0] + Jinv[1][0]*Jinv[0][0]);
  const double IstXY = J31013*Jinv[0][0]*Jinv[0][1] + J31310*Jinv[1][0]*Jinv[1][1] + J21411*Jinv[2][0]*Jinv[1][1] + J11512*Jinv[2][0]*Jinv[0][1] + J16*(Jinv[0][0]*Jinv[1][1] + Jinv[1][0]*Jinv[0][1]);
  const double IstXZ = J31013*Jinv[0][0]*Jinv[0][2] + J31310*Jinv[1][0]*Jinv[1][2] + J21411*Jinv[2][0]*Jinv[1][2] + J11512*Jinv[2][0]*Jinv[0][2] + J16*(Jinv[0][0]*Jinv[1][2] + Jinv[1][0]*Jinv[0][2]);
  const double IstYX = J31013*Jinv[0][1]*Jinv[0][0] + J31310*Jinv[1][1]*Jinv[1][0] + J21411*Jinv[2][1]*Jinv[1][0] + J11512*Jinv[2][1]*Jinv[0][0] + J16*(Jinv[0][1]*Jinv[1][0] + Jinv[1][1]*Jinv[0][0]);
  const double IstYY = J31013*Jinv[0][1]*Jinv[0][1] + J31310*Jinv[1][1]*Jinv[1][1] + J21411*Jinv[2][1]*Jinv[1][1] + J11512*Jinv[2][1]*Jinv[0][1] + J16*(Jinv[0][1]*Jinv[1][1] + Jinv[1][1]*Jinv[0][1]);
  const double IstYZ = J31013*Jinv[0][1]*Jinv[0][2] + J31310*Jinv[1][1]*Jinv[1][2] + J21411*Jinv[2][1]*Jinv[1][2] + J11512*Jinv[2][1]*Jinv[0][2] + J16*(Jinv[0][1]*Jinv[1][2] + Jinv[1][1]*Jinv[0][2]);
  const double IstZX = J31013*Jinv[0][2]*Jinv[0][0] + J31310*Jinv[1][2]*Jinv[1][0] + J21411*Jinv[2][2]*Jinv[1][0] + J11512*Jinv[2][2]*Jinv[0][0] + J16*(Jinv[0][2]*Jinv[1][0] + Jinv[1][2]*Jinv[0][0]);
  const double IstZY = J31013*Jinv[0][2]*Jinv[0][1] + J31310*Jinv[1][2]*Jinv[1][1] + J21411*Jinv[2][2]*Jinv[1][1] + J11512*Jinv[2][2]*Jinv[0][1] + J16*(Jinv[0][2]*Jinv[1][1] + Jinv[1][2]*Jinv[0][1]);
  const double IstZZ = J31013*Jinv[0][2]*Jinv[0][2] + J31310*Jinv[1][2]*Jinv[1][2] + J21411*Jinv[2][2]*Jinv[1][2] + J11512*Jinv[2][2]*Jinv[0][2] + J16*(Jinv[0][2]*Jinv[1][2] + Jinv[1][2]*Jinv[0][2]);
  const double IutXX = J21114*Jinv[0][0]*Jinv[1][0] + J31013*Jinv[0][0]*Jinv[2][0] + J11215*Jinv[1][0]*Jinv[1][0] + J11512*Jinv[2][0]*Jinv[2][0] + J16*(Jinv[1][0]*Jinv[2][0] + Jinv[2][0]*Jinv[1][0]);
  const double IutXY = J21114*Jinv[0][0]*Jinv[1][1] + J31013*Jinv[0][0]*Jinv[2][1] + J11215*Jinv[1][0]*Jinv[1][1] + J11512*Jinv[2][0]*Jinv[2][1] + J16*(Jinv[1][0]*Jinv[2][1] + Jinv[2][0]*Jinv[1][1]);
  const double IutXZ = J21114*Jinv[0][0]*Jinv[1][2] + J31013*Jinv[0][0]*Jinv[2][2] + J11215*Jinv[1][0]*Jinv[1][2] + J11512*Jinv[2][0]*Jinv[2][2] + J16*(Jinv[1][0]*Jinv[2][2] + Jinv[2][0]*Jinv[1][2]);
  const double IutYX = J21114*Jinv[0][1]*Jinv[1][0] + J31013*Jinv[0][1]*Jinv[2][0] + J11215*Jinv[1][1]*Jinv[1][0] + J11512*Jinv[2][1]*Jinv[2][0] + J16*(Jinv[1][1]*Jinv[2][0] + Jinv[2][1]*Jinv[1][0]);
  const double IutYY = J21114*Jinv[0][1]*Jinv[1][1] + J31013*Jinv[0][1]*Jinv[2][1] + J11215*Jinv[1][1]*Jinv[1][1] + J11512*Jinv[2][1]*Jinv[2][1] + J16*(Jinv[1][1]*Jinv[2][1] + Jinv[2][1]*Jinv[1][1]);
  const double IutYZ = J21114*Jinv[0][1]*Jinv[1][2] + J31013*Jinv[0][1]*Jinv[2][2] + J11215*Jinv[1][1]*Jinv[1][2] + J11512*Jinv[2][1]*Jinv[2][2] + J16*(Jinv[1][1]*Jinv[2][2] + Jinv[2][1]*Jinv[1][2]);
  const double IutZX = J21114*Jinv[0][2]*Jinv[1][0] + J31013*Jinv[0][2]*Jinv[2][0] + J11215*Jinv[1][2]*Jinv[1][0] + J11512*Jinv[2][2]*Jinv[2][0] + J16*(Jinv[1][2]*Jinv[2][0] + Jinv[2][2]*Jinv[1][0]);
  const double IutZY = J21114*Jinv[0][2]*Jinv[1][1] + J31013*Jinv[0][2]*Jinv[2][1] + J11215*Jinv[1][2]*Jinv[1][1] + J11512*Jinv[2][2]*Jinv[2][1] + J16*(Jinv[1][2]*Jinv[2][1] + Jinv[2][2]*Jinv[1][1]);
  const double IutZZ = J21114*Jinv[0][2]*Jinv[1][2] + J31013*Jinv[0][2]*Jinv[2][2] + J11215*Jinv[1][2]*Jinv[1][2] + J11512*Jinv[2][2]*Jinv[2][2] + J16*(Jinv[1][2]*Jinv[2][2] + Jinv[2][2]*Jinv[1][2]);
  const double IusXX = J21114*Jinv[0][0]*Jinv[0][0] + J11215*Jinv[1][0]*Jinv[0][0] + J31310*Jinv[1][0]*Jinv[2][0] + J21411*Jinv[2][0]*Jinv[2][0] + J16*(Jinv[0][0]*Jinv[2][0] + Jinv[2][0]*Jinv[0][0]);
  const double IusXY = J21114*Jinv[0][0]*Jinv[0][1] + J11215*Jinv[1][0]*Jinv[0][1] + J31310*Jinv[1][0]*Jinv[2][1] + J21411*Jinv[2][0]*Jinv[2][1] + J16*(Jinv[0][0]*Jinv[2][1] + Jinv[2][0]*Jinv[0][1]);
  const double IusXZ = J21114*Jinv[0][0]*Jinv[0][2] + J11215*Jinv[1][0]*Jinv[0][2] + J31310*Jinv[1][0]*Jinv[2][2] + J21411*Jinv[2][0]*Jinv[2][2] + J16*(Jinv[0][0]*Jinv[2][2] + Jinv[2][0]*Jinv[0][2]);
  const double IusYX = J21114*Jinv[0][1]*Jinv[0][0] + J11215*Jinv[1][1]*Jinv[0][0] + J31310*Jinv[1][1]*Jinv[2][0] + J21411*Jinv[2][1]*Jinv[2][0] + J16*(Jinv[0][1]*Jinv[2][0] + Jinv[2][1]*Jinv[0][0]);
  const double IusYY = J21114*Jinv[0][1]*Jinv[0][1] + J11215*Jinv[1][1]*Jinv[0][1] + J31310*Jinv[1][1]*Jinv[2][1] + J21411*Jinv[2][1]*Jinv[2][1] + J16*(Jinv[0][1]*Jinv[2][1] + Jinv[2][1]*Jinv[0][1]);
  const double IusYZ = J21114*Jinv[0][1]*Jinv[0][2] + J11215*Jinv[1][1]*Jinv[0][2] + J31310*Jinv[1][1]*Jinv[2][2] + J21411*Jinv[2][1]*Jinv[2][2] + J16*(Jinv[0][1]*Jinv[2][2] + Jinv[2][1]*Jinv[0][2]);
  const double IusZX = J21114*Jinv[0][2]*Jinv[0][0] + J11215*Jinv[1][2]*Jinv[0][0] + J31310*Jinv[1][2]*Jinv[2][0] + J21411*Jinv[2][2]*Jinv[2][0] + J16*(Jinv[0][2]*Jinv[2][0] + Jinv[2][2]*Jinv[0][0]);
  const double IusZY = J21114*Jinv[0][2]*Jinv[0][1] + J11215*Jinv[1][2]*Jinv[0][1] + J31310*Jinv[1][2]*Jinv[2][1] + J21411*Jinv[2][2]*Jinv[2][1] + J16*(Jinv[0][2]*Jinv[2][1] + Jinv[2][2]*Jinv[0][1]);
  const double IusZZ = J21114*Jinv[0][2]*Jinv[0][2] + J11215*Jinv[1][2]*Jinv[0][2] + J31310*Jinv[1][2]*Jinv[2][2] + J21411*Jinv[2][2]*Jinv[2][2] + J16*(Jinv[0][2]*Jinv[2][2] + Jinv[2][2]*Jinv[0][2]);

  // define the 12x12 matrix FCF in 3x3 blocks (the active blocks of GetStab)
  double FCF[12][12];
  FCF[0][0] = C1*HstXX + C3*(HstYY + HstZZ);
  FCF[0][1] = C4*HstXY;
  FCF[0][2] = C4*HstXZ;
  FCF[1][0] = FCF[0][1];
  FCF[1][1] = C1*HstYY + C3*(HstXX + HstZZ);
  FCF[1][2] = C4*HstYZ;
  FCF[2][0] = FCF[0][2];
  FCF[2][1] = FCF[1][2];
  FCF[2][2] = C1*HstZZ + C3*(HstYY + HstXX);
  FCF[0][3] = C1*IttXX + C3*(IttYY + IttZZ);
  FCF[0][4] = C2*IttXY + C3*IttYX;
  FCF[0][5] = C2*IttXZ + C3*IttZX;
  FCF[1][3] = C2*IttYX + C3*IttXY;
  FCF[1][4] = C1*IttYY + C3*(IttXX + IttZZ);
  FCF[1][5] = C2*IttYZ + C3*IttZY;
  FCF[2][3] = C2*IttZX + C3*IttXZ;
  FCF[2][4] = C2*IttZY + C3*IttYZ;
  FCF[2][5] = C1*IttZZ + C3*(IttYY + IttXX);
  FCF[0][6] = C1*IssXX + C3*(IssYY + IssZZ);
  FCF[0][7] = C2*IssXY + C3*IssYX;
  FCF[0][8] = C2*IssXZ + C3*IssZX;
  FCF[1][6] = C2*IssYX + C3*IssXY;
  FCF[1][7] = C1*IssYY + C3*(IssXX + IssZZ);
  FCF[1][8] = C2*IssYZ + C3*IssZY;
  FCF[2][6] = C2*IssZX + C3*IssXZ;
  FCF[2][7] = C2*IssZY + C3*IssYZ;
  FCF[2][8] = C1*IssZZ + C3*(IssYY + IssXX);
  FCF[0][9]  = C3*(IstYY + IstZZ);
  FCF[0][10] = C3*IstXY;
  FCF[0][11] = C3*IstXZ;
  FCF[1][9]  = C3*IstYX;
  FCF[1][10] = C3*(IstXX + IstZZ);
  FCF[1][11] = C3*IstYZ;
  FCF[2][9]  = C3*IstZX;
  FCF[2][10] = C3*IstZY;
  FCF[2][11] = C3*(IstYY + IstXX);
  FCF[3][0] = C1*IttXX + C3*(IttYY + IttZZ);
  FCF[3][1] = C2*IttYX + C3*IttXY;
  FCF[3][2] = C2*IttZX + C3*IttXZ;
  FCF[4][0] = C2*IttXY + C3*IttYX;
  FCF[4][1] = C1*IttYY + C3*(IttXX + IttZZ);
  FCF[4][2] = C2*IttZY + C3*IttYZ;
  FCF[5][0] = C2*IttXZ + C3*IttZX;
  FCF[5][1] = C2*IttYZ + C3*IttZY;
  FCF[5][2] = C1*IttZZ + C3*(IttYY + IttXX);
  FCF[3][3] = C1*HutXX + C3*(HutYY + HutZZ);
  FCF[3][4] = C4*HutXY;
  FCF[3][5] = C4*HutXZ;
  FCF[4][3] = FCF[3][4];
  FCF[4][4] = C1*HutYY + C3*(HutXX + HutZZ);
  FCF[4][5] = C4*HutYZ;
  FCF[5][3] = FCF[3][5];
  FCF[5][4] = FCF[4][5];
  FCF[5][5] = C1*HutZZ + C3*(HutYY + HutXX);
  FCF[3][6] = C1*IuuXX + C3*(IuuYY + IuuZZ);
  FCF[3][7] = C2*IuuXY + C3*IuuYX;
  FCF[3][8] = C2*IuuXZ + C3*IuuZX;
  FCF[4][6] = C2*IuuYX + C3*IuuXY;
  FCF[4][7] = C1*IuuYY + C3*(IuuXX + IuuZZ);
  FCF[4][8] = C2*IuuYZ + C3*IuuZY;
  FCF[5][6] = C2*IuuZX + C3*IuuXZ;
  FCF[5][7] = C2*IuuZY + C3*IuuYZ;
  FCF[5][8] = C1*IuuZZ + C3*(IuuYY + IuuXX);
  FCF[3][9]  = C3*(IutYY + IutZZ);
  FCF[3][10] = C3*IutXY;
  FCF[3][11] = C3*IutXZ;
  FCF[4][9]  = C3*IutYX;
  FCF[4][10] = C3*(IutXX + IutZZ);
  FCF[4][11] = C3*IutYZ;
  FCF[5][9]  = C3*IutZX;
  FCF[5][10] = C3*IutZY;
  FCF[5][11] = C3*(IutYY + IutXX);
  FCF[6][0] = C1*IssXX + C3*(IssYY + IssZZ);
  FCF[6][1] = C2*IssYX + C3*IssXY;
  FCF[6][2] = C2*IssZX + C3*IssXZ;
  FCF[7][0] = C2*IssXY + C3*IssYX;
  FCF[7][1] = C1*IssYY + C3*(IssXX + IssZZ);
  FCF[7][2] = C2*IssZY + C3*IssYZ;
  FCF[8][0] = C2*IssXZ + C3*IssZX;
  FCF[8][1] = C2*IssYZ + C3*IssZY;
  FCF[8][2] = C1*IssZZ + C3*(IssYY + IssXX);
  FCF[6][3] = C1*IuuXX + C3*(IuuYY + IuuZZ);
  FCF[6][4] = C2*IuuYX + C3*IuuXY;
  FCF[6][5] = C2*IuuZX + C3*IuuXZ;
  FCF[7][3] = C2*IuuXY + C3*IuuYX;
  FCF[7][4] = C1*IuuYY + C3*(IuuXX + IuuZZ);
  FCF[7][5] = C2*IuuZY + C3*IuuYZ;
  FCF[8][3] = C2*IuuXZ + C3*IuuZX;
  FCF[8][4] = C2*IuuYZ + C3*IuuZY;
  FCF[8][5] = C1*IuuZZ + C3*(IuuYY + IuuXX);
  FCF[6][6] = C1*HusXX + C3*(HusYY + HusZZ);
  FCF[6][7] = C4*HusXY;
  FCF[6][8] = C4*HusXZ;
  FCF[7][6] = FCF[6][7];
  FCF[7][7] = C1*HusYY + C3*(HusXX + HusZZ);
  FCF[7][8] = C4*HusYZ;
  FCF[8][6] = FCF[6][8];
  FCF[8][7] = FCF[7][8];
  FCF[8][8] = C1*HusZZ + C3*(HusYY + HusXX);
  FCF[6][9]  = C3*(IusYY + IusZZ);
  FCF[6][10] = C3*IusXY;
  FCF[6][11] = C3*IusXZ;
  FCF[7][9]  = C3*IusYX;
  FCF[7][10] = C3*(IusXX + IusZZ);
  FCF[7][11] = C3*IusYZ;
  FCF[8][9]  = C3*IusZX;
  FCF[8][10] = C3*IusZY;
  FCF[8][11] = C3*(IusYY + IusXX);
  FCF[9][0]  = C3*(IstYY + IstZZ);
  FCF[9][1]  = C3*IstYX;
  FCF[9][2]  = C3*IstZX;
  FCF[10][0] = C3*IstXY;
  FCF[10][1] = C3*(IstXX + IstZZ);
  FCF[10][2] = C3*IstZY;
  FCF[11][0] = C3*IstXZ;
  FCF[11][1] = C3*IstYZ;
  FCF[11][2] = C3*(IstYY + IstXX);
  FCF[9][3]  = C3*(IutYY + IutZZ);
  FCF[9][4]  = C3*IutYX;
  FCF[9][5]  = C3*IutZX;
  FCF[10][3] = C3*IutXY;
  FCF[10][4] = C3*(IutXX + IutZZ);
  FCF[10][5] = C3*IutZY;
  FCF[11][3] = C3*IutXZ;
  FCF[11][4] = C3*IutYZ;
  FCF[11][5] = C3*(IutYY + IutXX);
  FCF[9][6]  = C3*(IusYY + IusZZ);
  FCF[9][7]  = C3*IusYX;
  FCF[9][8]  = C3*IusZX;
  FCF[10][6] = C3*IusXY;
  FCF[10][7] = C3*(IusXX + IusZZ);
  FCF[10][8] = C3*IusZY;
  FCF[11][6] = C3*IusXZ;
  FCF[11][7] = C3*IusYZ;
  FCF[11][8] = C3*(IusYY + IusXX);
  FCF[9][9]   = C3*(HstuYY + HstuZZ);
  FCF[9][10]  = C3*HstuXY;
  FCF[9][11]  = C3*HstuXZ;
  FCF[10][9]  = FCF[9][10];
  FCF[10][10] = C3*(HstuXX + HstuZZ);
  FCF[10][11] = C3*HstuYZ;
  FCF[11][9]  = FCF[9][11];
  FCF[11][10] = FCF[10][11];
  FCF[11][11] = C3*(HstuYY + HstuXX);

  // enhanced strain portion of the stabilization stiffness matrix: the constitutive
  // coefficients
  const double CssXX = C1*Jinv[0][0]*Jinv[0][0] + C3*(Jinv[0][1]*Jinv[0][1] + Jinv[0][2]*Jinv[0][2]);
  const double CssXY = C4*Jinv[0][0]*Jinv[0][1];
  const double CssXZ = C4*Jinv[0][0]*Jinv[0][2];
  const double CssYY = C1*Jinv[0][1]*Jinv[0][1] + C3*(Jinv[0][0]*Jinv[0][0] + Jinv[0][2]*Jinv[0][2]);
  const double CssYZ = C4*Jinv[0][1]*Jinv[0][2];
  const double CssZZ = C1*Jinv[0][2]*Jinv[0][2] + C3*(Jinv[0][0]*Jinv[0][0] + Jinv[0][1]*Jinv[0][1]);
  const double CttXX = C1*Jinv[1][0]*Jinv[1][0] + C3*(Jinv[1][1]*Jinv[1][1] + Jinv[1][2]*Jinv[1][2]);
  const double CttXY = C4*Jinv[1][0]*Jinv[1][1];
  const double CttXZ = C4*Jinv[1][0]*Jinv[1][2];
  const double CttYY = C1*Jinv[1][1]*Jinv[1][1] + C3*(Jinv[1][0]*Jinv[1][0] + Jinv[1][2]*Jinv[1][2]);
  const double CttYZ = C4*Jinv[1][1]*Jinv[1][2];
  const double CttZZ = C1*Jinv[1][2]*Jinv[1][2] + C3*(Jinv[1][0]*Jinv[1][0] + Jinv[1][1]*Jinv[1][1]);
  const double CuuXX = C1*Jinv[2][0]*Jinv[2][0] + C3*(Jinv[2][1]*Jinv[2][1] + Jinv[2][2]*Jinv[2][2]);
  const double CuuXY = C4*Jinv[2][0]*Jinv[2][1];
  const double CuuXZ = C4*Jinv[2][0]*Jinv[2][2];
  const double CuuYY = C1*Jinv[2][1]*Jinv[2][1] + C3*(Jinv[2][0]*Jinv[2][0] + Jinv[2][2]*Jinv[2][2]);
  const double CuuYZ = C4*Jinv[2][1]*Jinv[2][2];
  const double CuuZZ = C1*Jinv[2][2]*Jinv[2][2] + C3*(Jinv[2][0]*Jinv[2][0] + Jinv[2][1]*Jinv[2][1]);
  const double CstXX = C1*Jinv[0][0]*Jinv[1][0] + C3*(Jinv[0][1]*Jinv[1][1] + Jinv[0][2]*Jinv[1][2]);
  const double CstXY = C2*Jinv[0][0]*Jinv[1][1] + C3*Jinv[0][1]*Jinv[1][0];
  const double CstXZ = C2*Jinv[0][0]*Jinv[1][2] + C3*Jinv[0][2]*Jinv[1][0];
  const double CstYX = C2*Jinv[0][1]*Jinv[1][0] + C3*Jinv[0][0]*Jinv[1][1];
  const double CstYY = C1*Jinv[0][1]*Jinv[1][1] + C3*(Jinv[0][0]*Jinv[1][0] + Jinv[0][2]*Jinv[1][2]);
  const double CstYZ = C2*Jinv[0][1]*Jinv[1][2] + C3*Jinv[0][2]*Jinv[1][1];
  const double CstZX = C2*Jinv[0][2]*Jinv[1][0] + C3*Jinv[0][0]*Jinv[1][2];
  const double CstZY = C2*Jinv[0][2]*Jinv[1][1] + C3*Jinv[0][1]*Jinv[1][2];
  const double CstZZ = C1*Jinv[0][2]*Jinv[1][2] + C3*(Jinv[0][0]*Jinv[1][0] + Jinv[0][1]*Jinv[1][1]);
  const double CsuXX = C1*Jinv[0][0]*Jinv[2][0] + C3*(Jinv[0][1]*Jinv[2][1] + Jinv[0][2]*Jinv[2][2]);
  const double CsuXY = C2*Jinv[0][0]*Jinv[2][1] + C3*Jinv[0][1]*Jinv[2][0];
  const double CsuXZ = C2*Jinv[0][0]*Jinv[2][2] + C3*Jinv[0][2]*Jinv[2][0];
  const double CsuYX = C2*Jinv[0][1]*Jinv[2][0] + C3*Jinv[0][0]*Jinv[2][1];
  const double CsuYY = C1*Jinv[0][1]*Jinv[2][1] + C3*(Jinv[0][0]*Jinv[2][0] + Jinv[0][2]*Jinv[2][2]);
  const double CsuYZ = C2*Jinv[0][1]*Jinv[2][2] + C3*Jinv[0][2]*Jinv[2][1];
  const double CsuZX = C2*Jinv[0][2]*Jinv[2][0] + C3*Jinv[0][0]*Jinv[2][2];
  const double CsuZY = C2*Jinv[0][2]*Jinv[2][1] + C3*Jinv[0][1]*Jinv[2][2];
  const double CsuZZ = C1*Jinv[0][2]*Jinv[2][2] + C3*(Jinv[0][0]*Jinv[2][0] + Jinv[0][1]*Jinv[2][1]);
  const double CtuXX = C1*Jinv[1][0]*Jinv[2][0] + C3*(Jinv[1][1]*Jinv[2][1] + Jinv[1][2]*Jinv[2][2]);
  const double CtuXY = C2*Jinv[1][0]*Jinv[2][1] + C3*Jinv[1][1]*Jinv[2][0];
  const double CtuXZ = C2*Jinv[1][0]*Jinv[2][2] + C3*Jinv[1][2]*Jinv[2][0];
  const double CtuYX = C2*Jinv[1][1]*Jinv[2][0] + C3*Jinv[1][0]*Jinv[2][1];
  const double CtuYY = C1*Jinv[1][1]*Jinv[2][1] + C3*(Jinv[1][0]*Jinv[2][0] + Jinv[1][2]*Jinv[2][2]);
  const double CtuYZ = C2*Jinv[1][1]*Jinv[2][2] + C3*Jinv[1][2]*Jinv[2][1];
  const double CtuZX = C2*Jinv[1][2]*Jinv[2][0] + C3*Jinv[1][0]*Jinv[2][2];
  const double CtuZY = C2*Jinv[1][2]*Jinv[2][1] + C3*Jinv[1][1]*Jinv[2][2];
  const double CtuZZ = C1*Jinv[1][2]*Jinv[2][2] + C3*(Jinv[1][0]*Jinv[2][0] + Jinv[1][1]*Jinv[2][1]);

  // the integrated matrix [FenT][C][Fen] and its inverse
  double FeCFe[9][9];
  FeCFe[0][0] = CssXX*J0789;
  FeCFe[0][1] = CssXY*J0789;
  FeCFe[0][2] = CssXZ*J0789;
  FeCFe[1][0] = FeCFe[0][1];
  FeCFe[1][1] = CssYY*J0789;
  FeCFe[1][2] = CssYZ*J0789;
  FeCFe[2][0] = FeCFe[0][2];
  FeCFe[2][1] = FeCFe[1][2];
  FeCFe[2][2] = CssZZ*J0789;
  FeCFe[0][3] = CstXX*J619;
  FeCFe[0][4] = CstXY*J619;
  FeCFe[0][5] = CstXZ*J619;
  FeCFe[1][3] = CstYX*J619;
  FeCFe[1][4] = CstYY*J619;
  FeCFe[1][5] = CstYZ*J619;
  FeCFe[2][3] = CstZX*J619;
  FeCFe[2][4] = CstZY*J619;
  FeCFe[2][5] = CstZZ*J619;
  FeCFe[0][6] = CsuXX*J518;
  FeCFe[0][7] = CsuXY*J518;
  FeCFe[0][8] = CsuXZ*J518;
  FeCFe[1][6] = CsuYX*J518;
  FeCFe[1][7] = CsuYY*J518;
  FeCFe[1][8] = CsuYZ*J518;
  FeCFe[2][6] = CsuZX*J518;
  FeCFe[2][7] = CsuZY*J518;
  FeCFe[2][8] = CsuZZ*J518;
  FeCFe[3][0] = CstXX*J619;
  FeCFe[3][1] = CstYX*J619;
  FeCFe[3][2] = CstZX*J619;
  FeCFe[4][0] = CstXY*J619;
  FeCFe[4][1] = CstYY*J619;
  FeCFe[4][2] = CstZY*J619;
  FeCFe[5][0] = CstXZ*J619;
  FeCFe[5][1] = CstYZ*J619;
  FeCFe[5][2] = CstZZ*J619;
  FeCFe[3][3] = CttXX*J0879;
  FeCFe[3][4] = CttXY*J0879;
  FeCFe[3][5] = CttXZ*J0879;
  FeCFe[4][3] = FeCFe[3][4];
  FeCFe[4][4] = CttYY*J0879;
  FeCFe[4][5] = CttYZ*J0879;
  FeCFe[5][3] = FeCFe[3][5];
  FeCFe[5][4] = FeCFe[4][5];
  FeCFe[5][5] = CttZZ*J0879;
  FeCFe[3][6] = CtuXX*J417;
  FeCFe[3][7] = CtuXY*J417;
  FeCFe[3][8] = CtuXZ*J417;
  FeCFe[4][6] = CtuYX*J417;
  FeCFe[4][7] = CtuYY*J417;
  FeCFe[4][8] = CtuYZ*J417;
  FeCFe[5][6] = CtuZX*J417;
  FeCFe[5][7] = CtuZY*J417;
  FeCFe[5][8] = CtuZZ*J417;
  FeCFe[6][0] = CsuXX*J518;
  FeCFe[6][1] = CsuYX*J518;
  FeCFe[6][2] = CsuZX*J518;
  FeCFe[7][0] = CsuXY*J518;
  FeCFe[7][1] = CsuYY*J518;
  FeCFe[7][2] = CsuZY*J518;
  FeCFe[8][0] = CsuXZ*J518;
  FeCFe[8][1] = CsuYZ*J518;
  FeCFe[8][2] = CsuZZ*J518;
  FeCFe[6][3] = CtuXX*J417;
  FeCFe[6][4] = CtuYX*J417;
  FeCFe[6][5] = CtuZX*J417;
  FeCFe[7][3] = CtuXY*J417;
  FeCFe[7][4] = CtuYY*J417;
  FeCFe[7][5] = CtuZY*J417;
  FeCFe[8][3] = CtuXZ*J417;
  FeCFe[8][4] = CtuYZ*J417;
  FeCFe[8][5] = CtuZZ*J417;
  FeCFe[6][6] = CuuXX*J0978;
  FeCFe[6][7] = CuuXY*J0978;
  FeCFe[6][8] = CuuXZ*J0978;
  FeCFe[7][6] = FeCFe[6][7];
  FeCFe[7][7] = CuuYY*J0978;
  FeCFe[7][8] = CuuYZ*J0978;
  FeCFe[8][6] = FeCFe[6][8];
  FeCFe[8][7] = FeCFe[7][8];
  FeCFe[8][8] = CuuZZ*J0978;
  double FeCFeInv[9][9];
  if (!invert_small<9>(FeCFe, FeCFeInv)) {
    for (int a = 0; a < 9; ++a)
      for (int c = 0; c < 9; ++c) FeCFeInv[a][c] = 0.0;
  }

  // the integrated matrix [FenT][C][Fhg] (last three columns zero)
  double FeCFhg[9][12];
  for (int a = 0; a < 9; ++a)
    for (int c = 0; c < 12; ++c) FeCFhg[a][c] = 0.0;
  FeCFhg[0][0] = CstXX*J0789 + CssXX*J619;
  FeCFhg[0][1] = CstXY*J0789 + CssXY*J619;
  FeCFhg[0][2] = CstXZ*J0789 + CssXZ*J619;
  FeCFhg[1][0] = CstYX*J0789 + CssXY*J619;
  FeCFhg[1][1] = CstYY*J0789 + CssYY*J619;
  FeCFhg[1][2] = CstYZ*J0789 + CssYZ*J619;
  FeCFhg[2][0] = CstZX*J0789 + CssXZ*J619;
  FeCFhg[2][1] = CstZY*J0789 + CssYZ*J619;
  FeCFhg[2][2] = CstZZ*J0789 + CssZZ*J619;
  FeCFhg[0][3] = CsuXX*J619 + CstXX*J518;
  FeCFhg[0][4] = CsuXY*J619 + CstXY*J518;
  FeCFhg[0][5] = CsuXZ*J619 + CstXZ*J518;
  FeCFhg[1][3] = CsuYX*J619 + CstYX*J518;
  FeCFhg[1][4] = CsuYY*J619 + CstYY*J518;
  FeCFhg[1][5] = CsuYZ*J619 + CstYZ*J518;
  FeCFhg[2][3] = CsuZX*J619 + CstZX*J518;
  FeCFhg[2][4] = CsuZY*J619 + CstZY*J518;
  FeCFhg[2][5] = CsuZZ*J619 + CstZZ*J518;
  FeCFhg[0][6] = CsuXX*J0789 + CssXX*J518;
  FeCFhg[0][7] = CsuXY*J0789 + CssXY*J518;
  FeCFhg[0][8] = CsuXZ*J0789 + CssXZ*J518;
  FeCFhg[1][6] = CsuYX*J0789 + CssXY*J518;
  FeCFhg[1][7] = CsuYY*J0789 + CssYY*J518;
  FeCFhg[1][8] = CsuYZ*J0789 + CssYZ*J518;
  FeCFhg[2][6] = CsuZX*J0789 + CssXZ*J518;
  FeCFhg[2][7] = CsuZY*J0789 + CssYZ*J518;
  FeCFhg[2][8] = CsuZZ*J0789 + CssZZ*J518;
  FeCFhg[3][0] = CstXX*J0879 + CttXX*J619;
  FeCFhg[3][1] = CstYX*J0879 + CttXY*J619;
  FeCFhg[3][2] = CstZX*J0879 + CttXZ*J619;
  FeCFhg[4][0] = CstXY*J0879 + CttXY*J619;
  FeCFhg[4][1] = CstYY*J0879 + CttYY*J619;
  FeCFhg[4][2] = CstZY*J0879 + CttYZ*J619;
  FeCFhg[5][0] = CstXZ*J0879 + CttXZ*J619;
  FeCFhg[5][1] = CstYZ*J0879 + CttYZ*J619;
  FeCFhg[5][2] = CstZZ*J0879 + CttZZ*J619;
  FeCFhg[3][3] = CtuXX*J0879 + CttXX*J417;
  FeCFhg[3][4] = CtuXY*J0879 + CttXY*J417;
  FeCFhg[3][5] = CtuXZ*J0879 + CttXZ*J417;
  FeCFhg[4][3] = CtuYX*J0879 + CttXY*J417;
  FeCFhg[4][4] = CtuYY*J0879 + CttYY*J417;
  FeCFhg[4][5] = CtuYZ*J0879 + CttYZ*J417;
  FeCFhg[5][3] = CtuZX*J0879 + CttXZ*J417;
  FeCFhg[5][4] = CtuZY*J0879 + CttYZ*J417;
  FeCFhg[5][5] = CtuZZ*J0879 + CttZZ*J417;
  FeCFhg[3][6] = CtuXX*J619 + CstXX*J417;
  FeCFhg[3][7] = CtuXY*J619 + CstYX*J417;
  FeCFhg[3][8] = CtuXZ*J619 + CstZX*J417;
  FeCFhg[4][6] = CtuYX*J619 + CstXY*J417;
  FeCFhg[4][7] = CtuYY*J619 + CstYY*J417;
  FeCFhg[4][8] = CtuYZ*J619 + CstZY*J417;
  FeCFhg[5][6] = CtuZX*J619 + CstXZ*J417;
  FeCFhg[5][7] = CtuZY*J619 + CstYZ*J417;
  FeCFhg[5][8] = CtuZZ*J619 + CstZZ*J417;
  FeCFhg[6][0] = CtuXX*J518 + CsuXX*J417;
  FeCFhg[6][1] = CtuYX*J518 + CsuYX*J417;
  FeCFhg[6][2] = CtuZX*J518 + CsuZX*J417;
  FeCFhg[7][0] = CtuXY*J518 + CsuXY*J417;
  FeCFhg[7][1] = CtuYY*J518 + CsuYY*J417;
  FeCFhg[7][2] = CtuZY*J518 + CsuZY*J417;
  FeCFhg[8][0] = CtuXZ*J518 + CsuXZ*J417;
  FeCFhg[8][1] = CtuYZ*J518 + CsuYZ*J417;
  FeCFhg[8][2] = CtuZZ*J518 + CsuZZ*J417;
  FeCFhg[6][3] = CtuXX*J0978 + CuuXX*J417;
  FeCFhg[6][4] = CtuYX*J0978 + CuuXY*J417;
  FeCFhg[6][5] = CtuZX*J0978 + CuuXZ*J417;
  FeCFhg[7][3] = CtuXY*J0978 + CuuXY*J417;
  FeCFhg[7][4] = CtuYY*J0978 + CuuYY*J417;
  FeCFhg[7][5] = CtuZY*J0978 + CuuYZ*J417;
  FeCFhg[8][3] = CtuXZ*J0978 + CuuXZ*J417;
  FeCFhg[8][4] = CtuYZ*J0978 + CuuYZ*J417;
  FeCFhg[8][5] = CtuZZ*J0978 + CuuZZ*J417;
  FeCFhg[6][6] = CsuXX*J0978 + CuuXX*J518;
  FeCFhg[6][7] = CsuYX*J0978 + CuuXY*J518;
  FeCFhg[6][8] = CsuZX*J0978 + CuuXZ*J518;
  FeCFhg[7][6] = CsuXY*J0978 + CuuXY*J518;
  FeCFhg[7][7] = CsuYY*J0978 + CuuYY*J518;
  FeCFhg[7][8] = CsuZY*J0978 + CuuYZ*J518;
  FeCFhg[8][6] = CsuXZ*J0978 + CuuXZ*J518;
  FeCFhg[8][7] = CsuYZ*J0978 + CuuYZ*J518;
  FeCFhg[8][8] = CsuZZ*J0978 + CuuZZ*J518;

  // interior = FCF - KuT * FeCFeInv * FeCFhg, KuT = FeCFhg^T (left to right, as OpenSees)
  double KuTinv[12][9];
  for (int i = 0; i < 12; ++i)
    for (int j = 0; j < 9; ++j) {
      double s = 0.0;
      for (int k = 0; k < 9; ++k) s += FeCFhg[k][i] * FeCFeInv[k][j];
      KuTinv[i][j] = s;
    }
  for (int i = 0; i < 12; ++i)
    for (int j = 0; j < 12; ++j) {
      double s = 0.0;
      for (int k = 0; k < 9; ++k) s += KuTinv[i][k] * FeCFhg[k][j];
      interior[i][j] = FCF[i][j] - s;
    }
}

}  // namespace ssp_detail

template <class Mat>
struct SSPbrick {
  static_assert(Material<Mat>);
  using Material = Mat;
  static_assert(Mat::KIND == MaterialKind::NDMaterial3D, "SSPbrick needs a 3D nD material");

  static constexpr int NDM = 3;
  static constexpr int NNODE = 8;
  static constexpr std::array<int, NNODE> NODE_NDF = {3, 3, 3, 3, 3, 3, 3, 3};
  static constexpr int NEXTRA = 0;
  static constexpr int NDOF = 24;
  static constexpr int NGP = 1;
  static constexpr int NSTRESS = 6;
  static constexpr int NSTATE = NGP * Mat::NSTATE;
  static constexpr MassType MASS = MassType::Lumped;
  static constexpr bool EXPLICIT = true;
  // Small strain: the reference geometry never changes, so it may be cached.
  static constexpr bool GEOMETRY_CACHEABLE = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
  // Kstab is symmetric positive semi-definite and adds no weaker symmetry than the material's.
  static constexpr TangentSymmetry TANGENT = Mat::TANGENT;
  // getInitialStiff returns getTangentStiff (betaKinit damping uses the current tangent).
  static constexpr bool INITIAL_STIFFNESS_IS_TANGENT = true;

  // Element constants (ElementTermElement): the hourglass vectors gst, gut, gus, gstu
  // (value 8 g + i: vector g at node i), then interior's upper triangle row by row.
  static constexpr int NCONST = 32 + 78;
  // Mben v has 12 values (4 hourglass modes x 3 components).
  static constexpr int NTERM = 12;

  struct Params {
    typename Mat::Params mat;
    double b[3];  // body force per unit volume (b1 b2 b3)
  };

  // The single integration point: Bnot's derivatives, the volume, J[0] and the per-node
  // bracket of the mass and body-force integrals.
  struct PointShape {
    double dNmod[8][3];
    double mVol;
    double J0;
    double poly[8];
  };

  FEMORAX_HD static void local_coords(const double (*xe)[NDM], double xl[3][8]) {
    for (int i = 0; i < 8; i++) {
      xl[0][i] = xe[i][0];
      xl[1][i] = xe[i][1];
      xl[2][i] = xe[i][2];
    }
  }

  FEMORAX_HD static void point_geometry(const double xl[3][8], int gp, PointShape& s) {
    (void)gp;
    ssp_detail::Geometry g;
    ssp_detail::geometry(xl, g, false);
    const ssp_detail::ShapeVectors h;
    for (int i = 0; i < 8; ++i)
      for (int k = 0; k < 3; ++k) s.dNmod[i][k] = g.dNmod[i][k];
    s.mVol = g.mVol;
    s.J0 = g.J[0];
    for (int i = 0; i < 8; ++i) s.poly[i] = ssp_detail::poly_bracket(g.J, h, i);
  }

  // --- Precomputed geometry ---------------------------------------------------------------
  // dNmod (24 values, node-major), mVol, J0, poly (8).
  static constexpr int NGEOM = 24 + 1 + 1 + 8;

  FEMORAX_HD static void point_store_geometry(const double xl[3][8], int gp, Strided<double> g) {
    PointShape s;
    point_geometry(xl, gp, s);
    for (int k = 0; k < 24; ++k) g[k] = s.dNmod[k / 3][k % 3];
    g[24] = s.mVol;
    g[25] = s.J0;
    for (int i = 0; i < 8; ++i) g[26 + i] = s.poly[i];
  }

  FEMORAX_HD static void point_load_geometry(int gp, Strided<const double> g, PointShape& s) {
    (void)gp;
    for (int k = 0; k < 24; ++k) s.dNmod[k / 3][k % 3] = g[k];
    s.mVol = g[24];
    s.J0 = g[25];
    for (int i = 0; i < 8; ++i) s.poly[i] = g[26 + i];
  }

  // --- Element constants (GetStab) ----------------------------------------------------------
  FEMORAX_HD static void element_constants(const Params& p, const double xl[3][8], Strided<double> c) {
    ssp_detail::Geometry g;
    ssp_detail::geometry(xl, g, true);
    double CmatI[NSTRESS][NSTRESS];
    Mat::initial_tangent(p.mat, CmatI);
    double interior[12][12];
    ssp_detail::interior_matrix(g, CmatI[0][0], CmatI[0][1], CmatI[3][3], interior);
    for (int i = 0; i < 8; ++i) {
      c[i] = g.gst[i];
      c[8 + i] = g.gut[i];
      c[16 + i] = g.gus[i];
      c[24 + i] = g.gstu[i];
    }
    int k = 32;
    for (int r = 0; r < 12; ++r)
      for (int q = r; q < 12; ++q) c[k++] = interior[r][q];
  }

  // interior(r, q) from the packed upper triangle.
  FEMORAX_HD static double interior_at(Strided<const double> c, int r, int q) {
    if (r > q) {
      const int t = r;
      r = q;
      q = t;
    }
    // offset of row r: sum_{i<r} (12 - i) = 12 r - r (r - 1) / 2
    return c[32 + 12 * r - r * (r - 1) / 2 + (q - r)];
  }

  // Geometry sources (MatrixFreeElement): recomputed from coordinates and parameters, or
  // loaded from the geometry cache and the element-constants array. Both give the same bits.
  struct RecomputedGeometry {
    const double (*xl)[8];
    double c[NCONST];
    FEMORAX_HD RecomputedGeometry(const double (*xl_)[8], const Params& p) : xl(xl_) {
      element_constants(p, xl_, Strided<double>{c, 1});
    }
    FEMORAX_HD void shape(int gp, PointShape& s) const { point_geometry(xl, gp, s); }
    FEMORAX_HD Strided<const double> constants() const { return {c, 1}; }
  };
  struct CachedGeometry {
    PointArray<const double> g;
    Strided<const double> c;
    FEMORAX_HD void shape(int gp, PointShape& s) const { point_load_geometry(gp, g.point(gp), s); }
    FEMORAX_HD Strided<const double> constants() const { return c; }
  };
  FEMORAX_HD static RecomputedGeometry recomputed_geometry(const double (*xl)[8], const Params& p) {
    return RecomputedGeometry(xl, p);
  }
  FEMORAX_HD static CachedGeometry cached_geometry(PointArray<const double> g, Strided<const double> c) {
    return {g, c};
  }

  // --- Kstab = Mben^T interior Mben (ElementTermElement) -----------------------------------
  // Mben(3 g + d, 3 i + d) = gamma_g(i). z = interior * (Mben v): 12 values.
  FEMORAX_HD static void element_term_reduce(const Params& p, Strided<const double> c, const double* v,
                                             double z[NTERM]) {
    (void)p;
    double w[12];
    for (int gi = 0; gi < 4; ++gi)
      for (int d = 0; d < 3; ++d) {
        double s = 0.0;
        for (int i = 0; i < 8; ++i) s += c[8 * gi + i] * v[3 * i + d];
        w[3 * gi + d] = s;
      }
    for (int r = 0; r < 12; ++r) {
      double s = 0.0;
      for (int q = 0; q < 12; ++q) s += interior_at(c, r, q) * w[q];
      z[r] = s;
    }
  }
  // (Kstab v) at node j from z: Mben_j^T z.
  FEMORAX_HD static void element_term_node(const Params& p, Strided<const double> c, const double z[NTERM],
                                           int j, double out[3]) {
    (void)p;
    for (int d = 0; d < 3; ++d) {
      double s = 0.0;
      for (int gi = 0; gi < 4; ++gi) s += c[8 * gi + j] * z[3 * gi + d];
      out[d] = s;
    }
  }
  // Kstab(a, b), summed in the order of (min(a, b), max(a, b)) so that it is exactly symmetric.
  FEMORAX_HD static double element_term_entry(const Params& p, Strided<const double> c, int a, int b) {
    (void)p;
    if (a > b) {
      const int t = a;
      a = b;
      b = t;
    }
    const int i = a / 3, d = a % 3, k = b / 3, e = b % 3;
    double s = 0.0;
    for (int gi = 0; gi < 4; ++gi)
      for (int hi = 0; hi < 4; ++hi) s += c[8 * gi + i] * interior_at(c, 3 * gi + d, 3 * hi + e) * c[8 * hi + k];
    return s;
  }

  // --- Point functions -------------------------------------------------------------------
  // update: strain = Bnot * u, accumulated over the DOFs in order (Matrix * Vector).
  FEMORAX_HD static void point_strain(const PointShape& s, const double ul[NDOF], double strain[NSTRESS]) {
    for (int c = 0; c < NSTRESS; ++c) strain[c] = 0.0;
    for (int i = 0; i < NNODE; i++) {
      const double bx = s.dNmod[i][0], by = s.dNmod[i][1], bz = s.dNmod[i][2];
      const double u0 = ul[3 * i + 0], u1 = ul[3 * i + 1], u2 = ul[3 * i + 2];
      // column 3i: rows 0 (bx), 3 (by), 5 (bz)
      strain[0] += bx * u0;
      strain[3] += by * u0;
      strain[5] += bz * u0;
      // column 3i+1: rows 1 (by), 3 (bx), 4 (bz)
      strain[1] += by * u1;
      strain[3] += bx * u1;
      strain[4] += bz * u1;
      // column 3i+2: rows 2 (bz), 4 (by), 5 (bx)
      strain[2] += bz * u2;
      strain[4] += by * u2;
      strain[5] += bx * u2;
    }
  }

  // getResistingForce: mInternalForces += mVol * Bnot^T stress (column sums, then * mVol).
  FEMORAX_HD static void point_node_force(const PointShape& s, const double stress[NSTRESS], int j,
                                          double fj[3]) {
    const double bx = s.dNmod[j][0], by = s.dNmod[j][1], bz = s.dNmod[j][2];
    fj[0] = (bx * stress[0] + by * stress[3] + bz * stress[5]) * s.mVol;
    fj[1] = (by * stress[1] + bx * stress[3] + bz * stress[4]) * s.mVol;
    fj[2] = (bz * stress[2] + by * stress[4] + bx * stress[5]) * s.mVol;
  }

  // getResistingForce (applyLoad == 0): mInternalForces(3i+q) -= b[q] * polyJac.
  FEMORAX_HD static double point_body_force(const Params& p, const PointShape& s, int j, int q) {
    const double polyJac = s.J0 * s.poly[j];
    return p.b[q] * polyJac;
  }

  // The point's share of the 3x3 block (J, K) of mVol Bnot^T D Bnot.
  FEMORAX_HD static void point_block_tangent(const PointShape& s, const double (*dd)[NSTRESS], int J, int K,
                                             double stiffJK[3][3]) {
    double BJ[NSTRESS][3], BK[NSTRESS][3];
    computeB(J, s, BJ);
    computeB(K, s, BK);
    double BJtranD[3][NSTRESS];
    for (int p = 0; p < 3; p++)
      for (int q = 0; q < NSTRESS; q++) {
        double sum = 0.0;
        for (int r = 0; r < NSTRESS; r++) sum += BJ[r][p] * (dd[r][q] * s.mVol);
        BJtranD[p][q] = sum;
      }
    for (int p = 0; p < 3; p++)
      for (int q = 0; q < 3; q++) {
        double sum = 0.0;
        for (int r = 0; r < NSTRESS; r++) sum += BJtranD[p][r] * BK[r][q];
        stiffJK[p][q] = sum;
      }
  }

  // getMass' massTerm at node j: density * J[0] * bracket.
  FEMORAX_HD static double node_mass(const Params& p, const PointShape& s, int j) {
    return Mat::density(p.mat) * s.J0 * s.poly[j];
  }

  // (M v) at node j: the lumped mass times v_j (one point holds the whole mass).
  FEMORAX_HD static void point_mass_apply(const Params& p, const PointShape& s, const double* v, int j,
                                          double* out) {
    const double m = node_mass(p, s, j);
    for (int q = 0; q < 3; q++) out[q] = m * v[3 * j + q];
  }
  FEMORAX_HD static double point_mass_entry(const Params& p, const PointShape& s, int j, int k) {
    return j == k ? node_mass(p, s, j) : 0.0;
  }

  // Everything the point contributes, before it is spread to the nodes.
  struct PointResult {
    PointShape s;
    double stress[NSTRESS];
    double dampStress[NSTRESS];  // ((betaK + betaKinit) Dt + betaKcomm Dc) * strain rate
    bool stiffDamping;
  };

  // Point update: strain, material update (writes trial state), damping stress. D0 is
  // unused (the initial stiffness is the tangent); the argument keeps the launch layer's
  // call the same for every element.
  FEMORAX_HD static void point_update_shape(const Params& p, const Rayleigh& damp, const PointShape& s,
                                            const double* ue, const double* ve,
                                            Strided<const double> committed_i, Strided<double> trial_i,
                                            const double (*D0)[NSTRESS], PointResult& r, double (*dd)[NSTRESS]) {
    (void)D0;
    r.s = s;
    double strain[NSTRESS];
    point_strain(r.s, ue, strain);
    Mat::update(p.mat, committed_i, trial_i, strain, r.stress, dd);
    r.stiffDamping = damp.betaK != 0.0 || damp.betaKinit != 0.0 || damp.betaKcomm != 0.0;
    for (int a = 0; a < NSTRESS; ++a) r.dampStress[a] = 0.0;
    if (r.stiffDamping) {
      double Dc[NSTRESS][NSTRESS];
      if (damp.betaKcomm != 0.0) Mat::committed_tangent(p.mat, committed_i, Dc);
      double rate[NSTRESS];
      point_strain(r.s, ve, rate);
      for (int a = 0; a < NSTRESS; ++a) {
        double sum = 0.0;
        for (int c = 0; c < NSTRESS; ++c) {
          double Dac = damp.betaK * dd[a][c];
          if (damp.betaKinit != 0.0) Dac += damp.betaKinit * dd[a][c];
          if (damp.betaKcomm != 0.0) Dac += damp.betaKcomm * Dc[a][c];
          sum += Dac * rate[c];
        }
        r.dampStress[a] = sum;
      }
    }
  }

  // What the point contributes to node j: residJ (mVol Bnot_j^T stress), body (b polyJac)
  // and dampJ (stiffness damping through the point, plus alphaM times the lumped mass).
  FEMORAX_HD static void point_node_contribution(const Params& p, const Rayleigh& damp, const PointResult& r,
                                                 const double* ve, int j, double residJ[3], double body[3],
                                                 double dampJ[3]) {
    point_node_force(r.s, r.stress, j, residJ);
    for (int q = 0; q < 3; q++) body[q] = point_body_force(p, r.s, j, q);
    dampJ[0] = dampJ[1] = dampJ[2] = 0.0;
    if (r.stiffDamping) point_node_force(r.s, r.dampStress, j, dampJ);
    if (damp.alphaM != 0.0) {
      const double m = node_mass(p, r.s, j);
      for (int q = 0; q < 3; q++) dampJ[q] += damp.alphaM * m * ve[3 * j + q];
    }
  }

  struct Work {
    double resid[NDOF];
    double fdamp[NDOF];
  };

  // Internal force (Kstab u + mVol Bnot^T stress - body force) plus the Rayleigh damping
  // force, and, when WantK, the tangent Kstab + mVol Bnot^T D Bnot.
  template <bool WantK, class Ctx, class Geometry>
  FEMORAX_HD static void compute_with(const Ctx& ctx, const Geometry& geometry, const Params& p,
                                      const Rayleigh& damp, const double* ue, const double* ve,
                                      PointArray<const double> committed, PointArray<double> trial, double* fe,
                                      double (*Ke)[NDOF], Work& work) {
    const Strided<const double> c = geometry.constants();
    // Kstab's share first (OpenSees: mInternalForces = Kstab * d, then the stress and body
    // terms). The element-wide reductions run redundantly on every lane (inner.hpp).
    double zu[NTERM], zv[NTERM];
    element_term_reduce(p, c, ue, zu);
    element_term_reduce(p, c, ve, zv);
    inner_for(ctx, NNODE, [&](int j) {
      element_term_force_node<SSPbrick>(p, damp, c, zu, zv, j, &work.resid[3 * j], &work.fdamp[3 * j]);
    });
    if constexpr (WantK) {
      inner_for(ctx, NNODE * NNODE, [&](int blk) {
        const int J = blk / NNODE, K = blk % NNODE;
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) Ke[3 * J + a][3 * K + b] = element_term_entry(p, c, 3 * J + a, 3 * K + b);
      });
    }
    for (int i = 0; i < NGP; i++) {
      PointShape s;
      geometry.shape(i, s);
      PointResult r;
      double dd[NSTRESS][NSTRESS];
      point_update_shape(p, damp, s, ue, ve, committed.point(i), trial.point(i), nullptr, r, dd);
      inner_for(ctx, NNODE, [&](int j) {
        double residJ[3], body[3], dampJ[3];
        point_node_contribution(p, damp, r, ve, j, residJ, body, dampJ);
        for (int q = 0; q < 3; q++) {
          work.resid[3 * j + q] += residJ[q];
          work.resid[3 * j + q] -= body[q];
          work.fdamp[3 * j + q] += dampJ[q];
        }
      });
      if constexpr (WantK) {
        inner_for(ctx, NNODE * NNODE, [&](int blk) {
          const int J = blk / NNODE, K = blk % NNODE;
          double stiffJK[3][3];
          point_block_tangent(r.s, dd, J, K, stiffJK);
          for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) Ke[3 * J + a][3 * K + b] += stiffJK[a][b];
        });
      }
    }
    inner_for(ctx, NNODE, [&](int j) {
      for (int q = 0; q < 3; q++) fe[3 * j + q] = work.resid[3 * j + q] + work.fdamp[3 * j + q];
    });
  }

  template <bool WantK, class Ctx>
  FEMORAX_HD static void compute(const Ctx& ctx, const Params& p, const Rayleigh& damp, const double (*xe)[NDM],
                                 const double* ue, const double* ve, PointArray<const double> committed,
                                 PointArray<double> trial, double* fe, double (*Ke)[NDOF], Work& work) {
    double xl[3][8];
    local_coords(xe, xl);
    compute_with<WantK>(ctx, RecomputedGeometry(xl, p), p, damp, ue, ve, committed, trial, fe, Ke, work);
  }

  // Serial form (the Element contract).
  template <bool WantK>
  FEMORAX_HD static void compute(const Params& p, const Rayleigh& damp, const double (*xe)[NDM], const double* ue,
                                 const double* ve, PointArray<const double> committed, PointArray<double> trial,
                                 double* fe, double (*Ke)[NDOF]) {
    Work work;
    compute<WantK>(SerialInner{}, p, damp, xe, ue, ve, committed, trial, fe, Ke, work);
  }

  // Serial form with the geometry cache and the element constants.
  template <bool WantK>
  FEMORAX_HD static void compute_cached(const Params& p, const Rayleigh& damp, PointArray<const double> geometry,
                                        Strided<const double> constants, const double* ue, const double* ve,
                                        PointArray<const double> committed, PointArray<double> trial, double* fe,
                                        double (*Ke)[NDOF]) {
    Work work;
    compute_with<WantK>(SerialInner{}, CachedGeometry{geometry, constants}, p, damp, ue, ve, committed, trial, fe,
                        Ke, work);
  }

  // getMass: the diagonal massTerm at every DOF of node i (row-major 24 x 24).
  FEMORAX_HD static void mass(const Params& p, const double (*xe)[NDM], double* Me) {
    for (int a = 0; a < NDOF * NDOF; ++a) Me[a] = 0.0;
    double m[NNODE];
    lumped_mass(p, xe, m);
    for (int i = 0; i < NNODE; ++i)
      for (int q = 0; q < 3; ++q) Me[(3 * i + q) * NDOF + 3 * i + q] = m[i];
  }

  FEMORAX_HD static void lumped_mass(const Params& p, const double (*xe)[NDM], double m[NNODE]) {
    double xl[3][8];
    local_coords(xe, xl);
    PointShape s;
    point_geometry(xl, 0, s);
    for (int j = 0; j < NNODE; ++j) m[j] = node_mass(p, s, j);
  }

  // Largest omega^2 of K phi = omega^2 M phi with the lumped mass and the committed tangent
  // (Kstab included). Runs once per element per stage, so the 24 x 24 matrix in one thread
  // is acceptable here.
  FEMORAX_HD static double max_omega_sq(const Params& p, const double (*xe)[NDM], PointArray<const double> committed) {
    double xl[3][8];
    local_coords(xe, xl);
    const RecomputedGeometry geo(xl, p);
    const Strided<const double> c = geo.constants();
    PointShape s;
    point_geometry(xl, 0, s);
    double A[NDOF][NDOF];
    for (int a = 0; a < NDOF; ++a)
      for (int b = 0; b < NDOF; ++b) A[a][b] = element_term_entry(p, c, a, b);
    double dd[NSTRESS][NSTRESS];
    Mat::committed_tangent(p.mat, committed.point(0), dd);
    for (int j = 0; j < NNODE; j++)
      for (int k = 0; k < NNODE; k++) {
        double stiffJK[3][3];
        point_block_tangent(s, dd, j, k, stiffJK);
        for (int a = 0; a < 3; a++)
          for (int b = 0; b < 3; b++) A[3 * j + a][3 * k + b] += stiffJK[a][b];
      }
    double inv_sqrt_m[NNODE];
    for (int j = 0; j < NNODE; j++) {
      const double m = node_mass(p, s, j);
      if (!(m > 0.0)) return 0.0;  // massless element: sets no limit
      inv_sqrt_m[j] = 1.0 / std::sqrt(m);
    }
    for (int a = 0; a < NDOF; ++a)
      for (int b = 0; b < NDOF; ++b) A[a][b] *= inv_sqrt_m[a / 3] * inv_sqrt_m[b / 3];
    return max_eigenvalue_symmetric(A);
  }

  FEMORAX_HD static double critical_dt(const Params& p, const Rayleigh& damp, ExplicitScheme scheme,
                                       const double (*xe)[NDM], PointArray<const double> committed) {
    return stable_dt(scheme, damp, max_omega_sq(p, xe, committed));
  }

 private:
  // Bnot's columns of node `node` (the same pattern as Brick::computeB).
  FEMORAX_HD static void computeB(int node, const PointShape& s, double B[NSTRESS][3]) {
    for (int r = 0; r < NSTRESS; ++r)
      for (int c = 0; c < 3; ++c) B[r][c] = 0.0;
    B[0][0] = s.dNmod[node][0];
    B[1][1] = s.dNmod[node][1];
    B[2][2] = s.dNmod[node][2];
    B[3][0] = s.dNmod[node][1];
    B[3][1] = s.dNmod[node][0];
    B[4][1] = s.dNmod[node][2];
    B[4][2] = s.dNmod[node][1];
    B[5][0] = s.dNmod[node][2];
    B[5][2] = s.dNmod[node][0];
  }
};

}  // namespace femorax
