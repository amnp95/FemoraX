#pragma once

// What the 3D beam-column elements share (docs/design.md, "Beams"): the mass (lumped with the
// rotary inertia, and consistent), the Rayleigh damping force, the matrix-free operator and the
// largest element eigenvalue, all from the element's parameters (rhoA, rot, cMass, transf), its
// transformation geometry and its basic stiffnesses: the current tangent kb_t (with the basic
// forces q for PDelta's geometric term), the initial kb_0 and the committed kb_c. The
// displacement-based beam (disp_beam_3d.hpp) integrates them from its sections; the force-based
// beam (force_beam_3d.hpp) has them as kv. Moved from disp_beam_3d.hpp unchanged (the same
// arithmetic in the same order); see there for the mass and damping choices against OpenSees.

#include <cmath>

#include "physics/common/sym_eigen.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"
#include "physics/operators/element_operator.hpp"
#include "physics/transformations/crd_transf_3d.hpp"

namespace femorax::beam3d {

// The lumped mass per DOF (12 values, node I then node J).
template <class P>
FEMORAX_HD inline void lumped_mass(const P& p, const CrdTransf3dGeometry& g, double m[12]) {
  const double half = 0.5 * g.L;
  for (int n = 0; n < 2; ++n) {
    for (int d = 0; d < 3; ++d) m[6 * n + d] = half * p.rhoA;
    // diag(R^T diag(rot) R): global axis d's rotary inertia.
    for (int d = 0; d < 3; ++d) {
      double s = 0.0;
      for (int a = 0; a < 3; ++a) s += g.R[a][d] * g.R[a][d] * p.rot[a];
      m[6 * n + 3 + d] = half * s;
    }
  }
}

// The consistent mass in global axes (getMass with cMass, plus FemoraX's torsional mass and
// the rotary inertia of bending).
template <class P>
FEMORAX_HD inline void consistent_mass(const P& p, const CrdTransf3dGeometry& g, double (*M)[12]) {
  const double L = g.L;
  double ml[12][12];
  for (int a = 0; a < 12; ++a)
    for (int b = 0; b < 12; ++b) ml[a][b] = 0.0;
  const double m = p.rhoA * L / 420.0;
  ml[0][0] = ml[6][6] = m * 140.0;
  ml[0][6] = ml[6][0] = m * 70.0;

  ml[2][2] = ml[8][8] = m * 156.0;
  ml[2][8] = ml[8][2] = m * 54.0;
  ml[4][4] = ml[10][10] = m * 4.0 * L * L;
  ml[4][10] = ml[10][4] = -m * 3.0 * L * L;
  ml[2][4] = ml[4][2] = -m * 22.0 * L;
  ml[8][10] = ml[10][8] = -ml[2][4];
  ml[2][10] = ml[10][2] = m * 13.0 * L;
  ml[4][8] = ml[8][4] = -ml[2][10];

  ml[1][1] = ml[7][7] = m * 156.0;
  ml[1][7] = ml[7][1] = m * 54.0;
  ml[5][5] = ml[11][11] = m * 4.0 * L * L;
  ml[5][11] = ml[11][5] = -m * 3.0 * L * L;
  ml[1][5] = ml[5][1] = m * 22.0 * L;
  ml[7][11] = ml[11][7] = -ml[1][5];
  ml[1][11] = ml[11][1] = -m * 13.0 * L;
  ml[5][7] = ml[7][5] = -ml[1][11];

  // FemoraX: torsional mass, linear twist.
  if (p.rot[0] != 0.0) {
    const double t = p.rot[0] * L / 6.0;
    ml[3][3] += 2.0 * t;
    ml[9][9] += 2.0 * t;
    ml[3][9] += t;
    ml[9][3] += t;
  }
  // FemoraX: rotary inertia of bending (DOFs w, theta of each plane; theta_y = -dw/dx flips
  // the coupling signs in the xz plane).
  const double R4[4][4] = {{36.0, 3.0 * L, -36.0, 3.0 * L},
                           {3.0 * L, 4.0 * L * L, -3.0 * L, -L * L},
                           {-36.0, -3.0 * L, 36.0, -3.0 * L},
                           {3.0 * L, -L * L, -3.0 * L, 4.0 * L * L}};
  const int xy[4] = {1, 5, 7, 11}, xz[4] = {2, 4, 8, 10};
  const double sxz[4] = {1.0, -1.0, 1.0, -1.0};
  const double cz = p.rot[2] / (30.0 * L), cy = p.rot[1] / (30.0 * L);
  if (cz != 0.0 || cy != 0.0)
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) {
        ml[xy[a]][xy[b]] += cz * R4[a][b];
        ml[xz[a]][xz[b]] += cy * sxz[a] * sxz[b] * R4[a][b];
      }
  crd_transf_3d_global_from_local(g, ml, M);
}

// The 12 x 12 mass matrix (row-major): consistent with cMass, else the lumped diagonal.
template <class P>
FEMORAX_HD inline void mass(const P& p, const CrdTransf3dGeometry& g, double* Me) {
  for (int a = 0; a < 144; ++a) Me[a] = 0.0;
  if (p.cMass) {
    double M[12][12];
    consistent_mass(p, g, M);
    for (int a = 0; a < 12; ++a)
      for (int b = 0; b < 12; ++b) Me[a * 12 + b] = M[a][b];
  } else {
    double m[12];
    lumped_mass(p, g, m);
    for (int a = 0; a < 12; ++a) Me[a * 12 + a] = m[a];
  }
}

// fd += (alphaM M + betaK Kt + betaK0 K0 + betaKc Kc) ve, in the basic system. kb_t: the current
// basic tangent; q: the current basic forces (PDelta's geometric term); kb_0, kb_c: the initial
// and committed basic tangents (read only when their factor is nonzero).
template <class P>
FEMORAX_HD inline void add_damping_force(const P& p, const Rayleigh& damp, const CrdTransf3dGeometry& g,
                                         const double* ve, const double q[6], const double (*kb_t)[6],
                                         const double (*kb0)[6], const double (*kbc)[6], double* fe) {
  const double p0[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  if (damp.betaK != 0.0 || damp.betaKinit != 0.0 || damp.betaKcomm != 0.0) {
    double vb[6];
    crd_transf_3d_basic_disp(p.transf, g, ve, vb);
    const bool need0 = damp.betaKinit != 0.0, needc = damp.betaKcomm != 0.0;
    double qd[6];
    for (int a = 0; a < 6; ++a) {
      double t = 0.0, t0 = 0.0, tc = 0.0;
      for (int b = 0; b < 6; ++b) {
        t += kb_t[a][b] * vb[b];
        if (need0) t0 += kb0[a][b] * vb[b];
        if (needc) tc += kbc[a][b] * vb[b];
      }
      qd[a] = damp.betaK * t + damp.betaKinit * t0 + damp.betaKcomm * tc;
    }
    // T^T qd (the transformation's linear part: no geometric term with qd).
    CrdTransf3dParams lin = p.transf;
    lin.type = CrdTransfType::Linear;
    double fd[12];
    crd_transf_3d_global_force(lin, g, ve, qd, p0, fd);
    for (int k = 0; k < 12; ++k) fe[k] += fd[k];
    if (damp.betaK != 0.0 && p.transf.type == CrdTransfType::PDelta) {
      // betaK Kgeo ve: PDelta's leaning-column force of ve with the current axial force q0,
      // minus its linear part.
      const double qa[6] = {q[0], 0.0, 0.0, 0.0, 0.0, 0.0};
      double fg[12], fl[12];
      crd_transf_3d_global_force(p.transf, g, ve, qa, p0, fg);
      crd_transf_3d_global_force(lin, g, ve, qa, p0, fl);
      for (int k = 0; k < 12; ++k) fe[k] += damp.betaK * (fg[k] - fl[k]);
    }
  }
  if (damp.alphaM != 0.0) {
    if (p.cMass) {
      double M[12][12];
      consistent_mass(p, g, M);
      for (int a = 0; a < 12; ++a) {
        double t = 0.0;
        for (int b = 0; b < 12; ++b) t += M[a][b] * ve[b];
        fe[a] += damp.alphaM * t;
      }
    } else {
      double m[12];
      lumped_mass(p, g, m);
      for (int k = 0; k < 12; ++k) fe[k] += damp.alphaM * m[k] * ve[k];
    }
  }
}

// The element's matrix-free operator (docs/design.md, "The matrix-free operator"):
//   y = cK (K_t or, with k.initial, K_0) x + cC (alphaM M + betaK K_t + betaK0 K_0 + betaKc K_c) x
//       + cM M x,
// every stiffness applied in the basic system (T^T kb T x, plus PDelta's geometric stiffness of
// the current axial force q with K_t). kb0 is read when k.initial or betaKinit, kbc when
// betaKcomm. With diagonal, y[a] is the operator's diagonal entry a.
template <class P>
FEMORAX_HD inline void operator_apply(const P& p, const Rayleigh& damp, const OperatorCoefficients& k,
                                      const CrdTransf3dGeometry& g, const double q[6], const double (*kbt)[6],
                                      const double (*kb0)[6], const double (*kbc)[6], const double* x, bool diagonal,
                                      double* y) {
  const double cS = k.stiffness, cC = k.damping, cM = k.mass;
  const bool need0 = k.initial || damp.betaKinit != 0.0, needc = damp.betaKcomm != 0.0;
  // The combined basic stiffness and the factor of K_t's geometric term.
  const double ct = (k.initial ? 0.0 : cS) + cC * damp.betaK;
  const double c0 = (k.initial ? cS : 0.0) + cC * damp.betaKinit;
  const double cc = cC * damp.betaKcomm;
  double kb[6][6];
  for (int a = 0; a < 6; ++a)
    for (int b = 0; b < 6; ++b)
      kb[a][b] = ct * kbt[a][b] + (need0 ? c0 * kb0[a][b] : 0.0) + (needc ? cc * kbc[a][b] : 0.0);
  const double mcoef = cM + cC * damp.alphaM;
  double ml[12] = {}, Mc[12][12];
  if (mcoef != 0.0) {
    if (p.cMass) consistent_mass(p, g, Mc);
    else lumped_mass(p, g, ml);
  }
  auto apply = [&](const double* xv, double* out) {
    double vb[6], qb[6];
    crd_transf_3d_basic_disp(p.transf, g, xv, vb);
    for (int a = 0; a < 6; ++a) {
      double t = 0.0;
      for (int b = 0; b < 6; ++b) t += kb[a][b] * vb[b];
      qb[a] = t;
    }
    CrdTransf3dParams lin = p.transf;
    lin.type = CrdTransfType::Linear;
    const double p0[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    crd_transf_3d_global_force(lin, g, xv, qb, p0, out);
    if (ct != 0.0 && p.transf.type == CrdTransfType::PDelta) {
      const double qa[6] = {q[0], 0.0, 0.0, 0.0, 0.0, 0.0};
      double fg[12], fl[12];
      crd_transf_3d_global_force(p.transf, g, xv, qa, p0, fg);
      crd_transf_3d_global_force(lin, g, xv, qa, p0, fl);
      for (int a = 0; a < 12; ++a) out[a] += ct * (fg[a] - fl[a]);
    }
    if (mcoef != 0.0) {
      for (int a = 0; a < 12; ++a) {
        double t = 0.0;
        if (p.cMass) {
          for (int b = 0; b < 12; ++b) t += Mc[a][b] * xv[b];
        } else {
          t = ml[a] * xv[a];
        }
        out[a] += mcoef * t;
      }
    }
  };
  if (!diagonal) {
    apply(x, y);
  } else {
    for (int a = 0; a < 12; ++a) {
      double e[12] = {}, col[12];
      e[a] = 1.0;
      apply(e, col);
      y[a] = col[a];
    }
  }
}

// Largest omega^2 of K phi = omega^2 M phi with the lumped mass, K from the basic stiffness kb
// and forces q (the element eigenvalue bound). DOFs without mass are left out; extra_mass adds
// the element's share of nodal masses (12 values, may be null).
template <class P>
FEMORAX_HD inline double max_omega_sq(const P& p, const CrdTransf3dGeometry& g, const double (*kb)[6],
                                      const double q[6], const double* extra_mass) {
  double K[12][12];
  crd_transf_3d_global_stiff(p.transf, g, kb, q, false, K);
  double m[12];
  lumped_mass(p, g, m);
  if (extra_mass)
    for (int k = 0; k < 12; ++k) m[k] += extra_mass[k];
  double A[12][12];
  for (int a = 0; a < 12; ++a)
    for (int b = 0; b < 12; ++b) {
      const bool massive = m[a] > 0.0 && m[b] > 0.0;
      A[a][b] = massive ? K[a][b] / std::sqrt(m[a] * m[b]) : 0.0;
    }
  return max_eigenvalue_symmetric(A);
}

}  // namespace femorax::beam3d
