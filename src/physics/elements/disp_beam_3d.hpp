#pragma once

// DispBeam3d<Section, NIP>: the displacement-based 3D beam-column (docs/design.md, "Beams").
//
// Port of OpenSees SRC/element/dispBeamColumn/DispBeamColumn3d.cpp (update, getTangentStiff,
// getResistingForce), OpenSees commit 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea, with
// LegendreBeamIntegration (the integration rule of `element dispBeamColumn tag i j nIP secTag
// transfTag`, the form Femora writes) and the transformations of
// physics/transformations/crd_transf_3d.hpp. Names follow OpenSees (v, e, xi6, kb, q, ka, wt).
//
// Two nodes x 6 DOFs (ux uy uz rx ry rz each). The section deformations at point i of
// natural coordinate xi (on [0, 1]) are, by the section's codes:
//   P: v0 / L;  MZ: ((6 xi - 4) v1 + (6 xi - 2) v2) / L;  MY: the same with v3, v4;  T: v5 / L;
// q = sum_i wt_i B_i^T s_i, kb = sum_i (wt_i / L) B'_i^T ks_i B'_i, then the transformation.
// No element loads (q0 = 0, p0 = 0) yet.
//
// Mass (deliberately not OpenSees', docs/design.md "Structures"): lumped, from the masses per
// length the adapter gives (rhoA for translation; rotary inertia per length about the local
// x, y and z axes: rho (Iy + Iz), rho Iy, rho Iz), half the length per node. OpenSees'
// lumped mass has no rotational part. The rotary inertia is a tensor R^T diag(ix, iy, iz) R in
// global axes; the lumped (diagonal) mass keeps its diagonal, exact for members along the
// global axes. The consistent mass (-cMass, implicit stages only): OpenSees' getMass (cubic
// Hermite translation, linear axial) plus what OpenSees leaves out: the torsional mass
// rho Ip L / 6 [2 1; 1 2] (OpenSees: "CURRENTLY NO TORSIONAL MASS") and the rotary inertia of
// bending, rho I / (30 L) [36 3L -36 3L; 3L 4L^2 -3L -L^2; ...] (the Rayleigh beam); both in
// local axes, turned by getGlobalMatrixFromLocal (which ignores joint offsets, as OpenSees).
//
// Rayleigh damping (Element::getRayleighDampingForces: (alphaM M + betaK Kt + betaK0 K0 +
// betaKc Kc) v), formed in the basic system so no 12 x 12 matrix is held (performance rule 3a):
// the damping basic force (betaK kb_t + betaK0 kb_0 + betaKc kb_c) T v through the transformation,
// plus betaK times PDelta's geometric stiffness (current axial force) times v, plus alphaM M v.
// Deliberate difference: kb_c is the sections' committed tangent, without PDelta's geometric
// term (OpenSees' Kc, the tangent at the last commit, has it with the committed axial force,
// which the element does not store); identical for Linear (docs/deviations.md).

#include <array>
#include <cmath>

#include "physics/common/strided.hpp"
#include "physics/common/sym_eigen.hpp"
#include "physics/contracts.hpp"
#include "physics/explicit_stability.hpp"
#include "physics/macros.hpp"
#include "physics/operators/element_operator.hpp"
#include "physics/transformations/crd_transf_3d.hpp"

namespace femorax {

// LegendreBeamIntegration::getSectionLocations / getSectionWeights (OpenSees' constants, on
// [0, 1], weights summing to 1).
template <int NIP>
struct LegendreBeamIntegration {
  static_assert(NIP >= 2 && NIP <= 5, "DispBeam3d: 2 to 5 integration points");
  FEMORAX_HD static void points(double xi[NIP], double wt[NIP]) {
    if constexpr (NIP == 2) {
      xi[0] = -0.577350269189626;
      xi[1] = 0.577350269189626;
      wt[0] = 1.0;
      wt[1] = 1.0;
    } else if constexpr (NIP == 3) {
      xi[0] = -0.774596669241483;
      xi[1] = 0.0;
      xi[2] = 0.774596669241483;
      wt[0] = 0.555555555555556;
      wt[1] = 0.888888888888889;
      wt[2] = 0.555555555555556;
    } else if constexpr (NIP == 4) {
      xi[0] = -0.861136311594053;
      xi[1] = -0.339981043584856;
      xi[2] = 0.339981043584856;
      xi[3] = 0.861136311594053;
      wt[0] = 0.347854845137454;
      wt[1] = 0.652145154862546;
      wt[2] = 0.652145154862546;
      wt[3] = 0.347854845137454;
    } else {
      xi[0] = -0.906179845938664;
      xi[1] = -0.538469310105683;
      xi[2] = 0.0;
      xi[3] = 0.538469310105683;
      xi[4] = 0.906179845938664;
      wt[0] = 0.236926885056189;
      wt[1] = 0.478628670499366;
      wt[2] = 0.568888888888889;
      wt[3] = 0.478628670499366;
      wt[4] = 0.236926885056189;
    }
    for (int i = 0; i < NIP; i++) {
      xi[i] = 0.5 * (xi[i] + 1.0);
      wt[i] *= 0.5;
    }
  }
};

template <class Sec, int NIP_>
  requires Section<Sec>
struct DispBeam3d {
  using Material = Sec;
  static constexpr int NDM = 3, NNODE = 2, NEXTRA = 0, NGP = NIP_, NDOF = 12;
  static constexpr std::array<int, NNODE> NODE_NDF = {6, 6};
  static constexpr int NSTATE = NGP * Sec::NSTATE;
  static constexpr int ORDER = Sec::NSTRAIN;
  static constexpr MassType MASS = MassType::Selectable;  // lumped, or consistent with cMass
  static constexpr bool EXPLICIT = true;
  // The geometry is two nodes and a transformation, recomputed in every kernel (no cache).
  static constexpr bool GEOMETRY_CACHEABLE = false;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::Section3d;
  static constexpr TangentSymmetry TANGENT = Sec::TANGENT;  // PDelta's geometric term is symmetric

  struct Params {
    typename Sec::Params mat;  // the section (named mat, as every element names its material)
    CrdTransf3dParams transf;
    double rhoA = 0.0;                     // translational mass per length
    double rot[3] = {0.0, 0.0, 0.0};      // rotary inertia per length about local x, y, z
    int cMass = 0;                         // 1: the consistent mass (implicit stages only)
  };

  // The section deformations of the basic deformations v at natural coordinate xi
  // (DispBeamColumn3d::update).
  FEMORAX_HD static void section_deformation(const double v[6], double oneOverL, double xi, double e[ORDER]) {
    const double xi6 = 6.0 * xi;
    for (int j = 0; j < ORDER; j++) {
      switch (Sec::CODE[j]) {
        case SectionResponse::P: e[j] = oneOverL * v[0]; break;
        case SectionResponse::MZ: e[j] = oneOverL * ((xi6 - 4.0) * v[1] + (xi6 - 2.0) * v[2]); break;
        case SectionResponse::MY: e[j] = oneOverL * ((xi6 - 4.0) * v[3] + (xi6 - 2.0) * v[4]); break;
        case SectionResponse::T: e[j] = oneOverL * v[5]; break;
        default: e[j] = 0.0; break;
      }
    }
  }

  // The basic forces q (and the basic stiffness kb when kb != nullptr) of the trial basic
  // deformations v: the section updates at every point, then the integrals of getTangentStiff.
  template <class SectionUpdate>
  FEMORAX_HD static void basic_response(double L, const double v[6], const SectionUpdate& section_at, double q[6],
                                        double (*kb)[6]) {
    double xi[NGP], wt[NGP];
    LegendreBeamIntegration<NGP>::points(xi, wt);
    const double oneOverL = 1.0 / L;
    for (int k = 0; k < 6; ++k) q[k] = 0.0;
    if (kb)
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b) kb[a][b] = 0.0;
    for (int i = 0; i < NGP; i++) {
      double e[ORDER], s[ORDER], ks[ORDER][ORDER];
      section_deformation(v, oneOverL, xi[i], e);
      section_at(i, e, s, ks);
      const double xi6 = 6.0 * xi[i];
      if (kb) {
        double ka[ORDER][6];
        for (int j = 0; j < ORDER; j++)
          for (int k = 0; k < 6; k++) ka[j][k] = 0.0;
        const double wti = wt[i] * oneOverL;
        double tmp;
        for (int j = 0; j < ORDER; j++) {
          switch (Sec::CODE[j]) {
            case SectionResponse::P:
              for (int k = 0; k < ORDER; k++) ka[k][0] += ks[k][j] * wti;
              break;
            case SectionResponse::MZ:
              for (int k = 0; k < ORDER; k++) {
                tmp = ks[k][j] * wti;
                ka[k][1] += (xi6 - 4.0) * tmp;
                ka[k][2] += (xi6 - 2.0) * tmp;
              }
              break;
            case SectionResponse::MY:
              for (int k = 0; k < ORDER; k++) {
                tmp = ks[k][j] * wti;
                ka[k][3] += (xi6 - 4.0) * tmp;
                ka[k][4] += (xi6 - 2.0) * tmp;
              }
              break;
            case SectionResponse::T:
              for (int k = 0; k < ORDER; k++) ka[k][5] += ks[k][j] * wti;
              break;
            default: break;
          }
        }
        for (int j = 0; j < ORDER; j++) {
          switch (Sec::CODE[j]) {
            case SectionResponse::P:
              for (int k = 0; k < 6; k++) kb[0][k] += ka[j][k];
              break;
            case SectionResponse::MZ:
              for (int k = 0; k < 6; k++) {
                tmp = ka[j][k];
                kb[1][k] += (xi6 - 4.0) * tmp;
                kb[2][k] += (xi6 - 2.0) * tmp;
              }
              break;
            case SectionResponse::MY:
              for (int k = 0; k < 6; k++) {
                tmp = ka[j][k];
                kb[3][k] += (xi6 - 4.0) * tmp;
                kb[4][k] += (xi6 - 2.0) * tmp;
              }
              break;
            case SectionResponse::T:
              for (int k = 0; k < 6; k++) kb[5][k] += ka[j][k];
              break;
            default: break;
          }
        }
      }
      double si;
      for (int j = 0; j < ORDER; j++) {
        si = s[j] * wt[i];
        switch (Sec::CODE[j]) {
          case SectionResponse::P: q[0] += si; break;
          case SectionResponse::MZ:
            q[1] += (xi6 - 4.0) * si;
            q[2] += (xi6 - 2.0) * si;
            break;
          case SectionResponse::MY:
            q[3] += (xi6 - 4.0) * si;
            q[4] += (xi6 - 2.0) * si;
            break;
          case SectionResponse::T: q[5] += si; break;
          default: break;
        }
      }
    }
  }

  FEMORAX_HD static int geometry(const Params& p, const double (*xe)[NDM], CrdTransf3dGeometry& g) {
    return crd_transf_3d_geometry(p.transf, xe[0], xe[1], g);
  }

  // Internal force plus Rayleigh damping force fe (getResistingForce +
  // getRayleighDampingForces) and, with WantK, the tangent Ke (getTangentStiff), the section
  // states updated from committed into trial. ve: the nodal velocities the damping uses.
  template <bool WantK>
  FEMORAX_HD static void compute(const Params& p, const Rayleigh& damp, const double (*xe)[NDM], const double* ue,
                                 const double* ve, PointArray<const double> committed, PointArray<double> trial,
                                 double* fe, double (*Ke)[NDOF]) {
    CrdTransf3dGeometry g;
    geometry(p, xe, g);
    double v[6];
    crd_transf_3d_basic_disp(p.transf, g, ue, v);
    const bool stiff_damping = damp.betaK != 0.0 || damp.betaKinit != 0.0 || damp.betaKcomm != 0.0;
    double q[6], kb[6][6];
    basic_response(
        g.L, v,
        [&](int i, const double* e, double* s, double (*ks)[ORDER]) {
          Sec::update(p.mat, committed.point(i), trial.point(i), e, s, ks);
        },
        q, (WantK || stiff_damping) ? kb : nullptr);
    const double p0[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    crd_transf_3d_global_force(p.transf, g, ue, q, p0, fe);
    if (stiff_damping || damp.alphaM != 0.0) add_damping_force(p, damp, g, xe, ve, committed, q, kb, fe);
    if constexpr (WantK) crd_transf_3d_global_stiff(p.transf, g, kb, q, true, Ke);
  }

  // The undamped force and tangent (no velocities).
  template <bool WantK>
  FEMORAX_HD static void compute(const Params& p, const double (*xe)[NDM], const double* ue,
                                 PointArray<const double> committed, PointArray<double> trial, double* fe,
                                 double (*Ke)[NDOF]) {
    const double ve[12] = {};
    compute<WantK>(p, Rayleigh{}, xe, ue, ve, committed, trial, fe, Ke);
  }

  // fd += (alphaM M + betaK Kt + betaK0 K0 + betaKc Kc) ve, in the basic system (see the top).
  // kb_t: the current basic tangent; q: the current basic forces (PDelta's geometric term).
  FEMORAX_HD static void add_damping_force(const Params& p, const Rayleigh& damp, const CrdTransf3dGeometry& g,
                                           const double (*xe)[NDM], const double* ve,
                                           PointArray<const double> committed, const double q[6],
                                           const double (*kb_t)[6], double* fe) {
    const double zero_q[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double p0[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if (damp.betaK != 0.0 || damp.betaKinit != 0.0 || damp.betaKcomm != 0.0) {
      double vb[6];
      crd_transf_3d_basic_disp(p.transf, g, ve, vb);
      double kb0[6][6], kbc[6][6], qq[6];
      const bool need0 = damp.betaKinit != 0.0, needc = damp.betaKcomm != 0.0;
      if (need0)
        basic_response(
            g.L, zero_q,
            [&](int, const double*, double* s, double (*ks)[ORDER]) {
              Sec::initial_tangent(p.mat, ks);
              for (int j = 0; j < ORDER; ++j) s[j] = 0.0;
            },
            qq, kb0);
      if (needc)
        basic_response(
            g.L, zero_q,
            [&](int i, const double*, double* s, double (*ks)[ORDER]) {
              Sec::committed_tangent(p.mat, committed.point(i), ks);
              for (int j = 0; j < ORDER; ++j) s[j] = 0.0;
            },
            qq, kbc);
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
        lumped_mass(p, xe, m);
        for (int k = 0; k < 12; ++k) fe[k] += damp.alphaM * m[k] * ve[k];
      }
    }
  }

  // The element's matrix-free operator (docs/design.md, "The matrix-free operator"):
  //   y = cK (K_t or, with k.initial, K_0) x + cC (alphaM M + betaK K_t + betaK0 K_0 + betaKc K_c) x
  //       + cM M x,
  // every stiffness applied in the basic system (T^T kb T x, plus PDelta's geometric stiffness
  // of the current axial force with K_t), so no 12 x 12 stiffness is formed (rule 3a); K_t at the
  // trial displacements ue (the sections updated from committed into trial), K_c from the
  // sections' committed tangent (no geometric term, as the damping force). With diagonal, y[a]
  // is the operator's diagonal entry a (the same function applied to unit vectors).
  FEMORAX_HD static void operator_apply(const Params& p, const Rayleigh& damp, const OperatorCoefficients& k,
                                        const double (*xe)[NDM], const double* ue, const double* x,
                                        PointArray<const double> committed, PointArray<double> trial, bool diagonal,
                                        double* y) {
    CrdTransf3dGeometry g;
    geometry(p, xe, g);
    const double cS = k.stiffness, cC = k.damping, cM = k.mass;
    const double zero_q[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    // The basic tangents: current (with q), initial, committed.
    double v[6], q[6], kbt[6][6], kb0[6][6], kbc[6][6], qq[6];
    crd_transf_3d_basic_disp(p.transf, g, ue, v);
    basic_response(
        g.L, v,
        [&](int i, const double* e, double* s, double (*ks)[ORDER]) {
          Sec::update(p.mat, committed.point(i), trial.point(i), e, s, ks);
        },
        q, kbt);
    const bool need0 = k.initial || damp.betaKinit != 0.0, needc = damp.betaKcomm != 0.0;
    if (need0)
      basic_response(
          g.L, zero_q,
          [&](int, const double*, double* s, double (*ks)[ORDER]) {
            Sec::initial_tangent(p.mat, ks);
            for (int j = 0; j < ORDER; ++j) s[j] = 0.0;
          },
          qq, kb0);
    if (needc)
      basic_response(
          g.L, zero_q,
          [&](int i, const double*, double* s, double (*ks)[ORDER]) {
            Sec::committed_tangent(p.mat, committed.point(i), ks);
            for (int j = 0; j < ORDER; ++j) s[j] = 0.0;
          },
          qq, kbc);
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
      else lumped_mass(p, xe, ml);
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

  // The consistent mass in global axes (getMass with cMass; see the top for what FemoraX adds).
  FEMORAX_HD static void consistent_mass(const Params& p, const CrdTransf3dGeometry& g, double (*M)[12]) {
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

  // The 12 x 12 mass matrix (row-major, Me[a * NDOF + c]): consistent with cMass, else the lumped
  // diagonal.
  FEMORAX_HD static void mass(const Params& p, const double (*xe)[NDM], double* Me) {
    for (int a = 0; a < NDOF * NDOF; ++a) Me[a] = 0.0;
    if (p.cMass) {
      CrdTransf3dGeometry g;
      geometry(p, xe, g);
      double M[12][12];
      consistent_mass(p, g, M);
      for (int a = 0; a < 12; ++a)
        for (int b = 0; b < 12; ++b) Me[a * NDOF + b] = M[a][b];
    } else {
      double m[12];
      lumped_mass_dof(p, xe, m);
      for (int a = 0; a < 12; ++a) Me[a * NDOF + a] = m[a];
    }
  }

  // The lumped mass per DOF (12 values, node I then node J): the rotations have their own
  // masses, so the per-node form of the solids does not fit (compile/batches.cpp reads this).
  FEMORAX_HD static void lumped_mass_dof(const Params& p, const double (*xe)[NDM], double m[NDOF]) {
    lumped_mass(p, xe, m);
  }
  FEMORAX_HD static void lumped_mass(const Params& p, const double (*xe)[NDM], double m[NDOF]) {
    CrdTransf3dGeometry g;
    geometry(p, xe, g);
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

  // Largest omega^2 of K phi = omega^2 M phi with the lumped mass and the tangent at the
  // committed state (the element eigenvalue bound). DOFs without mass are left out (their
  // stiffness gives no inertia-driven mode); extra_mass adds the element's share of nodal
  // masses (12 values, may be null).
  FEMORAX_HD static double max_omega_sq(const Params& p, const double (*xe)[NDM], PointArray<const double> committed,
                                        const double* extra_mass = nullptr) {
    CrdTransf3dGeometry g;
    geometry(p, xe, g);
    double kb[6][6], q[6];
    const double v[6] = {0, 0, 0, 0, 0, 0};
    basic_response(
        g.L, v,
        [&](int i, const double*, double* s, double (*ks)[ORDER]) {
          Sec::committed_tangent(p.mat, committed.point(i), ks);
          for (int j = 0; j < ORDER; ++j) s[j] = 0.0;
        },
        q, kb);
    double K[12][12];
    crd_transf_3d_global_stiff(p.transf, g, kb, q, false, K);
    double m[12];
    lumped_mass(p, xe, m);
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

  FEMORAX_HD static double critical_dt(const Params& p, const Rayleigh& damp, ExplicitScheme scheme,
                                       const double (*xe)[NDM], PointArray<const double> committed,
                                       const double* extra_mass = nullptr) {
    return stable_dt(scheme, damp, max_omega_sq(p, xe, committed, extra_mass));
  }
};

}  // namespace femorax
