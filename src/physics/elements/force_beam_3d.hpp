#pragma once

// ForceBeam3d<Section, NIP>: the force-based beam-column (docs/design.md, "Beams: the
// force-based beam"), step 1: the physics with any Section (the elastic section first).
// Ported from OpenSees 955ceb0, SRC/element/forceBeamColumn/ForceBeamColumn3d.cpp (update,
// getTangentStiff, getResistingForce, getInitialFlexibility) and LobattoBeamIntegration.cpp;
// variable names kept close to the originals (Se, kv, vs, fs, Ssr, dvTrial, SeTrial, ...).
//
// Deliberate difference (design.md): OpenSees' update is incremental from the last trial state;
// here every evaluation iterates from the committed state with the whole increment v - v_commit
// (the element contract: compute(committed, trial)). The elastic element is linear: the same
// result to rounding.
//
// Point state (POINT_STATE doubles per point, physics/contracts.hpp point_state_size):
//   [0, ORDER) vs, [ORDER, ORDER + ORDER^2) fs (row major), then ORDER Ssr, the section's own
//   Sec::NSTATE values, and EPP slots of the element values spread over the points:
//   Se (6), kv (36), v (6), and a flag (1 once the state was formed; zeros are a fresh state).
//
// Not yet (later steps): the Rayleigh damping, the mass and the stable step (step 2), the
// registry and kernels (step 3), element loads (FemoraX has none), subdivision of the increment
// (OpenSees' maxSubdivisions defaults to 1: none).

#include <array>
#include <cmath>

#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"
#include "physics/transformations/crd_transf_3d.hpp"

namespace femorax {

// LobattoBeamIntegration::getSectionLocations (mapped to [0, 1]) and getSectionWeights (halved:
// they sum to 1), 3 to 7 points.
template <int NIP>
struct LobattoBeamIntegration {
  static_assert(NIP >= 3 && NIP <= 7, "ForceBeam3d: 3 to 7 Lobatto points");
  FEMORAX_HD static void points(double xi[NIP], double wt[NIP]) {
    if constexpr (NIP == 3) {
      const double x[3] = {-1.0, 0.0, 1.0};
      const double w[3] = {3.33333333333333e-01, 1.33333333333333e+00, 3.33333333333334e-01};
      fill(x, w, xi, wt);
    } else if constexpr (NIP == 4) {
      const double x[4] = {-1.0, -4.47213595499958e-01, 4.47213595499958e-01, 1.0};
      const double w[4] = {1.66666666666667e-01, 8.33333333333333e-01, 8.33333333333334e-01,
                           1.66666666666666e-01};
      fill(x, w, xi, wt);
    } else if constexpr (NIP == 5) {
      const double x[5] = {-1.0, -6.54653670707977e-01, 0.0, 6.54653670707977e-01, 1.0};
      const double w[5] = {1.00000000000000e-01, 5.44444444444445e-01, 7.11111111111111e-01,
                           5.44444444444444e-01, 1.00000000000000e-01};
      fill(x, w, xi, wt);
    } else if constexpr (NIP == 6) {
      const double x[6] = {-1.0, -7.65055323929465e-01, -2.85231516480645e-01, 2.85231516480645e-01,
                           7.65055323929465e-01, 1.0};
      const double w[6] = {6.66666666666666e-02, 3.78474956297848e-01, 5.54858377035486e-01,
                           5.54858377035487e-01, 3.78474956297848e-01, 6.66666666666668e-02};
      fill(x, w, xi, wt);
    } else {
      const double x[7] = {-1.0, -8.30223896278567e-01, -4.68848793470714e-01, 0.0, 4.68848793470714e-01,
                           8.30223896278567e-01, 1.0};
      const double w[7] = {4.76190476190473e-02, 2.76826047361566e-01, 4.31745381209863e-01,
                           4.87619047619048e-01, 4.31745381209863e-01, 2.76826047361567e-01,
                           4.76190476190480e-02};
      fill(x, w, xi, wt);
    }
  }
  FEMORAX_HD static void fill(const double* x, const double* w, double* xi, double* wt) {
    for (int i = 0; i < NIP; ++i) {
      xi[i] = 0.5 * (x[i] + 1.0);
      wt[i] = w[i] * 0.5;
    }
  }
};

namespace force_beam_detail {

// The inverse of an n x n matrix (n <= 6) by Gauss-Jordan with partial pivoting; false when
// singular (OpenSees' Matrix::Solve with the identity).
template <int N>
FEMORAX_HD inline bool invert(const double (&A)[N][N], double (&Ainv)[N][N]) {
  double a[N][2 * N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      a[i][j] = A[i][j];
      a[i][N + j] = i == j ? 1.0 : 0.0;
    }
  for (int c = 0; c < N; ++c) {
    int piv = c;
    for (int r = c + 1; r < N; ++r)
      if (std::fabs(a[r][c]) > std::fabs(a[piv][c])) piv = r;
    if (a[piv][c] == 0.0) return false;
    if (piv != c)
      for (int j = 0; j < 2 * N; ++j) {
        const double t = a[c][j];
        a[c][j] = a[piv][j];
        a[piv][j] = t;
      }
    const double d = 1.0 / a[c][c];
    for (int j = 0; j < 2 * N; ++j) a[c][j] *= d;
    for (int r = 0; r < N; ++r) {
      if (r == c) continue;
      const double f = a[r][c];
      if (f == 0.0) continue;
      for (int j = 0; j < 2 * N; ++j) a[r][j] -= f * a[c][j];
    }
  }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) Ainv[i][j] = a[i][N + j];
  return true;
}

}  // namespace force_beam_detail

template <class Sec, int NIP_>
struct ForceBeam3d {
  using Material = Sec;
  static constexpr int NDM = 3, NNODE = 2, NEXTRA = 0, NGP = NIP_, NDOF = 12, NEBD = 6;
  static constexpr std::array<int, NNODE> NODE_NDF = {6, 6};
  static constexpr int ORDER = Sec::NSTRAIN;
  static constexpr int NEV = 6 + 36 + 6 + 1;              // Se, kv, v, flag
  static constexpr int EPP = (NEV + NGP - 1) / NGP;       // element values per point
  static constexpr int OFF_FS = ORDER, OFF_SSR = ORDER + ORDER * ORDER, OFF_SEC = OFF_SSR + ORDER;
  static constexpr int OFF_EV = OFF_SEC + Sec::NSTATE;
  static constexpr int POINT_STATE = OFF_EV + EPP;
  static constexpr int NSTATE = NGP * POINT_STATE;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::Section3d;
  static constexpr bool GEOMETRY_CACHEABLE = false;
  static constexpr double DefaultLoverGJ = 1.0e-10;  // OpenSees, for sections without torsion

  struct Params {
    typename Sec::Params mat;
    CrdTransf3dParams transf;
    double rhoA = 0.0;
    double rot[3] = {0.0, 0.0, 0.0};
    int cMass = 0;
    int maxIters = 10;    // -iter maxIters tol (Femora's defaults)
    double tol = 1.0e-12;
  };

  // The element state, gathered from the points (or written back to them).
  struct State {
    double Se[NEBD], kv[NEBD][NEBD], v[NEBD];
    double vs[NGP][ORDER], fs[NGP][ORDER][ORDER], Ssr[NGP][ORDER];
  };

  FEMORAX_HD static bool has_torsion() {
    for (int j = 0; j < ORDER; ++j)
      if (Sec::CODE[j] == SectionResponse::T) return true;
    return false;
  }

  // A point's values from k on (the section's own state from OFF_SEC).
  template <class T>
  FEMORAX_HD static Strided<T> from(Strided<T> pt, int k) {
    return {pt.base + static_cast<std::ptrdiff_t>(k) * pt.stride, pt.stride};
  }
  // The element value k (Se, kv, v, flag) in the points' extra slots.
  template <class T>
  FEMORAX_HD static T& ev(PointArray<T> s, int k) {
    return s.point(k / EPP)[OFF_EV + k % EPP];
  }

  // Section forces Ss = b(xL) Se at a point (update: Ss(ii) by code).
  FEMORAX_HD static void section_forces(double xL, double oneOverL, const double Se[NEBD], double Ss[ORDER]) {
    const double xL1 = xL - 1.0;
    for (int ii = 0; ii < ORDER; ii++) {
      switch (Sec::CODE[ii]) {
        case SectionResponse::P: Ss[ii] = Se[0]; break;
        case SectionResponse::MZ: Ss[ii] = xL1 * Se[1] + xL * Se[2]; break;
        case SectionResponse::VY: Ss[ii] = oneOverL * (Se[1] + Se[2]); break;
        case SectionResponse::MY: Ss[ii] = xL1 * Se[3] + xL * Se[4]; break;
        case SectionResponse::VZ: Ss[ii] = oneOverL * (Se[3] + Se[4]); break;
        case SectionResponse::T: Ss[ii] = Se[5]; break;
        default: Ss[ii] = 0.0; break;
      }
    }
  }

  // f += b^T fSec b wtL (update's fb and f loops, by code).
  FEMORAX_HD static void add_flexibility(double xL, double oneOverL, double wtL, const double (&fSec)[ORDER][ORDER],
                                         double (&f)[NEBD][NEBD]) {
    const double xL1 = xL - 1.0;
    double fb[ORDER][NEBD] = {};
    double tmp;
    for (int ii = 0; ii < ORDER; ii++) {
      switch (Sec::CODE[ii]) {
        case SectionResponse::P:
          for (int jj = 0; jj < ORDER; jj++) fb[jj][0] += fSec[jj][ii] * wtL;
          break;
        case SectionResponse::MZ:
          for (int jj = 0; jj < ORDER; jj++) {
            tmp = fSec[jj][ii] * wtL;
            fb[jj][1] += xL1 * tmp;
            fb[jj][2] += xL * tmp;
          }
          break;
        case SectionResponse::VY:
          for (int jj = 0; jj < ORDER; jj++) {
            tmp = oneOverL * fSec[jj][ii] * wtL;
            fb[jj][1] += tmp;
            fb[jj][2] += tmp;
          }
          break;
        case SectionResponse::MY:
          for (int jj = 0; jj < ORDER; jj++) {
            tmp = fSec[jj][ii] * wtL;
            fb[jj][3] += xL1 * tmp;
            fb[jj][4] += xL * tmp;
          }
          break;
        case SectionResponse::VZ:
          for (int jj = 0; jj < ORDER; jj++) {
            tmp = oneOverL * fSec[jj][ii] * wtL;
            fb[jj][3] += tmp;
            fb[jj][4] += tmp;
          }
          break;
        case SectionResponse::T:
          for (int jj = 0; jj < ORDER; jj++) fb[jj][5] += fSec[jj][ii] * wtL;
          break;
        default: break;
      }
    }
    for (int ii = 0; ii < ORDER; ii++) {
      switch (Sec::CODE[ii]) {
        case SectionResponse::P:
          for (int jj = 0; jj < NEBD; jj++) f[0][jj] += fb[ii][jj];
          break;
        case SectionResponse::MZ:
          for (int jj = 0; jj < NEBD; jj++) {
            tmp = fb[ii][jj];
            f[1][jj] += xL1 * tmp;
            f[2][jj] += xL * tmp;
          }
          break;
        case SectionResponse::VY:
          for (int jj = 0; jj < NEBD; jj++) {
            tmp = oneOverL * fb[ii][jj];
            f[1][jj] += tmp;
            f[2][jj] += tmp;
          }
          break;
        case SectionResponse::MY:
          for (int jj = 0; jj < NEBD; jj++) {
            tmp = fb[ii][jj];
            f[3][jj] += xL1 * tmp;
            f[4][jj] += xL * tmp;
          }
          break;
        case SectionResponse::VZ:
          for (int jj = 0; jj < NEBD; jj++) {
            tmp = oneOverL * fb[ii][jj];
            f[3][jj] += tmp;
            f[4][jj] += tmp;
          }
          break;
        case SectionResponse::T:
          for (int jj = 0; jj < NEBD; jj++) f[5][jj] += fb[ii][jj];
          break;
        default: break;
      }
    }
  }

  // vr += b^T dvs wtL (update's residual integration, by code).
  FEMORAX_HD static void add_deformation(double xL, double oneOverL, double wtL, const double dvs[ORDER],
                                         double vr[NEBD]) {
    const double xL1 = xL - 1.0;
    for (int ii = 0; ii < ORDER; ii++) {
      const double dei = dvs[ii] * wtL;
      double tmp;
      switch (Sec::CODE[ii]) {
        case SectionResponse::P: vr[0] += dei; break;
        case SectionResponse::MZ:
          vr[1] += xL1 * dei;
          vr[2] += xL * dei;
          break;
        case SectionResponse::VY:
          tmp = oneOverL * dei;
          vr[1] += tmp;
          vr[2] += tmp;
          break;
        case SectionResponse::MY:
          vr[3] += xL1 * dei;
          vr[4] += xL * dei;
          break;
        case SectionResponse::VZ:
          tmp = oneOverL * dei;
          vr[3] += tmp;
          vr[4] += tmp;
          break;
        case SectionResponse::T: vr[5] += dei; break;
        default: break;
      }
    }
  }

  // The section's initial flexibility fs0 = ks0^-1.
  FEMORAX_HD static void initial_flexibility(const typename Sec::Params& p, double (&fs0)[ORDER][ORDER]) {
    double ks0[ORDER][ORDER];
    Sec::initial_tangent(p, ks0);
    force_beam_detail::invert<ORDER>(ks0, fs0);
  }

  // getInitialFlexibility, inverted: the initial basic stiffness kv0.
  FEMORAX_HD static void initial_basic_stiffness(const Params& p, double L, double (&kv0)[NEBD][NEBD]) {
    double xi[NGP], wt[NGP];
    LobattoBeamIntegration<NGP>::points(xi, wt);
    double fs0[ORDER][ORDER];
    initial_flexibility(p.mat, fs0);
    double f[NEBD][NEBD] = {};
    for (int i = 0; i < NGP; ++i) add_flexibility(xi[i], 1.0 / L, wt[i] * L, fs0, f);
    if (!has_torsion()) f[5][5] = DefaultLoverGJ;
    force_beam_detail::invert<NEBD>(f, kv0);
  }

  // The committed state, a fresh one (zeros) made the initial state of setDomain
  // (initializeSectionHistoryVariables: vs = 0, fs = fs0, Ssr = 0; kv from the initial
  // flexibility; Se = 0).
  FEMORAX_HD static void load_state(const Params& p, double L, PointArray<const double> c, State& s) {
    if (ev(c, NEV - 1) == 0.0) {
      double fs0[ORDER][ORDER];
      initial_flexibility(p.mat, fs0);
      for (int a = 0; a < NEBD; ++a) {
        s.Se[a] = 0.0;
        s.v[a] = 0.0;
      }
      initial_basic_stiffness(p, L, s.kv);
      for (int i = 0; i < NGP; ++i)
        for (int a = 0; a < ORDER; ++a) {
          s.vs[i][a] = 0.0;
          s.Ssr[i][a] = 0.0;
          for (int b = 0; b < ORDER; ++b) s.fs[i][a][b] = fs0[a][b];
        }
      return;
    }
    for (int a = 0; a < NEBD; ++a) {
      s.Se[a] = ev(c, a);
      s.v[a] = ev(c, 42 + a);
      for (int b = 0; b < NEBD; ++b) s.kv[a][b] = ev(c, 6 + 6 * a + b);
    }
    for (int i = 0; i < NGP; ++i) {
      const auto pt = c.point(i);
      for (int a = 0; a < ORDER; ++a) {
        s.vs[i][a] = pt[a];
        s.Ssr[i][a] = pt[OFF_SSR + a];
        for (int b = 0; b < ORDER; ++b) s.fs[i][a][b] = pt[OFF_FS + ORDER * a + b];
      }
    }
  }

  FEMORAX_HD static void store_state(const State& s, PointArray<double> t) {
    for (int a = 0; a < NEBD; ++a) {
      ev(t, a) = s.Se[a];
      ev(t, 42 + a) = s.v[a];
      for (int b = 0; b < NEBD; ++b) ev(t, 6 + 6 * a + b) = s.kv[a][b];
    }
    ev(t, NEV - 1) = 1.0;
    for (int i = 0; i < NGP; ++i) {
      const auto pt = t.point(i);
      for (int a = 0; a < ORDER; ++a) {
        pt[a] = s.vs[i][a];
        pt[OFF_SSR + a] = s.Ssr[i][a];
        for (int b = 0; b < ORDER; ++b) pt[OFF_FS + ORDER * a + b] = s.fs[i][a][b];
      }
    }
  }

  // update(): the element state compatible with the basic displacement v, from the committed
  // state s (overwritten with the result). Returns false when no scheme converged (the step
  // fails). The sections' own state goes from committed to trial at every point.
  FEMORAX_HD static bool update(const Params& p, double L, const double v[NEBD], PointArray<const double> committed,
                                PointArray<double> trial, State& s) {
    double dv[NEBD], vin[NEBD];
    for (int a = 0; a < NEBD; ++a) {
      dv[a] = v[a] - s.v[a];
      vin[a] = s.v[a];
    }
    const double oneOverL = 1.0 / L;
    double xi[NGP], wt[NGP];
    LobattoBeamIntegration<NGP>::points(xi, wt);
    double fs0[ORDER][ORDER];
    initial_flexibility(p.mat, fs0);
    const State start = s;
    for (int l = 0; l < 3; l++) {
      State t = start;  // SeTrial = Se; kvTrial = kv; vsSubdivide = vs; ...
      double dSe[NEBD];
      for (int a = 0; a < NEBD; ++a) {
        dSe[a] = 0.0;
        for (int b = 0; b < NEBD; ++b) dSe[a] += t.kv[a][b] * dv[b];
      }
      for (int a = 0; a < NEBD; ++a) t.Se[a] += dSe[a];
      const int numIters = l == 1 ? 10 * p.maxIters : p.maxIters;
      for (int j = 0; j < numIters; j++) {
        double f[NEBD][NEBD] = {}, vr[NEBD] = {};
        for (int i = 0; i < NGP; i++) {
          const double xL = xi[i], wtL = wt[i] * L;
          double Ss[ORDER], dSs[ORDER], dvs[ORDER];
          section_forces(xL, oneOverL, t.Se, Ss);
          for (int a = 0; a < ORDER; ++a) dSs[a] = Ss[a] - t.Ssr[i][a];
          const bool initial = l == 1 || (l == 2 && j == 0);
          for (int a = 0; a < ORDER; ++a) {
            dvs[a] = 0.0;
            for (int b = 0; b < ORDER; ++b) dvs[a] += (initial ? fs0[a][b] : t.fs[i][a][b]) * dSs[b];
          }
          for (int a = 0; a < ORDER; ++a) t.vs[i][a] += dvs[a];
          double ks[ORDER][ORDER];
          Sec::update(p.mat, from(committed.point(i), OFF_SEC), from(trial.point(i), OFF_SEC), t.vs[i], t.Ssr[i],
                      ks);
          force_beam_detail::invert<ORDER>(ks, t.fs[i]);
          for (int a = 0; a < ORDER; ++a) dSs[a] = Ss[a] - t.Ssr[i][a];
          for (int a = 0; a < ORDER; ++a) {
            dvs[a] = 0.0;
            for (int b = 0; b < ORDER; ++b) dvs[a] += t.fs[i][a][b] * dSs[b];
          }
          add_flexibility(xL, oneOverL, wtL, t.fs[i], f);
          for (int a = 0; a < ORDER; ++a) dvs[a] += t.vs[i][a];
          add_deformation(xL, oneOverL, wtL, dvs, vr);
        }
        if (!has_torsion()) {
          f[5][5] = DefaultLoverGJ;
          vr[5] = t.Se[5] * DefaultLoverGJ;
        }
        if (!force_beam_detail::invert<NEBD>(f, t.kv)) return false;
        double r[NEBD];
        for (int a = 0; a < NEBD; ++a) r[a] = vin[a] + dv[a] - vr[a];  // dv = vin + dvTrial - vr
        double dW = 0.0;
        for (int a = 0; a < NEBD; ++a) {
          dSe[a] = 0.0;
          for (int b = 0; b < NEBD; ++b) dSe[a] += t.kv[a][b] * r[b];
          dW += r[a] * dSe[a];
        }
        for (int a = 0; a < NEBD; ++a) t.Se[a] += dSe[a];
        if (std::fabs(dW) < p.tol) {
          for (int a = 0; a < NEBD; ++a) t.v[a] = v[a];
          s = t;
          return true;
        }
      }
    }
    return false;
  }

  FEMORAX_HD static int geometry(const Params& p, const double (*xe)[NDM], CrdTransf3dGeometry& g) {
    return crd_transf_3d_geometry(p.transf, xe[0], xe[1], g);
  }

  // getResistingForce (q = Se) and, with WantK, getTangentStiff (kb = kv) through the
  // transformation, the state updated from committed into trial. Returns false when the
  // element iteration did not converge.
  template <bool WantK>
  FEMORAX_HD static bool compute(const Params& p, const double (*xe)[NDM], const double* ue,
                                 PointArray<const double> committed, PointArray<double> trial, double* fe,
                                 double (*Ke)[NDOF]) {
    CrdTransf3dGeometry g;
    geometry(p, xe, g);
    double v[NEBD];
    crd_transf_3d_basic_disp(p.transf, g, ue, v);
    State s;
    load_state(p, g.L, committed, s);
    const bool ok = update(p, g.L, v, committed, trial, s);
    store_state(s, trial);
    const double p0[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    crd_transf_3d_global_force(p.transf, g, ue, s.Se, p0, fe);
    if constexpr (WantK) crd_transf_3d_global_stiff(p.transf, g, s.kv, s.Se, true, Ke);
    return ok;
  }
};

}  // namespace femorax
