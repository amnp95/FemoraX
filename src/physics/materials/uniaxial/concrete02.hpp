#pragma once

// Concrete02: concrete with a linear tension softening branch (Kent-Scott-Park compression
// envelope, Yassin's unloading and reloading rules).
//
// Port of OpenSees SRC/material/uniaxial/Concrete02.cpp, OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea (`uniaxialMaterial Concrete02 tag fc epsc0 fcu epscu
// <rat ft Ets>`; the short form sets rat = 0.1, ft = 0.1 |fc|, Ets = 0.1 fc / epsc0; the
// constructor makes fc, epsc0, fcu, epscu negative: the adapter does both). update is
// setTrialStrain with getStress and getTangent, from the committed state, with OpenSees' names
// and operation order (its single-precision constant .5f included: 0.5 exactly).
//
// State (NSTATE doubles per buffer; zeros are the constructor's state):
//   0 ecmin   1 dept   2 eps   3 sig   4 e (the tangent)   5 formed (0: e is still 2 fc / epsc0)
// OpenSees' quirk kept: a trial strain within DBL_EPSILON of the committed one returns the
// committed state (revertToLastCommit), the committed strain included. Not ported: the energy
// and the sensitivities.

#include <cfloat>
#include <cmath>

#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

struct Concrete02 {
  static constexpr MaterialKind KIND = MaterialKind::Uniaxial;
  static constexpr int NSTATE = 6;
  static constexpr int NPARAM = 7;

  struct Params {
    double fc, epsc0, fcu, epscu, rat, ft, Ets;  // OpenSees member names; fc ... epscu negative
  };
  FEMORAX_HD static Params unpack(const double* v) { return {v[0], v[1], v[2], v[3], v[4], v[5], v[6]}; }
  FEMORAX_HD static void pack(const Params& p, double* v) {
    v[0] = p.fc;
    v[1] = p.epsc0;
    v[2] = p.fcu;
    v[3] = p.epscu;
    v[4] = p.rat;
    v[5] = p.ft;
    v[6] = p.Ets;
  }

  // The monotonic envelope in tension (positive envelope).
  FEMORAX_HD static void Tens_Envlp(const Params& p, double epsc, double& sigc, double& Ect) {
    const double Ec0 = 2.0 * p.fc / p.epsc0;
    const double eps0 = p.ft / Ec0;
    const double epsu = p.ft * (1.0 / p.Ets + 1.0 / Ec0);
    if (epsc <= eps0) {
      sigc = epsc * Ec0;
      Ect = Ec0;
    } else {
      if (epsc <= epsu) {
        Ect = -p.Ets;
        sigc = p.ft - p.Ets * (epsc - eps0);
      } else {
        Ect = 1.0e-10;
        sigc = 0.0;
      }
    }
  }

  // The monotonic envelope in compression (negative envelope).
  FEMORAX_HD static void Compr_Envlp(const Params& p, double epsc, double& sigc, double& Ect) {
    const double Ec0 = 2.0 * p.fc / p.epsc0;
    const double ratLocal = epsc / p.epsc0;
    if (epsc >= p.epsc0) {
      sigc = p.fc * ratLocal * (2.0 - ratLocal);
      Ect = Ec0 * (1.0 - ratLocal);
    } else {
      // linear descending branch between epsc0 and epscu
      if (epsc > p.epscu) {
        sigc = (p.fcu - p.fc) * (epsc - p.epsc0) / (p.epscu - p.epsc0) + p.fc;
        Ect = (p.fcu - p.fc) / (p.epscu - p.epsc0);
      } else {
        // flat friction branch for strains larger than epscu
        sigc = p.fcu;
        Ect = 1.0e-10;
      }
    }
  }

  FEMORAX_HD static void update(const Params& p, Strided<const double> committed, Strided<double> trial,
                                double trialStrain, double& stress, double& tangent) {
    const double fc = p.fc, epsc0 = p.epsc0, fcu = p.fcu, epscu = p.epscu, rat = p.rat;
    const double ec0 = fc * 2. / epsc0;
    const double ecminP = committed[0], deptP = committed[1], epsP = committed[2], sigP = committed[3];
    const double eP = committed[5] != 0.0 ? committed[4] : 2.0 * fc / epsc0;

    // retrieve concrete history variables
    double ecmin = ecminP;
    double dept = deptP;
    // calculate current strain
    double eps = trialStrain;
    double sig = 0.0, e = 0.0;
    const double deps = eps - epsP;

    if (std::fabs(deps) < DBL_EPSILON) {
      // revertToLastCommit
      e = eP;
      sig = sigP;
      eps = epsP;
    } else if (eps < ecmin) {
      // the monotonic envelope in compression; reset the minimum strain
      Compr_Envlp(p, eps, sig, e);
      ecmin = eps;
    } else {
      // point R that determines the reloading slope (Fig. 2.11, Eqs. 2.31 and 2.32 of the EERC report)
      const double epsr = (fcu - rat * ec0 * epscu) / (ec0 * (1.0 - rat));
      const double sigmr = ec0 * epsr;
      // the previous minimum stress from the minimum previous strain and the envelope
      double sigmm;
      double dumy;
      Compr_Envlp(p, ecmin, sigmm, dumy);
      // the reloading slope (Eq. 2.35) and its intersection with the zero stress axis (Eq. 2.36)
      const double er = (sigmm - sigmr) / (ecmin - epsr);
      const double ept = ecmin - sigmm / er;

      if (eps <= ept) {
        const double sigmin = sigmm + er * (eps - ecmin);
        const double sigmax = er * .5f * (eps - ept);
        sig = sigP + ec0 * deps;
        e = ec0;
        if (sig <= sigmin) {
          sig = sigmin;
          e = er;
        }
        if (sig >= sigmax) {
          sig = sigmax;
          e = 0.5 * er;
        }
      } else {
        // the reloading branch in tension: the strain at the peak of the tensile relation (Eq. 2.42)
        const double epn = ept + dept;
        double sicn;
        if (eps <= epn) {
          Tens_Envlp(p, dept, sicn, e);
          if (dept != 0.0) {
            e = sicn / dept;
          } else {
            e = ec0;
          }
          sig = e * (eps - ept);
        } else {
          // the tensile envelope curve shifted by ept
          const double epstmp = eps - ept;
          Tens_Envlp(p, epstmp, sig, e);
          dept = eps - ept;
        }
      }
    }

    trial[0] = ecmin;
    trial[1] = dept;
    trial[2] = eps;
    trial[3] = sig;
    trial[4] = e;
    trial[5] = 1.0;
    stress = sig;
    tangent = e;
  }

  FEMORAX_HD static double initial_tangent(const Params& p) { return 2.0 * p.fc / p.epsc0; }
};

}  // namespace femorax
