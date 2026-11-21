#pragma once

// Steel01: bilinear steel with kinematic hardening and optional isotropic hardening.
//
// Port of OpenSees SRC/material/uniaxial/Steel01.cpp, OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea (`uniaxialMaterial Steel01 tag fy E0 b <a1 a2 a3 a4>`,
// defaults a1 = 0, a2 = 55, a3 = 0, a4 = 55). update is setTrialStrain (determineTrialState) with
// getStress and getTangent, from the committed state, with OpenSees' names and operation order.
//
// State (NSTATE doubles per buffer; a state of zeros is the constructor's state, so a fresh
// buffer needs no initialization):
//   0 minStrain   1 maxStrain   2 shiftP - 1   3 shiftN - 1   4 loading (0, 1, -1)
//   5 strain      6 stress      7 hardening (0: the tangent is E0, 1: it is Esh = b E0)
// shift - 1 is exact (shift >= 1 comes from 1 + a pow(...)), and the tangent is one of two
// values, so both round-trip bit for bit. Not ported: the energy and the sensitivities.

#include <cfloat>
#include <cmath>

#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

struct Steel01 {
  static constexpr MaterialKind KIND = MaterialKind::Uniaxial;
  static constexpr int NSTATE = 8;
  static constexpr int NPARAM = 7;

  struct Params {
    double fy, E0, b, a1, a2, a3, a4;  // OpenSees member names
  };
  FEMORAX_HD static Params unpack(const double* v) { return {v[0], v[1], v[2], v[3], v[4], v[5], v[6]}; }
  FEMORAX_HD static void pack(const Params& p, double* v) {
    v[0] = p.fy;
    v[1] = p.E0;
    v[2] = p.b;
    v[3] = p.a1;
    v[4] = p.a2;
    v[5] = p.a3;
    v[6] = p.a4;
  }

  FEMORAX_HD static void update(const Params& p, Strided<const double> committed, Strided<double> trial,
                                double strain, double& stress, double& tangent) {
    const double fy = p.fy, E0 = p.E0, b = p.b;
    const double Esh = b * E0;
    // Reset history variables to last converged state
    const double CminStrain = committed[0], CmaxStrain = committed[1];
    const double CshiftP = committed[2] + 1.0, CshiftN = committed[3] + 1.0;
    const int Cloading = static_cast<int>(committed[4]);
    const double Cstrain = committed[5], Cstress = committed[6];
    const double Ctangent = committed[7] != 0.0 ? Esh : E0;
    double TminStrain = CminStrain, TmaxStrain = CmaxStrain, TshiftP = CshiftP, TshiftN = CshiftN;
    int Tloading = Cloading;
    double Tstrain = Cstrain, Tstress = Cstress, Ttangent = Ctangent;

    // Determine change in strain from last converged state
    const double dStrain = strain - Cstrain;

    if (std::fabs(dStrain) > DBL_EPSILON) {
      Tstrain = strain;
      // determineTrialState(dStrain)
      const double fyOneMinusB = fy * (1.0 - b);
      const double epsy = fy / E0;
      const double c1 = Esh * Tstrain;
      const double c2 = TshiftN * fyOneMinusB;
      const double c3 = TshiftP * fyOneMinusB;
      const double c = Cstress + E0 * dStrain;

      const double c1c3 = c1 + c3;
      if (c1c3 < c)
        Tstress = c1c3;
      else
        Tstress = c;
      const double c1c2 = c1 - c2;
      if (c1c2 > Tstress) Tstress = c1c2;

      if (std::fabs(Tstress - c) < DBL_EPSILON)
        Ttangent = E0;
      else
        Ttangent = Esh;

      // Determine initial loading condition
      if (Tloading == 0 && dStrain != 0.0) {
        if (dStrain > 0.0)
          Tloading = 1;
        else
          Tloading = -1;
      }
      // Transition from loading to unloading
      if (Tloading == 1 && dStrain < 0.0) {
        Tloading = -1;
        if (Cstrain > TmaxStrain) TmaxStrain = Cstrain;
        TshiftN = 1 + p.a1 * std::pow((TmaxStrain - TminStrain) / (2.0 * p.a2 * epsy), 0.8);
      }
      // Transition from unloading to loading
      if (Tloading == -1 && dStrain > 0.0) {
        Tloading = 1;
        if (Cstrain < TminStrain) TminStrain = Cstrain;
        TshiftP = 1 + p.a3 * std::pow((TmaxStrain - TminStrain) / (2.0 * p.a4 * epsy), 0.8);
      }
    }

    trial[0] = TminStrain;
    trial[1] = TmaxStrain;
    trial[2] = TshiftP - 1.0;
    trial[3] = TshiftN - 1.0;
    trial[4] = static_cast<double>(Tloading);
    trial[5] = Tstrain;
    trial[6] = Tstress;
    // Ttangent is E0 or Esh (or the committed one of the two); with b = 1 both are E0.
    trial[7] = (Ttangent == E0) ? 0.0 : 1.0;
    stress = Tstress;
    tangent = Ttangent;
  }

  FEMORAX_HD static double initial_tangent(const Params& p) { return p.E0; }
};

}  // namespace femorax
