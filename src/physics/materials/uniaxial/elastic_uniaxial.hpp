#pragma once

// ElasticUniaxial: OpenSees' uniaxial Elastic material.
//
// Port of OpenSees SRC/material/uniaxial/ElasticMaterial.cpp, OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea (`uniaxialMaterial Elastic tag E <eta> <Eneg>`).
// getStress: Epos * strain for strain >= 0, Eneg * strain otherwise; getTangent the modulus of
// the strain's sign. State: the strain (for the committed tangent of Epos != Eneg). Not ported: eta (the damping tangent; a fiber section never
// passes a strain rate, so it has no effect there).

#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

struct ElasticUniaxial {
  static constexpr MaterialKind KIND = MaterialKind::Uniaxial;
  static constexpr int NSTATE = 1;
  static constexpr int NPARAM = 2;

  struct Params {
    double Epos, Eneg;  // OpenSees member names
  };
  FEMORAX_HD static Params unpack(const double* v) { return {v[0], v[1]}; }
  FEMORAX_HD static void pack(const Params& p, double* v) {
    v[0] = p.Epos;
    v[1] = p.Eneg;
  }

  // setTrialStrain(strain), then getStress() and getTangent().
  FEMORAX_HD static void update(const Params& p, Strided<const double> committed, Strided<double> trial,
                                double strain, double& stress, double& tangent) {
    (void)committed;
    trial[0] = strain;
    if (strain >= 0.0) {
      stress = p.Epos * strain;
      tangent = p.Epos;
    } else {
      stress = p.Eneg * strain;
      tangent = p.Eneg;
    }
  }

  FEMORAX_HD static double initial_tangent(const Params& p) { return p.Epos; }
  FEMORAX_HD static double committed_tangent(const Params& p, Strided<const double> committed) {
    return committed[0] >= 0.0 ? p.Epos : p.Eneg;
  }
};

}  // namespace femorax
