#pragma once

// ElasticIsotropic, three-dimensional form.
//
// Port of OpenSees SRC/material/nD/ElasticIsotropicThreeDimensional.cpp
// (with ElasticIsotropicMaterial.cpp for the parameters and density), OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea. stdBrick and SSPbrick request the
// "ThreeDimensional" copy of nDMaterial ElasticIsotropic, which is this class.
//
// Variable names and operation order follow getStress()/getTangent(), so results match
// OpenSees bit for bit: stress is computed from the explicit formulas, not as D*eps.
// Strain order is OpenSees': 11 22 33 12 23 31, with engineering shear strains.
//
// OpenSees keeps the trial and committed strain (epsilon, Cepsilon), but the stress is a
// function of the total trial strain only, so no history is needed: NSTATE = 0.

#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

struct ElasticIsotropic {
  static constexpr MaterialKind KIND = MaterialKind::NDMaterial3D;
  static constexpr int NSTRAIN = 6;
  static constexpr int NSTATE = 0;
  static constexpr TangentSymmetry TANGENT = TangentSymmetry::SymmetricPositiveDefinite;
  static constexpr bool CONSTANT_TANGENT = true;

  struct Params {
    double E;    // Young's modulus
    double v;    // Poisson's ratio (OpenSees member name)
    double rho;  // mass density
  };

  // setTrialStrain(strain) followed by getStress() and getTangent().
  // committed and trial are unused (NSTATE = 0); D is written in full (zeros included).
  // State, for materials that have it, is read and written through the strided views:
  // value k of this point is committed[k] (docs/design.md, memory contract rule 1).
  FEMORAX_HD static void update(const Params& p, Strided<const double> committed,
                                Strided<double> trial, const double epsilon[NSTRAIN],
                                double sigma[NSTRAIN], double (*D)[NSTRAIN]) {
    (void)committed;
    (void)trial;
    double mu2 = p.E / (1.0 + p.v);
    const double lam = p.v * mu2 / (1.0 - 2.0 * p.v);
    const double mu = 0.50 * mu2;
    mu2 += lam;

    const double eps0 = epsilon[0];
    const double eps1 = epsilon[1];
    const double eps2 = epsilon[2];

    // sigma = D*epsilon, written out as in getStress()
    sigma[0] = mu2 * eps0 + lam * (eps1 + eps2);
    sigma[1] = mu2 * eps1 + lam * (eps0 + eps2);
    sigma[2] = mu2 * eps2 + lam * (eps0 + eps1);
    sigma[3] = mu * epsilon[3];
    sigma[4] = mu * epsilon[4];
    sigma[5] = mu * epsilon[5];

    fill_tangent(mu2, lam, mu, D);
  }

  // getInitialTangent(): the same matrix.
  FEMORAX_HD static void initial_tangent(const Params& p, double (*D)[NSTRAIN]) {
    double mu2 = p.E / (1.0 + p.v);
    const double lam = p.v * mu2 / (1.0 - 2.0 * p.v);
    const double mu = 0.50 * mu2;
    mu2 += lam;
    fill_tangent(mu2, lam, mu, D);
  }

  // Tangent at the committed state: constant for this material.
  FEMORAX_HD static void committed_tangent(const Params& p, Strided<const double> committed,
                                           double (*D)[NSTRAIN]) {
    (void)committed;
    initial_tangent(p, D);
  }

  // getRho()
  FEMORAX_HD static double density(const Params& p) { return p.rho; }

 private:
  FEMORAX_HD static void fill_tangent(double mu2, double lam, double mu,
                                      double (*D)[NSTRAIN]) {
    for (int i = 0; i < NSTRAIN; ++i)
      for (int j = 0; j < NSTRAIN; ++j) D[i][j] = 0.0;
    D[0][0] = D[1][1] = D[2][2] = mu2;
    D[0][1] = D[1][0] = D[0][2] = D[2][0] = D[1][2] = D[2][1] = lam;
    D[3][3] = mu;
    D[4][4] = mu;
    D[5][5] = mu;
  }
};

static_assert(Material<ElasticIsotropic>);

}  // namespace femorax
