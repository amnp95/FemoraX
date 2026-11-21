#pragma once

// ElasticSection3d: an elastic beam section of a 3D frame.
//
// Port of OpenSees SRC/material/section/ElasticSection3d.cpp, OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea (`section Elastic tag E A Iz Iy G J`, the form
// Femora writes). Deformations and resultants in getType() order: P, Mz, My, T
// (axial strain, curvatures about z and y, twist). getStressResultant: s = (EA e0, EIz e1,
// EIy e2, GJ e3); getSectionTangent and getInitialTangent: diag(EA, EIz, EIy, GJ). No history.

#include <array>

#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

struct ElasticSection3d {
  static constexpr MaterialKind KIND = MaterialKind::Section3d;
  static constexpr int NSTRAIN = 4;
  static constexpr int NSTATE = 0;
  // The response named by each deformation (getType order), as a function: nvcc does not use
  // static arrays of the section in device code.
  FEMORAX_HD static constexpr SectionResponse code(int j) {
    constexpr SectionResponse c[NSTRAIN] = {SectionResponse::P, SectionResponse::MZ, SectionResponse::MY,
                                            SectionResponse::T};
    return c[j];
  }
  static constexpr TangentSymmetry TANGENT = TangentSymmetry::SymmetricPositiveDefinite;
  static constexpr bool CONSTANT_TANGENT = true;

  struct Params {
    double E, A, Iz, Iy, G, J;  // OpenSees member names
  };

  // setTrialSectionDeformation(e), then getStressResultant() and getSectionTangent().
  FEMORAX_HD static void update(const Params& p, Strided<const double> committed, Strided<double> trial,
                                const double e[NSTRAIN], double s[NSTRAIN], double (*ks)[NSTRAIN]) {
    (void)committed;
    (void)trial;
    s[0] = p.E * p.A * e[0];
    s[1] = p.E * p.Iz * e[1];
    s[2] = p.E * p.Iy * e[2];
    s[3] = p.G * p.J * e[3];
    initial_tangent(p, ks);
  }

  FEMORAX_HD static void initial_tangent(const Params& p, double (*ks)[NSTRAIN]) {
    for (int i = 0; i < NSTRAIN; ++i)
      for (int j = 0; j < NSTRAIN; ++j) ks[i][j] = 0.0;
    ks[0][0] = p.E * p.A;
    ks[1][1] = p.E * p.Iz;
    ks[2][2] = p.E * p.Iy;
    ks[3][3] = p.G * p.J;
  }

  FEMORAX_HD static void committed_tangent(const Params& p, Strided<const double> committed,
                                           double (*ks)[NSTRAIN]) {
    (void)committed;
    initial_tangent(p, ks);
  }

  // The geometry for the beam's own mass.
  FEMORAX_HD static SectionProperties properties(const Params& p) { return {p.A, p.Iz, p.Iy, p.J}; }
};

static_assert(Section<ElasticSection3d>);

}  // namespace femorax
