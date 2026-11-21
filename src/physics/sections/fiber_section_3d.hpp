#pragma once

// FiberSection3d: a beam section of a 3D frame made of uniaxial fibers, with elastic torsion.
//
// Port of OpenSees SRC/material/section/FiberSection3d.cpp, OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea (`section Fiber tag [-GJ GJ] { fiber, patch, layer }`,
// the form Femora writes; every patch and layer is a list of fibers). Deformations and
// resultants in getType() order: P, Mz, My, T. setTrialSectionDeformation: each fiber's strain
// is d0 - y d1 + z d2 with y, z from the centroid (yBar = sum(y A) / sum(A), accumulated fiber
// by fiber as addFiber does; the adapter computes it, or 0 with -noCentroid), the resultants and
// the tangent are sums over the fibers in the section's fiber order (kept: a different order
// changes the last bits), torsion is s3 = GJ d3 (OpenSees' ElasticMaterial(0, GJ)).
//
// Each fiber's material is chosen at run time (physics/materials/uniaxial/uniaxial.hpp): the
// section holds up to kFiberMaterials parameter sets and an index per fiber. NF is the size
// class (the compile-time maximum number of fibers); numFibers of them are used. State: fiber
// i's kUniaxialState doubles at i * kUniaxialState.
//
// The fiber loop is an inner loop (docs/design.md, "Memory and parallelism contract", rule 3):
// one inner_for updates the fibers (each reads and writes its own state and its two work
// values), then inner_reduce adds each resultant and tangent entry in fiber order, so every
// lane split gives the same bits. The caller provides the work array (2 NF doubles; team
// scratch in lane-parallel layouts).
// Not ported: section integrations (SectionIntegration), a uniaxial torsion material other than
// elastic (-torsion), sensitivities, the energy.

#include "physics/common/inner.hpp"
#include "physics/common/strided.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"
#include "physics/materials/uniaxial/uniaxial.hpp"

namespace femorax {

inline constexpr int kFiberMaterials = 4;  // uniaxial parameter sets per section

template <int NF>
struct FiberSection3d {
  static constexpr MaterialKind KIND = MaterialKind::Section3d;
  static constexpr int NSTRAIN = 4;
  static constexpr int MAX_FIBERS = NF;
  static constexpr int NSTATE = NF * kUniaxialState;
  static constexpr int NWORK = 2 * NF;
  FEMORAX_HD static constexpr SectionResponse code(int j) {
    constexpr SectionResponse c[NSTRAIN] = {SectionResponse::P, SectionResponse::MZ, SectionResponse::MY,
                                            SectionResponse::T};
    return c[j];
  }
  // Softening fibers (concrete past its peak, in tension) make the tangent indefinite.
  static constexpr TangentSymmetry TANGENT = TangentSymmetry::Symmetric;
  static constexpr bool CONSTANT_TANGENT = false;

  struct Params {
    int numFibers = 0;
    double yBar = 0.0, zBar = 0.0;  // the centroid OpenSees subtracts (0 with -noCentroid)
    double GJ = 0.0;                // elastic torsion
    double matData[3 * NF] = {};    // yLoc, zLoc, Area per fiber (OpenSees' layout)
    int material[NF] = {};          // each fiber's parameter set
    UniaxialParams materials[kFiberMaterials] = {};
  };

  // The centroid as OpenSees accumulates it in addFiber (Abar, QzBar, QyBar fiber by fiber).
  FEMORAX_HD static void compute_centroid(Params& p) {
    double Abar = 0.0, QzBar = 0.0, QyBar = 0.0;
    for (int i = 0; i < p.numFibers; ++i) {
      const double yLoc = p.matData[3 * i], zLoc = p.matData[3 * i + 1], Area = p.matData[3 * i + 2];
      Abar += Area;
      QzBar += yLoc * Area;
      QyBar += zLoc * Area;
      p.yBar = QzBar / Abar;
      p.zBar = QyBar / Abar;
    }
  }

  FEMORAX_HD static Strided<const double> fiber(Strided<const double> s, int i) {
    return {s.base + static_cast<std::ptrdiff_t>(i) * kUniaxialState * s.stride, s.stride};
  }
  FEMORAX_HD static Strided<double> fiber(Strided<double> s, int i) {
    return {s.base + static_cast<std::ptrdiff_t>(i) * kUniaxialState * s.stride, s.stride};
  }

  // Sums the section tangent from each fiber's tangent * area (value(i)) in fiber order.
  template <class Inner, class Value>
  FEMORAX_HD static void sum_tangent(const Inner& ctx, const Params& p, const Value& value, double (*ks)[NSTRAIN]) {
    const int n = p.numFibers;
    auto y = [&](int i) { return p.matData[3 * i] - p.yBar; };
    auto z = [&](int i) { return p.matData[3 * i + 1] - p.zBar; };
    for (int a = 0; a < NSTRAIN; ++a)
      for (int b = 0; b < NSTRAIN; ++b) ks[a][b] = 0.0;
    ks[0][0] = inner_reduce(ctx, n, [&](int i) { return value(i); });
    ks[0][1] = inner_reduce(ctx, n, [&](int i) { return -y(i) * value(i); });                 // vas1
    ks[0][2] = inner_reduce(ctx, n, [&](int i) { return z(i) * value(i); });                  // vas2
    ks[1][1] = inner_reduce(ctx, n, [&](int i) { return (-y(i) * value(i)) * -y(i); });       // vas1 * -y
    ks[1][2] = inner_reduce(ctx, n, [&](int i) { return (-y(i) * value(i)) * z(i); });        // vas1as2
    ks[2][2] = inner_reduce(ctx, n, [&](int i) { return (z(i) * value(i)) * z(i); });         // vas2 * z
    ks[1][0] = ks[0][1];
    ks[2][0] = ks[0][2];
    ks[2][1] = ks[1][2];
  }

  // setTrialSectionDeformation(e), then getStressResultant() and getSectionTangent().
  // work: NWORK doubles the caller provides (each fiber's stress * area and tangent * area).
  template <class Inner>
  FEMORAX_HD static void update(const Inner& ctx, const Params& p, Strided<const double> committed,
                                Strided<double> trial, const double e[NSTRAIN], double s[NSTRAIN],
                                double (*ks)[NSTRAIN], double* work) {
    const int n = p.numFibers;
    const double d0 = e[0], d1 = e[1], d2 = e[2], d3 = e[3];
    inner_for(ctx, n, [&](int i) {
      const double y = p.matData[3 * i] - p.yBar;
      const double z = p.matData[3 * i + 1] - p.zBar;
      const double A = p.matData[3 * i + 2];
      // determine material strain and set it
      const double strain = d0 - y * d1 + z * d2;
      double stress, tangent;
      uniaxial_update(p.materials[p.material[i]], fiber(committed, i), fiber(trial, i), strain, stress, tangent);
      work[2 * i] = stress * A;       // fs0
      work[2 * i + 1] = tangent * A;  // value
    });
    sum_tangent(ctx, p, [&](int i) { return work[2 * i + 1]; }, ks);
    s[0] = inner_reduce(ctx, n, [&](int i) { return work[2 * i]; });
    s[1] = inner_reduce(ctx, n, [&](int i) { return work[2 * i] * -(p.matData[3 * i] - p.yBar); });
    s[2] = inner_reduce(ctx, n, [&](int i) { return work[2 * i] * (p.matData[3 * i + 1] - p.zBar); });
    s[3] = p.GJ * d3;
    ks[3][3] = p.GJ;
  }

  // The Section contract's form: plain loops, the work array on the stack.
  FEMORAX_HD static void update(const Params& p, Strided<const double> committed, Strided<double> trial,
                                const double e[NSTRAIN], double s[NSTRAIN], double (*ks)[NSTRAIN]) {
    double work[NWORK];
    update(SerialInner{}, p, committed, trial, e, s, ks, work);
  }

  // getInitialTangent().
  FEMORAX_HD static void initial_tangent(const Params& p, double (*ks)[NSTRAIN]) {
    sum_tangent(SerialInner{}, p,
                [&](int i) { return uniaxial_initial_tangent(p.materials[p.material[i]]) * p.matData[3 * i + 2]; }, ks);
    ks[3][3] = p.GJ;
  }

  // The tangent at the committed state (each fiber's committed tangent).
  FEMORAX_HD static void committed_tangent(const Params& p, Strided<const double> committed, double (*ks)[NSTRAIN]) {
    sum_tangent(SerialInner{}, p,
                [&](int i) {
                  return uniaxial_committed_tangent(p.materials[p.material[i]], fiber(committed, i)) *
                         p.matData[3 * i + 2];
                },
                ks);
    ks[3][3] = p.GJ;
  }

  // The section's geometry for the beam's own mass: the fibers' area and second moments about
  // the centroid; the element's density multiplies them (one density for the member).
  FEMORAX_HD static SectionProperties properties(const Params& p) {
    SectionProperties g;
    for (int i = 0; i < p.numFibers; ++i) {
      const double y = p.matData[3 * i] - p.yBar, z = p.matData[3 * i + 1] - p.zBar, A = p.matData[3 * i + 2];
      g.A += A;
      g.Iz += A * y * y;
      g.Iy += A * z * z;
    }
    g.J = g.Iz + g.Iy;
    return g;
  }
};

}  // namespace femorax
