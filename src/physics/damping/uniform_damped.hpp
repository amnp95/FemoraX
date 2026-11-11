#pragma once

// UniformDamped<Mat, NF>: Uniform damping as a material wrapper (docs/design.md, "The step
// context and the damped-material wrapper"). The wrapped material is called unchanged; its
// stress at each point is filtered (physics/damping/uniform_damping.hpp) and the damping
// stress qd is added to it, and the tangent is multiplied by the filters' stiffness
// multiplier, as OpenSees' Brick does (`dd *= theDamping->getStiffnessMultiplier()`). Every
// element that updates its material through material_update (physics/contracts.hpp) then
// carries the damping force B^T qd dvol in its own force, in every kernel variant.
//
// NF is the most filters a definition may use (a compile-time bucket, like the yield
// surfaces of the multi-yield materials); the definition's own count, Params::fl.nFilter,
// may be smaller.
//
// State per point: the wrapped material's Mat::NSTATE values, then the filter state of its
// NSTRAIN stresses, NSTRAIN * (NF + 2) values (uniform_damping_update's layout).
//
// Not yet here (docs/design.md, step 2b-2): the stiffness correction at the band's geometric
// mean, the -fact time series, the material stage of wrapped multi-yield materials, and the
// compacted (predictor / corrector) path.

#include "physics/contracts.hpp"
#include "physics/damping/uniform_damping.hpp"
#include "physics/macros.hpp"

namespace femorax {

template <class Mat, int NF>
struct UniformDamped {
  static_assert(Material<Mat>);
  static_assert(NF >= 2 && NF <= kUniformDampingMaxFilters);
  using Base = Mat;
  static constexpr MaterialKind KIND = Mat::KIND;
  static constexpr int NSTRAIN = Mat::NSTRAIN;
  static constexpr int NDAMP = NSTRAIN * (NF + 2);
  static constexpr int NSTATE = Mat::NSTATE + NDAMP;
  static constexpr TangentSymmetry TANGENT = Mat::TANGENT;
  // The tangent depends on the step size (the stiffness multiplier).
  static constexpr bool CONSTANT_TANGENT = false;

  struct Params {
    typename Mat::Params mat;
    UniformDampingFilters fl;  // fl.nFilter <= NF
    double eta = 0.0;          // the loss factor, uniform_damping_eta(ratio)
    double ta = 0.0;           // activation time
    double td = 1e20;          // deactivation time
  };

  // The wrapped material's part of a point's state, and the filters' part.
  template <class T>
  FEMORAX_HD static Strided<T> material_part(Strided<T> s) {
    return s;
  }
  template <class T>
  FEMORAX_HD static Strided<T> damping_part(Strided<T> s) {
    return {s.base + static_cast<std::ptrdiff_t>(Mat::NSTATE) * s.stride, s.stride};
  }

  FEMORAX_HD static UniformDampingStep step_of(const StepContext& step, const Params& p) {
    UniformDampingStep s;
    s.dynamic = step.dynamic != 0;
    s.t = step.t;
    s.dT = step.dT;
    s.ta = p.ta;
    s.td = p.td;
    return s;
  }

  // The update with the step: the material's stress plus the damping stress, the material's
  // tangent times the stiffness multiplier.
  FEMORAX_HD static void update(const StepContext& step, const Params& p, Strided<const double> committed,
                                Strided<double> trial, const double* eps, double* sig, double (*D)[NSTRAIN]) {
    Mat::update(p.mat, material_part(committed), material_part(trial), eps, sig, D);
    const UniformDampingStep s = step_of(step, p);
    double qd[NSTRAIN];
    uniform_damping_update<NSTRAIN>(p.fl, p.eta, s, damping_part(committed), damping_part(trial), sig, qd);
    const double km = uniform_damping_stiffness_multiplier(p.fl, p.eta, s);
    for (int a = 0; a < NSTRAIN; ++a) {
      sig[a] += qd[a];
      for (int c = 0; c < NSTRAIN; ++c) D[a][c] *= km;
    }
  }
  // Without a step (the material concept's form): a static update.
  FEMORAX_HD static void update(const Params& p, Strided<const double> committed, Strided<double> trial,
                                const double* eps, double* sig, double (*D)[NSTRAIN]) {
    update(StepContext{}, p, committed, trial, eps, sig, D);
  }

  // The material's initial state; the filters start at rest (OpenSees: revertToStart zeros).
  FEMORAX_HD static void initial_state(const Params& p, Strided<double> s) {
    if constexpr (requires { Mat::initial_state(p.mat, s); }) Mat::initial_state(p.mat, material_part(s));
    const Strided<double> d = damping_part(s);
    for (int k = 0; k < NDAMP; ++k) d[k] = 0.0;
  }

  FEMORAX_HD static void initial_tangent(const Params& p, double (*D)[NSTRAIN]) { Mat::initial_tangent(p.mat, D); }
  FEMORAX_HD static void committed_tangent(const Params& p, Strided<const double> committed, double (*D)[NSTRAIN]) {
    Mat::committed_tangent(p.mat, material_part(committed), D);
  }
  FEMORAX_HD static double density(const Params& p) { return Mat::density(p.mat); }

  // The stable step's tangent: the material's, times the largest stiffness multiplier
  // (1 + 2 eta sum alpha_i, its limit for a vanishing step), a bound that needs no step size.
  FEMORAX_HD static void stability_tangent(const Params& p, Strided<const double> committed, double (*D)[NSTRAIN]) {
    femorax::stability_tangent<Mat>(p.mat, material_part(committed), D);
    double km = 1.0;
    for (int i = 0; i < p.fl.nFilter; ++i) km += 2.0 * p.eta * p.fl.alpha[i];
    for (int a = 0; a < NSTRAIN; ++a)
      for (int c = 0; c < NSTRAIN; ++c) D[a][c] *= km;
  }

  // What a recorder reads (getStress): the material's own stress, without the damping stress,
  // as OpenSees' recorders report it.
  FEMORAX_HD static void recorded_stress(const Params& p, Strided<const double> committed, Strided<double> scratch,
                                         const double* eps, double* out) {
    if constexpr (requires { Mat::recorded_stress(p.mat, committed, scratch, eps, out); }) {
      Mat::recorded_stress(p.mat, material_part(committed), scratch, eps, out);
    } else {
      static_assert(Mat::NSTATE == 0, "a material with history state needs recorded_stress");
      double D[NSTRAIN][NSTRAIN];
      Mat::update(p.mat, material_part(committed), {}, eps, out, D);
    }
  }
};

}  // namespace femorax
