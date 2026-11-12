#pragma once

// PressureIndependMultiYield, three-dimensional form (Elgamal, Yang and Parra's multi-yield
// surface plasticity with a von Mises type surface, for clay and other pressure-independent
// materials; friction angle > 0 makes the moduli and surfaces depend on the confinement at
// the switch to the plastic stage).
//
// Port of OpenSees SRC/material/nD/soil/PressureIndependMultiYield.cpp (with T2Vector.cpp
// and MultiYieldSurface.cpp: physics/common/t2_vector.hpp), OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea. Function and variable names follow the
// original (getStress, setTrialStress, stressCorrection, updateActiveSurface, paramScaling,
// ...), and every operation keeps its order, so the two can be read side by side and agree
// bit for bit on host compilers that do not contract multiply-adds.
//
// What changes, and why (docs/deviations.md, "PressureIndependMultiYield"):
// - OpenSees' per-material static arrays (loadStagex, frictionAnglex, ...) are the
//   material's Params; its static workspaces (theTangent, subStrainRate and the
//   function-local statics) are locals. Each function's inputs were checked so that no
//   value crosses functions through a workspace (getTangent's devia only meets a zero
//   coefficient when it is stale).
// - Per point, OpenSees' instance members refShearModulus, refBulkModulus and the surface
//   sizes and moduli change at the switch to the plastic stage (paramScaling scales them by
//   the confinement there). They are point state here (G, K, sizes, moduli), starting from
//   the material's values. The yield surfaces are a fixed-size array (NS, a compile-time
//   bucket); surface 0 is OpenSees' unused surface (size 0, modulus 0, center 0).
// - elast2Plast changes the committed state on the first call after the switch. Here it
//   runs inside update on a copy of the committed state, whose result becomes committed at
//   commit; a pure function of the committed state, so the result is the same and a step
//   that is reverted and redone sees the same values.
// - Recursion (stressCorrection, deviatorScaling) is written as loops; OpenSees' exit() on
//   impossible states (surface motion errors) gives NaN stress here, so the step fails
//   instead of the program.
// - The initial tangent (getInitialTangent) is the material's reference elastic tangent;
//   OpenSees returns the point's scaled one after the switch. The two places that read it
//   after the switch (betaKinit damping and ModifiedNewton -initial) are refused with this
//   material (io/support.cpp).
//
// State per point (STATE layout below): committed and trial buffers of NSTATE doubles,
// 30 + 8 NS. update writes every trial value, so commit is a swap of the buffers.
//
// Strain order: OpenSees', 11 22 33 12 23 31 with engineering shear strains; stresses in the
// same order. Compression is negative.

#include <cmath>
#include <limits>

#include "physics/common/inner.hpp"
#include "physics/common/strided.hpp"
#include "physics/common/t2_vector.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

// The constructor's arguments (OpenSees order and the Tcl parser's defaults).
struct PressureIndependMultiYieldInput {
  double rho, refShearModul, refBulkModul, cohesi, peakShearStra, frictionAng, refPress, pressDependCoe;
  int numberOfYieldSurf;  // positive; with gredu, the number of user surfaces
  const double* gredu;    // nullptr, or numberOfYieldSurf (gamma, G/Gmax) pairs
};

template <int NS>
struct PressureIndependMultiYield {
  static_assert(NS >= 1 && NS <= 100, "OpenSees allows at most 100 yield surfaces");
  static constexpr MaterialKind KIND = MaterialKind::NDMaterial3D;
  static constexpr int NSTRAIN = 6;
  static constexpr int MAX_SURFACES = NS;
  // Symmetric (elastic minus a rank-one term along the active surface's normal); positive
  // semi-definite: the outermost surface has plastic modulus 0.
  static constexpr TangentSymmetry TANGENT = TangentSymmetry::SymmetricPositiveSemidefinite;
  static constexpr bool CONSTANT_TANGENT = false;

  // --- State layout (offsets in one buffer) -------------------------------------------
  struct STATE {
    static constexpr int STRESS = 0;   // currentStress / trialStress
    static constexpr int STRAIN = 13;  // currentStrain (trial: the strain at commit)
    static constexpr int ACTIVE = 26;  // activeSurfaceNum / committedActiveSurf
    static constexpr int E2P = 27;
    static constexpr int G = 28;  // refShearModulus of the instance (scaled at the switch)
    static constexpr int K = 29;  // refBulkModulus of the instance
    static constexpr int SIZES = 30;        // surface i (1..NS): SIZES + i - 1
    static constexpr int MODULI = 30 + NS;  // surface i: MODULI + i - 1
    static constexpr int CENTERS = 30 + 2 * NS;  // surface i: CENTERS + 6 (i - 1)
  };
  static constexpr int NSTATE = STATE::CENTERS + 6 * NS;

  struct Params {
    int loadStage;      // 0 elastic, 1 plastic (updateMaterialStage)
    int numOfSurfaces;  // <= NS
    double rho, refShearModulus, refBulkModulus, frictionAngle, peakShearStrain;
    double refPressure;  // negative (compression), as OpenSees stores it
    double cohesion, pressDependCoeff, residualPress;
    double size[NS + 1];     // the surfaces setUpSurfaces builds; [0] = 0 (unused surface)
    double modulus[NS + 1];  // their plastic moduli; [0] = 0
  };

  using Input = PressureIndependMultiYieldInput;

  enum Setup : int {
    Ok = 0,
    TooManySurfaces = 1,       // numberOfYieldSurf > NS (choose a larger bucket)
    InvalidFrictionAngle = 2,  // user surfaces: sin(phi) outside [0, 1] (OpenSees exits)
    NegativeModulus = 3,       // user surfaces: a plastic modulus <= 0 (OpenSees exits)
    TooFewPairs = 4,           // user surfaces: fewer than 2 pairs (OpenSees reads garbage)
    BadInput = 5,              // a value OpenSees treats as fatal
  };

  // Constructor checks plus setUpSurfaces. Values OpenSees resets with a warning are reset
  // the same way.
  FEMORAX_HD static int setup(const Input& in, Params& p) {
    double r = in.rho, frictionAng = in.frictionAng, cohesi = in.cohesi, pressDependCoe = in.pressDependCoe;
    int numberOfYieldSurf = in.numberOfYieldSurf;
    if (in.refShearModul <= 0 || in.refBulkModul <= 0) return BadInput;
    if (frictionAng < 0.) frictionAng = 0.;
    if (frictionAng == 0. && cohesi <= 0.) return BadInput;
    if (cohesi <= 0) cohesi = 0.;
    if (in.peakShearStra <= 0 || in.refPress <= 0) return BadInput;
    if (pressDependCoe < 0) pressDependCoe = 0.;
    if (pressDependCoe > 0 && frictionAng == 0) pressDependCoe = 0.;
    if (numberOfYieldSurf <= 0) numberOfYieldSurf = 10;
    if (numberOfYieldSurf > 100) numberOfYieldSurf = 100;
    if (r < 0) r = 0.;
    if (numberOfYieldSurf > NS) return TooManySurfaces;
    if (in.gredu != nullptr && numberOfYieldSurf < 2) return TooFewPairs;

    p.loadStage = 0;
    p.numOfSurfaces = numberOfYieldSurf;
    p.rho = r;
    p.refShearModulus = in.refShearModul;
    p.refBulkModulus = in.refBulkModul;
    p.frictionAngle = frictionAng;
    p.peakShearStrain = in.peakShearStra;
    p.refPressure = -in.refPress;  // compression is negative
    p.cohesion = cohesi;
    p.pressDependCoeff = pressDependCoe;
    p.residualPress = 0.;
    for (int i = 0; i <= NS; ++i) p.size[i] = p.modulus[i] = 0.0;
    return setUpSurfaces(in.gredu, p);
  }

  // Default state of a new point (the constructor's member values and its surfaces).
  FEMORAX_HD static void initial_state(const Params& p, Strided<double> s) {
    for (int k = 0; k < NSTATE; ++k) s[k] = 0.0;
    s[STATE::G] = p.refShearModulus;
    s[STATE::K] = p.refBulkModulus;
    for (int i = 1; i <= NS; ++i) {
      s[STATE::SIZES + i - 1] = p.size[i];
      s[STATE::MODULI + i - 1] = p.modulus[i];
    }
  }

  FEMORAX_HD static double density(const Params& p) { return p.rho; }

  // getInitialTangent of a new instance: the reference elastic tangent (the header says when
  // OpenSees' differs).
  FEMORAX_HD static void initial_tangent(const Params& p, double (*D)[NSTRAIN]) {
    elastic_tangent(p.refShearModulus, p.refBulkModulus, D);
  }

  // The tangent at the committed state (betaKcomm damping): getTangent after commitState,
  // when the trial stress and surfaces equal the committed ones.
  FEMORAX_HD static void committed_tangent(const Params& p, Strided<const double> committed, double (*D)[NSTRAIN]) {
    Point<SerialInner> w(p, SerialInner{}, committed, Strided<double>{});
    w.load_committed();
    w.centers_from_committed = true;
    w.trialStress = w.currentStress;
    w.getTangent(D);
  }

  // The stable step's tangent (contracts.hpp, stability_tangent): the point's elastic moduli,
  // as paramScaling will set them when the switch to the plastic stage is pending (the
  // stiffest response the next step can show; unloading is elastic). Not an OpenSees
  // function.
  FEMORAX_HD static void stability_tangent(const Params& p, Strided<const double> committed, double (*D)[NSTRAIN]) {
    double G, K;
    stability_moduli(p, committed, G, K);
    elastic_tangent(G, K, D);
  }
  // The same tangent as a factor of the reference elastic tangent (contracts.hpp,
  // ScaledStability): G and K are the reference moduli times one factor (paramScaling scales
  // both by the same value), so the factor is G over the reference shear modulus.
  FEMORAX_HD static void stability_reference(const Params& p, double (*D)[NSTRAIN]) {
    elastic_tangent(p.refShearModulus, p.refBulkModulus, D);
  }
  FEMORAX_HD static double stability_scale(const Params& p, Strided<const double> committed) {
    double G, K;
    stability_moduli(p, committed, G, K);
    return G / p.refShearModulus;
  }
  FEMORAX_HD static void stability_moduli(const Params& p, Strided<const double> committed, double& G, double& K) {
    G = committed[STATE::G];
    K = committed[STATE::K];
    if (p.loadStage == 1 && committed[STATE::E2P] == 0.0 && p.frictionAngle != 0.) {
      soil::T2 stress;
      stress.load(committed, STATE::STRESS);
      double vol = stress.volume();
      if (vol > 0.) vol = 0.;  // elast2Plast drops the tension first
      const double conHeig = -(vol - p.residualPress);
      const double scale = std::pow(-conHeig / (p.refPressure - p.residualPress), p.pressDependCoeff);
      G *= scale;
      K *= scale;
    }
  }

  // What a recorder reads after a commit (element output): getStress, which redoes the
  // update from the committed state with a zero strain increment, and getStrain, the
  // committed strain. `scratch` receives the trial state of that update (NSTATE values).
  FEMORAX_HD static void recorded_stress(const Params& p, Strided<const double> committed, Strided<double> scratch,
                                         const double eps[NSTRAIN], double sig[NSTRAIN]) {
    (void)eps;
    double e[NSTRAIN], D[NSTRAIN][NSTRAIN];
    soil::T2 strain;
    strain.load(committed, STATE::STRAIN);
    strain.t2_engr(e);
    update(p, committed, scratch, e, sig, D);
  }
  FEMORAX_HD static void recorded_strain(const Params& p, Strided<const double> committed, const double eps[NSTRAIN],
                                         double out[NSTRAIN]) {
    (void)p;
    (void)eps;
    soil::T2 strain;
    strain.load(committed, STATE::STRAIN);
    strain.t2_engr(out);
  }

  // setTrialStrain(eps) + getStress() + getTangent(): sig and D at the trial strain, trial
  // state written in full.
  FEMORAX_HD static void update(const Params& p, Strided<const double> committed, Strided<double> trial,
                                const double eps[NSTRAIN], double sig[NSTRAIN], double (*D)[NSTRAIN]) {
    update_with(SerialInner{}, p, committed, trial, eps, sig, D);
  }

  template <class Ctx>
  FEMORAX_HD static void update_with(const Ctx& ctx, const Params& p, Strided<const double> committed,
                                     Strided<double> trial, const double eps[NSTRAIN], double sig[NSTRAIN],
                                     double (*D)[NSTRAIN]) {
    Point<Ctx> w(p, ctx, committed, trial);
    w.getStress(eps, false);
    w.finish(sig, D);
  }

  // The same work split for the launch layer's compacted plastic-corrector pass: predict
  // runs getStress until its first stress correction; a point that never needs one is
  // finished and predict returns false; otherwise it returns true and correct redoes the
  // whole update (the same function, so the bits are those of update).
  template <class Ctx>
  FEMORAX_HD static bool predict(const Ctx& ctx, const Params& p, Strided<const double> committed,
                                 Strided<double> trial, const double eps[NSTRAIN], double sig[NSTRAIN],
                                 double (*D)[NSTRAIN]) {
    Point<Ctx> w(p, ctx, committed, trial);
    const bool plastic = w.getStress(eps, true);
    if (!plastic) w.finish(sig, D);
    return plastic;
  }
  template <class Ctx>
  FEMORAX_HD static void correct(const Ctx& ctx, const Params& p, Strided<const double> committed,
                                 Strided<double> trial, const double eps[NSTRAIN], double sig[NSTRAIN],
                                 double (*D)[NSTRAIN]) {
    update_with(ctx, p, committed, trial, eps, sig, D);
  }

 private:
  FEMORAX_HD static void elastic_tangent(double refShearModulus, double refBulkModulus, double (*D)[NSTRAIN]) {
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 6; j++) {
        D[i][j] = 0.;
        if (i == j) D[i][j] += refShearModulus;
        if (i < 3 && j < 3 && i == j) D[i][j] += refShearModulus;
        if (i < 3 && j < 3) D[i][j] += (refBulkModulus - 2. * refShearModulus / 3.);
      }
  }

  // setUpSurfaces (automatic surfaces, or the user's G/Gmax curve). Writes back the
  // residual pressure, friction angle and cohesion, as OpenSees does.
  FEMORAX_HD static int setUpSurfaces(const double* gredu, Params& p) {
    double residualPress = p.residualPress;
    const double refPressure = p.refPressure;
    const int numOfSurfaces = p.numOfSurfaces;
    double frictionAngle = p.frictionAngle;
    double cohesion = p.cohesion;
    const double peakShearStrain = p.peakShearStrain;
    const double refShearModulus = p.refShearModulus;

    double stress1, stress2, strain1, strain2, size = 0., elasto_plast_modul, plast_modul;
    const double pi = 3.14159265358979;
    double refStrain = 0., peakShear = 0., coneHeight = 0.;

    if (gredu == nullptr) {  // automatic generation of surfaces
      if (frictionAngle > 0) {
        double sinPhi = std::sin(frictionAngle * pi / 180.);
        double Mnys = 6. * sinPhi / (3. - sinPhi);
        residualPress = 3. * cohesion / (std::sqrt(2.) * Mnys);
        coneHeight = -(refPressure - residualPress);
        peakShear = std::sqrt(2.) * coneHeight * Mnys / 3.;
        refStrain = (peakShearStrain * peakShear) / (refShearModulus * peakShearStrain - peakShear);
      } else if (frictionAngle == 0.) {  // cohesion = peakShearStrength
        peakShear = 2 * std::sqrt(2.) * cohesion / 3;
        refStrain = (peakShearStrain * peakShear) / (refShearModulus * peakShearStrain - peakShear);
        residualPress = 0.;
      }

      double stressInc = peakShear / numOfSurfaces;

      for (int ii = 1; ii <= numOfSurfaces; ii++) {
        stress1 = ii * stressInc;
        stress2 = stress1 + stressInc;
        strain1 = stress1 * refStrain / (refShearModulus * refStrain - stress1);
        strain2 = stress2 * refStrain / (refShearModulus * refStrain - stress2);
        if (frictionAngle > 0.) size = 3. * stress1 / std::sqrt(2.) / coneHeight;
        else if (frictionAngle == 0.) size = 3. * stress1 / std::sqrt(2.);

        elasto_plast_modul = 2. * (stress2 - stress1) / (strain2 - strain1);

        if ((2. * refShearModulus - elasto_plast_modul) <= 0) plast_modul = soil::UP_LIMIT;
        else plast_modul = (2. * refShearModulus * elasto_plast_modul) / (2. * refShearModulus - elasto_plast_modul);
        if (plast_modul < 0) plast_modul = 0;
        if (plast_modul > soil::UP_LIMIT) plast_modul = soil::UP_LIMIT;
        if (ii == numOfSurfaces) plast_modul = 0;

        p.size[ii] = size;
        p.modulus[ii] = plast_modul;
      }
    } else {  // user defined surfaces
      if (frictionAngle > 0) {  // ignore user defined frictionAngle
        int ii = 2 * (numOfSurfaces - 1);
        double tmax = refShearModulus * gredu[ii] * gredu[ii + 1];
        double Mnys = -(std::sqrt(3.) * tmax - 2. * cohesion) / refPressure;
        if (Mnys <= 0) {  // also ignore user defined cohesion
          cohesion = std::sqrt(3.) / 2 * tmax;
          frictionAngle = 0.;
          coneHeight = 1.;
          residualPress = 0.;
        } else {
          double sinPhi = 3 * Mnys / (6 + Mnys);
          if (sinPhi < 0. || sinPhi > 1.) return InvalidFrictionAngle;
          residualPress = 2. * cohesion / Mnys;
          if (residualPress < 0.01 * refPressure) residualPress = 0.01 * refPressure;
          coneHeight = -(refPressure - residualPress);
          frictionAngle = std::asin(sinPhi) * 180 / pi;
        }
      } else if (frictionAngle == 0.) {  // ignore user defined cohesion
        int ii = 2 * (numOfSurfaces - 1);
        double tmax = refShearModulus * gredu[ii] * gredu[ii + 1];
        cohesion = std::sqrt(3.) / 2 * tmax;
        coneHeight = 1.;
        residualPress = 0.;
      }
      // OpenSees: if (frictionAngle == 0.) pressDependCoeff = 0. (a local copy; no effect,
      // paramScaling returns early when the friction angle is 0).

      for (int i = 1; i < numOfSurfaces; i++) {
        int ii = 2 * (i - 1);
        strain1 = gredu[ii];
        stress1 = refShearModulus * gredu[ii + 1] * strain1;
        strain2 = gredu[ii + 2];
        stress2 = refShearModulus * gredu[ii + 3] * strain2;

        size = std::sqrt(3.) * stress1 / coneHeight;
        elasto_plast_modul = 2. * (stress2 - stress1) / (strain2 - strain1);
        if ((2. * refShearModulus - elasto_plast_modul) <= 0) plast_modul = soil::UP_LIMIT;
        else plast_modul = (2. * refShearModulus * elasto_plast_modul) / (2. * refShearModulus - elasto_plast_modul);
        if (plast_modul <= 0) return NegativeModulus;
        if (plast_modul > soil::UP_LIMIT) plast_modul = soil::UP_LIMIT;

        p.size[i] = size;
        p.modulus[i] = plast_modul;

        if (i == (numOfSurfaces - 1)) {
          plast_modul = 0;
          size = std::sqrt(3.) * stress2 / coneHeight;
          p.size[i + 1] = size;
          p.modulus[i + 1] = plast_modul;
        }
      }
    }

    p.residualPress = residualPress;
    p.frictionAngle = frictionAngle;
    p.cohesion = cohesion;
    return Ok;
  }

 public:
  // ---------------------------------------------------------------------------------------
  // One point's update: OpenSees' member functions, with the instance's members as locals.
  // Surface centers, sizes and moduli: the trial buffer (theSurfaces) is worked in place;
  // committedSurfaces are the committed buffer, copied into the trial buffer first so that
  // elast2Plast can work on them (see the header).
  // ---------------------------------------------------------------------------------------
  template <class Ctx>
  struct Point {
    const Params& p;
    Ctx ctx;
    Strided<const double> C;
    Strided<double> T;
    bool ok = true;                       // false after an impossible state (OpenSees exits)
    bool centers_from_committed = false;  // committed_tangent: read the surfaces from C

    soil::T2 currentStress, currentStrain, trialStress;
    int committedActiveSurf = 0, activeSurfaceNum = 0, e2p = 0;
    double refShearModulus = 0., refBulkModulus = 0.;  // the instance's (scaled at the switch)

    // OpenSees' static workspaces that carry values within one call.
    soil::T2 strainRate, subStrainRate;

    FEMORAX_HD Point(const Params& p_, const Ctx& ctx_, Strided<const double> C_, Strided<double> T_)
        : p(p_), ctx(ctx_), C(C_), T(T_) {}

    // --- surfaces ---------------------------------------------------------------------
    FEMORAX_HD double surf(int base, int i) const {
      if (i == 0) return 0.;
      return centers_from_committed ? C[base + i - 1] : T[base + i - 1];
    }
    FEMORAX_HD double size(int i) const { return surf(STATE::SIZES, i); }
    FEMORAX_HD double modulus(int i) const { return surf(STATE::MODULI, i); }
    FEMORAX_HD void center(int i, double* out) const {
      if (i == 0) {
        soil::vzero(out);
        return;
      }
      const int k = STATE::CENTERS + 6 * (i - 1);
      if (centers_from_committed) {
        for (int c = 0; c < 6; ++c) out[c] = C[k + c];
      } else {
        for (int c = 0; c < 6; ++c) out[c] = T[k + c];
      }
    }
    FEMORAX_HD void setCenter(int i, const double* v) {
      const int k = STATE::CENTERS + 6 * (i - 1);
      for (int c = 0; c < 6; ++c) T[k + c] = v[c];
    }

    FEMORAX_HD void load_committed() {
      currentStress.load(C, STATE::STRESS);
      currentStrain.load(C, STATE::STRAIN);
      committedActiveSurf = static_cast<int>(C[STATE::ACTIVE]);
      activeSurfaceNum = committedActiveSurf;
      e2p = static_cast<int>(C[STATE::E2P]);
      refShearModulus = C[STATE::G];
      refBulkModulus = C[STATE::K];
    }

    // setTrialStrain: strainRate = strain - currentStrain (engineering, stored as tensor).
    FEMORAX_HD void setTrialStrain(const double* strain) {
      double workV6[6], cs[6];
      soil::vcopy(workV6, strain);
      currentStrain.t2_engr(cs);
      soil::vsub_eq(workV6, cs);
      strainRate.set(workV6, 1);
    }

    // setTrialStrain + getStress. With stop_at_correction, returns true (and stops) where
    // getStress would make its first stress correction; otherwise runs to the end and
    // returns false.
    FEMORAX_HD bool getStress(const double* eps, bool stop_at_correction) {
      load_committed();
      setTrialStrain(eps);
      const int loadStage = p.loadStage;
      const int numOfSurfaces = p.numOfSurfaces;
      // committedSurfaces into the trial buffer (theSurfaces), before elast2Plast: see the
      // header.
      inner_for(ctx, NS, [&](int s) {
        T[STATE::SIZES + s] = C[STATE::SIZES + s];
        T[STATE::MODULI + s] = C[STATE::MODULI + s];
        const int k = STATE::CENTERS + 6 * s;
        for (int c = 0; c < 6; ++c) T[k + c] = C[k + c];
      });
      if (loadStage == 1 && e2p == 0) elast2Plast();

      if (loadStage != 1) {  // linear elastic
        double D[6][6];
        getTangent(D);
        double a[6], rate[6];
        soil::vcopy(a, currentStress.t2);
        strainRate.t2_engr(rate);
        // a.addMatrixVector(1.0, theTangent, rate, 1.0): column by column
        for (int i = 0; i < 6; i++)
          for (int j = 0; j < 6; j++) a[j] += D[j][i] * rate[i];
        trialStress.set(a);
        return false;
      }
      // theSurfaces = committedSurfaces: the trial buffer already holds them.
      activeSurfaceNum = committedActiveSurf;
      subStrainRate = strainRate;
      setTrialStress(currentStress);
      if (isLoadReversal()) {
        updateInnerSurface();
        activeSurfaceNum = 0;
      }
      int numSubIncre = setSubStrainRate();

      for (int i = 0; i < numSubIncre; i++) {
        if (i == 0) setTrialStress(currentStress);
        else setTrialStress(trialStress);
        if (activeSurfaceNum == 0 && !isCrossingNextSurface()) continue;
        if (stop_at_correction) return true;
        if (activeSurfaceNum == 0) activeSurfaceNum++;
        stressCorrection(0);
        updateActiveSurface();
      }
      (void)numOfSurfaces;
      // volume stress change
      double volum = refBulkModulus * (strainRate.volume() * 3.);
      volum += currentStress.volume();
      double d[6];
      soil::vcopy(d, trialStress.dev);
      trialStress.set_dev(d, volum);
      return false;
    }

    // --- finish: the tangent, the trial buffer, the outputs ---------------------------
    FEMORAX_HD void finish(double* sig, double (*D)[NSTRAIN]) {
      getTangent(D);
      for (int i = 0; i < 6; ++i) sig[i] = trialStress.t2[i];
      if (!ok) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        for (int i = 0; i < 6; ++i) sig[i] = nan;
      }
      // commitState: currentStress = trialStress; currentStrain += strainRate; in the
      // plastic stage committedActiveSurf and the surfaces. The instance's moduli, sizes,
      // moduli of the surfaces and e2p are members (elast2Plast's changes persist).
      double workV6[6];
      soil::vcopy(workV6, currentStrain.t2);
      soil::vadd_eq(workV6, strainRate.t2);
      soil::T2 strainAtCommit;
      strainAtCommit.set(workV6);
      trialStress.store(T, STATE::STRESS);
      strainAtCommit.store(T, STATE::STRAIN);
      T[STATE::E2P] = e2p;
      T[STATE::G] = refShearModulus;
      T[STATE::K] = refBulkModulus;
      if (p.loadStage == 1) {
        T[STATE::ACTIVE] = activeSurfaceNum;
      } else {
        // committedActiveSurf as elast2Plast may have set it (not in stage 0: it runs only
        // in the plastic stage), and the copied surfaces stay.
        T[STATE::ACTIVE] = committedActiveSurf;
      }
    }

    // --- OpenSees' member functions -----------------------------------------------------
    FEMORAX_HD void elast2Plast() {
      const int loadStage = p.loadStage;
      const double frictionAngle = p.frictionAngle;
      const int numOfSurfaces = p.numOfSurfaces;
      if (loadStage != 1 || e2p == 1) return;
      e2p = 1;

      if (currentStress.volume() > 0. && frictionAngle > 0.) {
        // setData(currentStress.deviator(), 0): the int overload (isEngrgStrain = 0).
        double d[6];
        soil::vcopy(d, currentStress.dev);
        currentStress.set(d, 0);
      }

      paramScaling();  // scale surface parameters corresponding to initial confinement

      // Active surface is 0, return
      if (currentStress.deviator_length() == 0.) return;

      // Find active surface (committedSurfaces = the trial buffer's copy here)
      while (yieldFunc(currentStress, ++committedActiveSurf) > 0) {
        if (committedActiveSurf == numOfSurfaces) {
          deviatorScaling(currentStress, numOfSurfaces);
          initSurfaceUpdate();
          return;
        }
      }
      committedActiveSurf--;
      initSurfaceUpdate();
    }

    FEMORAX_HD double yieldFunc(const soil::T2& stress, int surfaceNum) const {
      double temp[6], c[6];
      soil::vcopy(temp, stress.dev);
      center(surfaceNum, c);
      soil::vsub_eq(temp, c);
      double sz = size(surfaceNum);
      return 3. / 2. * soil::ddot(temp, temp) - sz * sz;
    }

    // deviatorScaling, its tail recursion as a loop.
    FEMORAX_HD void deviatorScaling(soil::T2& stress, int surfaceNum) {
      const int numOfSurfaces = p.numOfSurfaces;
      for (int iter = 0; iter < 100000; ++iter) {
        double diff = yieldFunc(stress, surfaceNum);
        if (surfaceNum < numOfSurfaces && diff < 0.) {
          double sz = size(surfaceNum);
          double deviaSz = std::sqrt(sz * sz + diff);
          double devia[6], temp[6], c[6];
          soil::vcopy(devia, stress.dev);
          center(surfaceNum, c);
          for (int k = 0; k < 6; ++k) temp[k] = devia[k] - c[k];  // devia - center (operator-)
          double coeff = (sz - deviaSz) / deviaSz;
          if (coeff < 1.e-13) coeff = 1.e-13;
          soil::add_vector(devia, 1.0, temp, coeff);
          stress.set_dev(devia, stress.volume());
          continue;  // deviatorScaling(stress, surfaces, surfaceNum, count)
        }
        if (surfaceNum == numOfSurfaces && std::fabs(diff) > soil::LOW_LIMIT) {
          double sz = size(surfaceNum);
          double newDevia[6] = {};
          soil::add_vector(newDevia, 0.0, stress.dev, sz / std::sqrt(diff + sz * sz));
          stress.set_dev(newDevia, stress.volume());
        }
        return;
      }
      ok = false;  // OpenSees would recurse until the stack overflows
    }

    FEMORAX_HD void initSurfaceUpdate() {
      if (committedActiveSurf == 0) return;
      const int numOfSurfaces = p.numOfSurfaces;
      double devia[6];
      soil::vcopy(devia, currentStress.dev);
      double Ms = std::sqrt(3. / 2. * soil::ddot(devia, devia));

      if (committedActiveSurf < numOfSurfaces) {  // failure surface can't move
        double newCenter[6] = {};
        soil::add_vector(newCenter, 0.0, devia, 1.0 - size(committedActiveSurf) / Ms);
        setCenter(committedActiveSurf, newCenter);
      }
      // committedSurfaces[i]: the trial buffer here.
      inner_for(ctx, committedActiveSurf - 1, [&](int k) {
        const int i = k + 1;
        double newCenter[6];
        const double f = 1. - size(i) / Ms;
        for (int c = 0; c < 6; ++c) newCenter[c] = devia[c] * f;  // devia * (...) (operator*)
        setCenter(i, newCenter);
      });
    }

    FEMORAX_HD void paramScaling() {
      const int numOfSurfaces = p.numOfSurfaces;
      const double frictionAngle = p.frictionAngle;
      const double residualPress = p.residualPress;
      const double refPressure = p.refPressure;
      const double pressDependCoeff = p.pressDependCoeff;
      if (frictionAngle == 0.) return;

      double conHeig = -(currentStress.volume() - residualPress);
      double scale = -conHeig / (refPressure - residualPress);

      scale = std::pow(scale, pressDependCoeff);
      refShearModulus *= scale;
      refBulkModulus *= scale;

      // MultiYieldSurface(temp, size, plastModul): center zero (an untouched static).
      inner_for(ctx, numOfSurfaces, [&](int k) {
        const int i = k + 1;
        const double plastModul = modulus(i) * scale;
        const double sz = size(i) * conHeig;
        T[STATE::SIZES + i - 1] = sz;
        T[STATE::MODULI + i - 1] = plastModul;
        const double zero[6] = {0., 0., 0., 0., 0., 0.};
        setCenter(i, zero);
      });
    }

    FEMORAX_HD void setTrialStress(const soil::T2& stress_in) {
      const soil::T2 stress = stress_in;  // may be trialStress itself
      double devia[6];
      soil::vcopy(devia, stress.dev);
      soil::add_vector(devia, 1.0, subStrainRate.dev, 2. * refShearModulus);
      trialStress.set_dev(devia, stress.volume());
    }

    FEMORAX_HD int setSubStrainRate() {
      const int numOfSurfaces = p.numOfSurfaces;
      if (strainRate.is_zero()) return 0;

      double elast_plast_modulus;
      if (activeSurfaceNum == 0) elast_plast_modulus = 2 * refShearModulus;
      else {
        double plast_modulus = modulus(activeSurfaceNum);
        elast_plast_modulus = 2 * refShearModulus * plast_modulus / (2 * refShearModulus + plast_modulus);
      }
      double incre[6] = {};
      soil::add_vector(incre, 0.0, strainRate.dev, elast_plast_modulus);

      soil::T2 increStress;
      increStress.set(incre, 0);
      double singleCross = size(numOfSurfaces) / numOfSurfaces;
      double totalCross = 3. * increStress.octahedral_shear() / std::sqrt(2.);
      int numOfSub = to_int(totalCross / singleCross + 1);
      if (numOfSub > numOfSurfaces) numOfSub = numOfSurfaces;
      soil::vcopy(incre, strainRate.t2);
      soil::vdiv_eq(incre, numOfSub);
      subStrainRate.set(incre);

      return numOfSub;
    }

    // double -> int as C++ truncation; values outside int are clamped (OpenSees' conversion
    // is undefined there).
    FEMORAX_HD static int to_int(double x) {
      if (!(x == x)) return 0;
      if (x >= 2147483647.0) return 2147483647;
      if (x <= -2147483648.0) return -2147483647 - 1;
      return static_cast<int>(x);
    }

    FEMORAX_HD void getContactStress(soil::T2& contactStress) const {
      double c[6], devia[6];
      center(activeSurfaceNum, c);
      soil::vcopy(devia, trialStress.dev);
      soil::vsub_eq(devia, c);
      double Ms = std::sqrt(3. / 2. * soil::ddot(devia, devia));
      soil::vmul_eq(devia, size(activeSurfaceNum) / Ms);
      soil::vadd_eq(devia, c);
      contactStress.set_dev(devia, trialStress.volume());
    }

    FEMORAX_HD int isLoadReversal() const {
      if (activeSurfaceNum == 0) return 0;
      double surfaceNormal[6], a[6];
      getSurfaceNormal(currentStress, surfaceNormal);
      soil::vcopy(a, trialStress.dev);
      soil::vsub_eq(a, currentStress.dev);
      if (soil::ddot(a, surfaceNormal) < 0) return 1;
      return 0;
    }

    FEMORAX_HD void getSurfaceNormal(const soil::T2& stress, double* surfaceNormal) const {
      double c[6];
      soil::vcopy(surfaceNormal, stress.dev);
      center(activeSurfaceNum, c);
      soil::vsub_eq(surfaceNormal, c);
      soil::vdiv_eq(surfaceNormal, std::sqrt(soil::ddot(surfaceNormal, surfaceNormal)));
    }

    FEMORAX_HD double getLoadingFunc(const soil::T2& contactStress, const double* surfaceNormal,
                                     int crossedSurface) const {
      double loadingFunc;
      double temp1 = 2. * refShearModulus;
      double temp2 = modulus(activeSurfaceNum);

      // for crossing first surface
      double temp = temp1 + temp2;
      double tmp[6];
      soil::vcopy(tmp, trialStress.dev);
      soil::vsub_eq(tmp, contactStress.dev);
      loadingFunc = soil::ddot(surfaceNormal, tmp) / temp;
      // for crossing more than one surface
      if (crossedSurface) {
        double temp3 = modulus(activeSurfaceNum - 1);
        loadingFunc *= (temp3 - temp2) / temp3;
      }
      return loadingFunc;
    }

    // stressCorrection, its tail recursion as a loop.
    FEMORAX_HD void stressCorrection(int crossedSurface) {
      for (;;) {
        soil::T2 contactStress;
        getContactStress(contactStress);
        double surfaceNormal[6];
        getSurfaceNormal(contactStress, surfaceNormal);
        double loadingFunc = getLoadingFunc(contactStress, surfaceNormal, crossedSurface);
        double devia[6] = {};
        soil::add_vector(devia, 0.0, surfaceNormal, -2 * refShearModulus * loadingFunc);
        soil::vadd_eq(devia, trialStress.dev);

        trialStress.set_dev(devia, trialStress.volume());
        deviatorScaling(trialStress, activeSurfaceNum);

        if (isCrossingNextSurface()) {
          activeSurfaceNum++;
          crossedSurface = 1;  // stressCorrection(1)
          continue;
        }
        return;
      }
    }

    FEMORAX_HD void updateActiveSurface() {
      const int numOfSurfaces = p.numOfSurfaces;
      if (activeSurfaceNum == numOfSurfaces) return;

      double A, B, Cc, X;
      double t1[6], t2[6], temp[6], center_[6], outcenter[6];
      center(activeSurfaceNum, center_);
      double sz = size(activeSurfaceNum);
      center(activeSurfaceNum + 1, outcenter);
      double outsize = size(activeSurfaceNum + 1);

      soil::vcopy(t1, trialStress.dev);
      soil::vsub_eq(t1, center_);
      soil::vcopy(t2, center_);
      soil::vsub_eq(t2, outcenter);

      A = soil::ddot(t1, t1);
      B = 2. * soil::ddot(t1, t2);
      Cc = soil::ddot(t2, t2) - 2. / 3. * outsize * outsize;
      X = soil::second_order_eqn(A, B, Cc, 0, ok);
      if (std::fabs(X - 1.) < soil::LOW_LIMIT) X = 1.;
      if (X < 1.) {  // OpenSees: "error in Direction of surface motion", exit
        ok = false;
        return;
      }

      soil::vcopy(temp, center_);
      soil::add_vector(temp, 1.0, t1, X);
      soil::vmul_eq(temp, (1.0 - sz / outsize));
      soil::vcopy(t2, center_);
      soil::add_vector(t2, 1.0, outcenter, -sz / outsize);
      soil::vsub_eq(temp, t2);

      soil::T2 direction;
      direction.set(temp);

      if (direction.deviator_length() < soil::LOW_LIMIT) return;

      soil::vcopy(temp, direction.dev);
      A = soil::ddot(temp, temp);
      B = -2 * soil::ddot(t1, temp);
      if (std::fabs(B) < soil::LOW_LIMIT) B = 0.;
      Cc = soil::ddot(t1, t1) - 2. / 3. * sz * sz;
      if (std::fabs(Cc) < soil::LOW_LIMIT || std::fabs(Cc) / soil::ddot(t1, t1) < soil::LOW_LIMIT) return;

      // Alborz Ghofrani: avoid quadratic equations with very small coefficients B and C
      if ((std::fabs(B) < 1.0e-10) && (std::fabs(Cc) < 1.0e-10)) return;

      if (B > 0. || Cc < 0.) {  // OpenSees: "error in surface motion", exit
        ok = false;
        return;
      }
      X = soil::second_order_eqn(A, B, Cc, 1, ok);

      soil::add_vector(center_, 1.0, temp, X);
      setCenter(activeSurfaceNum, center_);
    }

    FEMORAX_HD void updateInnerSurface() {
      if (activeSurfaceNum <= 1) return;
      double devia[6], center_[6];
      soil::vcopy(devia, currentStress.dev);
      center(activeSurfaceNum, center_);
      const double sz = size(activeSurfaceNum);
      inner_for(ctx, activeSurfaceNum - 1, [&](int k) {
        const int i = k + 1;
        double newcenter[6];
        soil::vcopy(newcenter, center_);
        soil::vsub_eq(newcenter, devia);
        soil::vmul_eq(newcenter, size(i) / sz);
        soil::vadd_eq(newcenter, devia);
        setCenter(i, newcenter);
      });
    }

    FEMORAX_HD int isCrossingNextSurface() const {
      const int numOfSurfaces = p.numOfSurfaces;
      if (activeSurfaceNum == numOfSurfaces) return 0;
      if (yieldFunc(trialStress, activeSurfaceNum + 1) > 0) return 1;
      return 0;
    }

    // getTangent (stage 0: elastic with the instance's moduli; stage 1: elastic minus the
    // active surface's term, from the trial stress and the trial surfaces, as OpenSees).
    FEMORAX_HD void getTangent(double (*D)[NSTRAIN]) {
      const int loadStage = p.loadStage;
      if (loadStage != 1) {
        elastic_tangent(refShearModulus, refBulkModulus, D);
        return;
      }
      double coeff;
      double devia[6] = {};
      if (activeSurfaceNum > 0) {
        double c[6];
        soil::vcopy(devia, trialStress.dev);
        center(activeSurfaceNum, c);
        soil::vsub_eq(devia, c);
        double sz = size(activeSurfaceNum);
        double plastModul = modulus(activeSurfaceNum);
        coeff = 6. * refShearModulus * refShearModulus / (2. * refShearModulus + plastModul) / sz / sz;
      } else {
        coeff = 0.;
      }
      for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
          D[i][j] = -coeff * devia[i] * devia[j];
          if (i == j) D[i][j] += refShearModulus;
          if (i < 3 && j < 3 && i == j) D[i][j] += refShearModulus;
          if (i < 3 && j < 3) D[i][j] += (refBulkModulus - 2. * refShearModulus / 3.);
        }
    }
  };
};

}  // namespace femorax
