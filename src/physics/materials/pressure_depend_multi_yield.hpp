#pragma once

// PressureDependMultiYield, three-dimensional form (Yang, Elgamal and Parra's multi-yield
// surface plasticity for pressure-sensitive soils, with dilatancy and liquefaction).
//
// Port of OpenSees SRC/material/nD/soil/PressureDependMultiYield.cpp (with T2Vector.cpp
// and MultiYieldSurface.cpp: physics/common/t2_vector.hpp), OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea. Function and variable names follow the
// original (getStress, setTrialStress, stressCorrection, updateActiveSurface, PPZPivot,
// cumuDilateStrainOcta, ...), and every operation keeps its order, so the two can be read
// side by side and agree bit for bit on host compilers that do not contract multiply-adds.
//
// What changes, and why (docs/deviations.md, "PressureDependMultiYield"):
// - OpenSees' per-material static arrays (loadStagex, refShearModulusx, ...) are the
//   material's Params; its static workspaces (theTangent, workV6, workT2V, trialStrain,
//   subStrainRate, the function-local statics) are locals. Each function's inputs were
//   checked so that no value crosses functions through a workspace: every workspace is
//   written before it is read within one call.
// - The yield surfaces are a fixed-size array: NS (a compile-time bucket, a batch's maximum
//   number of surfaces) centers per point in the state; sizes and plastic moduli, equal for
//   every point of a material, are in Params. Surface 0 is OpenSees' unused, default
//   surface (size 0, modulus 0, center 0), which getTangent can reach.
// - pa (pAtm) is per material; OpenSees keeps one static pAtm, the last constructed
//   PressureDependMultiYield's, for every material at run time.
// - elast2Plast changes the committed state on the first call after the switch to the
//   plastic stage. Here it runs inside update on a copy of the committed state, whose
//   result becomes committed at commit. It is a pure function of the committed state, so
//   the result is the same, and a step that is reverted and redone sees the same values.
// - Recursion (stressCorrection, deviatorScaling) is written as loops; OpenSees' exit() on
//   impossible states (secondOrderEqn, surface motion errors) gives NaN stress here, so the
//   step fails instead of the program.
// - Load stage 2 (elastic with the initial pressure, used by u-p elements) is not
//   supported; schema 0.4 has stages 0 and 1 only. The volumetric-plasticity terms (Hv, Pv)
//   are ported but Femora never sets them (Hv = 0).
//
// State per point (STATE layout below): committed and trial buffers of NSTATE doubles,
// 89 + 6 NS: six T2 tensors (13 each), eleven scalars, NS surface centers. update writes
// every trial value, so commit is a swap of the buffers and revert costs nothing (OpenSees'
// getStress always restarts from the committed state; its revertToLastCommit is empty).
//
// Strain order: OpenSees', 11 22 33 12 23 31 with engineering shear strains; stresses
// in the same order. Compression is negative.

#include <cfloat>
#include <cmath>

#include "physics/common/inner.hpp"
#include "physics/common/strided.hpp"
#include "physics/common/t2_vector.hpp"
#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

// The constructor's arguments (OpenSees order and defaults of the Tcl parser), the same
// for every surface bucket.
struct PressureDependMultiYieldInput {
  double rho, refShearModul, refBulkModul, frictionAng, peakShearStra, refPress, pressDependCoe,
      phaseTransformAng, contractionParam1, dilationParam1, dilationParam2,
      liquefactionParam1, liquefactionParam2, liquefactionParam4;
  int numberOfYieldSurf;  // positive; with gredu, the number of user surfaces
  const double* gredu;    // nullptr, or numberOfYieldSurf (gamma, G/Gmax) pairs
  double e, volLimit1, volLimit2, volLimit3, atm, cohesi, hv, pv;
};
template <int NS>
struct PressureDependMultiYield {
  static_assert(NS >= 1 && NS <= 40, "OpenSees allows at most 40 yield surfaces");
  static constexpr MaterialKind KIND = MaterialKind::NDMaterial3D;
  static constexpr int NSTRAIN = 6;
  static constexpr int MAX_SURFACES = NS;
  // Symmetric (OpenSees' non-symmetric form is commented out in getTangent); positive
  // semi-definite: the outermost surface has plastic modulus 0.
  static constexpr TangentSymmetry TANGENT = TangentSymmetry::SymmetricPositiveSemidefinite;
  static constexpr bool CONSTANT_TANGENT = false;

  // --- State layout (offsets in one buffer) -------------------------------------------
  struct STATE {
    static constexpr int STRESS = 0;            // currentStress / trialStress
    static constexpr int STRAIN = 13;           // currentStrain (trial: the strain at commit)
    static constexpr int REVERSAL = 26;         // reversalStress
    static constexpr int PPZPIVOT = 39;
    static constexpr int PPZCENTER = 52;
    static constexpr int LOCK = 65;             // lockStress
    static constexpr int ACTIVE = 78;           // activeSurfaceNum / committedActiveSurf
    static constexpr int PRESSURED = 79;
    static constexpr int ONPPZ = 80;
    static constexpr int PPZSIZE = 81;
    static constexpr int CUMUDILATE = 82;       // cumuDilateStrainOcta
    static constexpr int MAXCUMUDILATE = 83;    // maxCumuDilateStrainOcta
    static constexpr int CUMUTRANSLATE = 84;    // cumuTranslateStrainOcta
    static constexpr int PREPPZ = 85;           // prePPZStrainOcta
    static constexpr int OPPOPREPPZ = 86;       // oppoPrePPZStrainOcta
    static constexpr int MAXPRESS = 87;
    static constexpr int E2P = 88;
    static constexpr int CENTERS = 89;          // surface i (1..NS): CENTERS + 6 (i - 1)
  };
  static constexpr int NSTATE = STATE::CENTERS + 6 * NS;

  struct Params {
    int loadStage;      // 0 elastic, 1 plastic (updateMaterialStage)
    int numOfSurfaces;  // <= NS
    double rho, refShearModulus, refBulkModulus, frictionAngle, peakShearStrain;
    double refPressure;  // negative (compression), as OpenSees stores it
    double cohesion, pressDependCoeff, phaseTransfAngle, contractParam1, dilateParam1,
        dilateParam2, liquefyParam1, liquefyParam2, liquefyParam4, einit, volLimit1, volLimit2,
        volLimit3, pAtm, Hv, Pv;
    double residualPress, stressRatioPT, strainPTOcta;  // from setUpSurfaces
    double size[NS + 1];     // surface sizes; [0] = 0 (OpenSees' unused surface)
    double modulus[NS + 1];  // plastic shear moduli; [0] = 0
  };

  using Input = PressureDependMultiYieldInput;


  enum Setup : int {
    Ok = 0,
    TooManySurfaces = 1,       // numberOfYieldSurf > NS (choose a larger bucket)
    InvalidFrictionAngle = 2,  // user surfaces: sin(phi) outside [0, 1] (OpenSees exits)
    NegativeModulus = 3,       // user surfaces: a plastic modulus <= 0 (OpenSees exits)
    TooFewPairs = 4,           // user surfaces: fewer than 2 pairs (OpenSees reads garbage)
    BadInput = 5,              // a value OpenSees treats as fatal
  };

  // Constructor checks plus setUpSurfaces. Values OpenSees resets with a warning are reset
  // the same way (PTAng > friction angle, negative cohesion, pressDependCoe or volLimit1).
  FEMORAX_HD static int setup(const Input& in, Params& p) {
    double frictionAng = in.frictionAng, phaseTransformAng = in.phaseTransformAng, cohesi = in.cohesi,
           pressDependCoe = in.pressDependCoe, volLim1 = in.volLimit1;
    int numberOfYieldSurf = in.numberOfYieldSurf;
    if (in.refShearModul <= 0 || in.refBulkModul <= 0 || frictionAng <= 0. || frictionAng >= 90. ||
        phaseTransformAng <= 0. || in.peakShearStra <= 0 || in.refPress <= 0 || in.rho < 0 || in.e < 0)
      return BadInput;
    if (phaseTransformAng > frictionAng) phaseTransformAng = frictionAng;
    if (cohesi < 0) cohesi = 0.;
    if (pressDependCoe < 0) pressDependCoe = 0.;
    if (numberOfYieldSurf <= 0) numberOfYieldSurf = 10;
    if (numberOfYieldSurf > 40) numberOfYieldSurf = 40;
    if (volLim1 < 0) volLim1 = 0.8;
    if (numberOfYieldSurf > NS) return TooManySurfaces;
    if (in.gredu != nullptr && numberOfYieldSurf < 2) return TooFewPairs;

    p.loadStage = 0;
    p.numOfSurfaces = numberOfYieldSurf;
    p.rho = in.rho;
    p.refShearModulus = in.refShearModul;
    p.refBulkModulus = in.refBulkModul;
    p.frictionAngle = frictionAng;
    p.peakShearStrain = in.peakShearStra;
    p.refPressure = -in.refPress;  // compression is negative
    p.cohesion = cohesi;
    p.pressDependCoeff = pressDependCoe;
    p.phaseTransfAngle = phaseTransformAng;
    p.contractParam1 = in.contractionParam1;
    p.dilateParam1 = in.dilationParam1;
    p.dilateParam2 = in.dilationParam2;
    p.liquefyParam1 = in.liquefactionParam1;
    p.liquefyParam2 = in.liquefactionParam2;
    p.liquefyParam4 = in.liquefactionParam4;
    p.einit = in.e;
    p.volLimit1 = volLim1;
    p.volLimit2 = in.volLimit2;
    p.volLimit3 = in.volLimit3;
    p.pAtm = in.atm;
    p.Hv = in.hv;
    p.Pv = in.pv;
    p.strainPTOcta = 0.;  // OpenSees leaves it unset if no surface pair brackets the PT ratio
    for (int i = 0; i <= NS; ++i) p.size[i] = p.modulus[i] = 0.0;
    return setUpSurfaces(in.gredu, p);
  }

  // Default state of a new point (the constructor's member values).
  FEMORAX_HD static void initial_state(const Params& p, Strided<double> s) {
    (void)p;
    for (int k = 0; k < NSTATE; ++k) s[k] = 0.0;
    s[STATE::ONPPZ] = -1;
  }

  FEMORAX_HD static double density(const Params& p) { return p.rho; }

  // getInitialTangent at the reference state (stage 0, factor 1): what SSPbrick's
  // stabilization reads when the element is created (OpenSees: setDomain, stage 0) and what
  // betaKinit damping uses.
  FEMORAX_HD static void initial_tangent(const Params& p, double (*D)[NSTRAIN]) {
    elastic_tangent(p, 1.0, D);
  }

  // The tangent at the committed state (betaKcomm damping, the stable step): getTangent
  // after commitState, when the trial surfaces equal the committed ones.
  FEMORAX_HD static void committed_tangent(const Params& p, Strided<const double> committed, double (*D)[NSTRAIN]) {
    if (p.loadStage != 1 || committed[STATE::E2P] == 0.0) {
      elastic_tangent(p, 1.0, D);
      return;
    }
    Point<SerialInner> w(p, SerialInner{}, committed, Strided<double>{});
    w.load_committed();
    w.activeSurfaceNum = w.committedActiveSurf;
    w.centers_from_committed = true;
    w.trialStress = w.currentStress;
    w.strainRate.zero();
    w.getTangent(D);
  }

  // What a recorder reads after a commit (element output): getStress, which redoes the
  // update from the committed state with a zero strain increment (commitState zeroes
  // strainRate), and getStrain, the committed strain. `scratch` receives the trial state
  // of that update (NSTATE values); eps (the strain of the displacement) is not used.
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
    if (w.predict(eps)) w.correct();
    w.finish(sig, D);
  }

  // The same work split for the launch layer's compacted plastic-corrector pass
  // (docs/design.md, "Hard parts"): predict runs the elastic predictor; for a point that
  // stays elastic it finishes the update (sig, D, trial state) and returns false; for a
  // point that needs the plastic corrector it returns true and leaves the point to
  // correct, which runs the predictor again (cheap, and the same function, so the bits are
  // those of update) and then the sub-increment loop. The trial buffer of a point that
  // predict sends on is complete only after correct.
  template <class Ctx>
  FEMORAX_HD static bool predict(const Ctx& ctx, const Params& p, Strided<const double> committed,
                                 Strided<double> trial, const double eps[NSTRAIN], double sig[NSTRAIN],
                                 double (*D)[NSTRAIN]) {
    Point<Ctx> w(p, ctx, committed, trial);
    const bool plastic = w.predict(eps);
    if (!plastic) w.finish(sig, D);
    return plastic;
  }
  template <class Ctx>
  FEMORAX_HD static void correct(const Ctx& ctx, const Params& p, Strided<const double> committed,
                                 Strided<double> trial, const double eps[NSTRAIN], double sig[NSTRAIN],
                                 double (*D)[NSTRAIN]) {
    Point<Ctx> w(p, ctx, committed, trial);
    if (w.predict(eps)) w.correct();
    w.finish(sig, D);
  }

 private:
  FEMORAX_HD static void elastic_tangent(const Params& p, double factor, double (*D)[NSTRAIN]) {
    const double refShearModulus = p.refShearModulus, refBulkModulus = p.refBulkModulus;
    for (int i = 0; i < 6; i++)
      for (int j = 0; j < 6; j++) {
        D[i][j] = 0.;
        if (i == j) D[i][j] += refShearModulus * factor;
        if (i < 3 && j < 3 && i == j) D[i][j] += refShearModulus * factor;
        if (i < 3 && j < 3) D[i][j] += (refBulkModulus - 2. * refShearModulus / 3.) * factor;
      }
  }

  // setUpSurfaces (automatic surfaces, or the user's G/Gmax curve).
  FEMORAX_HD static int setUpSurfaces(const double* gredu, Params& p) {
    double residualPress = 0.0;
    const double refPressure = p.refPressure;
    const double refShearModulus = p.refShearModulus;
    const int numOfSurfaces = p.numOfSurfaces;
    double frictionAngle = p.frictionAngle;
    const double cohesion = p.cohesion;
    const double peakShearStrain = p.peakShearStrain;
    double phaseTransfAngle = p.phaseTransfAngle;
    double stressRatioPT = 0.0;
    const double pAtm = p.pAtm;
    const double pi = soil::pi;

    double refStrain, peakShear, coneHeight;
    double stress1, stress2, strain1, strain2, size, elasto_plast_modul, plast_modul;
    double ratio1, ratio2;

    if (gredu == nullptr) {
      double sinPhi = std::sin(frictionAngle * pi / 180.);
      double Mnys = 6. * sinPhi / (3. - sinPhi);
      double sinPhiPT = std::sin(phaseTransfAngle * pi / 180.);
      stressRatioPT = 6. * sinPhiPT / (3. - sinPhiPT);
      residualPress = 2 * cohesion / Mnys;
      if (residualPress < 0.0001 * pAtm) residualPress = 0.0001 * pAtm;
      coneHeight = -(refPressure - residualPress);
      peakShear = std::sqrt(2.) * coneHeight * Mnys / 3.;
      refStrain = (peakShearStrain * peakShear) / (refShearModulus * peakShearStrain - peakShear);

      double stressInc = peakShear / numOfSurfaces;

      for (int ii = 1; ii <= numOfSurfaces; ii++) {
        stress1 = ii * stressInc;
        stress2 = stress1 + stressInc;
        ratio1 = 3. * stress1 / std::sqrt(2.) / coneHeight;
        ratio2 = 3. * stress2 / std::sqrt(2.) / coneHeight;
        strain1 = stress1 * refStrain / (refShearModulus * refStrain - stress1);
        strain2 = stress2 * refStrain / (refShearModulus * refStrain - stress2);

        if (ratio1 <= stressRatioPT && ratio2 >= stressRatioPT) {
          double ratio = (ratio2 - stressRatioPT) / (ratio2 - ratio1);
          p.strainPTOcta = strain2 - ratio * (strain2 - strain1);
        }

        size = ratio1;
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
      int ii = 2 * (numOfSurfaces - 1);
      double tmax = refShearModulus * gredu[ii] * gredu[ii + 1];
      double Mnys = -(std::sqrt(3.) * tmax - 2. * cohesion) / refPressure;
      residualPress = 2 * cohesion / Mnys;
      if (residualPress < 0.0001 * pAtm) residualPress = 0.0001 * pAtm;
      coneHeight = -(refPressure - residualPress);

      double sinPhi = 3 * Mnys / (6 + Mnys);
      if (sinPhi < 0. || sinPhi > 1.) return InvalidFrictionAngle;

      frictionAngle = std::asin(sinPhi) * 180 / pi;
      if (phaseTransfAngle > frictionAngle) phaseTransfAngle = frictionAngle;
      double sinPhiPT = std::sin(phaseTransfAngle * pi / 180.);
      stressRatioPT = 6. * sinPhiPT / (3. - sinPhiPT);

      for (int i = 1; i < numOfSurfaces; i++) {
        int jj = 2 * (i - 1);
        strain1 = gredu[jj];
        stress1 = refShearModulus * gredu[jj + 1] * strain1;
        strain2 = gredu[jj + 2];
        stress2 = refShearModulus * gredu[jj + 3] * strain2;

        ratio1 = std::sqrt(3.) * stress1 / coneHeight;
        ratio2 = std::sqrt(3.) * stress2 / coneHeight;
        if (ratio1 <= stressRatioPT && ratio2 >= stressRatioPT) {
          double ratio = (ratio2 - stressRatioPT) / (ratio2 - ratio1);
          // gamma_oct = sqrt(6.0)/3*gamma12
          p.strainPTOcta = std::sqrt(6.) / 3 * (strain2 - ratio * (strain2 - strain1));
        }

        size = ratio1;
        elasto_plast_modul = 2. * (stress2 - stress1) / (strain2 - strain1);

        if ((2. * refShearModulus - elasto_plast_modul) <= 0) plast_modul = soil::UP_LIMIT;
        else plast_modul = (2. * refShearModulus * elasto_plast_modul) / (2. * refShearModulus - elasto_plast_modul);
        if (plast_modul <= 0) return NegativeModulus;
        if (plast_modul > soil::UP_LIMIT) plast_modul = soil::UP_LIMIT;
        p.size[i] = size;
        p.modulus[i] = plast_modul;

        if (i == (numOfSurfaces - 1)) {
          plast_modul = 0;
          size = ratio2;
          p.size[i + 1] = size;
          p.modulus[i + 1] = plast_modul;
        }
      }
    }
    p.residualPress = residualPress;
    p.frictionAngle = frictionAngle;
    p.phaseTransfAngle = phaseTransfAngle;
    p.stressRatioPT = stressRatioPT;
    return Ok;
  }

  // ---------------------------------------------------------------------------------------
  // One point's update: OpenSees' member functions, with the instance's members as locals.
  // Surface centers: the trial buffer (theSurfaces) is worked in place; committedSurfaces
  // are the committed buffer (read only), except during elast2Plast (see the header).
  // ---------------------------------------------------------------------------------------
  template <class Ctx>
  struct Point {
    const Params& p;
    Ctx ctx;
    Strided<const double> C;
    Strided<double> T;
    bool ok = true;                      // false after an impossible state (OpenSees exits)
    bool centers_from_committed = false;  // committed_tangent: read centers from C

    // Committed values (currentStress, ...) and trial values (trialStress, ...).
    soil::T2 currentStress, currentStrain;
    int committedActiveSurf = 0, e2p = 0;
    double maxPress = 0.;

    soil::T2 trialStress, reversalStress, PPZPivot, PPZCenter, lockStress;
    int activeSurfaceNum = 0, onPPZ = -1;
    double pressureD = 0., PPZSize = 0., cumuDilateStrainOcta = 0., maxCumuDilateStrainOcta = 0.,
           cumuTranslateStrainOcta = 0., prePPZStrainOcta = 0., oppoPrePPZStrainOcta = 0.;

    // OpenSees' static workspaces that carry values within one call.
    soil::T2 strainRate, subStrainRate, trialStrain;
    double modulusFactor = 0.;

    FEMORAX_HD Point(const Params& p_, const Ctx& ctx_, Strided<const double> C_, Strided<double> T_)
        : p(p_), ctx(ctx_), C(C_), T(T_) {}

    // --- surfaces ---------------------------------------------------------------------
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
      e2p = static_cast<int>(C[STATE::E2P]);
      maxPress = C[STATE::MAXPRESS];
      // The trial plastic values start as the committed ones (getStress copies them).
      reversalStress.load(C, STATE::REVERSAL);
      PPZPivot.load(C, STATE::PPZPIVOT);
      PPZCenter.load(C, STATE::PPZCENTER);
      lockStress.load(C, STATE::LOCK);
      activeSurfaceNum = committedActiveSurf;
      pressureD = C[STATE::PRESSURED];
      onPPZ = static_cast<int>(C[STATE::ONPPZ]);
      PPZSize = C[STATE::PPZSIZE];
      cumuDilateStrainOcta = C[STATE::CUMUDILATE];
      maxCumuDilateStrainOcta = C[STATE::MAXCUMUDILATE];
      cumuTranslateStrainOcta = C[STATE::CUMUTRANSLATE];
      prePPZStrainOcta = C[STATE::PREPPZ];
      oppoPrePPZStrainOcta = C[STATE::OPPOPREPPZ];
    }

    // setTrialStrain: strainRate = strain - currentStrain (engineering, stored as tensor).
    FEMORAX_HD void setTrialStrain(const double* strain) {
      double workV6[6], cs[6];
      soil::vcopy(workV6, strain);
      currentStrain.t2_engr(cs);
      soil::vsub_eq(workV6, cs);
      strainRate.set(workV6, 1);
    }

    // --- predict / correct ------------------------------------------------------------
    // Everything of getStress up to the elastic-or-plastic decision. Returns true if the
    // plastic corrector (the sub-increment loop) must run.
    FEMORAX_HD bool predict(const double* eps) {
      load_committed();
      setTrialStrain(eps);
      const int loadStage = p.loadStage;
      const int numOfSurfaces = p.numOfSurfaces;
      // theSurfaces = committedSurfaces (OpenSees: in getStress, after elast2Plast; here
      // first, so that elast2Plast works on the copy: see the header).
      inner_for(ctx, numOfSurfaces, [&](int s) {
        const int k = STATE::CENTERS + 6 * s;
        for (int c = 0; c < 6; ++c) T[k + c] = C[k + c];
      });
      if (loadStage == 1 && e2p == 0) elast2Plast();

      if (loadStage != 1) {  // linear elastic
        double D[6][6];
        elastic_tangent(p, 1.0, D);
        double workV6[6], rate[6];
        soil::vcopy(workV6, currentStress.t2);
        strainRate.t2_engr(rate);
        // addMatrixVector(1.0, theTangent, rate, 1.0): column by column
        for (int i = 0; i < 6; i++)
          for (int j = 0; j < 6; j++) workV6[j] += D[j][i] * rate[i];
        trialStress.set(workV6);
        return false;
      }
      activeSurfaceNum = committedActiveSurf;
      subStrainRate = strainRate;
      setTrialStress(currentStress);
      if (activeSurfaceNum > 0 && isLoadReversal(currentStress)) {
        updateInnerSurface();
        activeSurfaceNum = 0;
      }
      if (activeSurfaceNum == 0 && !isCrossingNextSurface()) {
        double workV6[6];
        soil::vcopy(workV6, currentStrain.t2);
        soil::add_vector(workV6, 1.0, strainRate.t2, 1.0);
        trialStrain.set(workV6);
        return false;
      }
      return true;
    }

    // The sub-increment loop of getStress.
    FEMORAX_HD void correct() {
      int is;
      int numSubIncre = setSubStrainRate();
      for (int i = 0; i < numSubIncre; i++) {
        double workV6[6];
        soil::vcopy(workV6, currentStrain.t2);
        soil::add_vector(workV6, 1.0, subStrainRate.t2, (i + 1));
        trialStrain.set(workV6);

        if (i == 0) {
          setTrialStress(currentStress);
          is = isLoadReversal(currentStress);
        } else {
          soil::T2 workT2V;
          workT2V.set(trialStress.t2);
          setTrialStress(trialStress);
          is = isLoadReversal(workT2V);
        }
        if (activeSurfaceNum > 0 && is) {
          updateInnerSurface();
          activeSurfaceNum = 0;
        }
        if (activeSurfaceNum == 0 && !isCrossingNextSurface()) continue;
        if (activeSurfaceNum == 0) activeSurfaceNum++;
        int lock = stressCorrection(0);
        if (lock == 0) updateActiveSurface();
      }
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
      // plastic stage the plastic values and maxPress.
      double workV6[6];
      soil::vcopy(workV6, currentStrain.t2);
      soil::vadd_eq(workV6, strainRate.t2);
      soil::T2 strainAtCommit;
      strainAtCommit.set(workV6);
      trialStress.store(T, STATE::STRESS);
      strainAtCommit.store(T, STATE::STRAIN);
      if (p.loadStage == 1) {
        store_trial_values();
        double mp = maxPress;
        if (trialStress.volume() < mp) mp = trialStress.volume();
        T[STATE::MAXPRESS] = mp;
        T[STATE::E2P] = e2p;
      } else {
        // The plastic values stay the committed ones (OpenSees: commitState copies them
        // only in the plastic stage); the centers were copied in predict.
        for (int k = STATE::REVERSAL; k < STATE::CENTERS; ++k) T[k] = C[k];
      }
    }

    FEMORAX_HD void store_trial_values() {
      reversalStress.store(T, STATE::REVERSAL);
      PPZPivot.store(T, STATE::PPZPIVOT);
      PPZCenter.store(T, STATE::PPZCENTER);
      lockStress.store(T, STATE::LOCK);
      T[STATE::ACTIVE] = activeSurfaceNum;
      T[STATE::PRESSURED] = pressureD;
      T[STATE::ONPPZ] = onPPZ;
      T[STATE::PPZSIZE] = PPZSize;
      T[STATE::CUMUDILATE] = cumuDilateStrainOcta;
      T[STATE::MAXCUMUDILATE] = maxCumuDilateStrainOcta;
      T[STATE::CUMUTRANSLATE] = cumuTranslateStrainOcta;
      T[STATE::PREPPZ] = prePPZStrainOcta;
      T[STATE::OPPOPREPPZ] = oppoPrePPZStrainOcta;
    }

    // --- OpenSees' member functions -----------------------------------------------------
    FEMORAX_HD void elast2Plast() {
      const int loadStage = p.loadStage;
      const int numOfSurfaces = p.numOfSurfaces;
      if (loadStage != 1 || e2p == 1) return;
      e2p = 1;
      if (currentStress.volume() > 0.) {
        // setData(currentStress.deviator(), 0): the int overload (isEngrgStrain = 0).
        double d[6];
        soil::vcopy(d, currentStress.dev);
        currentStress.set(d, 0);
      }
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

    FEMORAX_HD double yieldFunc(const soil::T2& stress, int surface_num) const {
      const double residualPress = p.residualPress;
      double coneHeight = stress.volume() - residualPress;
      double workV6[6], c[6];
      soil::vcopy(workV6, stress.dev);
      center(surface_num, c);
      soil::add_vector(workV6, 1.0, c, -coneHeight);
      double sz = p.size[surface_num] * coneHeight;
      return 3. / 2. * soil::ddot(workV6, workV6) - sz * sz;
    }

    // deviatorScaling, its tail recursion as a loop.
    FEMORAX_HD void deviatorScaling(soil::T2& stress, int surfaceNum) {
      const double residualPress = p.residualPress;
      const int numOfSurfaces = p.numOfSurfaces;
      for (int iter = 0; iter < 100000; ++iter) {
        double diff = yieldFunc(stress, surfaceNum);
        double coneHeight = stress.volume() - residualPress;
        if (surfaceNum < numOfSurfaces && diff < 0.) {
          double sz = -p.size[surfaceNum] * coneHeight;
          double deviaSz = std::sqrt(sz * sz + diff);
          double devia[6], workV6[6], c[6];
          soil::vcopy(devia, stress.dev);
          soil::vcopy(workV6, devia);
          center(surfaceNum, c);
          soil::add_vector(workV6, 1.0, c, -coneHeight);
          double coeff = (sz - deviaSz) / deviaSz;
          if (coeff < 1.e-13) coeff = 1.e-13;
          soil::add_vector(devia, 1.0, workV6, coeff);
          stress.set_dev(devia, stress.volume());
          continue;  // deviatorScaling(stress, surfaces, surfaceNum)
        }
        if (surfaceNum == numOfSurfaces && std::fabs(diff) > soil::LOW_LIMIT) {
          double sz = -p.size[surfaceNum] * coneHeight;
          double workV6[6];
          soil::vcopy(workV6, stress.dev);
          soil::vmul_eq(workV6, sz / std::sqrt(diff + sz * sz));
          stress.set_dev(workV6, stress.volume());
        }
        return;
      }
      ok = false;  // OpenSees would recurse until the stack overflows
    }

    FEMORAX_HD void initSurfaceUpdate() {
      const double residualPress = p.residualPress;
      const int numOfSurfaces = p.numOfSurfaces;
      if (committedActiveSurf == 0) return;
      double coneHeight = -(currentStress.volume() - residualPress);
      double devia[6];
      soil::vcopy(devia, currentStress.dev);
      double Ms = std::sqrt(3. / 2. * soil::ddot(devia, devia));
      if (committedActiveSurf < numOfSurfaces) {  // failure surface can't move
        double workV6[6] = {};
        soil::add_vector(workV6, 0.0, devia, (1. - p.size[committedActiveSurf] * coneHeight / Ms));
        soil::vdiv_eq(workV6, -coneHeight);
        setCenter(committedActiveSurf, workV6);
      }
      // committedSurfaces[i] and theSurfaces[i]: the same buffer here.
      inner_for(ctx, committedActiveSurf - 1, [&](int k) {
        const int i = k + 1;
        double workV6[6] = {};
        soil::add_vector(workV6, 0.0, devia, (1. - p.size[i] * coneHeight / Ms));
        soil::vdiv_eq(workV6, -coneHeight);
        setCenter(i, workV6);
      });
      activeSurfaceNum = committedActiveSurf;
    }

    FEMORAX_HD double getModulusFactor(const soil::T2& stress) const {
      const double residualPress = p.residualPress;
      const double refPressure = p.refPressure;
      const double pressDependCoeff = p.pressDependCoeff;
      double conHeig = stress.volume() - residualPress;
      double scale = conHeig / (refPressure - residualPress);
      scale = std::pow(scale, pressDependCoeff);
      return (1.e-10 > scale) ? 1.e-10 : scale;
    }

    FEMORAX_HD void setTrialStress(const soil::T2& stress_in) {
      const soil::T2 stress = stress_in;  // may be trialStress itself
      const double refShearModulus = p.refShearModulus;
      const double refBulkModulus = p.refBulkModulus;
      modulusFactor = getModulusFactor(stress);
      double workV6[6];
      soil::vcopy(workV6, stress.dev);
      soil::add_vector(workV6, 1.0, subStrainRate.dev, 2 * refShearModulus * modulusFactor);
      double B = refBulkModulus * modulusFactor;
      if (p.Hv != 0. && trialStress.volume() <= maxPress && subStrainRate.volume() < 0.) {
        double tp = std::fabs(trialStress.volume() - p.residualPress);
        B = (B * p.Hv * std::pow(tp, p.Pv)) / (B + p.Hv * std::pow(tp, p.Pv));
      }
      double volume = stress.volume() + subStrainRate.volume() * 3. * B;
      if (volume > 0.) volume = 0.;
      trialStress.set_dev(workV6, volume);
    }

    FEMORAX_HD int setSubStrainRate() {
      const double residualPress = p.residualPress;
      const double refShearModulus = p.refShearModulus;
      const int numOfSurfaces = p.numOfSurfaces;
      if (activeSurfaceNum == numOfSurfaces) return 1;
      if (strainRate.is_zero()) return 0;

      double elast_plast_modulus;
      double conHeig = -(currentStress.volume() - residualPress);
      double factor = getModulusFactor(currentStress);
      if (activeSurfaceNum == 0) elast_plast_modulus = 2 * refShearModulus * factor;
      else {
        double plast_modulus = p.modulus[activeSurfaceNum] * factor;
        elast_plast_modulus = 2 * refShearModulus * factor * plast_modulus / (2 * refShearModulus * factor + plast_modulus);
      }
      double workV6[6] = {};
      soil::add_vector(workV6, 0.0, strainRate.dev, elast_plast_modulus);
      soil::T2 workT2V;
      workT2V.set(workV6, 0);

      double singleCross = p.size[numOfSurfaces] * conHeig / numOfSurfaces;
      double totalCross = 3. * workT2V.octahedral_shear() / std::sqrt(2.);
      int numOfSub = to_int(totalCross / singleCross + 1);
      if (numOfSub > numOfSurfaces) numOfSub = numOfSurfaces;

      int numOfSub1 = to_int(strainRate.octahedral_shear(1) / 1.0e-4);
      int numOfSub2 = to_int(strainRate.volume() / 1.e-5);
      if (numOfSub1 > numOfSub) numOfSub = numOfSub1;
      if (numOfSub2 > numOfSub) numOfSub = numOfSub2;

      soil::add_vector(workV6, 0.0, strainRate.t2, 1.0 / numOfSub);
      subStrainRate.set(workV6);
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
      const double residualPress = p.residualPress;
      double conHeig = trialStress.volume() - residualPress;
      double c[6], workV6[6];
      center(activeSurfaceNum, c);
      soil::vcopy(workV6, trialStress.dev);
      soil::add_vector(workV6, 1.0, c, -conHeig);
      double Ms = std::sqrt(3. / 2. * soil::ddot(workV6, workV6));
      soil::add_vector(workV6, p.size[activeSurfaceNum] * (-conHeig) / Ms, c, conHeig);
      contactStress.set_dev(workV6, trialStress.volume());
    }

    FEMORAX_HD int isLoadReversal(const soil::T2& stress) const {
      if (activeSurfaceNum == 0) return 0;
      soil::T2 workT2V;
      getSurfaceNormal(stress, workT2V);
      double workV6[6];
      soil::vcopy(workV6, trialStress.t2);
      soil::vsub_eq(workV6, currentStress.t2);
      if (soil::ddot(workV6, workT2V.t2) < 0) return 1;
      return 0;
    }

    FEMORAX_HD void getSurfaceNormal(const soil::T2& stress, soil::T2& normal) const {
      const double residualPress = p.residualPress;
      double conHeig = stress.volume() - residualPress;
      double workV6[6], c[6];
      soil::vcopy(workV6, stress.dev);
      center(activeSurfaceNum, c);
      double sz = p.size[activeSurfaceNum];
      double volume = conHeig * (soil::ddot(c, c) - 2. / 3. * sz * sz) - soil::ddot(workV6, c);
      soil::add_vector(workV6, 1.0, c, -conHeig);
      soil::vmul_eq(workV6, 3.0);
      soil::T2 workT2V;
      workT2V.set_dev(workV6, volume);
      double u[6];
      workT2V.unit_t2(u);
      normal.set(u);
    }

    FEMORAX_HD double getPlasticPotential(const soil::T2& contactStress, const soil::T2& surfaceNormal) {
      const double residualPress = p.residualPress;
      const double stressRatioPT = p.stressRatioPT;
      const int numOfSurfaces = p.numOfSurfaces;
      const double contractParam1 = p.contractParam1;
      const double dilateParam1 = p.dilateParam1;
      const double dilateParam2 = p.dilateParam2;

      double plasticPotential, contractRule, unloadRule, dilateRule, shearLoading, temp;

      double contactRatio = contactStress.deviator_ratio(residualPress);
      temp = contactRatio / stressRatioPT;
      double factorPT = (temp * temp - 1) / (temp * temp + 1) / 3.;
      contractRule = factorPT * contractParam1;
      if (contractRule > 0.) contractRule = -contractRule;
      if (contractRule < -5.0e4) contractRule = -5.0e4;
      temp = currentStress.volume() - pressureD;
      if (temp >= 0.) unloadRule = 0.;
      else {
        double conHeiD = pressureD - residualPress;
        double temp1 = std::sqrt(3. / 2.) * currentStress.deviator_length() + stressRatioPT * conHeiD;
        temp = temp1 / (-temp);
        if (temp < p.size[numOfSurfaces]) temp = p.size[numOfSurfaces];
        temp1 = (reversalStress.volume() - residualPress) / conHeiD;
        unloadRule = -std::sqrt(3. / 2.) * surfaceNormal.deviator_length() * temp1 / temp;
      }

      double currentRatio = currentStress.deviator_ratio(residualPress);
      double trialRatio = trialStress.deviator_ratio(residualPress);
      shearLoading = soil::ddot(currentStress.dev, trialStress.dev);

      if (factorPT < 0.) {  // below PT
        if (pressureD == 0.) plasticPotential = contractRule;
        else if (trialStress.volume() >= pressureD) {
          pressureD = 0.;
          plasticPotential = contractRule;
        } else if (trialRatio > currentRatio && shearLoading >= 0.)
          plasticPotential = contractRule;
        else
          plasticPotential = unloadRule;
      } else {  // above PT
        if (trialRatio > currentRatio && shearLoading >= 0.) {  // dilation
          if (pressureD == 0.) pressureD = currentStress.volume();
          reversalStress = currentStress;
          updatePPZ(contactStress);
          if (onPPZ == -1 || onPPZ == 1) return soil::LOCK_VALUE;
          if (isCriticalState(contactStress))
            dilateRule = 0;
          else
            dilateRule = factorPT * dilateParam1 * std::exp(dilateParam2 * cumuDilateStrainOcta);
          if (dilateRule > 5.0e4) dilateRule = 5.0e4;
          return dilateRule;
        } else if (pressureD == 0.)
          plasticPotential = contractRule;
        else if (trialStress.volume() >= pressureD) {
          pressureD = 0.;
          plasticPotential = contractRule;
        } else
          plasticPotential = unloadRule;
      }

      if (onPPZ > 0) onPPZ = 0;
      if (onPPZ != -1) PPZTranslation(contactStress);
      if (isCriticalState(contactStress)) return 0.;
      return plasticPotential;
    }

    FEMORAX_HD int isCriticalState(const soil::T2& stress) const {
      const double einit = p.einit;
      const double volLimit1 = p.volLimit1;
      const double volLimit2 = p.volLimit2;
      const double volLimit3 = p.volLimit3;
      const double pAtm = p.pAtm;  // per material (docs/deviations.md)

      double vol = trialStrain.volume() * 3.0;
      double etria = einit + vol + vol * einit;
      vol = currentStrain.volume() * 3.0;
      double ecurr = einit + vol + vol * einit;

      double ecr1, ecr2;
      if (volLimit3 != 0.) {
        ecr1 = volLimit1 - volLimit2 * std::pow(std::fabs(-stress.volume() / pAtm), volLimit3);
        ecr2 = volLimit1 - volLimit2 * std::pow(std::fabs(-currentStress.volume() / pAtm), volLimit3);
      } else {
        ecr1 = volLimit1 - volLimit2 * std::log(std::fabs(-stress.volume() / pAtm));
        ecr2 = volLimit1 - volLimit2 * std::log(std::fabs(-currentStress.volume() / pAtm));
      }

      if (ecurr < ecr2 && etria < ecr1) return 0;
      if (ecurr > ecr2 && etria > ecr1) return 0;
      return 1;
    }

    FEMORAX_HD void updatePPZ(const soil::T2& contactStress) {
      const double liquefyParam1 = p.liquefyParam1;
      const double residualPress = p.residualPress;
      const double refPressure = p.refPressure;
      const double pressDependCoeff = p.pressDependCoeff;
      soil::T2 workT2V;
      double workV6[6];

      // PPZ inactive if liquefyParam1==0.
      if (liquefyParam1 == 0.) {
        if (onPPZ == 2) {
          double d[6];
          for (int i = 0; i < 6; ++i) d[i] = trialStrain.t2[i] - PPZPivot.t2[i];
          workT2V.set(d);
          cumuDilateStrainOcta = workT2V.octahedral_shear(1);
        } else if (onPPZ != 2) {
          onPPZ = 2;
          PPZPivot = trialStrain;
          cumuDilateStrainOcta = 0.;
        }
        return;
      }

      // dilation: calc. cumulated dilative strain
      if (onPPZ == 2) {
        PPZPivot = trialStrain;
        soil::vcopy(workV6, PPZPivot.t2);
        soil::vsub_eq(workV6, PPZCenter.t2);
        workT2V.set(workV6);
        cumuDilateStrainOcta += subStrainRate.octahedral_shear(1);
        if (cumuDilateStrainOcta > maxCumuDilateStrainOcta) maxCumuDilateStrainOcta = cumuDilateStrainOcta;
        return;
      }

      // calc. PPZ size.
      double PPZLimit = getPPZLimits(1, contactStress);
      double TransLimit = getPPZLimits(2, contactStress);

      if (onPPZ == -1 || onPPZ == 0) {
        soil::vcopy(workV6, trialStrain.t2);
        soil::vsub_eq(workV6, PPZPivot.t2);
        workT2V.set(workV6);
        double temp = workT2V.octahedral_shear(1);
        if (temp > cumuDilateStrainOcta) {
          double volume = -contactStress.volume();
          oppoPrePPZStrainOcta = prePPZStrainOcta;
          double ratio = (volume + residualPress) / (-refPressure + residualPress);
          ratio = std::pow(ratio, 1. - pressDependCoeff);
          prePPZStrainOcta = ratio * p.strainPTOcta;
          if (oppoPrePPZStrainOcta == 0.) oppoPrePPZStrainOcta = prePPZStrainOcta;
        }
      }
      if (onPPZ > -1)
        PPZSize = PPZLimit + (prePPZStrainOcta + oppoPrePPZStrainOcta + TransLimit + maxCumuDilateStrainOcta) / 2.;
      else
        PPZSize = PPZLimit + (prePPZStrainOcta + oppoPrePPZStrainOcta + maxCumuDilateStrainOcta) / 2.;

      // calc. new PPZ center.
      if (onPPZ == 0 || onPPZ == 1) {
        soil::vcopy(workV6, PPZPivot.t2);
        soil::vsub_eq(workV6, PPZCenter.t2);
        workT2V.set(workV6);

        double coeff = (PPZSize - cumuTranslateStrainOcta) / workT2V.octahedral_shear(1);
        soil::vcopy(workV6, PPZPivot.t2);
        soil::add_vector(workV6, 1.0, workT2V.t2, -coeff);
        PPZCenter.set(workV6);
      }

      soil::vcopy(workV6, trialStrain.t2);
      soil::vsub_eq(workV6, PPZCenter.t2);
      workT2V.set(workV6);
      double temp = soil::ddot(subStrainRate.t2, workV6);

      if ((workT2V.octahedral_shear(1) > PPZSize && temp > 0.) || PPZLimit == 0.) {  // outside PPZ
        soil::vcopy(workV6, trialStrain.t2);
        soil::vsub_eq(workV6, PPZPivot.t2);
        workT2V.set(workV6);
        double temp1 = workT2V.octahedral_shear(1);
        if (temp1 > cumuDilateStrainOcta) {
          cumuDilateStrainOcta = 0.;
          if (PPZLimit == 0.) maxCumuDilateStrainOcta = 0.;
        }
        onPPZ = 2;
        PPZPivot = trialStrain;
        cumuTranslateStrainOcta = 0.;
      } else {  // inside PPZ
        if (onPPZ == 0 || onPPZ == 1) PPZTranslation(contactStress);
        if (onPPZ == -1 || onPPZ == 0) lockStress = contactStress;
        if (onPPZ == 0) onPPZ = 1;
      }
    }

    FEMORAX_HD void PPZTranslation(const soil::T2& contactStress) {
      const double liquefyParam1 = p.liquefyParam1;
      if (liquefyParam1 == 0.) return;
      double PPZLimit = getPPZLimits(1, contactStress);
      if (PPZLimit == 0.) return;
      double PPZTransLimit = getPPZLimits(2, contactStress);

      double workV6[6];
      soil::vcopy(workV6, trialStrain.dev);
      soil::vsub_eq(workV6, PPZPivot.dev);
      soil::T2 workT2V;
      workT2V.set(workV6);

      double temp = workT2V.octahedral_shear(1);
      if (cumuTranslateStrainOcta < temp) cumuTranslateStrainOcta = temp;
      if (maxCumuDilateStrainOcta == 0.) temp = PPZTransLimit;
      else temp = PPZTransLimit * cumuDilateStrainOcta / maxCumuDilateStrainOcta;
      if (cumuTranslateStrainOcta > temp) cumuTranslateStrainOcta = temp;
    }

    FEMORAX_HD double getPPZLimits(int which, const soil::T2& contactStress) const {
      const double liquefyParam1 = p.liquefyParam1;
      const double liquefyParam2 = p.liquefyParam2;
      const double liquefyParam4 = p.liquefyParam4;
      double PPZLimit, temp;
      double volume = -contactStress.volume();
      if (volume >= liquefyParam1) PPZLimit = 0.;
      else {
        temp = volume * soil::pi / liquefyParam1 / 2.;
        PPZLimit = liquefyParam2 * std::pow(std::cos(temp), 3.);
      }
      if (which == 1) return PPZLimit;
      return liquefyParam4 * PPZLimit;
    }

    FEMORAX_HD double getLoadingFunc(const soil::T2& contactStress, const soil::T2& surfaceNormal,
                                     double plasticPotential, int crossedSurface) const {
      const int numOfSurfaces = p.numOfSurfaces;
      const double refShearModulus = p.refShearModulus;
      const double refBulkModulus = p.refBulkModulus;

      double loadingFunc, limit;
      double modul = p.modulus[activeSurfaceNum];
      double temp1 = 2. * refShearModulus * modulusFactor * soil::ddot(surfaceNormal.dev, surfaceNormal.dev);
      double temp2 = 9. * refBulkModulus * modulusFactor * surfaceNormal.volume() * plasticPotential;

      // for the first crossing
      double temp = temp1 + temp2 + modul * modulusFactor;
      if (activeSurfaceNum == numOfSurfaces) limit = p.modulus[activeSurfaceNum - 1] * modulusFactor / 2.;
      else limit = modul * modulusFactor / 2.;
      if (temp < limit) {
        plasticPotential = (temp2 + limit - temp) / (9. * refBulkModulus * modulusFactor * surfaceNormal.volume());
        temp = limit;
      }
      (void)plasticPotential;
      double workV6[6];
      soil::vcopy(workV6, trialStress.dev);
      soil::vsub_eq(workV6, contactStress.dev);
      loadingFunc = soil::ddot(surfaceNormal.t2, workV6) / temp;

      if (loadingFunc < 0.) loadingFunc = 0;

      // for more than one crossing
      if (crossedSurface) {
        temp = (p.modulus[activeSurfaceNum - 1] - modul) / p.modulus[activeSurfaceNum - 1];
        loadingFunc *= temp;
      }
      return loadingFunc;
    }

    // stressCorrection, its tail recursion as a loop.
    FEMORAX_HD int stressCorrection(int crossedSurface) {
      const double refShearModulus = p.refShearModulus;
      const double refBulkModulus = p.refBulkModulus;
      for (;;) {
        soil::T2 contactStress, surfNormal;
        getContactStress(contactStress);
        getSurfaceNormal(contactStress, surfNormal);
        double plasticPotential = getPlasticPotential(contactStress, surfNormal);
        if (plasticPotential == soil::LOCK_VALUE && (onPPZ == -1 || onPPZ == 1)) {
          trialStress = lockStress;
          return 1;
        }

        double tVolume = trialStress.volume();
        double loadingFunc = getLoadingFunc(contactStress, surfNormal, plasticPotential, crossedSurface);
        double volume = tVolume - plasticPotential * 3 * refBulkModulus * modulusFactor * loadingFunc;

        double workV6[6];
        soil::vcopy(workV6, trialStress.dev);

        if (volume > 0. && volume != tVolume) {
          double coeff = tVolume / (tVolume - volume);
          coeff *= -2 * refShearModulus * modulusFactor * loadingFunc;
          soil::add_vector(workV6, 1.0, surfNormal.dev, coeff);
          volume = 0.;
        } else if (volume > 0.) {
          volume = 0.;
        } else {
          double coeff = -2 * refShearModulus * modulusFactor * loadingFunc;
          soil::add_vector(workV6, 1.0, surfNormal.dev, coeff);
        }
        trialStress.set_dev(workV6, volume);
        deviatorScaling(trialStress, activeSurfaceNum);

        if (isCrossingNextSurface()) {
          activeSurfaceNum++;
          crossedSurface = 1;  // return stressCorrection(1)
          continue;
        }
        return 0;
      }
    }

    FEMORAX_HD void updateActiveSurface() {
      const double residualPress = p.residualPress;
      const int numOfSurfaces = p.numOfSurfaces;
      if (activeSurfaceNum == numOfSurfaces) return;

      double A, B, Cc, X;
      double t1[6], t2[6], center_[6], outcenter[6], workV6[6];
      double conHeig = trialStress.volume() - residualPress;
      center(activeSurfaceNum, center_);
      double size = p.size[activeSurfaceNum];
      center(activeSurfaceNum + 1, outcenter);
      double outsize = p.size[activeSurfaceNum + 1];

      soil::vcopy(t1, trialStress.dev);
      soil::add_vector(t1, 1.0, center_, -conHeig);
      soil::vcopy(t2, center_);
      soil::vsub_eq(t2, outcenter);
      soil::vmul_eq(t2, conHeig);

      A = soil::ddot(t1, t1);
      B = 2. * soil::ddot(t1, t2);
      Cc = soil::ddot(t2, t2) - 2. / 3. * outsize * outsize * conHeig * conHeig;
      X = soil::second_order_eqn(A, B, Cc, 0, ok);
      if (std::fabs(X - 1.) < soil::LOW_LIMIT) X = 1.;
      if (X < 1.) return;

      soil::add_vector(workV6, 0.0, t1, X);
      soil::add_vector(workV6, 1.0, center_, conHeig);
      soil::vmul_eq(workV6, (1.0 - size / outsize));
      soil::vcopy(t2, center_);
      soil::add_vector(t2, 1.0, outcenter, -size / outsize);
      soil::vmul_eq(t2, conHeig);
      soil::vsub_eq(workV6, t2);

      soil::T2 workT2V;
      workT2V.set(workV6);
      if (workT2V.deviator_length() < soil::LOW_LIMIT) return;

      soil::vcopy(workV6, workT2V.dev);
      A = conHeig * conHeig * soil::ddot(workV6, workV6);
      B = 2 * conHeig * soil::ddot(t1, workV6);
      if (std::fabs(B) < soil::LOW_LIMIT) B = 0.;
      Cc = soil::ddot(t1, t1) - 2. / 3. * size * size * conHeig * conHeig;
      if (std::fabs(Cc) < soil::LOW_LIMIT || std::fabs(Cc) / soil::ddot(t1, t1) < soil::LOW_LIMIT) return;
      if (B > 0. || Cc < 0.) {  // OpenSees: "error in surface motion", exit
        ok = false;
        return;
      }
      X = soil::second_order_eqn(A, B, Cc, 1, ok);

      soil::add_vector(center_, 1.0, workV6, -X);
      setCenter(activeSurfaceNum, center_);
    }

    FEMORAX_HD void updateInnerSurface() {
      const double residualPress = p.residualPress;
      if (activeSurfaceNum <= 1) return;
      double devia[6], center_[6];
      double conHeig = currentStress.volume() - residualPress;
      soil::vcopy(devia, currentStress.dev);
      center(activeSurfaceNum, center_);
      double size = p.size[activeSurfaceNum];
      inner_for(ctx, activeSurfaceNum - 1, [&](int k) {
        const int i = k + 1;
        double workV6[6] = {};
        soil::add_vector(workV6, 0.0, center_, conHeig);
        soil::vsub_eq(workV6, devia);
        soil::vmul_eq(workV6, p.size[i] / size);
        soil::vadd_eq(workV6, devia);
        soil::vdiv_eq(workV6, conHeig);
        setCenter(i, workV6);
      });
    }

    FEMORAX_HD int isCrossingNextSurface() const {
      const int numOfSurfaces = p.numOfSurfaces;
      if (activeSurfaceNum == numOfSurfaces) return 0;
      if (yieldFunc(trialStress, activeSurfaceNum + 1) > 0) return 1;
      return 0;
    }

    // getTangent (stage 0: elastic; stage 1: elastoplastic, from the committed stress and
    // committedActiveSurf, with the trial surface activeSurfaceNum, as OpenSees).
    FEMORAX_HD void getTangent(double (*D)[NSTRAIN]) {
      const int loadStage = p.loadStage;
      const double refShearModulus = p.refShearModulus;
      const double refBulkModulus = p.refBulkModulus;
      const double residualPress = p.residualPress;
      if (loadStage != 1) {
        elastic_tangent(p, 1.0, D);
        return;
      }
      double coeff1, coeff2;
      double factor = getModulusFactor(currentStress);
      double shearModulus = factor * refShearModulus;
      double bulkModulus = factor * refBulkModulus;

      // volumetric plasticity
      if (p.Hv != 0. && trialStress.volume() <= maxPress && strainRate.volume() < 0.) {
        double tp = std::fabs(trialStress.volume() - residualPress);
        bulkModulus = (bulkModulus * p.Hv * std::pow(tp, p.Pv)) / (bulkModulus + p.Hv * std::pow(tp, p.Pv));
      }

      double workV6[6];
      if (loadStage != 0 && committedActiveSurf > 0) {
        soil::T2 workT2V;
        getSurfaceNormal(currentStress, workT2V);
        soil::vcopy(workV6, workT2V.dev);
        double volume = workT2V.volume();
        double Ho = 9. * bulkModulus * volume * volume + 2. * shearModulus * soil::ddot(workV6, workV6);
        double plastModul = factor * p.modulus[committedActiveSurf];
        coeff1 = 9. * bulkModulus * bulkModulus * volume * volume / (Ho + plastModul);
        coeff2 = 4. * shearModulus * shearModulus / (Ho + plastModul);
      } else {
        coeff1 = coeff2 = 0.;
        soil::vzero(workV6);
      }

      for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) {
          D[i][j] = -coeff2 * workV6[i] * workV6[j];
          if (i == j) D[i][j] += shearModulus;
          if (i < 3 && j < 3 && i == j) D[i][j] += shearModulus;
          if (i < 3 && j < 3) D[i][j] += (bulkModulus - 2. * shearModulus / 3. - coeff1);
        }
    }
  };
};

}  // namespace femorax
