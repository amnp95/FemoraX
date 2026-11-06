// The element and material contracts accept well-formed types with any DOF layout and
// reject inconsistent ones; the explicit stable step matches the schemes it describes.

#include "physics/contracts.hpp"
#include "physics/explicit_stability.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

using femorax::ExplicitScheme;
using femorax::MassType;
using femorax::MaterialKind;
using femorax::Rayleigh;

// ---------------------------------------------------------------------------
// Mock materials and elements (signatures only matter here)
// ---------------------------------------------------------------------------
struct Mat3D {
  static constexpr MaterialKind KIND = MaterialKind::NDMaterial3D;
  static constexpr int NSTRAIN = 6;
  static constexpr int NSTATE = 0;
  struct Params { double E, nu, rho; };
  static void update(const Params&, const double*, double*, const double*, double*,
                     double (*)[NSTRAIN]) {}
  static void initial_tangent(const Params&, double (*)[NSTRAIN]) {}
  static double density(const Params& p) { return p.rho; }
};

struct MatUniaxial {
  static constexpr MaterialKind KIND = MaterialKind::Uniaxial;
  static constexpr int NSTRAIN = 1;
  static constexpr int NSTATE = 2;
  struct Params { double E; };
  static void update(const Params&, const double*, double*, const double*, double*,
                     double (*)[NSTRAIN]) {}
  static void initial_tangent(const Params&, double (*)[NSTRAIN]) {}
  static double density(const Params&) { return 0.0; }
};

// Shared function set for mock elements of any size.
template <int NDM_, int NNODE_, int NDOF_>
struct MockFunctions {
  struct Params { double dummy; };
  template <bool WantK>
  static void compute(const Params&, const Rayleigh&, const double (*)[NDM_], const double*,
                      const double*, const double*, double*, double*, double (*)[NDOF_]) {}
  static void mass(const Params&, const double (*)[NDM_], double*) {}
  static double max_omega_sq(const Params&, const double (*)[NDM_], const double*) {
    return 1.0;
  }
  static double critical_dt(const Params&, const Rayleigh&, ExplicitScheme,
                            const double (*)[NDM_], const double*) {
    return 1.0;
  }
};

// 8-node brick, 3 DOFs per node.
struct Brick : MockFunctions<3, 8, 24> {
  static constexpr int NDM = 3, NNODE = 8, NEXTRA = 0, NDOF = 24, NGP = 8, NSTATE = 0;
  static constexpr std::array<int, NNODE> NODE_NDF = {3, 3, 3, 3, 3, 3, 3, 3};
  static constexpr MassType MASS = MassType::Selectable;
  static constexpr bool EXPLICIT = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
};

// u-p style: 4 DOFs at the 8 corner nodes, 3 at the 12 mid-side nodes (20-node brick).
struct BrickUP20 : MockFunctions<3, 20, 8 * 4 + 12 * 3> {
  static constexpr int NDM = 3, NNODE = 20, NEXTRA = 0, NDOF = 68, NGP = 27, NSTATE = 0;
  static constexpr std::array<int, NNODE> NODE_NDF = {4, 4, 4, 4, 4, 4, 4, 4, 3, 3,
                                                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
  static constexpr MassType MASS = MassType::Consistent;
  static constexpr bool EXPLICIT = false;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
};

// 3D beam: 2 nodes with 6 DOFs.
struct Beam3D : MockFunctions<3, 2, 12> {
  static constexpr int NDM = 3, NNODE = 2, NEXTRA = 0, NDOF = 12, NGP = 5, NSTATE = 10;
  static constexpr std::array<int, NNODE> NODE_NDF = {6, 6};
  static constexpr MassType MASS = MassType::Lumped;
  static constexpr bool EXPLICIT = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::Uniaxial;
};

// Element that owns 3 DOFs of its own besides its nodes' DOFs.
struct WithExtra : MockFunctions<3, 8, 8 * 3 + 3> {
  static constexpr int NDM = 3, NNODE = 8, NEXTRA = 3, NDOF = 27, NGP = 8, NSTATE = 0;
  static constexpr std::array<int, NNODE> NODE_NDF = {3, 3, 3, 3, 3, 3, 3, 3};
  static constexpr MassType MASS = MassType::Lumped;
  static constexpr bool EXPLICIT = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
};

// --- Broken elements: each one breaks exactly one rule ---------------------
struct WrongNDOF : MockFunctions<3, 8, 25> {  // 8*3 != 25
  static constexpr int NDM = 3, NNODE = 8, NEXTRA = 0, NDOF = 25, NGP = 8, NSTATE = 0;
  static constexpr std::array<int, NNODE> NODE_NDF = {3, 3, 3, 3, 3, 3, 3, 3};
  static constexpr MassType MASS = MassType::Lumped;
  static constexpr bool EXPLICIT = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
};

struct WrongNodeNdfLength : MockFunctions<3, 8, 21> {  // NODE_NDF has 7 entries, NNODE 8
  static constexpr int NDM = 3, NNODE = 8, NEXTRA = 0, NDOF = 21, NGP = 8, NSTATE = 0;
  static constexpr std::array<int, 7> NODE_NDF = {3, 3, 3, 3, 3, 3, 3};
  static constexpr MassType MASS = MassType::Lumped;
  static constexpr bool EXPLICIT = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
};

struct ZeroNdfNode : MockFunctions<3, 2, 3> {  // a node with 0 DOFs
  static constexpr int NDM = 3, NNODE = 2, NEXTRA = 0, NDOF = 3, NGP = 1, NSTATE = 0;
  static constexpr std::array<int, NNODE> NODE_NDF = {3, 0};
  static constexpr MassType MASS = MassType::Lumped;
  static constexpr bool EXPLICIT = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
};

struct NoCriticalDt {  // missing critical_dt and max_omega_sq
  static constexpr int NDM = 3, NNODE = 1, NEXTRA = 0, NDOF = 3, NGP = 1, NSTATE = 0;
  static constexpr std::array<int, NNODE> NODE_NDF = {3};
  static constexpr MassType MASS = MassType::Lumped;
  static constexpr bool EXPLICIT = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
  struct Params { double dummy; };
  template <bool WantK>
  static void compute(const Params&, const Rayleigh&, const double (*)[3], const double*,
                      const double*, const double*, double*, double*, double (*)[3]) {}
  static void mass(const Params&, const double (*)[3], double*) {}
};

struct RuntimeNdof : Brick {  // NDOF is not a compile-time constant
  static inline int NDOF = 24;
};

struct RuntimeExplicitFlag : Brick {  // EXPLICIT is not a compile-time constant
  static inline bool EXPLICIT = true;
};

struct ConsistentOnlyExplicit : Brick {  // claims EXPLICIT but cannot lump
  static constexpr MassType MASS = MassType::Consistent;
};

// ---------------------------------------------------------------------------
// Compile-time checks
// ---------------------------------------------------------------------------
static_assert(femorax::Material<Mat3D>);
static_assert(femorax::Material<MatUniaxial>);

static_assert(femorax::Element<Brick>);
static_assert(femorax::Element<BrickUP20>);
static_assert(femorax::Element<Beam3D>);
static_assert(femorax::Element<WithExtra>);

static_assert(!femorax::Element<WrongNDOF>);
static_assert(!femorax::Element<WrongNodeNdfLength>);
static_assert(!femorax::Element<ZeroNdfNode>);
static_assert(!femorax::Element<NoCriticalDt>);
static_assert(!femorax::Element<RuntimeNdof>);
static_assert(!femorax::Element<RuntimeExplicitFlag>);
static_assert(!femorax::ExplicitElement<RuntimeExplicitFlag>);

static_assert(femorax::ExplicitElement<Brick>);
static_assert(femorax::ExplicitElement<Beam3D>);
static_assert(!femorax::ExplicitElement<BrickUP20>);           // EXPLICIT = false
static_assert(!femorax::ExplicitElement<ConsistentOnlyExplicit>);

static_assert(femorax::CompatiblePair<Brick, Mat3D>);
static_assert(!femorax::CompatiblePair<Brick, MatUniaxial>);
static_assert(femorax::CompatiblePair<Beam3D, MatUniaxial>);

// ---------------------------------------------------------------------------
// Stable time step against direct simulation of each scheme on one damped mode
// (m = 1, k = w^2, c = alphaM + beta*w^2).
// ---------------------------------------------------------------------------

// OpenSees ExplicitDifference: v(n+1/2) = v(n-1/2) + dt*a(n); u(n+1) = u(n) + dt*v(n+1/2);
// a(n+1) = -k*u(n+1) - c*v(n+1/2).
bool explicit_difference_bounded(double w, double c, double dt, int nsteps) {
  double u = 1.0, v = 0.0, a = -w * w;
  for (int n = 0; n < nsteps; ++n) {
    v += dt * a;
    u += dt * v;
    a = -w * w * u - c * v;
    if (!(std::abs(u) < 1e8)) return false;
  }
  return true;
}

// OpenSees CentralDifference with centred damping, c = alphaM only (milestone 1a).
bool central_difference_bounded(double w, double c, double dt, int nsteps) {
  double um = 1.0, u = 1.0;
  const double a0 = 1.0 / (dt * dt), a1 = c / (2.0 * dt);
  for (int n = 0; n < nsteps; ++n) {
    const double up = ((2.0 * a0 - w * w) * u - (a0 - a1) * um) / (a0 + a1);
    um = u;
    u = up;
    if (!(std::abs(u) < 1e8)) return false;
  }
  return true;
}

struct ModeCase {
  double w, alphaM, betaK, betaKinit, betaKcomm;
};

class StableDt : public ::testing::TestWithParam<ModeCase> {};

TEST_P(StableDt, ExplicitDifferenceLimitIsSharp) {
  const ModeCase m = GetParam();
  const Rayleigh d{m.alphaM, m.betaK, m.betaKinit, m.betaKcomm};
  const double dt = femorax::stable_dt(ExplicitScheme::ExplicitDifference, d, m.w * m.w);
  const double c = m.alphaM + (m.betaK + m.betaKinit + m.betaKcomm) * m.w * m.w;
  EXPECT_TRUE(explicit_difference_bounded(m.w, c, 0.999 * dt, 20000));
  EXPECT_FALSE(explicit_difference_bounded(m.w, c, 1.01 * dt, 20000));
}

TEST_P(StableDt, CentralDifferenceIsTwoOverOmega) {
  const ModeCase m = GetParam();
  const Rayleigh d{m.alphaM, 0.0, 0.0, 0.0};
  const double dt = femorax::stable_dt(ExplicitScheme::CentralDifference, d, m.w * m.w);
  EXPECT_DOUBLE_EQ(dt, 2.0 / m.w);
  EXPECT_TRUE(central_difference_bounded(m.w, m.alphaM, 0.999 * dt, 20000));
  EXPECT_FALSE(central_difference_bounded(m.w, m.alphaM, 1.01 * dt, 20000));
}

INSTANTIATE_TEST_SUITE_P(
    Modes, StableDt,
    ::testing::Values(ModeCase{100.0, 0.0, 0.0, 0.0, 0.0},      // undamped
                      ModeCase{100.0, 0.5, 0.0, 0.0, 0.0},      // mass-proportional
                      ModeCase{100.0, 0.0, 2e-3, 0.0, 0.0},     // stiffness-proportional
                      ModeCase{300.0, 1.2, 6e-4, 0.0, 0.0},     // both
                      ModeCase{50.0, 3.0, 4e-4, 3e-4, 3e-4},    // all four terms
                      ModeCase{2000.0, 0.8, 1e-3, 0.0, 0.0}));  // heavy damping at high w

TEST(StableDt, UndampedExplicitDifferenceIsTwoOverOmega) {
  EXPECT_DOUBLE_EQ(femorax::stable_dt(ExplicitScheme::ExplicitDifference, Rayleigh{}, 400.0),
                   2.0 / 20.0);
}

TEST(StableDt, StiffnessDampingLowersExplicitDifferenceStep) {
  const double w2 = 1.0e6;
  const double undamped = femorax::stable_dt(ExplicitScheme::ExplicitDifference, {}, w2);
  const double damped =
      femorax::stable_dt(ExplicitScheme::ExplicitDifference, {0.0, 1e-4, 0.0, 0.0}, w2);
  EXPECT_LT(damped, undamped);
}

TEST(StableDt, NoStiffnessSetsNoLimit) {
  EXPECT_TRUE(std::isinf(femorax::stable_dt(ExplicitScheme::ExplicitDifference, {}, 0.0)));
}

}  // namespace
