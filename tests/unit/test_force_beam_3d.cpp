// ForceBeam3d (docs/design.md, "Beams: the force-based beam", step 1): the physics with the
// elastic section. One element of an elastic cantilever is exact for end forces and moments
// (the force-based element's interpolation of the section forces is exact without element
// loads): the tip deflections and rotations equal Euler-Bernoulli's to 1e-12 for 3 to 7 Lobatto
// points; the tangent times the displacement gives the force (linear); the stiffness equals the
// displacement beam's exact two-node stiffness; rigid motion gives no force; a fresh state and a
// committed state give the same answer.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "physics/elements/disp_beam_3d.hpp"
#include "physics/elements/force_beam_3d.hpp"
#include "physics/sections/elastic_section_3d.hpp"
#include "reference_data.hpp"

namespace {

using femorax::ElasticSection3d;
constexpr double E = 2.5e7, A = 0.5, Iz = 0.02, Iy = 0.015, G = 1.0e7, J = 0.03, L = 3.0;

template <int NIP>
struct Fixture {
  using Beam = femorax::ForceBeam3d<ElasticSection3d, NIP>;
  typename Beam::Params p{};
  std::vector<double> sc, st;  // committed and trial point state (contiguous per point)
  double xe[2][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, L}};  // vertical
  Fixture() : sc(Beam::NSTATE, 0.0), st(Beam::NSTATE, 0.0) {
    p.mat = {E, A, Iz, Iy, G, J};
    p.transf.vecxz[0] = 1.0;
    p.transf.vecxz[1] = 0.0;
    p.transf.vecxz[2] = 0.0;
  }
  femorax::PointArray<const double> committed() const { return {sc.data(), 1, Beam::POINT_STATE}; }
  femorax::PointArray<double> trial() { return {st.data(), 1, Beam::POINT_STATE}; }
  // K (12 x 12) and f at the nodal displacements u.
  bool eval(const double* u, double* f, double (*K)[12]) {
    Beam::template compute<true>(p, xe, u, committed(), trial(), f, K);
    return Beam::converged(femorax::PointArray<const double>{st.data(), 1, Beam::POINT_STATE});
  }
};

template <int NIP>
void cantilever() {
  Fixture<NIP> fx;
  // The base node fixed: solve K_jj u_j = F at the tip node (DOFs 6..11) for unit end loads.
  double u0[12] = {}, f[12], K[12][12];
  ASSERT_TRUE(fx.eval(u0, f, K));
  for (int k = 0; k < 12; ++k) EXPECT_NEAR(f[k], 0.0, 1e-30);
  double Kt[6][6];
  for (int a = 0; a < 6; ++a)
    for (int b = 0; b < 6; ++b) Kt[a][b] = K[6 + a][6 + b];
  double Ft[6][6];
  femorax::force_beam_detail::invert<6>(Kt, Ft);
  // The member's local axes: x along global z; vecxz = global x, so local z ~ global x, local y ~ ...
  // Check against closed forms in global terms: a tip force P along global x bends about the axis
  // whose moment of inertia couples x: here (vecxz = x) the local z axis is global x, so a force
  // along global x is a local z force: deflection P L^3 / (3 E Iy), rotation P L^2 / (2 E Iy).
  const double defl_x = L * L * L / (3.0 * E * Iy), defl_y = L * L * L / (3.0 * E * Iz);
  EXPECT_NEAR(std::fabs(Ft[0][0]), defl_x, 1e-12 * defl_x) << NIP;
  EXPECT_NEAR(std::fabs(Ft[1][1]), defl_y, 1e-12 * defl_y) << NIP;
  EXPECT_NEAR(Ft[2][2], L / (E * A), 1e-12 * L / (E * A)) << NIP;
  EXPECT_NEAR(Ft[5][5], L / (G * J), 1e-12 * L / (G * J)) << NIP;
  EXPECT_NEAR(std::fabs(Ft[3][3]), L / (E * Iz), 1e-12 * L / (E * Iz)) << NIP;
  EXPECT_NEAR(std::fabs(Ft[4][4]), L / (E * Iy), 1e-12 * L / (E * Iy)) << NIP;
  EXPECT_NEAR(std::fabs(Ft[0][4]), L * L / (2.0 * E * Iy), 1e-12 * L * L / (2.0 * E * Iy)) << NIP;

  // Linear: the force at a displacement equals K u (to rounding), from the fresh state and from
  // a committed state.
  double u[12];
  for (int k = 0; k < 12; ++k) u[k] = 1e-4 * std::sin(0.7 * k + 0.3);
  double fu[12], Ku[12][12];
  ASSERT_TRUE(fx.eval(u, fu, Ku));
  double peak = 0.0;
  for (int a = 0; a < 12; ++a) {
    double ku = 0.0;
    for (int b = 0; b < 12; ++b) ku += K[a][b] * u[b];
    peak = std::max(peak, std::fabs(ku));
    EXPECT_NEAR(fu[a], ku, 1e-10 * std::fabs(ku) + 1e-9) << NIP << " " << a;
  }
  // Commit (trial -> committed), then the same displacement again: the same force; then twice it.
  fx.sc = fx.st;
  double f2[12], K2[12][12];
  ASSERT_TRUE(fx.eval(u, f2, K2));
  for (int a = 0; a < 12; ++a) EXPECT_NEAR(f2[a], fu[a], 1e-12 * peak) << a;
  double uu[12];
  for (int k = 0; k < 12; ++k) uu[k] = 2.0 * u[k];
  ASSERT_TRUE(fx.eval(uu, f2, K2));
  for (int a = 0; a < 12; ++a) EXPECT_NEAR(f2[a], 2.0 * fu[a], 1e-11 * peak) << a;
  // Rigid translation: no force.
  double ur[12] = {1e-3, -2e-3, 3e-3, 0, 0, 0, 1e-3, -2e-3, 3e-3, 0, 0, 0};
  Fixture<NIP> fr;
  double fr_[12], Kr[12][12];
  ASSERT_TRUE(fr.eval(ur, fr_, Kr));
  for (int a = 0; a < 12; ++a) EXPECT_NEAR(fr_[a], 0.0, 1e-9) << a;
}

TEST(ForceBeam3d, OneElementCantileverIsExactFor3To7Points) {
  cantilever<3>();
  cantilever<4>();
  cantilever<5>();
  cantilever<6>();
  cantilever<7>();
}

// Against OpenSees' ForceBeamColumn3d with ElasticSection3d and LobattoBeamIntegration in a
// Domain (tests/reference/drivers/force_beam_column.cpp): the force, the tangent and the initial
// stiffness for four geometries (joint offsets included), 3, 5, 7 points and two displacement
// fields, Linear and PDelta.
double rel_diff(const double* a, const double* b, int n) {
  double d = 0.0, m = 0.0;
  for (int k = 0; k < n; ++k) {
    d = std::max(d, std::fabs(a[k] - b[k]));
    m = std::max(m, std::fabs(b[k]));
  }
  return m > 0.0 ? d / m : d;
}

template <int NIP>
void compare_row(const femorax::test::ReferenceFile& ref, const std::vector<double>& r, femorax::CrdTransfType type,
                 double worst[3]) {
  using Beam = femorax::ForceBeam3d<ElasticSection3d, NIP>;
  auto at = [&](const std::string& col) { return r[static_cast<std::size_t>(ref.column(col))]; };
  auto atk = [&](const std::string& col, int k) { return at(col + std::to_string(k)); };
  typename Beam::Params p{};
  p.mat = {at("E"), at("A"), at("Iz"), at("Iy"), at("G"), at("J")};
  p.transf.type = type;
  double xe[2][3];
  for (int i = 0; i < 3; ++i) {
    xe[0][i] = atk("xi", i);
    xe[1][i] = atk("xj", i);
    p.transf.vecxz[i] = atk("vecxz", i);
    p.transf.nodeIOffset[i] = atk("offi", i);
    p.transf.nodeJOffset[i] = atk("offj", i);
  }
  double ug[12], os_f[12], os_K[144], os_K0[144];
  for (int k = 0; k < 12; ++k) ug[k] = atk("ug", k), os_f[k] = atk("f", k);
  for (int k = 0; k < 144; ++k) os_K[k] = atk("K", k), os_K0[k] = atk("K0", k);
  std::vector<double> sc(Beam::NSTATE, 0.0), st(Beam::NSTATE, 0.0), st0(Beam::NSTATE, 0.0);
  const femorax::PointArray<const double> c{sc.data(), 1, Beam::POINT_STATE};
  double f[12], K[12][12], f0[12], K0[12][12];
  Beam::template compute<true>(p, xe, ug, c, {st.data(), 1, Beam::POINT_STATE}, f, K);
  const double zero[12] = {};
  Beam::template compute<true>(p, xe, zero, c, {st0.data(), 1, Beam::POINT_STATE}, f0, K0);
  worst[0] = std::max(worst[0], rel_diff(f, os_f, 12));
  worst[1] = std::max(worst[1], rel_diff(&K[0][0], os_K, 144));
  worst[2] = std::max(worst[2], rel_diff(&K0[0][0], os_K0, 144));
}

class ForceBeam3dReference : public ::testing::TestWithParam<femorax::CrdTransfType> {};

TEST_P(ForceBeam3dReference, MatchesOpenSees) {
  const auto type = GetParam();
  const std::string name = type == femorax::CrdTransfType::Linear ? "Linear" : "PDelta";
  const auto ref = femorax::test::read_reference(std::string(FEMORAX_REFERENCE_DATA) + "/element/ForceBeamColumn3d_" +
                                                 name + ".txt");
  ASSERT_EQ(ref.rows.size(), 24u);
  double worst[3] = {0, 0, 0};
  for (const auto& r : ref.rows) {
    const int nip = static_cast<int>(r[static_cast<std::size_t>(ref.column("nip"))]);
    if (nip == 3) compare_row<3>(ref, r, type, worst);
    else if (nip == 5) compare_row<5>(ref, r, type, worst);
    else compare_row<7>(ref, r, type, worst);
  }
  std::printf("[ ForceBeamColumn3d %s against OpenSees: f %.2e, K %.2e, K0 %.2e ]\n", name.c_str(), worst[0], worst[1],
              worst[2]);
  for (int k = 0; k < 3; ++k) EXPECT_LE(worst[k], 1e-14) << k;  // measured 3.4e-16
}

INSTANTIATE_TEST_SUITE_P(Types, ForceBeam3dReference,
                         ::testing::Values(femorax::CrdTransfType::Linear, femorax::CrdTransfType::PDelta),
                         [](const auto& tp) { return tp.param == femorax::CrdTransfType::Linear ? "Linear" : "PDelta"; });

// Step 2: mass, damping, operator, stable step (beam_common_3d.hpp with the force beam's own
// basic stiffnesses). An elastic prismatic member: the displacement beam (2+ points) has the
// exact stiffness too, so both elements must give the same stiffness, the same mass (the same
// code) and the same largest omega^2; the damping force and the operator equal their matrices.
template <class Beam>
void skew_params(typename Beam::Params& p, femorax::CrdTransfType type) {
  p.mat = {E, A, Iz, Iy, G, J};
  p.transf.type = type;
  p.transf.vecxz[0] = -0.2;
  p.transf.vecxz[1] = 0.3;
  p.transf.vecxz[2] = 1.0;
  p.transf.nodeIOffset[0] = 0.25;
  p.transf.nodeJOffset[2] = -0.15;
  p.rhoA = 0.39;
  p.rot[0] = 7.8 * (Iz + Iy);
  p.rot[1] = 7.8 * Iy;
  p.rot[2] = 7.8 * Iz;
}

TEST(ForceBeam3d, MassStiffnessAndStableStepEqualTheElasticDisplacementBeam) {
  using FB = femorax::ForceBeam3d<ElasticSection3d, 5>;
  using DB = femorax::DispBeam3d<ElasticSection3d, 3>;
  const double xe[2][3] = {{0.3, -1.2, 0.7}, {2.9, 0.4, 2.2}};
  for (const auto type : {femorax::CrdTransfType::Linear, femorax::CrdTransfType::PDelta}) {
    for (const int cmass : {0, 1}) {
      FB::Params pf{};
      DB::Params pd{};
      skew_params<FB>(pf, type);
      skew_params<DB>(pd, type);
      pf.cMass = pd.cMass = cmass;
      double Mf[144], Md[144];
      FB::mass(pf, xe, Mf);
      DB::mass(pd, xe, Md);
      for (int k = 0; k < 144; ++k) EXPECT_EQ(Mf[k], Md[k]) << k;
    }
    FB::Params pf{};
    DB::Params pd{};
    skew_params<FB>(pf, type);
    skew_params<DB>(pd, type);
    std::vector<double> sc(FB::NSTATE, 0.0), st(FB::NSTATE, 0.0);
    double u0[12] = {}, ff[12], Kf[12][12], fd[12], Kd[12][12];
    FB::compute<true>(pf, xe, u0, {sc.data(), 1, FB::POINT_STATE}, {st.data(), 1, FB::POINT_STATE}, ff,
                                   Kf);
    DB::compute<true>(pd, xe, u0, {nullptr, 1, 0}, {nullptr, 1, 0}, fd, Kd);
    double kmax = 0.0, kd = 0.0;
    for (int a = 0; a < 12; ++a)
      for (int b = 0; b < 12; ++b) {
        kmax = std::max(kmax, std::fabs(Kd[a][b]));
        kd = std::max(kd, std::fabs(Kf[a][b] - Kd[a][b]));
      }
    EXPECT_LE(kd, 1e-12 * kmax);
    const double wf = FB::max_omega_sq(pf, xe, {sc.data(), 1, FB::POINT_STATE});
    const double wd = DB::max_omega_sq(pd, xe, {nullptr, 1, 0});
    std::printf("[ force beam against the displacement beam (%s): K %.2e, omega^2 %.6e / %.6e ]\n",
                type == femorax::CrdTransfType::Linear ? "Linear" : "PDelta", kd / kmax, wf, wd);
    EXPECT_NEAR(wf, wd, 1e-10 * wd);
  }
}

TEST(ForceBeam3d, DampingAndOperatorEqualTheirMatrices) {
  using FB = femorax::ForceBeam3d<ElasticSection3d, 4>;
  const double xe[2][3] = {{0.3, -1.2, 0.7}, {2.9, 0.4, 2.2}};
  for (const auto type : {femorax::CrdTransfType::Linear, femorax::CrdTransfType::PDelta}) {
    FB::Params p{};
    skew_params<FB>(p, type);
    // The committed state at u1, the trial at u2.
    double u1[12], u2[12], ve[12], x[12];
    for (int k = 0; k < 12; ++k) {
      u1[k] = 1e-3 * std::sin(0.9 * k + 0.2);
      u2[k] = u1[k] + 5e-4 * std::cos(1.3 * k);
      ve[k] = 0.1 * std::sin(0.4 * k + 1.0);
      x[k] = std::cos(0.7 * k + 0.5);
    }
    std::vector<double> s0(FB::NSTATE, 0.0), sc(FB::NSTATE, 0.0), st(FB::NSTATE, 0.0), sx(FB::NSTATE, 0.0);
    auto pa = [](std::vector<double>& v) { return femorax::PointArray<double>{v.data(), 1, FB::POINT_STATE}; };
    auto pc = [](const std::vector<double>& v) {
      return femorax::PointArray<const double>{v.data(), 1, FB::POINT_STATE};
    };
    double f[12], Kc[12][12], Kt[12][12], K0[12][12], zero[12] = {};
    FB::compute<true>(p, xe, u1, pc(s0), pa(sc), f, Kc);      // commit at u1
    FB::compute<true>(p, xe, u2, pc(sc), pa(st), f, Kt);      // trial at u2
    FB::compute<true>(p, xe, zero, pc(s0), pa(sx), f, K0);    // initial
    // Kc without PDelta's geometric term (the damping's committed stiffness, as the
    // displacement beam: docs/deviations.md): the committed state's kv through T alone.
    double Kc_lin[12][12];
    {
      femorax::CrdTransf3dGeometry g;
      FB::geometry(p, xe, g);
      FB::State s;
      FB::load_state(p, g.L, pc(sc), s);
      femorax::crd_transf_3d_global_stiff(p.transf, g, s.kv, s.Se, false, Kc_lin);
    }
    double M[144];
    FB::mass(p, xe, M);
    double peak = 0.0, worst = 0.0;
    const double factors[4][4] = {{0.7, 0, 0, 0}, {0, 2e-3, 0, 0}, {0, 0, 1e-3, 0}, {0, 0, 0, 1.5e-3}};
    for (const auto& fc : factors) {
      femorax::Rayleigh d{};
      d.alphaM = fc[0];
      d.betaK = fc[1];
      d.betaKinit = fc[2];
      d.betaKcomm = fc[3];
      double fdmp[12], fund[12];
      std::vector<double> t1(FB::NSTATE, 0.0), t2(FB::NSTATE, 0.0);
      FB::compute<false>(p, d, xe, u2, ve, pc(sc), pa(t1), fdmp, nullptr);
      FB::compute<false>(p, femorax::Rayleigh{}, xe, u2, ve, pc(sc), pa(t2), fund, nullptr);
      for (int a = 0; a < 12; ++a) {
        double want = 0.0;
        for (int b = 0; b < 12; ++b)
          want += (d.alphaM * M[12 * a + b] + d.betaK * Kt[a][b] + d.betaKinit * K0[a][b] +
                   d.betaKcomm * Kc_lin[a][b]) *
                  ve[b];
        peak = std::max(peak, std::fabs(want));
        worst = std::max(worst, std::fabs(fdmp[a] - fund[a] - want));
      }
    }
    // The operator: cK Kt + cC (alphaM M + betaK Kt + betaK0 K0 + betaKc Kc) + cM M, and its diagonal.
    femorax::Rayleigh d{};
    d.alphaM = 0.3;
    d.betaK = 1e-3;
    d.betaKinit = 2e-3;
    d.betaKcomm = 3e-3;
    const femorax::OperatorCoefficients k{1.7, 2.3, 400.0, false};
    double y[12], yd[12];
    std::vector<double> t3(FB::NSTATE, 0.0);
    FB::operator_apply(p, d, k, xe, u2, x, pc(sc), pa(t3), false, y);
    FB::operator_apply(p, d, k, xe, u2, x, pc(sc), pa(t3), true, yd);
    double opeak = 0.0, oworst = 0.0, dworst = 0.0;
    for (int a = 0; a < 12; ++a) {
      double want = 0.0, diag = 0.0;
      for (int b = 0; b < 12; ++b) {
        const double Kab = k.stiffness * Kt[a][b] +
                           k.damping * (d.alphaM * M[12 * a + b] + d.betaK * Kt[a][b] + d.betaKinit * K0[a][b] +
                                        d.betaKcomm * Kc_lin[a][b]) +
                           k.mass * M[12 * a + b];
        want += Kab * x[b];
        if (a == b) diag = Kab;
      }
      opeak = std::max(opeak, std::fabs(want));
      oworst = std::max(oworst, std::fabs(y[a] - want));
      dworst = std::max(dworst, std::fabs(yd[a] - diag) / std::max(1.0, std::fabs(diag)));
    }
    std::printf("[ force beam (%s): damping %.2e of %.3e; operator %.2e of %.3e; diagonal %.2e ]\n",
                type == femorax::CrdTransfType::Linear ? "Linear" : "PDelta", worst / peak, peak, oworst / opeak, opeak,
                dworst);
    EXPECT_LE(worst, 1e-11 * peak);
    EXPECT_LE(oworst, 1e-11 * opeak);
    EXPECT_LE(dworst, 1e-11);
  }
}

}  // namespace
