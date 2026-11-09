#pragma once

// stdBrick: 8-node trilinear brick, 2x2x2 Gauss integration.
//
// Port of OpenSees SRC/element/brick/Brick.cpp and shp3d.cpp, OpenSees commit
// 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea. Names follow the original (xl, shp, xsj,
// dvol, sg, wg, b, massType, ...) so the two can be read side by side.
//
// What is ported, and the OpenSees function it matches:
//   point_geometry      shp3d at Gauss point gp, dvol = wg * xsj
//   point_strain        Brick::update (strain from trial displacements)
//   point_node_force    Brick::formResidAndTangent residual: B^T (stress * dvol)
//   point_body_force    Brick::formResidAndTangent body force: dvol * b * N
//   point_block_tangent Brick::formResidAndTangent tangent: B_J^T (D * dvol) B_K
//   compute             getResistingForce + getRayleighDampingForces (+ getTangentStiff)
//   mass, lumped_mass   Brick::formInertiaTerms (massType 0 consistent, 1 lumped)
//
// Deliberate differences (docs/deviations.md, "stdBrick"):
// - Rayleigh damping forces are summed point by point (matrix-free) instead of forming
//   C = alphaM M + betaK K + betaK0 K0 + betaKc Kc and multiplying by v. Same quantity,
//   different rounding order.
// - Only the body-force path with no eleLoad (applyLoad == 0) exists; schema v0 has no
//   eleLoad.
// - The optional element Damping objects (theDamping, OpenSees -damp) are not supported.
//
// Gauss points are numbered as in OpenSees: gp = 4*i + 2*j + k with natural coordinates
// (sg[i], sg[j], sg[k]), sg = {-1/sqrt(3), +1/sqrt(3)}, weights 1.

#include <array>
#include <cmath>

#include "physics/common/inner.hpp"
#include "physics/common/strided.hpp"
#include "physics/common/sym_eigen.hpp"
#include "physics/contracts.hpp"
#include "physics/explicit_stability.hpp"
#include "physics/macros.hpp"

namespace femorax {

namespace brick_detail {

// shp3d.cpp: shape functions and global derivatives at natural point ss.
//   shp[0][i] = dN_i/dx, shp[1][i] = dN_i/dy, shp[2][i] = dN_i/dz, shp[3][i] = N_i
FEMORAX_HD inline void shp3d(const double ss[3], double& xsj, double shp[4][8],
                             const double xl[3][8]) {
  double rxsj, ap1, am1, ap2, am2, ap3, am3, c1, c2, c3;
  double xs[3][3];
  double ad[3][3];

  ap1 = 1.0 + ss[0];
  am1 = 1.0 - ss[0];
  ap2 = 1.0 + ss[1];
  am2 = 1.0 - ss[1];
  ap3 = 1.0 + ss[2];
  am3 = 1.0 - ss[2];

  // ( - , - )
  c1 = 0.125 * am1 * am2;
  c2 = 0.125 * am2 * am3;
  c3 = 0.125 * am1 * am3;
  shp[0][0] = -c2;
  shp[0][1] = c2;
  shp[1][0] = -c3;
  shp[1][3] = c3;
  shp[2][0] = -c1;
  shp[2][4] = c1;
  shp[3][0] = c1 * am3;
  shp[3][4] = c1 * ap3;

  // ( + , + )
  c1 = 0.125 * ap1 * ap2;
  c2 = 0.125 * ap2 * ap3;
  c3 = 0.125 * ap1 * ap3;
  shp[0][7] = -c2;
  shp[0][6] = c2;
  shp[1][5] = -c3;
  shp[1][6] = c3;
  shp[2][2] = -c1;
  shp[2][6] = c1;
  shp[3][2] = c1 * am3;
  shp[3][6] = c1 * ap3;

  // ( - , + )
  c1 = 0.125 * am1 * ap2;
  c2 = 0.125 * am2 * ap3;
  c3 = 0.125 * am1 * ap3;
  shp[0][4] = -c2;
  shp[0][5] = c2;
  shp[1][4] = -c3;
  shp[1][7] = c3;
  shp[2][3] = -c1;
  shp[2][7] = c1;
  shp[3][3] = c1 * am3;
  shp[3][7] = c1 * ap3;

  // ( + , - )
  c1 = 0.125 * ap1 * am2;
  c2 = 0.125 * ap2 * am3;
  c3 = 0.125 * ap1 * am3;
  shp[0][3] = -c2;
  shp[0][2] = c2;
  shp[1][1] = -c3;
  shp[1][2] = c3;
  shp[2][1] = -c1;
  shp[2][5] = c1;
  shp[3][1] = c1 * am3;
  shp[3][5] = c1 * ap3;

  // Jacobian
  for (int j = 0; j < 3; j++) {
    xs[j][0] = (xl[j][1] - xl[j][0]) * shp[0][1] + (xl[j][2] - xl[j][3]) * shp[0][2] +
               (xl[j][5] - xl[j][4]) * shp[0][5] + (xl[j][6] - xl[j][7]) * shp[0][6];

    xs[j][1] = (xl[j][2] - xl[j][1]) * shp[1][2] + (xl[j][3] - xl[j][0]) * shp[1][3] +
               (xl[j][6] - xl[j][5]) * shp[1][6] + (xl[j][7] - xl[j][4]) * shp[1][7];

    xs[j][2] = (xl[j][4] - xl[j][0]) * shp[2][4] + (xl[j][5] - xl[j][1]) * shp[2][5] +
               (xl[j][6] - xl[j][2]) * shp[2][6] + (xl[j][7] - xl[j][3]) * shp[2][7];
  }

  // Adjoint
  ad[0][0] = xs[1][1] * xs[2][2] - xs[1][2] * xs[2][1];
  ad[0][1] = xs[2][1] * xs[0][2] - xs[2][2] * xs[0][1];
  ad[0][2] = xs[0][1] * xs[1][2] - xs[0][2] * xs[1][1];

  ad[1][0] = xs[1][2] * xs[2][0] - xs[1][0] * xs[2][2];
  ad[1][1] = xs[2][2] * xs[0][0] - xs[2][0] * xs[0][2];
  ad[1][2] = xs[0][2] * xs[1][0] - xs[0][0] * xs[1][2];

  ad[2][0] = xs[1][0] * xs[2][1] - xs[1][1] * xs[2][0];
  ad[2][1] = xs[2][0] * xs[0][1] - xs[2][1] * xs[0][0];
  ad[2][2] = xs[0][0] * xs[1][1] - xs[0][1] * xs[1][0];

  // Determinant and inverse
  xsj = xs[0][0] * ad[0][0] + xs[0][1] * ad[1][0] + xs[0][2] * ad[2][0];
  rxsj = 1.0 / xsj;

  for (int j = 0; j < 3; j++)
    for (int i = 0; i < 3; i++) xs[i][j] = ad[i][j] * rxsj;

  // Global derivatives
  for (int k = 0; k < 8; k++) {
    c1 = shp[0][k] * xs[0][0] + shp[1][k] * xs[1][0] + shp[2][k] * xs[2][0];
    c2 = shp[0][k] * xs[0][1] + shp[1][k] * xs[1][1] + shp[2][k] * xs[2][1];
    c3 = shp[0][k] * xs[0][2] + shp[1][k] * xs[1][2] + shp[2][k] * xs[2][2];

    shp[0][k] = c1;
    shp[1][k] = c2;
    shp[2][k] = c3;
  }
}

}  // namespace brick_detail

template <class Mat>
struct StdBrick {
  static_assert(Material<Mat>);
  using Material = Mat;
  static_assert(Mat::KIND == MaterialKind::NDMaterial3D, "stdBrick needs a 3D nD material");

  static constexpr int NDM = 3;
  static constexpr int NNODE = 8;
  static constexpr std::array<int, NNODE> NODE_NDF = {3, 3, 3, 3, 3, 3, 3, 3};
  static constexpr int NEXTRA = 0;
  static constexpr int NDOF = 24;
  static constexpr int NGP = 8;
  static constexpr int NSTRESS = 6;
  static constexpr int NSTATE = NGP * Mat::NSTATE;
  static constexpr MassType MASS = MassType::Selectable;
  static constexpr bool EXPLICIT = true;
  // Small strain: the reference geometry never changes, so it may be cached.
  static constexpr bool GEOMETRY_CACHEABLE = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
  // No element terms of its own: the stiffness is B^T D B.
  static constexpr TangentSymmetry TANGENT = Mat::TANGENT;

  struct Params {
    typename Mat::Params mat;
    double b[3];   // body force per unit volume (b1 b2 b3)
    int massType;  // 0 consistent, 1 lumped (-lumped), as Brick::massType
  };

  // Shape functions and volume weight of one Gauss point.
  struct PointShape {
    double shp[4][8];
    double dvol;
  };

  // Brick::computeBasis: nodal coordinates as xl[dim][node].
  FEMORAX_HD static void local_coords(const double (*xe)[NDM], double xl[3][8]) {
    for (int i = 0; i < 8; i++) {
      xl[0][i] = xe[i][0];
      xl[1][i] = xe[i][1];
      xl[2][i] = xe[i][2];
    }
  }

  // shp3d at Gauss point gp = 4*i + 2*j + k; dvol = wg[gp] * xsj with wg = 1.
  FEMORAX_HD static void point_geometry(const double xl[3][8], int gp, PointShape& s) {
    const double one_over_root3 = 1.0 / std::sqrt(3.0);  // Brick::one_over_root3
    const double sg[2] = {-one_over_root3, one_over_root3};
    const double gaussPoint[3] = {sg[(gp >> 2) & 1], sg[(gp >> 1) & 1], sg[gp & 1]};
    double xsj;
    brick_detail::shp3d(gaussPoint, xsj, s.shp, xl);
    const double wg = 1.0;
    s.dvol = wg * xsj;
  }

  // --- Precomputed geometry (small strain: the reference geometry never changes) --------
  // Per Gauss point: the 24 global derivatives shp[0..2][0..7] in that order, then dvol.
  static constexpr int NGEOM = 25;

  // Shape function values N_j at point gp: shp3d's shp[3][*], with the same arithmetic
  // (they depend on the natural coordinates only, so they are not stored).
  FEMORAX_HD static void point_shape_values(int gp, double N[8]) {
    const double one_over_root3 = 1.0 / std::sqrt(3.0);
    const double sg[2] = {-one_over_root3, one_over_root3};
    const double ss[3] = {sg[(gp >> 2) & 1], sg[(gp >> 1) & 1], sg[gp & 1]};
    const double ap1 = 1.0 + ss[0], am1 = 1.0 - ss[0];
    const double ap2 = 1.0 + ss[1], am2 = 1.0 - ss[1];
    const double ap3 = 1.0 + ss[2], am3 = 1.0 - ss[2];
    double c1 = 0.125 * am1 * am2;
    N[0] = c1 * am3;
    N[4] = c1 * ap3;
    c1 = 0.125 * ap1 * ap2;
    N[2] = c1 * am3;
    N[6] = c1 * ap3;
    c1 = 0.125 * am1 * ap2;
    N[3] = c1 * am3;
    N[7] = c1 * ap3;
    c1 = 0.125 * ap1 * am2;
    N[1] = c1 * am3;
    N[5] = c1 * ap3;
  }

  // Compute point gp's geometry and store it (compile step).
  FEMORAX_HD static void point_store_geometry(const double xl[3][8], int gp, Strided<double> g) {
    PointShape s;
    point_geometry(xl, gp, s);
    for (int k = 0; k < 24; ++k) g[k] = s.shp[k / 8][k % 8];
    g[24] = s.dvol;
  }

  // Rebuild point gp's shape from stored geometry: bit-identical to point_geometry.
  FEMORAX_HD static void point_load_geometry(int gp, Strided<const double> g, PointShape& s) {
    for (int k = 0; k < 24; ++k) s.shp[k / 8][k % 8] = g[k];
    point_shape_values(gp, s.shp[3]);
    s.dvol = g[24];
  }

  // Geometry sources for compute: recomputed from coordinates, or loaded from the cache.
  struct RecomputedGeometry {
    const double (*xl)[8];
    FEMORAX_HD void shape(int gp, PointShape& s) const { point_geometry(xl, gp, s); }
  };
  struct CachedGeometry {
    PointArray<const double> g;
    FEMORAX_HD void shape(int gp, PointShape& s) const { point_load_geometry(gp, g.point(gp), s); }
  };
  // Factories (MatrixFreeElement): stdBrick has no element constants, so p and c are unused.
  FEMORAX_HD static RecomputedGeometry recomputed_geometry(const double (*xl)[8], const Params& p) {
    (void)p;
    return {xl};
  }
  FEMORAX_HD static CachedGeometry cached_geometry(PointArray<const double> g, Strided<const double> c) {
    (void)c;
    return {g};
  }

  // Brick::update: strain (11 22 33 12 23 31, engineering shear) from nodal values u.
  // The same operator gives the strain rate from nodal velocities.
  FEMORAX_HD static void point_strain(const PointShape& s, const double ul[NDOF],
                                      double strain[NSTRESS]) {
    for (int c = 0; c < NSTRESS; ++c) strain[c] = 0.0;
    for (int j = 0; j < NNODE; j++) {
      const double b00 = s.shp[0][j];
      const double b11 = s.shp[1][j];
      const double b22 = s.shp[2][j];
      const double b30 = s.shp[1][j];
      const double b31 = s.shp[0][j];
      const double b41 = s.shp[2][j];
      const double b42 = s.shp[1][j];
      const double b50 = s.shp[2][j];
      const double b52 = s.shp[0][j];

      const double ul0 = ul[3 * j + 0];
      const double ul1 = ul[3 * j + 1];
      const double ul2 = ul[3 * j + 2];

      strain[0] += b00 * ul0;
      strain[1] += b11 * ul1;
      strain[2] += b22 * ul2;
      strain[3] += b30 * ul0 + b31 * ul1;
      strain[4] += b41 * ul1 + b42 * ul2;
      strain[5] += b50 * ul0 + b52 * ul2;
    }
  }

  // Contribution of one point to node j's force: B_j^T (stress * dvol)
  // (Brick::formResidAndTangent, residJ).
  FEMORAX_HD static void point_node_force(const PointShape& s, const double stress_in[NSTRESS],
                                          int j, double residJ[3]) {
    const double stress0 = stress_in[0] * s.dvol;
    const double stress1 = stress_in[1] * s.dvol;
    const double stress2 = stress_in[2] * s.dvol;
    const double stress3 = stress_in[3] * s.dvol;
    const double stress4 = stress_in[4] * s.dvol;
    const double stress5 = stress_in[5] * s.dvol;

    const double b00 = s.shp[0][j];
    const double b11 = s.shp[1][j];
    const double b22 = s.shp[2][j];
    const double b30 = s.shp[1][j];
    const double b31 = s.shp[0][j];
    const double b41 = s.shp[2][j];
    const double b42 = s.shp[1][j];
    const double b50 = s.shp[2][j];
    const double b52 = s.shp[0][j];

    residJ[0] = b00 * stress0 + b30 * stress3 + b50 * stress5;
    residJ[1] = b11 * stress1 + b31 * stress3 + b41 * stress4;
    residJ[2] = b22 * stress2 + b42 * stress4 + b52 * stress5;
  }

  // Body force of one point at node j, component q: dvol * b[q] * N_j. OpenSees subtracts it
  // from the residual after adding residJ: resid += residJ; resid -= body (applyLoad == 0).
  FEMORAX_HD static double point_body_force(const Params& p, const PointShape& s, int j, int q) {
    return s.dvol * p.b[q] * s.shp[3][j];
  }

  // Contribution of one point to the 3x3 block (J, K) of the stiffness: B_J^T (dd dvol) B_K.
  FEMORAX_HD static void point_block_tangent(const PointShape& s,
                                             const double (*dd)[NSTRESS], int J, int K,
                                             double stiffJK[3][3]) {
    double BJ[NSTRESS][3], BK[NSTRESS][3];
    computeB(J, s, BJ);
    computeB(K, s, BK);
    // BJtranD = BJ^T * (dd * dvol), then stiffJK = BJtranD * BK, as the OpenSees loops.
    double BJtranD[3][NSTRESS];
    for (int p = 0; p < 3; p++)
      for (int q = 0; q < NSTRESS; q++) {
        double sum = 0.0;
        for (int r = 0; r < NSTRESS; r++) sum += BJ[r][p] * (dd[r][q] * s.dvol);
        BJtranD[p][q] = sum;
      }
    for (int p = 0; p < 3; p++)
      for (int q = 0; q < 3; q++) {
        double sum = 0.0;
        for (int r = 0; r < NSTRESS; r++) sum += BJtranD[p][r] * BK[r][q];
        stiffJK[p][q] = sum;
      }
  }

  // Point s's share of (M v) at node j (Brick::formInertiaTerms' mass, without forming it):
  // consistent: rho dvol N_j sum_k N_k v_k; lumped: rho dvol N_j v_j. Summed over the
  // points, this is mass() times v.
  FEMORAX_HD static void point_mass_apply(const Params& p, const PointShape& s, const double* v,
                                          int j, double* out) {
    const double temp = s.shp[3][j] * s.dvol * Mat::density(p.mat);
    for (int q = 0; q < 3; q++) {
      double vq;
      if (p.massType == 0) {
        vq = 0.0;
        for (int k = 0; k < NNODE; k++) vq += s.shp[3][k] * v[3 * k + q];
      } else {
        vq = v[3 * j + q];
      }
      out[q] = temp * vq;
    }
  }

  // Point s's share of the mass entry between nodes j and k (the same component; the mass
  // couples no components): rho dvol N_j N_k (consistent) or rho dvol N_j for j == k, 0
  // otherwise (lumped).
  FEMORAX_HD static double point_mass_entry(const Params& p, const PointShape& s, int j, int k) {
    const double temp = s.shp[3][j] * s.dvol * Mat::density(p.mat);
    if (p.massType == 0) return temp * s.shp[3][k];
    return j == k ? temp : 0.0;
  }

  // Everything one Gauss point contributes, before it is spread to the nodes.
  struct PointResult {
    PointShape s;
    double stress[NSTRESS];      // material stress
    double dampStress[NSTRESS];  // (betaK Dt + betaKinit D0 + betaKcomm Dc) * strain rate
    double v_gp[3];              // velocity at the point (consistent-mass alphaM damping)
    double rho;
    bool stiffDamping;
  };

  // Point i: geometry, strain, material update (writes trial state), damping stress.
  // dd receives the material tangent. D0 must hold the initial tangent when
  // damp.betaKinit != 0 (computed once per element by the caller).
  FEMORAX_HD static void point_update(const Params& p, const Rayleigh& damp,
                                      const double xl[3][8], const double* ue, const double* ve,
                                      int i, Strided<const double> committed_i,
                                      Strided<double> trial_i, const double (*D0)[NSTRESS],
                                      PointResult& r, double (*dd)[NSTRESS]) {
    PointShape s;
    point_geometry(xl, i, s);
    point_update_shape(p, damp, s, ue, ve, committed_i, trial_i, D0, r, dd);
  }

  // The same with the point's shape already built (recomputed or from the cache).
  FEMORAX_HD static void point_update_shape(const Params& p, const Rayleigh& damp,
                                            const PointShape& s, const double* ue,
                                            const double* ve, Strided<const double> committed_i,
                                            Strided<double> trial_i, const double (*D0)[NSTRESS],
                                            PointResult& r, double (*dd)[NSTRESS]) {
    r.s = s;
    double strain[NSTRESS];
    point_strain(r.s, ue, strain);
    Mat::update(p.mat, committed_i, trial_i, strain, r.stress, dd);

    r.stiffDamping = damp.betaK != 0.0 || damp.betaKinit != 0.0 || damp.betaKcomm != 0.0;
    for (int a = 0; a < NSTRESS; ++a) r.dampStress[a] = 0.0;
    if (r.stiffDamping) {
      double Dc[NSTRESS][NSTRESS];
      if (damp.betaKcomm != 0.0) Mat::committed_tangent(p.mat, committed_i, Dc);
      double rate[NSTRESS];
      point_strain(r.s, ve, rate);
      for (int a = 0; a < NSTRESS; ++a) {
        double sum = 0.0;
        for (int c = 0; c < NSTRESS; ++c) {
          double Dac = damp.betaK * dd[a][c];
          if (damp.betaKinit != 0.0) Dac += damp.betaKinit * D0[a][c];
          if (damp.betaKcomm != 0.0) Dac += damp.betaKcomm * Dc[a][c];
          sum += Dac * rate[c];
        }
        r.dampStress[a] = sum;
      }
    }
    r.rho = Mat::density(p.mat);
    r.v_gp[0] = r.v_gp[1] = r.v_gp[2] = 0.0;
    if (damp.alphaM != 0.0 && p.massType == 0) {
      for (int j = 0; j < NNODE; j++)
        for (int q = 0; q < 3; q++) r.v_gp[q] += r.s.shp[3][j] * ve[3 * j + q];
    }
  }

  // What point r contributes to node j: residJ (B_j^T stress dvol), body (dvol b N_j) and
  // dampJ (damping force). Accumulate as resid += residJ; resid -= body; fdamp += dampJ, in
  // Gauss point order, then fe = resid + fdamp: the OpenSees order (bit-identical force).
  FEMORAX_HD static void point_node_contribution(const Params& p, const Rayleigh& damp,
                                                 const PointResult& r, const double* ve, int j,
                                                 double residJ[3], double body[3],
                                                 double dampJ[3]) {
    point_node_force(r.s, r.stress, j, residJ);
    for (int q = 0; q < 3; q++) body[q] = point_body_force(p, r.s, j, q);
    dampJ[0] = dampJ[1] = dampJ[2] = 0.0;
    if (r.stiffDamping) point_node_force(r.s, r.dampStress, j, dampJ);
    if (damp.alphaM != 0.0) {
      // alphaM * M * v with this point's mass: temp = N_j * dvol * rho.
      const double temp = r.s.shp[3][j] * r.s.dvol * r.rho;
      for (int q = 0; q < 3; q++) {
        const double vq = p.massType == 0 ? r.v_gp[q] : ve[3 * j + q];
        dampJ[q] += damp.alphaM * temp * vq;
      }
    }
  }

  // Buffers written by the inner loops of compute. In lane-parallel layouts the launch
  // layer places them in team scratch (inner.hpp: locals are private per lane).
  struct Work {
    double resid[NDOF];  // internal force with body force
    double fdamp[NDOF];  // Rayleigh damping force
  };

  // Internal force including the body force, plus all four Rayleigh damping forces
  // (OpenSees getResistingForce + getRayleighDampingForces), and, when WantK, the tangent
  // stiffness (getTangentStiff; damping terms are not added to Ke).
  //
  // Inner loops: the 8 nodes (force) and the 64 node-pair 3x3 blocks (tangent) go through
  // inner_for(ctx, ...); everything per Gauss point runs on every lane. fe, Ke and work
  // must be shared between the lanes of the calling thread.
  template <bool WantK, class Ctx>
  FEMORAX_HD static void compute(const Ctx& ctx, const Params& p, const Rayleigh& damp,
                                 const double (*xe)[NDM], const double* ue, const double* ve,
                                 PointArray<const double> committed, PointArray<double> trial,
                                 double* fe, double (*Ke)[NDOF], Work& work) {
    double xl[3][8];
    local_coords(xe, xl);
    compute_with<WantK>(ctx, RecomputedGeometry{xl}, p, damp, ue, ve, committed, trial, fe, Ke,
                        work);
  }

  // compute with any geometry source (RecomputedGeometry or CachedGeometry).
  template <bool WantK, class Ctx, class Geometry>
  FEMORAX_HD static void compute_with(const Ctx& ctx, const Geometry& geometry, const Params& p,
                                      const Rayleigh& damp, const double* ue, const double* ve,
                                      PointArray<const double> committed,
                                      PointArray<double> trial, double* fe,
                                      double (*Ke)[NDOF], Work& work) {
    inner_for(ctx, NNODE, [&](int j) {
      for (int q = 0; q < 3; q++) work.resid[3 * j + q] = work.fdamp[3 * j + q] = 0.0;
    });
    if constexpr (WantK) {
      inner_for(ctx, NNODE * NNODE, [&](int blk) {
        const int J = blk / NNODE, K = blk % NNODE;
        for (int a = 0; a < 3; a++)
          for (int c = 0; c < 3; c++) Ke[3 * J + a][3 * K + c] = 0.0;
      });
    }

    double D0[NSTRESS][NSTRESS];
    if (damp.betaKinit != 0.0) Mat::initial_tangent(p.mat, D0);

    for (int i = 0; i < NGP; i++) {
      PointShape s;
      geometry.shape(i, s);
      PointResult r;
      double dd[NSTRESS][NSTRESS];
      point_update_shape(p, damp, s, ue, ve, committed.point(i), trial.point(i), D0, r, dd);

      inner_for(ctx, NNODE, [&](int j) {
        double residJ[3], body[3], dampJ[3];
        point_node_contribution(p, damp, r, ve, j, residJ, body, dampJ);
        for (int q = 0; q < 3; q++) {
          work.resid[3 * j + q] += residJ[q];
          work.resid[3 * j + q] -= body[q];
          work.fdamp[3 * j + q] += dampJ[q];
        }
      });

      if constexpr (WantK) {
        inner_for(ctx, NNODE * NNODE, [&](int blk) {
          const int J = blk / NNODE, K = blk % NNODE;
          double stiffJK[3][3];
          point_block_tangent(r.s, dd, J, K, stiffJK);
          for (int a = 0; a < 3; a++)
            for (int c = 0; c < 3; c++) Ke[3 * J + a][3 * K + c] += stiffJK[a][c];
        });
      }
    }
    // Same index space as the writes above, so each lane reads only its own nodes.
    inner_for(ctx, NNODE, [&](int j) {
      for (int q = 0; q < 3; q++) fe[3 * j + q] = work.resid[3 * j + q] + work.fdamp[3 * j + q];
    });
  }

  // Serial form (the Element contract): plain loops, local work buffers.
  template <bool WantK>
  FEMORAX_HD static void compute(const Params& p, const Rayleigh& damp, const double (*xe)[NDM],
                                 const double* ue, const double* ve,
                                 PointArray<const double> committed, PointArray<double> trial,
                                 double* fe, double (*Ke)[NDOF]) {
    Work work;
    compute<WantK>(SerialInner{}, p, damp, xe, ue, ve, committed, trial, fe, Ke, work);
  }

  // Serial form with precomputed geometry (geometry.point(gp) holds NGEOM values).
  template <bool WantK>
  FEMORAX_HD static void compute_cached(const Params& p, const Rayleigh& damp,
                                        PointArray<const double> geometry, const double* ue,
                                        const double* ve, PointArray<const double> committed,
                                        PointArray<double> trial, double* fe, double (*Ke)[NDOF]) {
    Work work;
    compute_with<WantK>(SerialInner{}, CachedGeometry{geometry}, p, damp, ue, ve, committed,
                        trial, fe, Ke, work);
  }

  // Element output (Brick::getResponse 6 and 7, the stress3D6 / strain3D6 responses of the
  // VTKHDF recorder): stress as the sum of sigma * 0.125, strain as the sum divided by 8.
  FEMORAX_HD static void average_stress(const double (*sig)[NSTRESS], double out[NSTRESS]) {
    for (int c = 0; c < NSTRESS; ++c) out[c] = 0.0;
    for (int i = 0; i < NGP; i++)
      for (int c = 0; c < NSTRESS; ++c) out[c] += sig[i][c] * 0.125;
  }
  FEMORAX_HD static void average_strain(const double (*eps)[NSTRESS], double out[NSTRESS]) {
    for (int c = 0; c < NSTRESS; ++c) out[c] = 0.0;
    for (int i = 0; i < NGP; i++)
      for (int c = 0; c < NSTRESS; ++c) out[c] += eps[i][c];
    for (int c = 0; c < NSTRESS; ++c) out[c] /= 8.0;
  }

  // Brick::formInertiaTerms with tangFlag = 1: the full 24x24 mass matrix (row-major,
  // Me[a * NDOF + c]), consistent or lumped per p.massType.
  FEMORAX_HD static void mass(const Params& p, const double (*xe)[NDM], double* Me) {
    double xl[3][8];
    local_coords(xe, xl);
    for (int a = 0; a < NDOF * NDOF; ++a) Me[a] = 0.0;
    const double rho = Mat::density(p.mat);
    for (int i = 0; i < NGP; i++) {
      PointShape s;
      point_geometry(xl, i, s);
      for (int j = 0; j < NNODE; j++) {
        double temp = s.shp[3][j] * s.dvol;
        temp *= rho;
        for (int k = 0; k < NNODE; k++) {
          double massJK;
          if (p.massType == 0) {
            massJK = temp * s.shp[3][k];
          } else {
            if (j != k) continue;
            massJK = temp;
          }
          for (int q = 0; q < 3; q++) Me[(3 * j + q) * NDOF + 3 * k + q] += massJK;
        }
      }
    }
  }

  // Lumped (row-sum) mass per node, the diagonal of mass() with massType 1. Used by
  // explicit stages and the stable time step whatever massType says.
  FEMORAX_HD static void lumped_mass(const Params& p, const double (*xe)[NDM], double m[NNODE]) {
    double xl[3][8];
    local_coords(xe, xl);
    for (int j = 0; j < NNODE; j++) m[j] = 0.0;
    const double rho = Mat::density(p.mat);
    for (int i = 0; i < NGP; i++) {
      PointShape s;
      point_geometry(xl, i, s);
      for (int j = 0; j < NNODE; j++) {
        double temp = s.shp[3][j] * s.dvol;
        temp *= rho;
        m[j] += temp;
      }
    }
  }

  // Largest omega^2 of K phi = omega^2 M phi with the lumped mass and the committed
  // tangent. Runs once per element per stage (stable step check), so holding the 24x24
  // matrix in one thread is acceptable here.
  FEMORAX_HD static double max_omega_sq(const Params& p, const double (*xe)[NDM],
                                        PointArray<const double> committed) {
    double xl[3][8];
    local_coords(xe, xl);
    double m[NNODE];
    lumped_mass(p, xe, m);
    double A[NDOF][NDOF];
    for (int a = 0; a < NDOF; ++a)
      for (int c = 0; c < NDOF; ++c) A[a][c] = 0.0;
    for (int i = 0; i < NGP; i++) {
      PointShape s;
      point_geometry(xl, i, s);
      double dd[NSTRESS][NSTRESS];
      Mat::committed_tangent(p.mat, committed.point(i), dd);
      for (int j = 0; j < NNODE; j++)
        for (int k = 0; k < NNODE; k++) {
          double stiffJK[3][3];
          point_block_tangent(s, dd, j, k, stiffJK);
          for (int a = 0; a < 3; a++)
            for (int c = 0; c < 3; c++) A[3 * j + a][3 * k + c] += stiffJK[a][c];
        }
    }
    double inv_sqrt_m[NNODE];
    for (int j = 0; j < NNODE; j++) {
      if (!(m[j] > 0.0)) return 0.0;  // massless element: sets no limit
      inv_sqrt_m[j] = 1.0 / std::sqrt(m[j]);
    }
    for (int a = 0; a < NDOF; ++a)
      for (int c = 0; c < NDOF; ++c) A[a][c] *= inv_sqrt_m[a / 3] * inv_sqrt_m[c / 3];
    return max_eigenvalue_symmetric(A);
  }

  FEMORAX_HD static double critical_dt(const Params& p, const Rayleigh& damp,
                                       ExplicitScheme scheme, const double (*xe)[NDM],
                                       PointArray<const double> committed) {
    return stable_dt(scheme, damp, max_omega_sq(p, xe, committed));
  }

 private:
  // Brick::computeB
  FEMORAX_HD static void computeB(int node, const PointShape& s, double B[NSTRESS][3]) {
    for (int r = 0; r < NSTRESS; ++r)
      for (int c = 0; c < 3; ++c) B[r][c] = 0.0;
    B[0][0] = s.shp[0][node];
    B[1][1] = s.shp[1][node];
    B[2][2] = s.shp[2][node];
    B[3][0] = s.shp[1][node];
    B[3][1] = s.shp[0][node];
    B[4][1] = s.shp[2][node];
    B[4][2] = s.shp[1][node];
    B[5][0] = s.shp[2][node];
    B[5][2] = s.shp[0][node];
  }
};

}  // namespace femorax
