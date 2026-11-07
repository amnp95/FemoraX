#pragma once

// The matrix-free operator of one element (docs/design.md, "The matrix-free operator"):
//
//   y_e = (cK K_t + cC C + cM M)_e v_e,   C = alphaM M + betaK K_t + betaKinit K_0 + betaKcomm K_c
//
// computed point by point without forming any matrix:
//   y_j += B_j^T (D_eff (B v)) dvol + cM' (M v)_j,   at every Gauss point,
//   D_eff = (cK + cC betaK) D_t + cC betaKinit D_0 + cC betaKcomm D_c,  cM' = cM + cC alphaM,
// and its diagonal the same way (B_j^T D_eff B_j dvol per node, plus the mass diagonal).
// Generic over any MatrixFreeElement: the element's pointwise functions do the work, so a new
// element or material gets the operator, its diagonal and the preconditioners for free.
//
// Where D_t (and D_c) come from is the caller's: a functor tangent_at(gp, s, D) (s: the point's
// shape functions) that recomputes
// it, loads it from storage, or copies a constant per material (the launch layer picks;
// physics never knows). Node DOFs are assumed to be 3 per node (the elements so far).

#include "physics/contracts.hpp"
#include "physics/macros.hpp"

namespace femorax {

// K_eff = stiffness K_t + damping C + mass M (an implicit step folds its integrator
// constants in here: Newmark c1, c2, c3).
struct OperatorCoefficients {
  double stiffness = 1.0;
  double damping = 0.0;
  double mass = 0.0;
};

namespace element_operator_detail {

template <class Elem>
struct Factors {
  double cK, c0, cc, cM;
  FEMORAX_HD Factors(const Rayleigh& d, const OperatorCoefficients& k)
      : cK(k.stiffness + k.damping * d.betaK),
        c0(k.damping * d.betaKinit),
        cc(k.damping * d.betaKcomm),
        cM(k.mass + k.damping * d.alphaM) {}
};

// D_eff at one point.
template <class Elem, class TangentAt, class CommittedAt>
FEMORAX_HD void effective_tangent(const Factors<Elem>& f, const double (*D0)[Elem::Material::NSTRAIN], int gp,
                                  const typename Elem::PointShape& s, const TangentAt& tangent_at,
                                  const CommittedAt& committed_at, double (*Deff)[Elem::Material::NSTRAIN]) {
  constexpr int NS = Elem::Material::NSTRAIN;
  double D[NS][NS];
  tangent_at(gp, s, D);
  double Dc[NS][NS];
  if (f.cc != 0.0) committed_at(gp, Dc);
  for (int a = 0; a < NS; ++a)
    for (int c = 0; c < NS; ++c) {
      double v = f.cK * D[a][c];
      if (f.c0 != 0.0) v += f.c0 * D0[a][c];
      if (f.cc != 0.0) v += f.cc * Dc[a][c];
      Deff[a][c] = v;
    }
}

}  // namespace element_operator_detail

// y_e = K_eff,e v_e (NDOF values each). geometry.shape(gp, s) gives each point's shape
// functions (recomputed or cached).
template <class Elem, class Geometry, class TangentAt, class CommittedAt>
FEMORAX_HD void element_operator_apply(const typename Elem::Params& p, const Rayleigh& damp,
                                       const OperatorCoefficients& k, const Geometry& geometry,
                                       const TangentAt& tangent_at, const CommittedAt& committed_at,
                                       const double* ve, double* ye) {
  static_assert(MatrixFreeElement<Elem>);
  using Mat = typename Elem::Material;
  constexpr int NS = Mat::NSTRAIN, NNODE = Elem::NNODE, NDOF = Elem::NDOF;
  const element_operator_detail::Factors<Elem> f(damp, k);
  for (int a = 0; a < NDOF; ++a) ye[a] = 0.0;
  double D0[NS][NS];
  if (f.c0 != 0.0) Mat::initial_tangent(p.mat, D0);
  for (int gp = 0; gp < Elem::NGP; ++gp) {
    typename Elem::PointShape s;
    geometry.shape(gp, s);
    double eps[NS];
    Elem::point_strain(s, ve, eps);
    double Deff[NS][NS];
    element_operator_detail::effective_tangent<Elem>(f, D0, gp, s, tangent_at, committed_at, Deff);
    double sig[NS];
    for (int a = 0; a < NS; ++a) {
      double sum = 0.0;
      for (int c = 0; c < NS; ++c) sum += Deff[a][c] * eps[c];
      sig[a] = sum;
    }
    for (int j = 0; j < NNODE; ++j) {
      double fj[3];
      Elem::point_node_force(s, sig, j, fj);
      for (int q = 0; q < 3; ++q) ye[3 * j + q] += fj[q];
      if (f.cM != 0.0) {
        double mj[3];
        Elem::point_mass_apply(p, s, ve, j, mj);
        for (int q = 0; q < 3; ++q) ye[3 * j + q] += f.cM * mj[q];
      }
    }
  }
}

// de: the element's contribution to diag(K_eff) in equation space, by local DOF. eq holds
// the element's equations (-1: fixed). Where constraints merge two local DOFs a != b into
// one equation (tied nodes of one element), the assembled diagonal of that equation also
// holds K_ab + K_ba; it is added to the first of the two, so that summing de into the
// equations gives exactly the diagonal of the assembled operator.
template <class Elem, class Geometry, class TangentAt, class CommittedAt>
FEMORAX_HD void element_operator_diagonal(const typename Elem::Params& p, const Rayleigh& damp,
                                          const OperatorCoefficients& k, const Geometry& geometry,
                                          const TangentAt& tangent_at, const CommittedAt& committed_at,
                                          const int* eq, double* de) {
  static_assert(MatrixFreeElement<Elem>);
  using Mat = typename Elem::Material;
  constexpr int NS = Mat::NSTRAIN, NNODE = Elem::NNODE, NDOF = Elem::NDOF;
  const element_operator_detail::Factors<Elem> f(damp, k);
  for (int a = 0; a < NDOF; ++a) de[a] = 0.0;
  // first[a]: the lowest local DOF with a's equation (a itself if none is lower).
  int first[NDOF];
  bool merged = false;
  for (int a = 0; a < NDOF; ++a) {
    first[a] = a;
    if (eq[a] < 0) continue;
    for (int b = 0; b < a; ++b)
      if (eq[b] == eq[a]) {
        first[a] = first[b];
        merged = true;
        break;
      }
  }
  double D0[NS][NS];
  if (f.c0 != 0.0) Mat::initial_tangent(p.mat, D0);
  for (int gp = 0; gp < Elem::NGP; ++gp) {
    typename Elem::PointShape s;
    geometry.shape(gp, s);
    double Deff[NS][NS];
    element_operator_detail::effective_tangent<Elem>(f, D0, gp, s, tangent_at, committed_at, Deff);
    for (int j = 0; j < NNODE; ++j) {
      double blk[3][3];
      Elem::point_block_tangent(s, Deff, j, j, blk);
      const double m = f.cM != 0.0 ? f.cM * Elem::point_mass_entry(p, s, j, j) : 0.0;
      for (int q = 0; q < 3; ++q) de[3 * j + q] += blk[q][q] + m;
    }
    if (!merged) continue;
    // Cross terms between distinct local DOFs that share an equation.
    for (int J = 0; J < NNODE; ++J)
      for (int K = 0; K < NNODE; ++K) {
        bool any = false;
        for (int a = 0; a < 3 && !any; ++a)
          for (int c = 0; c < 3; ++c) {
            const int ia = 3 * J + a, ic = 3 * K + c;
            if (ia != ic && eq[ia] >= 0 && eq[ia] == eq[ic]) any = true;
          }
        if (!any) continue;
        double blk[3][3];
        Elem::point_block_tangent(s, Deff, J, K, blk);
        const double m = (f.cM != 0.0 && J != K) ? f.cM * Elem::point_mass_entry(p, s, J, K) : 0.0;
        for (int a = 0; a < 3; ++a)
          for (int c = 0; c < 3; ++c) {
            const int ia = 3 * J + a, ic = 3 * K + c;
            if (ia == ic || eq[ia] < 0 || eq[ia] != eq[ic]) continue;
            de[first[ia]] += blk[a][c] + (a == c ? m : 0.0);
          }
      }
  }
}

}  // namespace femorax
