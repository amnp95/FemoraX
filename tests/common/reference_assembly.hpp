#pragma once

// Test-only reference assembly (docs/design.md, "Linear solvers": assembled matrices exist
// only in tests). Global K_t, K_0, K_c, M and the Rayleigh C are assembled on the host from
// the element matrices of every registered element-material pair, into the sparsity
// pattern of compile/sparsity.hpp, in batch and element order. They are the reference for
// the matrix-free operator and, through Eigen, the reference solutions for the solvers.
//
//   K_t  tangent at the given displacement (Element::compute<true>, OpenSees
//        getTangentStiff; bit-identical to OpenSees for stdBrick)
//   K_0  B^T D_0 B (Material::initial_tangent), K_c  B^T D_c B (committed_tangent)
//   M    Element::mass (consistent or lumped per element) plus nodal masses
//   C    per element alphaM M + betaK K_t + betaKinit K_0 + betaKcomm K_c, with the
//        element's damping region

#include <Eigen/Sparse>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "compile/batches.hpp"
#include "compile/registry.hpp"
#include "compile/sparsity.hpp"

namespace femorax::testing {

struct ReferenceSystem {
  compile::SparsityPattern pattern;
  std::vector<double> K, K0, Kc, M, C;  // values in pattern order
  // M r_d for the rigid unit translation r_d of every element DOF in direction d, fixed DOFs
  // included (the UniformExcitation load is -ag M r_d), plus the nodal masses.
  std::vector<double> Mr[3];

  // c1 K + c2 C + c3 M, the effective stiffness of an implicit step.
  std::vector<double> effective(double c1, double c2, double c3) const {
    std::vector<double> v(K.size());
    for (std::size_t k = 0; k < v.size(); ++k) v[k] = c1 * K[k] + c2 * C[k] + c3 * M[k];
    return v;
  }

  // y = A x for values A in pattern order.
  std::vector<double> multiply(const std::vector<double>& a, const std::vector<double>& x) const {
    std::vector<double> y(static_cast<std::size_t>(pattern.rows), 0.0);
    for (std::int32_t i = 0; i < pattern.rows; ++i)
      for (std::int64_t k = pattern.row_offsets[i]; k < pattern.row_offsets[i + 1]; ++k)
        y[i] += a[k] * x[pattern.columns[k]];
    return y;
  }

  Eigen::SparseMatrix<double> to_eigen(const std::vector<double>& a) const {
    std::vector<Eigen::Triplet<double>> t;
    t.reserve(a.size());
    for (std::int32_t i = 0; i < pattern.rows; ++i)
      for (std::int64_t k = pattern.row_offsets[i]; k < pattern.row_offsets[i + 1]; ++k)
        t.emplace_back(i, pattern.columns[k], a[k]);
    Eigen::SparseMatrix<double> m(pattern.rows, pattern.rows);
    m.setFromTriplets(t.begin(), t.end());
    return m;
  }
};

namespace detail {

template <class Elem>
void assemble_batch(const compile::Compiled& c, const compile::Batch& b, const std::vector<double>& u,
                    ReferenceSystem& r) {
  using Mat = typename Elem::Material;
  constexpr int NDOF = Elem::NDOF, NNODE = Elem::NNODE, NS = Elem::NSTRESS;
  const std::int32_t N = c.dofs.num_nodes();
  std::vector<double> committed(Elem::NSTATE > 0 ? Elem::NSTATE : 1, 0.0), trial = committed;
  for (std::int32_t e = 0; e < b.count; ++e) {
    const auto& ps = b.params[static_cast<std::size_t>(b.param[e])];
    const auto p = ps.params<Elem>();
    double xe[NNODE][Elem::NDM];
    for (int j = 0; j < NNODE; ++j) {
      const std::int32_t n = b.node[static_cast<std::size_t>(j) * b.padded + e];
      for (int d = 0; d < Elem::NDM; ++d) xe[j][d] = c.coords[static_cast<std::size_t>(d) * N + n];
    }
    std::int32_t eq[NDOF];
    double ue[NDOF], ve[NDOF] = {};
    for (int a = 0; a < NDOF; ++a) {
      eq[a] = b.eq[static_cast<std::size_t>(a) * b.padded + e];
      ue[a] = eq[a] >= 0 ? u[eq[a]] : 0.0;
    }
    const PointArray<double> tr{trial.data(), 1, Mat::NSTATE};
    const PointArray<const double> co{committed.data(), 1, Mat::NSTATE};
    static thread_local double Kt[NDOF][NDOF], K0[NDOF][NDOF], Kc[NDOF][NDOF], Me[NDOF * NDOF];
    double fe[NDOF];
    Elem::template compute<true>(p, Rayleigh{}, xe, ue, ve, co, tr, fe, Kt);
    // K_0 and K_c from the pointwise functions, with the materials' D_0 and D_c.
    double xl[3][8];
    Elem::local_coords(xe, xl);
    double D0[NS][NS], Dc[NS][NS];
    Mat::initial_tangent(p.mat, D0);
    for (int a = 0; a < NDOF; ++a)
      for (int k = 0; k < NDOF; ++k) K0[a][k] = Kc[a][k] = 0.0;
    for (int gp = 0; gp < Elem::NGP; ++gp) {
      typename Elem::PointShape s;
      Elem::point_geometry(xl, gp, s);
      Mat::committed_tangent(p.mat, co.point(gp), Dc);
      for (int J = 0; J < NNODE; ++J)
        for (int K = 0; K < NNODE; ++K) {
          double b0[3][3], bc[3][3];
          Elem::point_block_tangent(s, D0, J, K, b0);
          Elem::point_block_tangent(s, Dc, J, K, bc);
          for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y) {
              K0[3 * J + x][3 * K + y] += b0[x][y];
              Kc[3 * J + x][3 * K + y] += bc[x][y];
            }
        }
    }
    Elem::mass(p, xe, Me);
    const Rayleigh& d = ps.damping;
    for (int a = 0; a < NDOF; ++a) {
      if (eq[a] < 0) continue;
      for (int k = 0; k < NDOF; ++k) r.Mr[k % 3][static_cast<std::size_t>(eq[a])] += Me[a * NDOF + k];
    }
    for (int a = 0; a < NDOF; ++a) {
      if (eq[a] < 0) continue;
      for (int k = 0; k < NDOF; ++k) {
        if (eq[k] < 0) continue;
        const std::int64_t pos = r.pattern.find(eq[a], eq[k]);
        const double m = Me[a * NDOF + k];
        r.K[pos] += Kt[a][k];
        r.K0[pos] += K0[a][k];
        r.Kc[pos] += Kc[a][k];
        r.M[pos] += m;
        r.C[pos] += d.alphaM * m + d.betaK * Kt[a][k] + d.betaKinit * K0[a][k] + d.betaKcomm * Kc[a][k];
      }
    }
  }
}

}  // namespace detail

// u: the displacement (equation numbering) at which K_t is formed; empty = zero.
inline ReferenceSystem assemble_reference(const io::Model& m, const compile::Compiled& c,
                                          std::vector<double> u = {}) {
  ReferenceSystem r;
  r.pattern = compile::build_pattern(c);
  const auto nnz = static_cast<std::size_t>(r.pattern.nnz());
  r.K.assign(nnz, 0.0);
  r.K0 = r.Kc = r.M = r.C = r.K;
  for (auto& v : r.Mr) v.assign(static_cast<std::size_t>(c.dofs.num_equations), 0.0);
  if (u.empty()) u.assign(static_cast<std::size_t>(c.dofs.num_equations), 0.0);
  for (const auto& b : c.batches) {
    registry::for_each_pair([&]<class E, class M>(int index) {
      if (index == b.pair)
        detail::assemble_batch<typename E::template Element<typename M::Material>>(c, b, u, r);
    });
  }
  // Nodal masses, as the compile step adds them to the lumped mass.
  std::unordered_map<std::int32_t, std::int32_t> internal;
  for (std::int32_t i = 0; i < c.dofs.num_nodes(); ++i) internal[c.dofs.node_tag[i]] = i;
  for (std::size_t k = 0; k < m.mass.node.size(); ++k) {
    const std::int32_t q = c.dofs.equation(internal.at(m.mass.node[k]), m.mass.dof[k] - 1);
    if (q >= 0) {
      r.M[static_cast<std::size_t>(r.pattern.find(q, q))] += m.mass.value[k];
      r.Mr[m.mass.dof[k] - 1][static_cast<std::size_t>(q)] += m.mass.value[k];
    }
  }
  return r;
}

}  // namespace femorax::testing
