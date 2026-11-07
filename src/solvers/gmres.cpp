#include "solvers/gmres.hpp"

#include <algorithm>
#include <cmath>

namespace femorax::solvers {

namespace {

Vector column(const Block& B, int c) { return Kokkos::subview(B, Kokkos::ALL, c); }

}  // namespace

Gmres::Gmres(const SolveOptions& options, int lanes, int restart)
    : options_(options), lanes_(lanes), m_(std::max(1, restart)) {}

void Gmres::setup(const LinearOperator& a, const Preconditioner* p) {
  a_ = &a;
  p_ = p;
  const auto n = a.info().rows;
  if (static_cast<std::int64_t>(w_.extent(0)) != n) {
    const auto un = static_cast<std::size_t>(n);
    V_ = Block("gmres_V", un, static_cast<std::size_t>(m_ + 1));
    Z_ = Block("gmres_Z", un, static_cast<std::size_t>(m_));
    w_ = Vector("gmres_w", un);
    r_ = Vector("gmres_r", un);
    ops_ = std::make_unique<VectorOps>(n, lanes_);
    block_ = std::make_unique<BlockOps>(n, m_ + 1, lanes_);
  }
}

SolveResult Gmres::solve(const ConstVector& b, const Vector& x) {
  SolveResult res;
  const VectorOps& ops = *ops_;
  const BlockOps& blk = *block_;
  res.rhs_norm = ops.norm2(b);
  const double tol = std::max(options_.relative_tolerance * res.rhs_norm, options_.absolute_tolerance);
  // Hessenberg matrix (column-major, (m+1) x m), Givens rotations, right-hand side.
  std::vector<double> H(static_cast<std::size_t>((m_ + 1) * m_)), cs(static_cast<std::size_t>(m_)),
      sn(static_cast<std::size_t>(m_)), g(static_cast<std::size_t>(m_ + 1)), h, h2, y;
  auto Hij = [&](int i, int j) -> double& { return H[static_cast<std::size_t>(j * (m_ + 1) + i)]; };
  bool first = true;
  while (true) {
    // r = b - A x (the true residual at every restart).
    a_->apply(x, w_);
    VectorOps::copy(b, r_);
    VectorOps::axpy(-1.0, w_, r_);
    const double beta = ops.norm2(r_);
    if (first) res.initial_residual = beta;
    first = false;
    res.final_residual = beta;
    if (beta <= tol) {
      res.converged = true;
      return res;
    }
    if (res.iterations >= options_.max_iterations) return res;
    const Vector v0 = column(V_, 0);
    VectorOps::fill(v0, 0.0);
    VectorOps::axpy(1.0 / beta, r_, v0);
    std::fill(g.begin(), g.end(), 0.0);
    g[0] = beta;
    int j = 0;
    for (; j < m_ && res.iterations < options_.max_iterations; ++j) {
      const Vector vj = column(V_, j), zj = column(Z_, j);
      if (p_) p_->apply(vj, zj);
      else VectorOps::copy(vj, zj);
      a_->apply(zj, w_);
      // CGS2: h = V^T w, w -= V h, then once more.
      blk.dots(V_, j + 1, w_, h);
      blk.combine(V_, j + 1, h, -1.0, w_);
      blk.dots(V_, j + 1, w_, h2);
      blk.combine(V_, j + 1, h2, -1.0, w_);
      for (int i = 0; i <= j; ++i) Hij(i, j) = h[static_cast<std::size_t>(i)] + h2[static_cast<std::size_t>(i)];
      const double hn = ops.norm2(w_);
      Hij(j + 1, j) = hn;
      if (hn > 0.0) {
        const Vector vn = column(V_, j + 1);
        VectorOps::fill(vn, 0.0);
        VectorOps::axpy(1.0 / hn, w_, vn);
      }
      // Apply the previous rotations to column j, then a new one to zero H(j+1, j).
      for (int i = 0; i < j; ++i) {
        const double a = Hij(i, j), c = Hij(i + 1, j);
        Hij(i, j) = cs[i] * a + sn[i] * c;
        Hij(i + 1, j) = -sn[i] * a + cs[i] * c;
      }
      const double a = Hij(j, j), c = Hij(j + 1, j);
      const double d = std::hypot(a, c);
      cs[j] = d > 0.0 ? a / d : 1.0;
      sn[j] = d > 0.0 ? c / d : 0.0;
      Hij(j, j) = d;
      Hij(j + 1, j) = 0.0;
      g[j + 1] = -sn[j] * g[j];
      g[j] = cs[j] * g[j];
      ++res.iterations;
      res.final_residual = std::abs(g[j + 1]);
      if (res.final_residual <= tol || hn == 0.0) {
        ++j;
        break;
      }
    }
    // y = H(0:j, 0:j)^-1 g(0:j); x += Z(:, 0:j) y.
    y.assign(static_cast<std::size_t>(j), 0.0);
    for (int i = j - 1; i >= 0; --i) {
      double s = g[static_cast<std::size_t>(i)];
      for (int k = i + 1; k < j; ++k) s -= Hij(i, k) * y[static_cast<std::size_t>(k)];
      y[static_cast<std::size_t>(i)] = s / Hij(i, i);
    }
    blk.combine(Z_, j, y, 1.0, x);
    if (res.final_residual <= tol) {
      // Confirm with the true residual (cheap: one apply) and report it.
      a_->apply(x, w_);
      VectorOps::copy(b, r_);
      VectorOps::axpy(-1.0, w_, r_);
      res.final_residual = ops.norm2(r_);
      if (res.final_residual <= tol) {
        res.converged = true;
        return res;
      }
    }
    if (res.iterations >= options_.max_iterations) return res;
  }
}

}  // namespace femorax::solvers
