#pragma once

// Small dense linear algebra for element-level setup (fixed sizes, no allocation):
// the inverse of an N x N matrix by LU factorization with partial pivoting, as LAPACK
// dgetrf/dgetri do it (OpenSees Matrix::Invert calls them). The operation order differs
// from LAPACK's blocked code, so results agree to rounding, not bit for bit.

#include <cmath>

#include "physics/macros.hpp"

namespace femorax {

// Ainv = A^-1. Returns false (Ainv undefined) if a pivot is exactly zero.
template <int N>
FEMORAX_HD inline bool invert_small(const double (&A)[N][N], double (&Ainv)[N][N]) {
  double LU[N][N];
  int piv[N];
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) LU[i][j] = A[i][j];
  for (int k = 0; k < N; ++k) {
    int p = k;
    double best = std::fabs(LU[k][k]);
    for (int i = k + 1; i < N; ++i)
      if (std::fabs(LU[i][k]) > best) {
        best = std::fabs(LU[i][k]);
        p = i;
      }
    piv[k] = p;
    if (best == 0.0) return false;
    if (p != k)
      for (int j = 0; j < N; ++j) {
        const double t = LU[k][j];
        LU[k][j] = LU[p][j];
        LU[p][j] = t;
      }
    const double r = 1.0 / LU[k][k];
    for (int i = k + 1; i < N; ++i) {
      LU[i][k] *= r;
      const double l = LU[i][k];
      for (int j = k + 1; j < N; ++j) LU[i][j] -= l * LU[k][j];
    }
  }
  // Solve LU X = P I column by column.
  for (int c = 0; c < N; ++c) {
    double x[N];
    for (int i = 0; i < N; ++i) x[i] = (i == c) ? 1.0 : 0.0;
    for (int k = 0; k < N; ++k)
      if (piv[k] != k) {
        const double t = x[k];
        x[k] = x[piv[k]];
        x[piv[k]] = t;
      }
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < i; ++j) x[i] -= LU[i][j] * x[j];
    for (int i = N - 1; i >= 0; --i) {
      for (int j = i + 1; j < N; ++j) x[i] -= LU[i][j] * x[j];
      x[i] /= LU[i][i];
    }
    for (int i = 0; i < N; ++i) Ainv[i][c] = x[i];
  }
  return true;
}

}  // namespace femorax
