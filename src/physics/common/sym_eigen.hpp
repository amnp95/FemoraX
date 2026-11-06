#pragma once

// Largest eigenvalue of a small symmetric matrix, fixed size, no heap (physics layer).
// Cyclic Jacobi rotations: deterministic and accurate to rounding, including repeated
// eigenvalues (a regular brick's largest frequency is repeated), where power iteration
// would converge slowly. Used once per element per stage for the stable time step, so
// the O(N^3) cost per sweep is acceptable.

#include <cmath>

#include "physics/macros.hpp"

namespace femorax {

// Overwrites A. Returns the largest eigenvalue.
template <int N>
FEMORAX_HD inline double max_eigenvalue_symmetric(double (&A)[N][N]) {
  constexpr int max_sweeps = 60;
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    double off = 0.0, diag = 0.0;
    for (int i = 0; i < N; ++i) {
      diag += A[i][i] * A[i][i];
      for (int j = i + 1; j < N; ++j) off += A[i][j] * A[i][j];
    }
    if (off <= 1e-30 * diag || off == 0.0) break;
    for (int p = 0; p < N - 1; ++p) {
      for (int q = p + 1; q < N; ++q) {
        const double apq = A[p][q];
        if (apq == 0.0) continue;
        // Rotation angle that zeroes A[p][q] (Golub & Van Loan, Algorithm 8.4.1).
        const double theta = (A[q][q] - A[p][p]) / (2.0 * apq);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        for (int k = 0; k < N; ++k) {  // A <- A J
          const double akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < N; ++k) {  // A <- J^T A
          const double apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        A[p][q] = A[q][p] = 0.0;
      }
    }
  }
  double lmax = A[0][0];
  for (int i = 1; i < N; ++i) lmax = A[i][i] > lmax ? A[i][i] : lmax;
  return lmax;
}

}  // namespace femorax
