#pragma once

// Storage of a material tangent D (NS x NS) at a Gauss point (docs/design.md, "The
// matrix-free operator"): packed upper triangle (NS (NS + 1) / 2 values, 21 for 6 strains)
// when the tangent is declared symmetric, all NS^2 values otherwise. Values are read and
// written through a Strided view (memory contract rule 1), entry k at k * stride.

#include "physics/common/strided.hpp"
#include "physics/macros.hpp"

namespace femorax {

template <int NS, bool SYMMETRIC>
struct TangentStorage {
  static constexpr int SIZE = SYMMETRIC ? NS * (NS + 1) / 2 : NS * NS;

  FEMORAX_HD static void store(const double (*D)[NS], Strided<double> out) {
    int k = 0;
    for (int a = 0; a < NS; ++a)
      for (int c = SYMMETRIC ? a : 0; c < NS; ++c) out[k++] = D[a][c];
  }

  // Symmetric: the lower triangle is the mirror of the stored upper one.
  FEMORAX_HD static void load(Strided<const double> in, double (*D)[NS]) {
    int k = 0;
    for (int a = 0; a < NS; ++a)
      for (int c = SYMMETRIC ? a : 0; c < NS; ++c) {
        D[a][c] = in[k++];
        if (SYMMETRIC) D[c][a] = D[a][c];
      }
  }
};

}  // namespace femorax
