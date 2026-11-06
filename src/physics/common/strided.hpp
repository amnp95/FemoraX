#pragma once

// Strided access to per-point and per-element arrays (docs/design.md, "Memory and
// parallelism contract", rule 1).
//
// Physics code never assumes the values of one point are contiguous. It receives a
// Strided reference and reads value k as ref[k], which is base[k * stride]:
//   stride = number of points  -> structure of arrays (coalesced on GPUs; the default)
//   stride = 1                 -> contiguous per point (array of structs; tests, CPU options)
// The launch layer chooses the stride per backend; physics code is the same for all.

#include <cstddef>

#include "physics/macros.hpp"

namespace femorax {

template <class T>
struct Strided {
  T* base = nullptr;
  int stride = 1;

  // 64-bit offset: k * stride can exceed 2^31 for large state (many points x many values).
  FEMORAX_HD T& operator[](int k) const {
    return base[static_cast<std::ptrdiff_t>(k) * stride];
  }
  FEMORAX_HD operator Strided<const T>() const { return {base, stride}; }
};

// The integration points of one element. Point gp starts at base + gp * point_step;
// within a point, values are stride apart.
//   SoA (value[k * nPoints + elem * NGP + gp]):  base = array + elem * NGP,
//                                                 point_step = 1, stride = nPoints
//   contiguous (value[(elem * NGP + gp) * NSTATE + k]): point_step = NSTATE, stride = 1
template <class T>
struct PointArray {
  T* base = nullptr;
  int stride = 1;
  int point_step = 0;

  FEMORAX_HD Strided<T> point(int gp) const {
    return {base + static_cast<std::ptrdiff_t>(gp) * point_step, stride};
  }
  FEMORAX_HD operator PointArray<const T>() const { return {base, stride, point_step}; }
};

}  // namespace femorax
