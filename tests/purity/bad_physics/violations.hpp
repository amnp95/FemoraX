// Fixture for the physics purity scanner: every marked line must be reported, every
// unmarked line must not. The expected count lives in tests/CMakeLists.txt.
// Mentions inside comments are ignored: Kokkos, new Foo, throw, virtual, std::vector.
#pragma once

#include <cmath>                          // allowed
#include <vector>                         // VIOLATION: include not allowed
#include "physics/macros.hpp"             // allowed
#include "exec/runtime.hpp"               // VIOLATION: include outside src/physics
#include <Kokkos_Core.hpp>                // VIOLATION x2: include not allowed, Kokkos

/* block comment: delete p; malloc(4); try { } */

namespace femorax {

struct Bad {
  static constexpr int NSTATE = 2;        // allowed
  static int counter;                     // VIOLATION: static mutable data
  mutable double cache;                   // VIOLATION: mutable state
  virtual void f();                       // VIOLATION: virtual
  FEMORAX_HD static void ok(const double a[3], double b[3]);  // allowed
};

inline void bad(double* p) {
  double* q = new double[3];              // VIOLATION: new
  delete[] q;                             // VIOLATION: delete
  void* r = malloc(8);                    // VIOLATION: malloc
  if (!p) throw 1;                        // VIOLATION: exceptions
  std::vector<double> v;                  // VIOLATION: heap-allocating std type
  const char* s = "new throw virtual";    // allowed: inside a string literal
  (void)r; (void)s; (void)v;
}

__host__ __device__ inline void g() {}    // VIOLATION: use FEMORAX_HD
inline void h() { cudaMalloc(nullptr, 0); }  // VIOLATION: native CUDA API

}  // namespace femorax
