#pragma once

// Kokkos mapping of the physics inner loops (src/physics/common/inner.hpp): vector lanes
// of one team thread. The three levels of the contract:
//   team             a group of elements          (Kokkos league member)
//   team thread      one element or Gauss point   (Kokkos::TeamThreadRange)
//   vector lane      work inside a point          (Kokkos::ThreadVectorRange, here)

#include <Kokkos_Core.hpp>

namespace femorax::exec {

// Context passed to physics code running on one team thread. scratch holds capacity
// values of the thread's scratch memory, used by inner_reduce to add terms in index order.
template <class Member>
struct VectorLanes {
  const Member& member;
  double* scratch;
  int capacity;
};

template <class Member, class F>
KOKKOS_INLINE_FUNCTION void inner_for(const VectorLanes<Member>& c, int n, F&& f) {
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(c.member, n), f);
}

// Terms are computed by the lanes into scratch, then added in index order by one lane and
// broadcast: the same result as the serial loop, bit for bit. Requires n <= capacity.
// Lanes are synchronized at the end of a ThreadVectorRange loop, so the single lane sees
// every term (to be confirmed on CUDA and HIP when the GPU builds run).
template <class Member, class F>
KOKKOS_INLINE_FUNCTION auto inner_reduce(const VectorLanes<Member>& c, int n, F&& f)
    -> decltype(f(0)) {
  using T = decltype(f(0));
  static_assert(sizeof(T) <= sizeof(double), "inner_reduce scratch holds doubles");
  if (n > c.capacity) Kokkos::abort("inner_reduce: n exceeds the thread scratch capacity");
  T* buf = reinterpret_cast<T*>(c.scratch);
  Kokkos::parallel_for(Kokkos::ThreadVectorRange(c.member, n), [&](int i) { buf[i] = f(i); });
  T acc = 0;
  Kokkos::single(
      Kokkos::PerThread(c.member),
      [&](T& a) {
        a = 0;
        for (int i = 0; i < n; ++i) a += buf[i];
      },
      acc);
  return acc;
}

}  // namespace femorax::exec
