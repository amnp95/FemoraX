#pragma once

// Assembly machinery shared by every kernel that adds element results into global arrays:
// the explicit internal force now, the implicit residual and stiffness (milestone 1b) next.
//
//   Out::Atomic  atomic add into the target      (fast, order varies between runs)
//   Out::Plain   plain add                       (colored launches: no two writers)
//   Out::Slots   write to the element's own slot (then SlotGather adds each target's slots
//                in a fixed precomputed order: deterministic, keeps the element order)
//
// For a vector, targets are equations. For a matrix, targets will be nonzero positions; the
// gather is the same.

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace femorax::exec {

using Space = Kokkos::DefaultExecutionSpace;
using Vec = Kokkos::View<double*, Space>;

enum class Out : int { Plain = 0, Atomic = 1, Slots = 2 };

// One element value out of a kernel: into target q (q < 0: dropped), or into the slot.
KOKKOS_INLINE_FUNCTION void emit(Out out, const Vec& target, const Vec& slots,
                                 std::int64_t slot, std::int32_t q, double value) {
  if (out == Out::Slots) {
    slots(slot) = value;
  } else if (q >= 0) {
    if (out == Out::Atomic) Kokkos::atomic_add(&target(q), value);
    else target(q) += value;
  }
}

// targets(t) = sum of t's slots, in the order the (target, slot) pairs were given.
class SlotGather {
 public:
  SlotGather() = default;
  // pairs: (target, slot) in summation order; slots are indices into a slot array of
  // num_slots entries. Host work, once per stage.
  SlotGather(std::int64_t num_slots, std::int32_t num_targets,
             const std::vector<std::pair<std::int32_t, std::int64_t>>& pairs);

  void gather(const Vec& targets) const;  // overwrites every target
  const Vec& slots() const { return slots_; }
  std::int64_t bytes() const;

 private:
  std::int32_t num_targets_ = 0;
  Vec slots_;
  Kokkos::View<std::int64_t*, Space> start_, list_;
};

}  // namespace femorax::exec
