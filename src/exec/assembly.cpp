#include "exec/assembly.hpp"

#include <string>

namespace femorax::exec {

SlotGather::SlotGather(std::int64_t num_slots, std::int32_t num_targets,
                       const std::vector<std::pair<std::int32_t, std::int64_t>>& pairs)
    : num_targets_(num_targets),
      slots_("gather_slots", static_cast<std::size_t>(num_slots)) {
  std::vector<std::int64_t> start(static_cast<std::size_t>(num_targets) + 1, 0);
  for (const auto& [t, s] : pairs) ++start[t + 1];
  for (std::int32_t t = 0; t < num_targets; ++t) start[t + 1] += start[t];
  std::vector<std::int64_t> list(pairs.size());
  std::vector<std::int64_t> fill(start.begin(), start.end() - 1);
  for (const auto& [t, s] : pairs) list[fill[t]++] = s;  // stable: keeps the given order

  start_ = Kokkos::View<std::int64_t*, Space>(std::string("gather_start"), start.size());
  list_ = Kokkos::View<std::int64_t*, Space>(std::string("gather_list"), list.size());
  auto hs = Kokkos::create_mirror_view(start_);
  auto hl = Kokkos::create_mirror_view(list_);
  for (std::size_t i = 0; i < start.size(); ++i) hs(i) = start[i];
  for (std::size_t i = 0; i < list.size(); ++i) hl(i) = list[i];
  Kokkos::deep_copy(start_, hs);
  Kokkos::deep_copy(list_, hl);
}

void SlotGather::gather(const Vec& targets) const {
  const auto slots = slots_;
  const auto start = start_, list = list_;
  Kokkos::parallel_for(
      "slot_gather", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int32_t>>(0, num_targets_),
      KOKKOS_LAMBDA(const std::int32_t t) {
        double sum = 0.0;
        for (std::int64_t j = start(t); j < start(t + 1); ++j) sum += slots(list(j));
        targets(t) = sum;
      });
}

std::int64_t SlotGather::bytes() const {
  return static_cast<std::int64_t>(slots_.extent(0) + start_.extent(0) + list_.extent(0)) * 8;
}

}  // namespace femorax::exec
