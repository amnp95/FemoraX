#include "kernels/timing.hpp"

#include <Kokkos_Core.hpp>

#include <chrono>

namespace femorax {

void fence() { Kokkos::fence("femorax::fence"); }

double wall_time() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace femorax
