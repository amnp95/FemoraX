#include "native/device_memory.hpp"

#include <Kokkos_Core.hpp>
#include <sycl/sycl.hpp>

namespace femorax::native {

// The device of Kokkos' default SYCL instance. Intel GPUs report free memory through the
// ext_intel_free_memory aspect, which the Level Zero runtime offers only with
// ZES_ENABLE_SYSMAN=1 in the environment; without it the estimate check reports "unknown".
std::int64_t sycl_free_memory_bytes() {
  try {
    const sycl::device dev = Kokkos::SYCL().sycl_queue().get_device();
    if (!dev.has(sycl::aspect::ext_intel_free_memory)) return -1;
    return static_cast<std::int64_t>(dev.get_info<sycl::ext::intel::info::device::free_memory>());
  } catch (const sycl::exception&) {
    return -1;
  }
}

}  // namespace femorax::native
