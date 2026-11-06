#include "exec/device_memory.hpp"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <string>

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
#include "native/device_memory.hpp"
#endif

namespace femorax::exec {

namespace {
std::int64_t host_mem_available() {
  std::ifstream f("/proc/meminfo");
  std::string key;
  std::int64_t kb = 0;
  std::string unit;
  while (f >> key >> kb >> unit)
    if (key == "MemAvailable:") return kb * 1024;
  return -1;
}
}  // namespace

std::int64_t device_free_memory_bytes() {
#if defined(KOKKOS_ENABLE_CUDA)
  if (!std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>)
    return native::cuda_free_memory_bytes();
#elif defined(KOKKOS_ENABLE_HIP)
  if (!std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>)
    return native::hip_free_memory_bytes();
#endif
  return host_mem_available();
}

}  // namespace femorax::exec
