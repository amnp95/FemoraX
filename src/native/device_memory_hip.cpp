#include "native/device_memory.hpp"

#include <hip/hip_runtime.h>

namespace femorax::native {

std::int64_t hip_free_memory_bytes() {
  std::size_t free_b = 0, total_b = 0;
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return -1;
  return static_cast<std::int64_t>(free_b);
}

}  // namespace femorax::native
