#include "native/device_memory.hpp"

#include <cuda_runtime.h>

namespace femorax::native {

std::int64_t cuda_free_memory_bytes() {
  std::size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return -1;
  return static_cast<std::int64_t>(free_b);
}

}  // namespace femorax::native
