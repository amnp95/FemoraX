#pragma once

// Vendor-runtime device queries that Kokkos does not expose (CLAUDE.md rule 3: src/native/
// also holds such queries; they contain no kernels).

#include <cstdint>

namespace femorax::native {

// Free bytes on the current device from the vendor runtime (cudaMemGetInfo,
// hipMemGetInfo); -1 on error. Each is built only with its backend.
std::int64_t cuda_free_memory_bytes();
std::int64_t hip_free_memory_bytes();

}  // namespace femorax::native
