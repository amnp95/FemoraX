#pragma once

// Vendor-runtime device queries that Kokkos does not expose (CLAUDE.md rule 3: src/native/
// also holds such queries; they contain no kernels).

#include <cstdint>

namespace femorax::native {

// cudaMemGetInfo's free bytes on the current device; -1 on error.
std::int64_t cuda_free_memory_bytes();

}  // namespace femorax::native
