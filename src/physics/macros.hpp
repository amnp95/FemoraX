#pragma once

// FEMORAX_HD marks a physics function as callable from host and device code.
// It expands to the compiler's host/device attributes directly, so the physics layer
// never includes Kokkos (CLAUDE.md rule 2).
#if defined(__CUDACC__) || defined(__HIPCC__)
#define FEMORAX_HD __host__ __device__
#else
#define FEMORAX_HD
#endif
