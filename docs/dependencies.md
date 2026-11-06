# Pinned versions

Every external version FemoraX builds against or ports from. Change a pin only together with
this file.

## Build dependencies

| Dependency | Version | Source | SHA-256 | Used for |
|---|---|---|---|---|
| Kokkos | 5.1.0 | `github.com/kokkos/kokkos/releases/download/5.1.0/kokkos-5.1.0.tar.gz` (FetchContent) | `7bdbdfc88033ed7d940c7940ed8919e1f2b78a9656c69276beb76ad45c41ec4e` | launch layer (`src/exec/`) |
| GoogleTest | 1.17.0 | `github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz` (FetchContent) | `65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c` | tests only |
| HDF5 | system (`libhdf5-dev`) | apt | n/a | model reader, recorders (from step 2) |
| Eigen | to be pinned in milestone 1b | | | CPU direct solver (`SimplicialLDLT`) |

### Kokkos 5.1.0 and CUDA 13

- Kokkos 5.x requires NVCC 12.2 or newer (`KOKKOS_NVCC_MINIMUM` in
  `cmake/kokkos_compiler_id.cmake`).
- CUDA 13 support arrived in 4.7.01 (kokkos#8298, kokkos#8707). A CUDA 13.1 compile fix
  (mdspan) is in 5.0.2 (kokkos#8801), so 5.1.0 has it.
- The development machine has CUDA 13.1 installed (no GPU). Compile check result:
  see "CUDA compile check" below.

### CUDA compile check

2026-11-05, compile only (no GPU on the machine): CUDA 13.1.115, GCC 13.3 as host compiler,
Kokkos `nvcc_wrapper`, `-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_SERIAL=ON
-DKokkos_ARCH_ADA89=ON`. Kokkos, `femorax_exec`, the `femorax` program and the unit tests
all build. The only warnings are `-Wundef` notes in the CUDA and glibc system headers, not in
FemoraX or Kokkos code. Running on a GPU waits for the hardware.

## Reference code (read-only, never built into FemoraX)

| Repository | Commit | Date | Used for |
|---|---|---|---|
| OpenSees (`../OpenSees`) | `955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea` | 2026-10-31 | source of every port; reference drivers in `tests/reference/` |
| Femora (`../Femora`) | `79bd4f67f59b5c27560bc0a2e4b483627aa07df8` | 2026-10-30 | model description and Tcl export behaviour; test-only cross-validation |

Each ported file names its OpenSees source file and this commit in its header comment. A
port made against a different commit records that commit there and in this table.
