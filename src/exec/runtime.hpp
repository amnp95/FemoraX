#pragma once

#include <memory>
#include <string>
#include <vector>

namespace femorax {

// Owns Kokkos initialization for the lifetime of the program (wraps Kokkos::ScopeGuard
// so callers outside src/exec/ never include Kokkos).
class Runtime {
 public:
  Runtime(int& argc, char* argv[]);
  ~Runtime();
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Names of the execution spaces enabled in this build, e.g. {"Serial", "OpenMP"}.
std::vector<std::string> enabled_execution_spaces();

// Name of Kokkos' default execution space (where kernels run unless told otherwise).
std::string default_execution_space();

// Number of threads (host) or concurrency (device) of the default execution space.
int default_concurrency();

// Number of threads that execute in lockstep on the default execution space: the warp
// (NVIDIA, 32) or wavefront (AMD, 64) width, 1 on CPU backends. Team sizes and batch
// padding derive from this at run time; nothing assumes 32 (memory contract, rule 4).
int simd_width();

// Kokkos version as "major.minor.patch".
std::string kokkos_version();

}  // namespace femorax
