#include "exec/runtime.hpp"

#include <Kokkos_Core.hpp>

#include <string>

namespace femorax {

struct Runtime::Impl {
  Kokkos::ScopeGuard guard;
  Impl(int& argc, char* argv[]) : guard(argc, argv) {}
};

Runtime::Runtime(int& argc, char* argv[]) : impl_(std::make_unique<Impl>(argc, argv)) {}

Runtime::~Runtime() = default;

std::vector<std::string> enabled_execution_spaces() {
  std::vector<std::string> spaces;
#ifdef KOKKOS_ENABLE_SERIAL
  spaces.emplace_back(Kokkos::Serial::name());
#endif
#ifdef KOKKOS_ENABLE_OPENMP
  spaces.emplace_back(Kokkos::OpenMP::name());
#endif
#ifdef KOKKOS_ENABLE_CUDA
  spaces.emplace_back(Kokkos::Cuda::name());
#endif
#ifdef KOKKOS_ENABLE_HIP
  spaces.emplace_back(Kokkos::HIP::name());
#endif
  return spaces;
}

std::string default_execution_space() { return Kokkos::DefaultExecutionSpace::name(); }

int default_concurrency() { return Kokkos::DefaultExecutionSpace().concurrency(); }

std::string kokkos_version() {
  constexpr int v = KOKKOS_VERSION;
  return std::to_string(v / 10000) + "." + std::to_string((v / 100) % 100) + "." +
         std::to_string(v % 100);
}

}  // namespace femorax
