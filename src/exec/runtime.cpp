#include "exec/runtime.hpp"

#include <Kokkos_Core.hpp>

#include <string>
#include <type_traits>

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

int simd_width() {
  // Kokkos has no public warp-size query (TeamPolicy::vector_length_max() is an upper bound
  // on vector length, 1024 on OpenMP). Its per-backend traits are internal (Impl), so this
  // is the one place that depends on them; recheck when Kokkos is upgraded.
#if defined(KOKKOS_ENABLE_CUDA)
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::Cuda>)
    return Kokkos::Impl::CudaTraits::WarpSize;
#endif
#if defined(KOKKOS_ENABLE_HIP)
  if constexpr (std::is_same_v<Kokkos::DefaultExecutionSpace, Kokkos::HIP>)
    return Kokkos::Impl::HIPTraits::WarpSize;  // 64 on CDNA (MI series), 32 on RDNA
#endif
  return 1;  // host backends: one lane per thread
}

std::string kokkos_version() {
  constexpr int v = KOKKOS_VERSION;
  return std::to_string(v / 10000) + "." + std::to_string((v / 100) % 100) + "." +
         std::to_string(v % 100);
}

}  // namespace femorax
