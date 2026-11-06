// Smoke tests for the Kokkos CPU build: both host backends are enabled and run kernels,
// and FP64 reductions agree between them.

#include "exec/runtime.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace {

bool has_space(const std::string& name) {
  const auto spaces = femorax::enabled_execution_spaces();
  return std::find(spaces.begin(), spaces.end(), name) != spaces.end();
}

// Sum of 1/(i+1)^2 for i in [0, n), computed by parallel_reduce on ExecSpace.
template <class ExecSpace>
double basel_sum(int n) {
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "basel_sum", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const int i, double& acc) {
        const double k = static_cast<double>(i + 1);
        acc += 1.0 / (k * k);
      },
      sum);
  return sum;
}

}  // namespace

TEST(Runtime, KokkosIsInitialized) { EXPECT_TRUE(Kokkos::is_initialized()); }

TEST(Runtime, ReportsPinnedKokkosVersion) { EXPECT_EQ(femorax::kokkos_version(), "5.1.0"); }

TEST(Runtime, ReportsEveryEnabledBackend) {
  EXPECT_TRUE(has_space("Serial"));  // every FemoraX build enables Serial
#ifdef KOKKOS_ENABLE_OPENMP
  EXPECT_TRUE(has_space("OpenMP"));
#endif
#ifdef KOKKOS_ENABLE_CUDA
  EXPECT_TRUE(has_space("Cuda"));
#endif
}

TEST(Runtime, DefaultSpaceIsOpenMPInCpuBuild) {
#if defined(KOKKOS_ENABLE_OPENMP) && !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  EXPECT_EQ(femorax::default_execution_space(), "OpenMP");
  EXPECT_GE(femorax::default_concurrency(), 1);
#else
  GTEST_SKIP() << "not a CPU-only build";
#endif
}

TEST(Runtime, ReductionAgreesAcrossBackends) {
  constexpr int n = 1'000'000;
  const double serial = basel_sum<Kokkos::Serial>(n);
  // Exact: sum_{k=1}^{inf} 1/k^2 = pi^2/6; the tail beyond n is about 1/n.
  const double pi = 3.14159265358979323846;
  EXPECT_NEAR(serial, pi * pi / 6.0, 1.01 / n);
  // A different backend sums in a different order: agree to rounding, not bitwise.
  const double other = basel_sum<Kokkos::DefaultExecutionSpace>(n);
  EXPECT_NEAR(other, serial, 1e-13);
}

TEST(Runtime, SimdWidthComesFromTheBackend) {
  const int w = femorax::simd_width();
  EXPECT_GE(w, 1);
#if defined(KOKKOS_ENABLE_CUDA)
  EXPECT_EQ(w, 32);
#elif !defined(KOKKOS_ENABLE_HIP) && !defined(KOKKOS_ENABLE_SYCL)
  EXPECT_EQ(w, 1);  // host backends execute one lane per thread
#endif
}

TEST(Runtime, ViewsAreFP64) {
  Kokkos::View<double*> v("v", 4);
  static_assert(sizeof(decltype(v)::value_type) == 8);
  Kokkos::deep_copy(v, 0.1);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  EXPECT_EQ(h(3), 0.1);
}
