#include "kernels/stream.hpp"

#include <Kokkos_Core.hpp>

namespace femorax {

namespace {
using Space = Kokkos::DefaultExecutionSpace;
using Policy = Kokkos::RangePolicy<Space, Kokkos::IndexType<int>>;
using Array = Kokkos::View<double*, Space::memory_space>;

// First touch in parallel, so pages land near the threads that use them. A free function:
// nvcc does not allow device lambdas inside members of a private nested class.
void first_touch(Array a, Array b, Array c, int n) {
  Kokkos::parallel_for(
      "stream_init", Policy(0, n), KOKKOS_LAMBDA(const int i) {
        a(i) = 1.0;
        b(i) = 2.0;
        c(i) = 0.5;
      });
  Kokkos::fence();
}
}  // namespace

struct StreamKernels::Impl {
  int n;
  Array a, b, c;
  explicit Impl(int n_)
      : n(n_),
        a(Kokkos::view_alloc(Kokkos::WithoutInitializing, "stream_a"), n_),
        b(Kokkos::view_alloc(Kokkos::WithoutInitializing, "stream_b"), n_),
        c(Kokkos::view_alloc(Kokkos::WithoutInitializing, "stream_c"), n_) {
    first_touch(a, b, c, n);
  }
};

StreamKernels::StreamKernels(int n) : impl_(std::make_unique<Impl>(n)) {}
StreamKernels::~StreamKernels() = default;

int StreamKernels::size() const { return impl_->n; }

void StreamKernels::copy() {
  auto a = impl_->a, c = impl_->c;
  Kokkos::parallel_for(
      "stream_copy", Policy(0, impl_->n), KOKKOS_LAMBDA(const int i) { c(i) = a(i); });
}

void StreamKernels::triad(double s) {
  auto a = impl_->a, b = impl_->b, c = impl_->c;
  Kokkos::parallel_for(
      "stream_triad", Policy(0, impl_->n), KOKKOS_LAMBDA(const int i) { a(i) = b(i) + s * c(i); });
}

double StreamKernels::checksum() const {
  auto a = impl_->a;
  double sum = 0.0;
  Kokkos::parallel_reduce(
      "stream_checksum", Policy(0, impl_->n),
      KOKKOS_LAMBDA(const int i, double& acc) { acc += a(i); }, sum);
  return sum;
}

}  // namespace femorax
