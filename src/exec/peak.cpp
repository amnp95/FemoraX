#include "exec/peak.hpp"

#include <Kokkos_Core.hpp>

namespace femorax::exec {

namespace {
using Space = Kokkos::DefaultExecutionSpace;
using Policy = Kokkos::RangePolicy<Space, Kokkos::IndexType<int>>;
using Array = Kokkos::View<double*, Space>;

void fma_chains(const Array& out, int n, int iters) {
  Kokkos::parallel_for(
      "peak_fp64_fma", Policy(0, n), KOKKOS_LAMBDA(const int i) {
        // Values chosen so the chains neither overflow nor flush to zero.
        const double b = 0.999999, c = 1.0e-7 * (i % 7 + 1);
        double a0 = 1.0, a1 = 1.1, a2 = 1.2, a3 = 1.3, a4 = 1.4, a5 = 1.5, a6 = 1.6, a7 = 1.7;
        for (int k = 0; k < iters; ++k) {
          a0 = a0 * b + c;
          a1 = a1 * b + c;
          a2 = a2 * b + c;
          a3 = a3 * b + c;
          a4 = a4 * b + c;
          a5 = a5 * b + c;
          a6 = a6 * b + c;
          a7 = a7 * b + c;
        }
        out(i) = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
      });
}
}  // namespace

struct PeakFp64::Impl {
  int n;
  Array out;
  explicit Impl(int n_) : n(n_), out("peak_out", n_) {}
};

PeakFp64::PeakFp64(int n) : impl_(std::make_unique<Impl>(n)) {}
PeakFp64::~PeakFp64() = default;

void PeakFp64::run(int iters) { fma_chains(impl_->out, impl_->n, iters); }
int PeakFp64::size() const { return impl_->n; }

double PeakFp64::checksum() const {
  auto out = impl_->out;
  double s = 0;
  Kokkos::parallel_reduce(
      "peak_checksum", Policy(0, impl_->n), KOKKOS_LAMBDA(const int i, double& acc) { acc += out(i); },
      s);
  return s;
}

}  // namespace femorax::exec
