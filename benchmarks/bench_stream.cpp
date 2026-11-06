// STREAM copy and triad: the bandwidth the hardware delivers, the reference for every
// bandwidth-bound kernel.

#include "kernels/stream.hpp"
#include "harness.hpp"

#include <memory>

namespace femorax::bench {

void register_stream(Registry& registry, const Options& options) {
  // 2^25 doubles = 256 MiB per array, far beyond any cache; 2^20 for the quick run.
  const int n = options.quick ? (1 << 20) : (1 << 25);
  auto k = std::make_shared<StreamKernels>(n);

  registry.add({.name = "stream/copy",
                .unit = "call",
                .bytes = std::int64_t{16} * n,
                .flops = 0.0,
                .items = n,
                .run = [k] { k->copy(); }});
  registry.add({.name = "stream/triad",
                .unit = "call",
                .bytes = std::int64_t{24} * n,
                .flops = 2.0 * n,
                .items = n,
                .run = [k] { k->triad(3.0); }});
}

}  // namespace femorax::bench

#include "kernels/peak.hpp"

namespace femorax::bench {

// FP64 arithmetic reference: 8 independent multiply-add chains per work item.
void register_peak(Registry& registry, const Options& options) {
  const int n = options.quick ? (1 << 14) : (1 << 20);
  const int iters = options.quick ? 64 : 1024;
  auto k = std::make_shared<PeakFp64>(n);
  registry.add({.name = "peak/fp64_fma",
                .unit = "call",
                .bytes = std::int64_t{8} * n,
                .flops = 16.0 * n * iters,
                .items = n,
                .run = [k, iters] { k->run(iters); }});
}

}  // namespace femorax::bench
