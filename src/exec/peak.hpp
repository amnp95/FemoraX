#pragma once

#include <memory>

namespace femorax::exec {

// FP64 arithmetic peak of the default execution space, as reached by code built with this
// build's flags: every work item runs 8 independent multiply-add chains, so throughput is
// limited by the FP64 units, not by latency or memory. One call does
// n * iters * 8 multiply-adds = 16 * n * iters FLOP (a fused multiply-add counts as 2).
class PeakFp64 {
 public:
  explicit PeakFp64(int n);
  ~PeakFp64();
  PeakFp64(const PeakFp64&) = delete;
  PeakFp64& operator=(const PeakFp64&) = delete;

  void run(int iters);
  int size() const;
  double checksum() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace femorax::exec
