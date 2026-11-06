#pragma once

#include <memory>

namespace femorax {

// STREAM-style kernels on the default execution space. They measure the memory bandwidth
// the hardware actually delivers, which is the yardstick for every other kernel: a
// bandwidth-bound kernel is judged by its fraction of the triad bandwidth.
class StreamKernels {
 public:
  explicit StreamKernels(int n);  // three FP64 arrays of n entries (32-bit indices)
  ~StreamKernels();
  StreamKernels(const StreamKernels&) = delete;
  StreamKernels& operator=(const StreamKernels&) = delete;

  int size() const;
  void copy();             // c[i] = a[i]               (16 bytes per entry)
  void triad(double s);    // a[i] = b[i] + s * c[i]    (24 bytes per entry)
  double checksum() const; // sum of a, so the compiler cannot drop the work

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace femorax
