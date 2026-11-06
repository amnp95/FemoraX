// Shared helpers for the OpenSees reference drivers: deterministic pseudo-random numbers
// and the reference-data text format.
//
// Format (one file per case):
//   # femorax-reference 1
//   # opensees_commit <hash>
//   # <key> <value...>            free-form header lines
//   # columns <name> <name> ...
//   <row of %.17g values>         one row per step; doubles round-trip exactly
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace refdrv {

// 64-bit LCG (Knuth's MMIX constants): the same sequence on every platform and compiler,
// unlike <random> distributions.
class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : s_(seed) {}
  // Uniform in [-1, 1).
  double symmetric() {
    s_ = s_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s_ >> 11) * (2.0 / 9007199254740992.0) - 1.0;
  }

 private:
  std::uint64_t s_;
};

class Writer {
 public:
  Writer(const std::string& path, const std::vector<std::string>& header,
         const std::vector<std::string>& columns);
  ~Writer();
  Writer(const Writer&) = delete;
  Writer& operator=(const Writer&) = delete;
  void row(const std::vector<double>& values);

 private:
  std::FILE* f_;
  std::size_t ncols_;
};

}  // namespace refdrv
