#include "solvers/vector_ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace femorax::solvers {

namespace {

using Team = Kokkos::TeamPolicy<Space>::member_type;
using Scratch = Kokkos::View<double*, Space::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

std::int64_t chunks_of(std::int64_t n) { return (n + VectorOps::kChunk - 1) / VectorOps::kChunk; }

// One level: out[c] = sum over chunk c of x[i] * y[i] (or x[i] if y is empty), in the
// fixed lane order described in the header.
void reduce_chunks(const ConstVector& x, const ConstVector& y, bool with_y, std::int64_t n, int lanes,
                   const Vector& out) {
  const std::int64_t nchunks = chunks_of(n);
  const std::int64_t C = VectorOps::kChunk;
  Kokkos::TeamPolicy<Space> policy(static_cast<int>(nchunks), lanes);
  policy.set_scratch_size(0, Kokkos::PerTeam(Scratch::shmem_size(lanes)));
  Kokkos::parallel_for(
      "solvers_reduce_chunks", policy, KOKKOS_LAMBDA(const Team& t) {
        Scratch lane_sum(t.team_scratch(0), t.team_size());
        const std::int64_t c = t.league_rank();
        const std::int64_t begin = c * C;
        const std::int64_t len = (n - begin < C) ? n - begin : C;
        const int T = t.team_size();
        const int l = t.team_rank();
        double s = 0.0;
        for (std::int64_t k = l; k < len; k += T) s += with_y ? x(begin + k) * y(begin + k) : x(begin + k);
        lane_sum(l) = s;
        t.team_barrier();
        if (l == 0) {
          double total = 0.0;
          for (int j = 0; j < T; ++j) total += lane_sum(j);
          out(c) = total;
        }
      });
}

}  // namespace

VectorOps::VectorOps(std::int64_t n, int lanes)
    : lanes_(lanes > 0 ? lanes : 1) {
  const std::int64_t level1 = std::max<std::int64_t>(1, chunks_of(n));
  partial_[0] = Vector("solvers_partial0", static_cast<std::size_t>(level1));
  partial_[1] = Vector("solvers_partial1", static_cast<std::size_t>(std::max<std::int64_t>(1, chunks_of(level1))));
}

double VectorOps::reduce_levels(std::int64_t n_partial) const {
  int cur = 0;
  while (n_partial > 1) {
    reduce_chunks(partial_[cur], partial_[cur], false, n_partial, lanes_, partial_[cur ^ 1]);
    n_partial = chunks_of(n_partial);
    cur ^= 1;
  }
  double r = 0.0;
  Kokkos::deep_copy(r, Kokkos::subview(partial_[cur], 0));
  return r;
}

double VectorOps::dot(const ConstVector& x, const ConstVector& y) const {
  const auto n = static_cast<std::int64_t>(x.extent(0));
  if (n == 0) return 0.0;
  if (chunks_of(n) > static_cast<std::int64_t>(partial_[0].extent(0)))
    throw std::invalid_argument("VectorOps::dot: vector of " + std::to_string(n) + " entries exceeds the capacity");
  reduce_chunks(x, y, true, n, lanes_, partial_[0]);
  return reduce_levels(chunks_of(n));
}

double VectorOps::norm2(const ConstVector& x) const { return std::sqrt(dot(x, x)); }

void VectorOps::fill(const Vector& y, double a) { Kokkos::deep_copy(y, a); }

void VectorOps::copy(const ConstVector& x, const Vector& y) { Kokkos::deep_copy(y, x); }

void VectorOps::axpy(double a, const ConstVector& x, const Vector& y) {
  Kokkos::parallel_for(
      "solvers_axpy", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, x.extent(0)),
      KOKKOS_LAMBDA(const std::int64_t i) { y(i) += a * x(i); });
}

void VectorOps::xpay(const ConstVector& x, double a, const Vector& y) {
  Kokkos::parallel_for(
      "solvers_xpay", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, x.extent(0)),
      KOKKOS_LAMBDA(const std::int64_t i) { y(i) = x(i) + a * y(i); });
}

void VectorOps::multiply(const ConstVector& d, const ConstVector& x, const Vector& y) {
  Kokkos::parallel_for(
      "solvers_multiply", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, x.extent(0)),
      KOKKOS_LAMBDA(const std::int64_t i) { y(i) = d(i) * x(i); });
}

void VectorOps::reciprocal(const ConstVector& d, const Vector& y) {
  Kokkos::parallel_for(
      "solvers_reciprocal", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, d.extent(0)),
      KOKKOS_LAMBDA(const std::int64_t i) { y(i) = 1.0 / d(i); });
}

double VectorOps::host_dot(const std::vector<double>& x, const std::vector<double>& y, int lanes) {
  auto level = [&](const std::vector<double>& a, const std::vector<double>* b) {
    const auto n = static_cast<std::int64_t>(a.size());
    std::vector<double> out(static_cast<std::size_t>(chunks_of(n)));
    for (std::int64_t c = 0; c < static_cast<std::int64_t>(out.size()); ++c) {
      const std::int64_t begin = c * kChunk, len = std::min(kChunk, n - begin);
      std::vector<double> lane(static_cast<std::size_t>(lanes), 0.0);
      for (int l = 0; l < lanes; ++l)
        for (std::int64_t k = l; k < len; k += lanes)
          lane[l] += b ? a[begin + k] * (*b)[begin + k] : a[begin + k];
      double total = 0.0;
      for (int l = 0; l < lanes; ++l) total += lane[l];
      out[c] = total;
    }
    return out;
  };
  if (x.empty()) return 0.0;
  std::vector<double> p = level(x, &y);
  while (p.size() > 1) p = level(p, nullptr);
  return p[0];
}

const char* to_string(Symmetry s) {
  switch (s) {
    case Symmetry::SymmetricPositiveDefinite: return "SymmetricPositiveDefinite";
    case Symmetry::Symmetric: return "Symmetric";
    case Symmetry::General: break;
  }
  return "General";
}

}  // namespace femorax::solvers
