#include "solvers/block_ops.hpp"

#include <algorithm>
#include <stdexcept>

#include "solvers/vector_ops.hpp"

namespace femorax::solvers {

namespace {

using Team = Kokkos::TeamPolicy<Space>::member_type;
using Scratch = Kokkos::View<double*, Space::scratch_memory_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

std::int64_t chunks_of(std::int64_t n) { return (n + VectorOps::kChunk - 1) / VectorOps::kChunk; }

// One level for k columns: out(c, col) = sum over chunk c of src(i, col) * w(i) (or of
// src(i, col)), one team per (chunk, column), lanes in the fixed order of VectorOps.
void reduce_block(const Block& src, const ConstVector& w, bool with_w, std::int64_t n, int k, int lanes,
                  const Block& out) {
  const std::int64_t nchunks = chunks_of(n);
  const std::int64_t C = VectorOps::kChunk;
  Kokkos::TeamPolicy<Space> policy(static_cast<int>(nchunks * k), lanes);
  policy.set_scratch_size(0, Kokkos::PerTeam(Scratch::shmem_size(lanes)));
  Kokkos::parallel_for(
      "solvers_reduce_block", policy, KOKKOS_LAMBDA(const Team& t) {
        Scratch lane_sum(t.team_scratch(0), t.team_size());
        const std::int64_t c = t.league_rank() / k;
        const int col = static_cast<int>(t.league_rank() % k);
        const std::int64_t begin = c * C;
        const std::int64_t len = (n - begin < C) ? n - begin : C;
        const int T = t.team_size();
        const int l = t.team_rank();
        double s = 0.0;
        for (std::int64_t j = l; j < len; j += T)
          s += with_w ? src(begin + j, col) * w(begin + j) : src(begin + j, col);
        lane_sum(l) = s;
        t.team_barrier();
        if (l == 0) {
          double total = 0.0;
          for (int j = 0; j < T; ++j) total += lane_sum(j);
          out(c, col) = total;
        }
      });
}

}  // namespace

BlockOps::BlockOps(std::int64_t n, int max_columns, int lanes)
    : lanes_(lanes > 0 ? lanes : 1), max_columns_(max_columns) {
  const std::int64_t level1 = std::max<std::int64_t>(1, chunks_of(n));
  const auto mc = static_cast<std::size_t>(max_columns);
  partial_[0] = Block("block_partial0", static_cast<std::size_t>(level1), mc);
  partial_[1] = Block("block_partial1", static_cast<std::size_t>(std::max<std::int64_t>(1, chunks_of(level1))), mc);
  coef_ = Kokkos::View<double*, Space>("block_coef", mc);
  coef_host_ = Kokkos::View<double*, Kokkos::HostSpace>("block_coef_host", mc);
}

void BlockOps::dots(const Block& V, int k, const ConstVector& w, std::vector<double>& h) const {
  h.assign(static_cast<std::size_t>(k), 0.0);
  if (k == 0) return;
  if (k > max_columns_) throw std::invalid_argument("BlockOps::dots: too many columns");
  const auto n = static_cast<std::int64_t>(w.extent(0));
  if (n == 0) return;
  reduce_block(V, w, true, n, k, lanes_, partial_[0]);
  std::int64_t len = chunks_of(n);
  int cur = 0;
  while (len > 1) {
    reduce_block(partial_[cur], w, false, len, k, lanes_, partial_[cur ^ 1]);
    len = chunks_of(len);
    cur ^= 1;
  }
  auto row = Kokkos::subview(partial_[cur], 0, std::make_pair(0, k));
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), row);
  for (int c = 0; c < k; ++c) h[static_cast<std::size_t>(c)] = hr(c);
}

void BlockOps::combine(const Block& V, int k, const std::vector<double>& y, double a, const Vector& w) const {
  if (k == 0) return;
  for (int c = 0; c < k; ++c) coef_host_(c) = a * y[static_cast<std::size_t>(c)];
  Kokkos::deep_copy(Kokkos::subview(coef_, std::make_pair(0, k)), Kokkos::subview(coef_host_, std::make_pair(0, k)));
  const auto coef = coef_;
  Kokkos::parallel_for(
      "solvers_block_combine", Kokkos::RangePolicy<Space, Kokkos::IndexType<std::int64_t>>(0, w.extent(0)),
      KOKKOS_LAMBDA(const std::int64_t i) {
        double s = 0.0;
        for (int c = 0; c < k; ++c) s += coef(c) * V(i, c);
        w(i) += s;
      });
}

double BlockOps::host_dot(const std::vector<double>& v, const std::vector<double>& w, int lanes) {
  return VectorOps::host_dot(v, w, lanes);
}

}  // namespace femorax::solvers
