#include "compile/sparsity.hpp"

#include <algorithm>

namespace femorax::compile {

std::int64_t SparsityPattern::find(std::int32_t i, std::int32_t j) const {
  if (i < 0 || i >= rows) return -1;
  const auto b = columns.begin() + row_offsets[static_cast<std::size_t>(i)];
  const auto e = columns.begin() + row_offsets[static_cast<std::size_t>(i) + 1];
  const auto it = std::lower_bound(b, e, j);
  return (it != e && *it == j) ? static_cast<std::int64_t>(it - columns.begin()) : -1;
}

SparsityPattern build_pattern(const Compiled& c) {
  const std::int32_t n = c.dofs.num_equations;
  // Pass 1: an upper bound of each row's length (sum of the ndof of the elements that touch
  // it), then the candidate columns of every row in one array; pass 2: sort and unique per
  // row. Memory: the candidates, about (element DOFs)^2 per element, once.
  std::vector<std::int64_t> bound(static_cast<std::size_t>(n) + 1, 0);
  for (const auto& b : c.batches)
    for (std::int32_t e = 0; e < b.count; ++e)
      for (int k = 0; k < b.ndof; ++k) {
        const std::int32_t q = b.eq[static_cast<std::size_t>(k) * b.padded + e];
        if (q >= 0) bound[static_cast<std::size_t>(q) + 1] += b.ndof;
      }
  for (std::int32_t i = 0; i < n; ++i) bound[i + 1] += bound[i];
  std::vector<std::int32_t> cand(static_cast<std::size_t>(bound[n]));
  std::vector<std::int64_t> fill(bound.begin(), bound.end() - 1);
  std::vector<std::int32_t> eqs;
  for (const auto& b : c.batches)
    for (std::int32_t e = 0; e < b.count; ++e) {
      eqs.clear();
      for (int k = 0; k < b.ndof; ++k) eqs.push_back(b.eq[static_cast<std::size_t>(k) * b.padded + e]);
      for (const auto qi : eqs) {
        if (qi < 0) continue;
        for (const auto qj : eqs)
          if (qj >= 0) cand[static_cast<std::size_t>(fill[qi]++)] = qj;
      }
    }

  SparsityPattern p;
  p.rows = n;
  p.row_offsets.assign(static_cast<std::size_t>(n) + 1, 0);
  // An equation no element touches (a node with nodal mass only) still gets its diagonal.
  for (std::int32_t i = 0; i < n; ++i) {
    auto b = cand.begin() + bound[i], e = cand.begin() + fill[i];
    std::sort(b, e);
    const auto u = std::unique(b, e);
    std::int64_t len = u - b;
    if (len == 0) len = 1;
    p.row_offsets[static_cast<std::size_t>(i) + 1] = p.row_offsets[i] + len;
  }
  p.columns.resize(static_cast<std::size_t>(p.row_offsets[n]));
  for (std::int32_t i = 0; i < n; ++i) {
    auto b = cand.begin() + bound[i];
    const std::int64_t len = p.row_offsets[i + 1] - p.row_offsets[i];
    if (fill[i] == bound[i]) {
      p.columns[static_cast<std::size_t>(p.row_offsets[i])] = i;
    } else {
      std::copy(b, b + len, p.columns.begin() + p.row_offsets[i]);
    }
  }
  return p;
}

std::int64_t pattern_bytes(const SparsityPattern& p) {
  return static_cast<std::int64_t>(sizeof(std::int64_t)) * (p.rows + 1) +
         static_cast<std::int64_t>(sizeof(std::int32_t) + sizeof(double)) * p.nnz();
}

}  // namespace femorax::compile
