#include "compile/lts_levels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace femorax::compile {

LtsLevels assign_lts_levels(const Compiled& c, const std::vector<std::vector<double>>& element_dt, double dt,
                            const LtsOptions& opt) {
  if (element_dt.size() != c.batches.size()) {
    throw std::invalid_argument("assign_lts_levels: one list of element steps per batch is needed");
  }
  if (!(dt > 0.0) || opt.ratio < 2 || opt.max_levels < 1 || opt.max_levels > 64 ||
      !(opt.margin > 0.0 && opt.margin <= 1.0)) {
    throw std::invalid_argument("assign_lts_levels: dt > 0, ratio >= 2, 1 <= max_levels <= 64, 0 < margin <= 1");
  }
  LtsLevels L;
  const auto neq = static_cast<std::size_t>(c.dofs.num_equations);
  L.equation_level.assign(neq, 0);
  L.min_element_dt = std::numeric_limits<double>::infinity();
  L.element_level.resize(c.batches.size());
  L.touched_level.resize(c.batches.size());
  int finest = 0;

  // 1. Each element's own level; each equation the finest level of its elements.
  for (std::size_t bi = 0; bi < c.batches.size(); ++bi) {
    const Batch& b = c.batches[bi];
    const auto& dts = element_dt[bi];
    if (dts.size() != static_cast<std::size_t>(b.count)) {
      throw std::invalid_argument("assign_lts_levels: batch " + std::to_string(bi) + " has " + std::to_string(b.count) +
                                  " elements but " + std::to_string(dts.size()) + " steps");
    }
    auto& lev = L.element_level[bi];
    lev.assign(static_cast<std::size_t>(b.count), 0);
    for (std::int32_t e = 0; e < b.count; ++e) {
      const double de = dts[static_cast<std::size_t>(e)];
      L.min_element_dt = std::min(L.min_element_dt, de);
      int l = 0;
      double step = dt;
      while (step > opt.margin * de && l + 1 < opt.max_levels) {
        step /= opt.ratio;
        ++l;
      }
      if (step > opt.margin * de) ++L.too_fine;
      lev[static_cast<std::size_t>(e)] = static_cast<std::int8_t>(l);
      finest = std::max(finest, l);
      for (int a = 0; a < b.ndof; ++a) {
        const std::int32_t q = b.eq[static_cast<std::size_t>(a) * static_cast<std::size_t>(b.padded) +
                                    static_cast<std::size_t>(e)];
        if (q >= 0) {
          auto& el = L.equation_level[static_cast<std::size_t>(q)];
          el = std::max(el, static_cast<std::int8_t>(l));
        }
      }
    }
  }
  L.levels = finest + 1;

  // 2. Each element's touched level (the finest of its equations), counts and work.
  L.elements_at_level.assign(static_cast<std::size_t>(L.levels), 0);
  L.equations_at_level.assign(static_cast<std::size_t>(L.levels), 0);
  for (const auto l : L.equation_level) ++L.equations_at_level[static_cast<std::size_t>(l)];
  const double finest_runs = std::pow(static_cast<double>(opt.ratio), finest);
  for (std::size_t bi = 0; bi < c.batches.size(); ++bi) {
    const Batch& b = c.batches[bi];
    auto& touched = L.touched_level[bi];
    touched.assign(static_cast<std::size_t>(b.count), 0);
    for (std::int32_t e = 0; e < b.count; ++e) {
      std::int8_t t = L.element_level[bi][static_cast<std::size_t>(e)];
      for (int a = 0; a < b.ndof; ++a) {
        const std::int32_t q = b.eq[static_cast<std::size_t>(a) * static_cast<std::size_t>(b.padded) +
                                    static_cast<std::size_t>(e)];
        if (q >= 0) t = std::max(t, L.equation_level[static_cast<std::size_t>(q)]);
      }
      touched[static_cast<std::size_t>(e)] = t;
      ++L.elements_at_level[static_cast<std::size_t>(L.element_level[bi][static_cast<std::size_t>(e)])];
      ++L.elements;
      L.global_work += finest_runs;
      L.lts_work += std::pow(static_cast<double>(opt.ratio), t);
    }
  }
  if (L.elements == 0) L.min_element_dt = 0.0;
  return L;
}

}  // namespace femorax::compile
