#include "exec/ground_motion.hpp"

#include <cmath>
#include <string>

namespace femorax::exec {

namespace {

using Space = Kokkos::DefaultExecutionSpace;

Kokkos::View<double*, Space> upload(const char* label, const std::vector<double>& h) {
  Kokkos::View<double*, Space> d(std::string(label), h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}

// PathSeries::getFactor (OpenSees), pseudoTime >= 0.
KOKKOS_INLINE_FUNCTION double path_uniform(const Kokkos::View<double*, Space>& path, double dt,
                                           double cFactor, double startTime, bool useLast,
                                           double pseudoTime) {
  if (pseudoTime < startTime) return 0.0;
  const double incr = (pseudoTime - startTime) / dt;
  const long long incr1 = static_cast<long long>(Kokkos::floor(incr));
  const long long incr2 = incr1 + 1;
  const long long size = static_cast<long long>(path.extent(0));
  if (incr2 >= size) {
    if (!useLast) return 0.0;
    return cFactor * path(size - 1);
  }
  const double value1 = path(incr1);
  const double value2 = path(incr2);
  return cFactor * (value1 + (value2 - value1) * (incr - incr1));
}

// PathTimeSeries::getFactor (OpenSees), with its currentTimeLoc kept between calls.
KOKKOS_INLINE_FUNCTION double path_time(const Kokkos::View<double*, Space>& thePath,
                                        const Kokkos::View<double*, Space>& time, double cFactor,
                                        int& currentTimeLoc, double pseudoTime) {
  double time1 = time(currentTimeLoc);
  if (pseudoTime < time1 && currentTimeLoc == 0) return 0.0;
  if (pseudoTime == time1) return cFactor * thePath(currentTimeLoc);
  const int size = static_cast<int>(time.extent(0));
  const int sizem1 = size - 1;
  const int sizem2 = size - 2;
  // useLast is never set for time-array series (OpenSees ignores the flag there).
  if (pseudoTime > time1 && currentTimeLoc == sizem1) return 0.0;
  double time2 = time(currentTimeLoc + 1);
  if (pseudoTime > time2) {
    while ((pseudoTime > time2) && (currentTimeLoc < sizem2)) {
      currentTimeLoc++;
      time1 = time2;
      time2 = time(currentTimeLoc + 1);
    }
    if (pseudoTime > time2) return 0.0;
  } else if (pseudoTime < time1) {
    while ((pseudoTime < time1) && (currentTimeLoc > 0)) {
      currentTimeLoc--;
      time2 = time1;
      time1 = time(currentTimeLoc);
    }
    if (pseudoTime < time1) return 0.0;
  }
  const double value1 = thePath(currentTimeLoc);
  const double value2 = thePath(currentTimeLoc + 1);
  return cFactor * (value1 + (value2 - value1) * (pseudoTime - time1) / (time2 - time1));
}

}  // namespace

GroundMotions::GroundMotions(const io::Model& m) {
  std::vector<std::int32_t> series_tag;
  for (const auto& s : m.time_series) {
    Series d;
    d.uniform = s.dt.has_value();
    d.dt = s.dt.value_or(0.0);
    d.factor = s.factor;
    d.use_last = s.use_last;
    std::vector<double> values = s.values;
    if (d.uniform && s.prepend_zero) values.insert(values.begin(), 0.0);  // PathSeries constructor
    d.values = upload("series_values", values);
    if (!d.uniform) d.time = upload("series_time", s.time);
    series_.push_back(d);
    series_tag.push_back(s.tag);
  }
  for (const auto& p : m.patterns) {
    int si = -1;
    for (std::size_t i = 0; i < series_tag.size(); ++i)
      if (series_tag[i] == p.accel) si = static_cast<int>(i);
    patterns_.push_back({p.tag, p.dof - 1, p.factor, p.vel0, si});
  }
}

bool GroundMotions::activate(std::int32_t tag) {
  for (std::size_t i = 0; i < patterns_.size(); ++i) {
    if (patterns_[i].tag != tag) continue;
    active_.push_back(static_cast<int>(i));
    const auto n = active_.size();
    accel_ = Kokkos::View<double*, Space>("ground_accel", n);
    dof_ = Kokkos::View<int*, Space>("ground_dof", n);
    time_loc_ = Kokkos::View<int*, Space>("ground_time_loc", n);
    auto hd = Kokkos::create_mirror_view(dof_);
    for (std::size_t k = 0; k < n; ++k) hd(k) = patterns_[active_[k]].dof;
    Kokkos::deep_copy(dof_, hd);
    return true;
  }
  return false;
}

std::vector<std::pair<int, double>> GroundMotions::initial_velocities() const {
  std::vector<std::pair<int, double>> out;
  for (const int i : active_)
    if (patterns_[i].vel0 != 0.0) out.emplace_back(patterns_[i].dof, patterns_[i].vel0);
  return out;
}

void GroundMotions::evaluate(double t) {
  // One tiny launch per active pattern: no host-device copy, the series stay on the device.
  for (std::size_t k = 0; k < active_.size(); ++k) {
    const Pattern& p = patterns_[active_[k]];
    const Series& s = series_[p.series];
    const auto accel = accel_;
    const auto loc = time_loc_;
    const auto values = s.values, time = s.time;
    const bool uniform = s.uniform, use_last = s.use_last;
    const double dt = s.dt, factor = s.factor, start = s.start, fact = p.fact;
    const int slot = static_cast<int>(k);
    Kokkos::parallel_for(
        "ground_motion", Kokkos::RangePolicy<Space>(0, 1), KOKKOS_LAMBDA(const int) {
          double a = 0.0;  // GroundMotion::getAccel: 0 for time < 0
          if (t >= 0.0) {
            if (uniform) {
              a = path_uniform(values, dt, factor, start, use_last, t);
            } else {
              int l = loc(slot);
              a = path_time(values, time, factor, l, t);
              loc(slot) = l;
            }
          }
          accel(slot) = fact * a;  // UniformExcitation: R = fact in its DOF
        });
  }
}

}  // namespace femorax::exec
