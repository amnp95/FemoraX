// Ground motions on the device against OpenSees' PathSeries / PathTimeSeries formulas
// (transcribed below on the host), for both series forms, times before, inside, at, and
// after the data, prepend_zero and use_last, and UniformExcitation's factor.

#include "exec/ground_motion.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using femorax::io::Model;

// PathSeries::getFactor
double path_uniform(const std::vector<double>& path, double dt, double c, bool use_last, double t) {
  if (t < 0.0) return 0.0;
  const double incr = t / dt;
  const long long i1 = static_cast<long long>(std::floor(incr)), i2 = i1 + 1;
  const long long n = static_cast<long long>(path.size());
  if (i2 >= n) return use_last ? c * path[n - 1] : 0.0;
  return c * (path[i1] + (path[i2] - path[i1]) * (incr - i1));
}

// PathTimeSeries::getFactor for nondecreasing calls (pure: the interval holding t).
double path_time(const std::vector<double>& path, const std::vector<double>& time, double c, double t) {
  if (t < time.front()) return 0.0;
  for (std::size_t i = 0; i < time.size(); ++i)
    if (t == time[i]) return c * path[i];
  if (t > time.back()) return 0.0;
  std::size_t i = 0;
  while (!(t >= time[i] && t <= time[i + 1])) ++i;
  return c * (path[i] + (path[i + 1] - path[i]) * (t - time[i]) / (time[i + 1] - time[i]));
}

Model model_with(bool uniform, bool prepend, bool use_last) {
  Model m;
  femorax::io::PathSeries s;
  s.tag = 1;
  s.factor = 9.81;
  s.values = {0.0, 0.3, -0.2, 0.5, 0.1};
  if (uniform) {
    s.dt = 0.02;
  } else {
    s.time = {0.0, 0.01, 0.035, 0.05, 0.08};
  }
  s.prepend_zero = prepend;
  s.use_last = use_last;
  m.time_series.push_back(s);
  m.patterns.push_back({7, 2, 1, 0.0, 1.5, {}});  // dof 2, factor 1.5
  return m;
}

// Host backends compute like the host formula, bit for bit. A GPU compiler fuses
// multiply-adds (nvcc --fmad=true), so there it holds to rounding only.
constexpr bool kHostBackend =
    std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;

void expect_same(double got, double want, double t) {
  if (kHostBackend) {
    EXPECT_EQ(got, want) << "t = " << t;
  } else {
    EXPECT_NEAR(got, want, 4e-16 * 9.81 * 1.5) << "t = " << t;
  }
}

double evaluate(femorax::exec::GroundMotions& g, double t) {
  g.evaluate(t);
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.accel());
  return h(0);
}

TEST(GroundMotion, MatchesOpenSeesPathSeries) {
  for (bool prepend : {false, true})
    for (bool use_last : {false, true}) {
      SCOPED_TRACE(std::string("prepend ") + (prepend ? "on" : "off") + ", use_last " + (use_last ? "on" : "off"));
      const Model m = model_with(true, prepend, use_last);
      femorax::exec::GroundMotions g(m);
      ASSERT_TRUE(g.activate(7));
      EXPECT_FALSE(g.activate(99));
      std::vector<double> path = m.time_series[0].values;
      if (prepend) path.insert(path.begin(), 0.0);
      for (double t = -0.01; t < 0.15; t += 0.0037) {
        expect_same(evaluate(g, t), 1.5 * path_uniform(path, 0.02, 9.81, use_last, t), t);
      }
      auto hd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), g.dof());
      EXPECT_EQ(hd(0), 1);
    }
}

TEST(GroundMotion, MatchesOpenSeesPathTimeSeries) {
  const Model m = model_with(false, false, false);
  femorax::exec::GroundMotions g(m);
  ASSERT_TRUE(g.activate(7));
  const auto& s = m.time_series[0];
  // Increasing times, as an analysis calls it, including the data points themselves.
  std::vector<double> times;
  for (double t = -0.01; t < 0.1; t += 0.0025) times.push_back(t);
  for (double t : s.time) times.push_back(t);
  std::sort(times.begin(), times.end());
  for (double t : times) {
    const double want = 1.5 * path_time(s.values, s.time, 9.81, t);
    EXPECT_NEAR(evaluate(g, t), want, 1e-15 * 9.81) << "t = " << t;
  }
}

}  // namespace
