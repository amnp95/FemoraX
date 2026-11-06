#pragma once

// UniformExcitation ground motions on the device: the Path time series are copied to the
// device once, and each step one small kernel evaluates every active pattern's ground
// acceleration at the current time into a device array (no host-device traffic per step).
//
// Port of OpenSees SRC/domain/pattern/PathSeries.cpp (uniform dt) and PathTimeSeries.cpp
// (time array, including its interval walk kept from call to call), with GroundMotion's
// time < 0 rule and UniformExcitation's factor, commit 955ceb0.

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <vector>

#include "io/model.hpp"

namespace femorax::exec {

class GroundMotions {
 public:
  using Space = Kokkos::DefaultExecutionSpace;

  GroundMotions() = default;
  explicit GroundMotions(const io::Model& model);

  // Activate pattern `tag` (a process step). Returns false if it is not a UniformExcitation.
  bool activate(std::int32_t tag);

  // Evaluate every active pattern at time t into accel(): accel(p) = fact_p * series_p(t).
  void evaluate(double t);

  int num_active() const { return static_cast<int>(active_.size()); }
  // Device arrays of the active patterns: their ground accelerations (after evaluate) and
  // the 0-based DOF each acts in.
  Kokkos::View<double*, Space> accel() const { return accel_; }
  Kokkos::View<int*, Space> dof() const { return dof_; }
  // Initial velocities (vel0) of the active patterns, host side.
  std::vector<std::pair<int, double>> initial_velocities() const;

 private:
  struct Series {
    bool uniform = true;
    double dt = 0, factor = 1, start = 0;
    bool use_last = false;
    Kokkos::View<double*, Space> values, time;
  };
  struct Pattern {
    std::int32_t tag;
    int dof;
    double fact, vel0;
    int series;
  };
  std::vector<Series> series_;
  std::vector<Pattern> patterns_;
  std::vector<int> active_;  // indices into patterns_
  Kokkos::View<double*, Space> accel_;
  Kokkos::View<int*, Space> dof_;
  Kokkos::View<int*, Space> time_loc_;  // PathTimeSeries::currentTimeLoc, per active pattern
};

}  // namespace femorax::exec
