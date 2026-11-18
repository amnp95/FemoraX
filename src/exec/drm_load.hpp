#pragma once

// H5DRM patterns on the device (milestone 4): OpenSees' H5DRMLoadPattern
// (SRC/domain/pattern/drm/H5DRMLoadPattern.cpp, commit 955ceb0).
//
// At each load evaluation (the time where the integrator applies loads: ExplicitDifference
// and Newmark at t + dt, CentralDifference at t), for t in [tstart, tend]:
//   1. the free-field motion of every matched node is interpolated linearly between the
//      two samples around t (drm_direct_read: i1 = floor((t - tstart) / dt), the file's z
//      components negated, d1 (1 - dtau) + d2 dtau);
//   2. each DRM element with boundary and exterior nodes gets its effective forces from
//      its own operator (kernels::launch_drm_forces), written to element slots;
//   3. the slots are summed per equation in a fixed order (deterministic) and the forces
//      are subtracted from the internal force vector f (they are external loads).
// Outside [tstart, tend] the pattern adds nothing, as OpenSees.
//
// The histories of the matched nodes live on the device as time blocks read from the file
// (DrmStream, exec/drm_stream.hpp in step 4; here the whole history at once).

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "compile/drm.hpp"
#include "exec/assembly.hpp"
#include "exec/internal_force.hpp"
#include "io/drm_file.hpp"
#include "io/model.hpp"

namespace femorax::exec {

// The histories are streamed (milestone 4, step 4): the device holds a window of W samples of
// every matched node, [s0, s0 + W), and apply() moves it forward to start at the earlier
// sample it needs when that or the later one leaves it (consecutive windows share a sample,
// so the interpolation never needs two windows). Each window is one HDF5 read of all the
// matched rows over its samples; the next window is read ahead on a thread when HDF5 is
// thread-safe (the recorders write HDF5 meanwhile), synchronously otherwise, and a request
// for any other window (a retry, a reset time) reads it directly. The results do not depend
// on W (tested: W = 2 against the whole history, bit for bit).
class DrmLoad {
 public:
  // window_samples: W; 0 chooses it from a memory budget (window_samples()).
  DrmLoad(const io::H5DrmPattern& p, const compile::Compiled& c, InternalForce& force,
          std::int64_t window_samples = 0);
  ~DrmLoad();
  DrmLoad(const DrmLoad&) = delete;
  DrmLoad& operator=(const DrmLoad&) = delete;
  // W for a pattern: requested if positive, else as many samples as fit in 256 MB for both
  // histories; at least 2 and at most the file's samples.
  static std::int64_t window_samples(std::int64_t nodes, std::int64_t samples, std::int64_t requested = 0);
  std::int64_t window() const { return W_; }
  std::int32_t tag() const { return pattern_.tag; }
  // f -= the DRM forces at time t.
  void apply(double t, const Vec& f);
  const compile::DrmSetup& setup() const { return setup_; }
  const io::DrmFile& file() const { return file_; }
  // Device bytes of a pattern with this many matched nodes, DRM element DOFs and history
  // samples (for the run's memory estimate).
  static std::int64_t device_bytes(std::int64_t nodes, std::int64_t element_dofs, std::int64_t samples,
                                   std::int32_t equations);

 private:
  void ensure(std::int64_t k1, std::int64_t k2);  // samples k1 and k2 in the window
  // Samples [s0, s0 + W) (those that exist) of every matched node into hu, ha, laid out as
  // the device window.
  void read_window(std::int64_t s0, std::vector<double>& hu, std::vector<double>& ha) const;
  io::H5DrmPattern pattern_;
  io::DrmFile file_;
  compile::DrmSetup setup_;
  InternalForce& force_;
  std::int32_t neq_ = 0;
  Kokkos::View<std::int32_t*, Space> node_slot_;
  struct BatchList {
    int batch;
    Kokkos::View<std::int32_t*, Space> elems;
    Kokkos::View<std::uint32_t*, Space> masks;
    std::int64_t base;
  };
  std::vector<BatchList> lists_;
  Vec du_, da_;            // the free-field motion of the matched nodes, [3 * nodes]
  Vec hist_u_, hist_a_;    // the window: [(3 * slot + c) * W + (k - s0)]
  std::int64_t W_ = 0, s0_ = -1;
  std::int64_t row_min_ = 0, row_count_ = 0;  // the file rows the matched nodes span
  std::unique_ptr<io::DrmReader> reader_;
  std::vector<double> win_u_, win_a_;  // host copies of the window being loaded
  std::vector<double> pf_u_, pf_a_;    // the window read ahead
  std::int64_t pf_s0_ = -1;
  std::future<void> prefetch_;
  bool async_ = false;
  Vec forces_;  // the gathered forces, [equations]
  SlotGather gather_;
};

}  // namespace femorax::exec
