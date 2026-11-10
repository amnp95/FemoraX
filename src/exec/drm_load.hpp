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
#include <memory>
#include <vector>

#include "compile/drm.hpp"
#include "exec/assembly.hpp"
#include "exec/internal_force.hpp"
#include "io/drm_file.hpp"
#include "io/model.hpp"

namespace femorax::exec {

class DrmLoad {
 public:
  DrmLoad(const io::H5DrmPattern& p, const compile::Compiled& c, InternalForce& force);
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
  Vec hist_u_, hist_a_;    // [(3 * slot + c) * samples + k]
  Vec forces_;  // the gathered forces, [equations]
  SlotGather gather_;
};

}  // namespace femorax::exec
