#include "exec/drm_load.hpp"

#include <cmath>
#include <map>
#include <utility>
#include <vector>

#include "solvers/vector_ops.hpp"

namespace femorax::exec {

namespace {
using Index = std::int32_t;

template <class T>
Kokkos::View<T*, Space> upload(const char* label, const std::vector<T>& h) {
  Kokkos::View<T*, Space> d(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), h.size());
  auto m = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < h.size(); ++i) m(i) = h[i];
  Kokkos::deep_copy(d, m);
  return d;
}
}  // namespace

std::int64_t DrmLoad::device_bytes(std::int64_t nodes, std::int64_t element_dofs, std::int64_t samples,
                                   std::int32_t equations) {
  return 2 * 3 * nodes * samples * 8 +           // histories
         2 * 3 * nodes * 8 +                     // the motion at t
         element_dofs * 8 + element_dofs * 8 +   // slots, gather list
         (static_cast<std::int64_t>(equations) + 1) * 8 + static_cast<std::int64_t>(equations) * 8;
}

DrmLoad::DrmLoad(const io::H5DrmPattern& p, const compile::Compiled& c, InternalForce& force)
    : pattern_(p), file_(io::read_drm_header(p.path)), setup_(compile::setup_drm(p, file_, c)), force_(force),
      neq_(c.dofs.num_equations) {
  node_slot_ = upload("drm_node_slot", setup_.node_slot);
  const auto nodes = static_cast<std::int64_t>(setup_.node.size());

  // Elements per batch, and the gather pairs in (batch, element, DOF) order.
  std::map<int, std::vector<const compile::DrmSetup::Element*>> per_batch;
  for (const auto& e : setup_.elements) per_batch[e.batch].push_back(&e);
  std::int64_t base = 0;
  std::vector<std::pair<std::int32_t, std::int64_t>> pairs;
  for (const auto& [bi, list] : per_batch) {
    const auto& hb = c.batches[static_cast<std::size_t>(bi)];
    const auto n = static_cast<std::int64_t>(list.size());
    std::vector<Index> elems;
    std::vector<std::uint32_t> masks;
    for (const auto* e : list) {
      elems.push_back(e->element);
      masks.push_back(e->boundary_mask);
    }
    lists_.push_back({bi, upload("drm_elements", elems), upload("drm_masks", masks), base});
    for (std::int64_t i = 0; i < n; ++i)
      for (int a = 0; a < hb.ndof; ++a) {
        const std::int32_t q = hb.eq[static_cast<std::size_t>(a) * static_cast<std::size_t>(hb.padded) +
                                     static_cast<std::size_t>(list[static_cast<std::size_t>(i)]->element)];
        if (q >= 0) pairs.emplace_back(q, base + a * n + i);
      }
    base += n * hb.ndof;
  }
  forces_ = Vec("drm_forces", static_cast<std::size_t>(neq_));
  gather_ = SlotGather(base, neq_, pairs);
  du_ = Vec("drm_u", static_cast<std::size_t>(3 * nodes));
  da_ = Vec("drm_a", static_cast<std::size_t>(3 * nodes));

  // The matched nodes' histories, whole (step 3; streamed in time blocks in step 4).
  const std::int64_t T = file_.samples;
  std::vector<double> hu(static_cast<std::size_t>(3 * nodes * T)), ha(hu.size()), block;
  for (std::int64_t sl = 0; sl < nodes; ++sl) {
    const std::int64_t row = setup_.row[static_cast<std::size_t>(sl)];
    io::read_drm_block(file_.path, "displacement", row, 3, 0, T, block);
    std::copy(block.begin(), block.end(), hu.begin() + 3 * sl * T);
    io::read_drm_block(file_.path, "acceleration", row, 3, 0, T, block);
    std::copy(block.begin(), block.end(), ha.begin() + 3 * sl * T);
  }
  hist_u_ = upload("drm_history_u", hu);
  hist_a_ = upload("drm_history_a", ha);
}

void DrmLoad::apply(double t, const Vec& f) {
  if (t < file_.tstart || t > file_.tend) return;  // OpenSees: no DRM forces outside the data
  const double dt = file_.dt, tstart = file_.tstart;
  const auto i1 = static_cast<std::int64_t>(std::floor((t - tstart) / dt));
  const std::int64_t T = file_.samples;
  std::int64_t i2 = i1 + 1;
  const double t1 = static_cast<double>(i1) * dt + tstart, t2 = static_cast<double>(i2) * dt + tstart;
  double dtau = (t - t1) / (t2 - t1);
  // OpenSees reads sample i2 even at t = tend, past the data when tend is the last sample's
  // time (an HDF5 error and exit there); FemoraX uses the last sample.
  if (i2 >= T) {
    i2 = T - 1;
    dtau = 0.0;
  }
  const std::int64_t k1 = std::min<std::int64_t>(i1, T - 1);
  const auto hu = hist_u_, ha = hist_a_, du = du_, da = da_;
  const auto nodes = static_cast<Index>(du_.extent(0) / 3);
  Kokkos::parallel_for(
      "drm_motion", Kokkos::RangePolicy<Space, Kokkos::IndexType<Index>>(0, nodes), KOKKOS_LAMBDA(const Index sl) {
        for (int c = 0; c < 3; ++c) {
          const std::int64_t r = (3 * static_cast<std::int64_t>(sl) + c) * T;
          double d1 = hu(r + k1), d2 = hu(r + i2), a1 = ha(r + k1), a2 = ha(r + i2);
          if (c == 2) {  // the file's z axis points down
            d1 = -d1;
            d2 = -d2;
            a1 = -a1;
            a2 = -a2;
          }
          du(3 * static_cast<std::int64_t>(sl) + c) = d1 * (1 - dtau) + d2 * (dtau);
          da(3 * static_cast<std::int64_t>(sl) + c) = a1 * (1 - dtau) + a2 * (dtau);
        }
      });
  for (const auto& l : lists_) force_.drm_forces(l.batch, l.elems, l.masks, node_slot_, du_, da_, gather_.slots(), l.base);
  gather_.gather(forces_);
  solvers::VectorOps::axpy(-1.0, forces_, f);
}

}  // namespace femorax::exec
