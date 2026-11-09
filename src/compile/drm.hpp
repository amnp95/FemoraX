#pragma once

// The compile step of an H5DRM pattern (milestone 4): the DRM stations placed in the model
// (OpenSees' coordinate transformation), each node matched to its nearest station within the
// distance tolerance (H5DRMLoadPattern::node_matching_BruteForce, with a spatial hash), and
// the DRM elements: the elements all of whose nodes are matched (OpenSees' rule); those
// with both boundary and exterior nodes receive the effective forces
//   boundary nodes: -M_be a_e - K_be u_e,   exterior nodes: M_eb a_b + K_eb u_b.
// The DRM layer must stay linear for the method to be valid, so a DRM element whose
// material has history state is refused. Host code.

#include <cstdint>
#include <string>
#include <vector>

#include "compile/batches.hpp"
#include "io/drm_file.hpp"
#include "io/model.hpp"

namespace femorax::compile {

struct DrmSetup {
  std::int32_t pattern = 0;  // the pattern's tag
  // Matched nodes (internal node indices), in node order, with their station, the station's
  // first history row and whether it is a boundary (1) or exterior (0) station.
  std::vector<std::int32_t> node, station, row;
  std::vector<std::uint8_t> boundary;
  // node_slot[n]: the position of internal node n in `node`, or -1.
  std::vector<std::int32_t> node_slot;
  // The DRM elements that receive forces (both kinds of node), per batch: the element's
  // index in the batch and a mask of its boundary nodes (bit a: element node a).
  struct Element {
    std::int32_t batch = 0, element = 0;
    std::uint32_t boundary_mask = 0;
  };
  std::vector<Element> elements;
  std::int64_t drm_elements = 0;  // all elements whose nodes are all matched
  std::int64_t stations_matched = 0;
  double max_match_distance = 0.0;
  // The stations' coordinates in the model (after the transformation), [3 * S].
  std::vector<double> station_xyz;
};

// Throws io::ModelError when a DRM element's material has history state, or when no
// element receives DRM forces.
DrmSetup setup_drm(const io::H5DrmPattern& p, const io::DrmFile& f, const Compiled& c);

}  // namespace femorax::compile
