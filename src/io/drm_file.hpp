#pragma once

// The DRM data file of an H5DRM pattern (OpenSees' layout, SRC/domain/pattern/drm/
// H5DRMLoadPattern.cpp at commit 955ceb0, and the files Femora's DRM tools write):
//
//   DRM_Data/xyz            [S, 3] float64   station coordinates
//   DRM_Data/internal       [S]    int       1: a boundary station (inside the DRM layer's
//                                            inner surface), 0: an exterior one
//   DRM_Data/data_location  [S]    int       the station's first row in the histories
//   DRM_Data/displacement   [R, T] float64   rows data_location + 0, 1, 2: x, y, z
//   DRM_Data/acceleration   [R, T] float64   (time along the rows)
//   DRM_Metadata/dt, tstart, tend            scalars
//   DRM_Metadata/drmbox_x0  [3]              the reference point of the coordinate transform
//   DRM_Metadata/drmbox_{x,y,z}{min,max}     the DRM box
//
// Other datasets (velocity, ...) may be present and are not read. Host code; HDF5's C API.

#include <cstdint>
#include <string>
#include <vector>

namespace femorax::io {

struct DrmFile {
  std::string path;
  std::vector<double> xyz;  // [3 * S], station by station
  std::vector<std::int32_t> internal, data_location;
  double dt = 0.0, tstart = 0.0, tend = 0.0;
  double x0[3] = {0, 0, 0};
  double box_min[3] = {0, 0, 0}, box_max[3] = {0, 0, 0};
  std::int64_t rows = 0, samples = 0;  // the histories' shape [rows, samples]
  std::int32_t stations() const { return static_cast<std::int32_t>(internal.size()); }
};

// Reads everything but the histories; checks the shapes (every station's three rows exist,
// the two histories have the same shape, dt > 0, tend >= tstart). Throws ModelError naming
// the file and the dataset.
DrmFile read_drm_header(const std::string& path);

// Reads rows [row_begin, row_begin + row_count) and samples [sample_begin,
// sample_begin + sample_count) of DRM_Data/<name> ("displacement" or "acceleration") into
// out (row-major, [row_count, sample_count]).
void read_drm_block(const std::string& path, const std::string& name, std::int64_t row_begin, std::int64_t row_count,
                    std::int64_t sample_begin, std::int64_t sample_count, std::vector<double>& out);

}  // namespace femorax::io
