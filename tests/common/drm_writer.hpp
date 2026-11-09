#pragma once

// Test helper: writes a DRM file in OpenSees' H5DRM layout (io/drm_file.hpp) from station
// coordinates, boundary flags and histories ([3 S, T] each, row-major).

#include <hdf5.h>

#include <cstdint>
#include <string>
#include <vector>

namespace femorax::testing {

struct DrmData {
  std::vector<double> xyz;            // [3 S]
  std::vector<std::int32_t> internal;  // [S]
  std::vector<double> disp, acc;       // [3 S * T], row r = 3 s + component
  std::int64_t samples = 2;
  double dt = 0.01, tstart = 0.0, tend = 0.01;
  double x0[3] = {0, 0, 0};
};

inline void write_drm_file(const std::string& path, const DrmData& d) {
  const auto S = static_cast<hsize_t>(d.internal.size());
  const hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  const hid_t gd = H5Gcreate2(f, "DRM_Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  const hid_t gm = H5Gcreate2(f, "DRM_Metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  auto put = [](hid_t g, const char* name, hid_t type, int rank, const hsize_t* dims, const void* data) {
    const hid_t sp = rank == 0 ? H5Screate(H5S_SCALAR) : H5Screate_simple(rank, dims, nullptr);
    const hid_t ds = H5Dcreate2(g, name, type, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(ds);
    H5Sclose(sp);
  };
  const hsize_t xyz_dims[2] = {S, 3}, s_dims[1] = {S}, h_dims[2] = {3 * S, static_cast<hsize_t>(d.samples)};
  put(gd, "xyz", H5T_NATIVE_DOUBLE, 2, xyz_dims, d.xyz.data());
  put(gd, "internal", H5T_NATIVE_INT32, 1, s_dims, d.internal.data());
  std::vector<std::int32_t> loc(S);
  for (hsize_t s = 0; s < S; ++s) loc[s] = static_cast<std::int32_t>(3 * s);
  put(gd, "data_location", H5T_NATIVE_INT32, 1, s_dims, loc.data());
  std::vector<double> zero(static_cast<std::size_t>(3 * S * static_cast<hsize_t>(d.samples)), 0.0);
  put(gd, "displacement", H5T_NATIVE_DOUBLE, 2, h_dims, d.disp.empty() ? zero.data() : d.disp.data());
  put(gd, "acceleration", H5T_NATIVE_DOUBLE, 2, h_dims, d.acc.empty() ? zero.data() : d.acc.data());
  put(gm, "dt", H5T_NATIVE_DOUBLE, 0, nullptr, &d.dt);
  put(gm, "tstart", H5T_NATIVE_DOUBLE, 0, nullptr, &d.tstart);
  put(gm, "tend", H5T_NATIVE_DOUBLE, 0, nullptr, &d.tend);
  const hsize_t three[1] = {3};
  put(gm, "drmbox_x0", H5T_NATIVE_DOUBLE, 1, three, d.x0);
  const double big = 1e9;
  for (const char* n : {"drmbox_xmax", "drmbox_ymax", "drmbox_zmax"}) put(gm, n, H5T_NATIVE_DOUBLE, 0, nullptr, &big);
  const double small = -1e9;
  for (const char* n : {"drmbox_xmin", "drmbox_ymin", "drmbox_zmin"}) put(gm, n, H5T_NATIVE_DOUBLE, 0, nullptr, &small);
  H5Gclose(gm);
  H5Gclose(gd);
  H5Fclose(f);
}

}  // namespace femorax::testing
