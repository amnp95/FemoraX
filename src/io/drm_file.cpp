#include "io/drm_file.hpp"

#include <hdf5.h>

#include <string>
#include <vector>

#include "io/h5.hpp"

namespace femorax::io {

namespace {

[[noreturn]] void fail(const std::string& path, const std::string& what) {
  throw ModelError("DRM file " + path + ": " + what);
}

struct Id {
  hid_t id = -1;
  herr_t (*close)(hid_t) = nullptr;
  Id(hid_t i, herr_t (*c)(hid_t)) : id(i), close(c) {}
  ~Id() {
    if (id >= 0 && close) close(id);
  }
  Id(const Id&) = delete;
  Id& operator=(const Id&) = delete;
};

std::vector<hsize_t> dims_of(hid_t dset) {
  Id space(H5Dget_space(dset), H5Sclose);
  const int rank = H5Sget_simple_extent_ndims(space.id);
  std::vector<hsize_t> d(static_cast<std::size_t>(rank > 0 ? rank : 0));
  if (rank > 0) H5Sget_simple_extent_dims(space.id, d.data(), nullptr);
  return d;
}

template <class T>
std::vector<T> read_all(hid_t file, const std::string& path, const std::string& name, hid_t type,
                        std::vector<hsize_t>& dims) {
  if (H5Lexists(file, name.c_str(), H5P_DEFAULT) <= 0) fail(path, "missing dataset " + name);
  Id d(H5Dopen2(file, name.c_str(), H5P_DEFAULT), H5Dclose);
  if (d.id < 0) fail(path, "cannot open " + name);
  dims = dims_of(d.id);
  std::size_t n = 1;
  for (auto v : dims) n *= static_cast<std::size_t>(v);
  std::vector<T> out(n);
  if (n > 0 && H5Dread(d.id, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) < 0) fail(path, "cannot read " + name);
  return out;
}

double scalar(hid_t file, const std::string& path, const std::string& name) {
  std::vector<hsize_t> dims;
  const auto v = read_all<double>(file, path, name, H5T_NATIVE_DOUBLE, dims);
  if (v.size() != 1) fail(path, name + " must be a scalar");
  return v[0];
}

}  // namespace

DrmFile read_drm_header(const std::string& path) {
  H5E_auto2_t old_func = nullptr;
  void* old_data = nullptr;
  H5Eget_auto2(H5E_DEFAULT, &old_func, &old_data);
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  struct Restore {
    H5E_auto2_t f;
    void* d;
    ~Restore() { H5Eset_auto2(H5E_DEFAULT, f, d); }
  } restore{old_func, old_data};

  Id file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (file.id < 0) fail(path, "cannot open as an HDF5 file");
  DrmFile f;
  f.path = path;
  std::vector<hsize_t> dims;
  f.xyz = read_all<double>(file.id, path, "DRM_Data/xyz", H5T_NATIVE_DOUBLE, dims);
  if (dims.size() != 2 || dims[1] != 3) fail(path, "DRM_Data/xyz must be [stations, 3]");
  const auto S = static_cast<std::size_t>(dims[0]);
  f.internal = read_all<std::int32_t>(file.id, path, "DRM_Data/internal", H5T_NATIVE_INT32, dims);
  if (f.internal.size() != S) fail(path, "DRM_Data/internal must have one value per station");
  f.data_location = read_all<std::int32_t>(file.id, path, "DRM_Data/data_location", H5T_NATIVE_INT32, dims);
  if (f.data_location.size() != S) fail(path, "DRM_Data/data_location must have one value per station");
  for (const auto v : f.internal)
    if (v != 0 && v != 1) fail(path, "DRM_Data/internal must be 0 or 1");
  f.dt = scalar(file.id, path, "DRM_Metadata/dt");
  f.tstart = scalar(file.id, path, "DRM_Metadata/tstart");
  f.tend = scalar(file.id, path, "DRM_Metadata/tend");
  if (!(f.dt > 0.0)) fail(path, "DRM_Metadata/dt must be positive");
  if (!(f.tend >= f.tstart)) fail(path, "DRM_Metadata/tend must not be before tstart");
  const auto x0 = read_all<double>(file.id, path, "DRM_Metadata/drmbox_x0", H5T_NATIVE_DOUBLE, dims);
  if (x0.size() != 3) fail(path, "DRM_Metadata/drmbox_x0 must have 3 values");
  for (int i = 0; i < 3; ++i) f.x0[i] = x0[static_cast<std::size_t>(i)];
  const char* axis[3] = {"x", "y", "z"};
  for (int i = 0; i < 3; ++i) {
    f.box_min[i] = scalar(file.id, path, std::string("DRM_Metadata/drmbox_") + axis[i] + "min");
    f.box_max[i] = scalar(file.id, path, std::string("DRM_Metadata/drmbox_") + axis[i] + "max");
  }
  std::vector<hsize_t> dd, da;
  for (const char* name : {"DRM_Data/displacement", "DRM_Data/acceleration"}) {
    if (H5Lexists(file.id, name, H5P_DEFAULT) <= 0) fail(path, std::string("missing dataset ") + name);
    Id d(H5Dopen2(file.id, name, H5P_DEFAULT), H5Dclose);
    (std::string(name).find("displacement") != std::string::npos ? dd : da) = dims_of(d.id);
  }
  if (dd.size() != 2) fail(path, "DRM_Data/displacement must be [rows, samples]");
  if (dd != da) fail(path, "DRM_Data/displacement and DRM_Data/acceleration must have the same shape");
  f.rows = static_cast<std::int64_t>(dd[0]);
  f.samples = static_cast<std::int64_t>(dd[1]);
  for (std::size_t s = 0; s < S; ++s) {
    const auto r = f.data_location[s];
    if (r < 0 || static_cast<std::int64_t>(r) + 3 > f.rows) {
      fail(path, "station " + std::to_string(s) + "'s data_location " + std::to_string(r) +
                     " is outside the histories' " + std::to_string(f.rows) + " rows");
    }
  }
  return f;
}

void read_drm_block(const std::string& path, const std::string& name, std::int64_t row_begin, std::int64_t row_count,
                    std::int64_t sample_begin, std::int64_t sample_count, std::vector<double>& out) {
  Id file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose);
  if (file.id < 0) fail(path, "cannot open as an HDF5 file");
  const std::string ds = "DRM_Data/" + name;
  Id d(H5Dopen2(file.id, ds.c_str(), H5P_DEFAULT), H5Dclose);
  if (d.id < 0) fail(path, "cannot open " + ds);
  Id space(H5Dget_space(d.id), H5Sclose);
  const hsize_t start[2] = {static_cast<hsize_t>(row_begin), static_cast<hsize_t>(sample_begin)};
  const hsize_t count[2] = {static_cast<hsize_t>(row_count), static_cast<hsize_t>(sample_count)};
  if (H5Sselect_hyperslab(space.id, H5S_SELECT_SET, start, nullptr, count, nullptr) < 0) {
    fail(path, "bad block of " + ds);
  }
  Id mem(H5Screate_simple(2, count, nullptr), H5Sclose);
  out.resize(static_cast<std::size_t>(row_count * sample_count));
  if (H5Dread(d.id, H5T_NATIVE_DOUBLE, mem.id, space.id, H5P_DEFAULT, out.data()) < 0) {
    fail(path, "cannot read a block of " + ds);
  }
}

}  // namespace femorax::io
