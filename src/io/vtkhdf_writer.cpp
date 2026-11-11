// Port of the file layout of OpenSees' VTKHDF recorder: SRC/recorder/VTKHDF_Recorder.cpp,
// OpenSees commit 955ceb0 (constructor, writeMesh, writeStep, writeDisp/Vel/Accel,
// createDataset, createOffsetDataset, extendDataset, extendOffsetDataset). Dataset names,
// element types, chunk shapes and the order of the per-step entries follow that file.

#include "io/vtkhdf_writer.hpp"

#include <hdf5.h>

#include <cstring>

#include "io/h5.hpp"

namespace femorax::io {

namespace {

[[noreturn]] void fail(const std::string& what) { throw ModelError("VTKHDF writer: " + what); }

void check(herr_t rc, const std::string& what) {
  if (rc < 0) fail(what);
}

hid_t checked(hid_t id, const std::string& what) {
  if (id < 0) fail(what);
  return id;
}

// createDataset / createOffsetDataset: empty, unlimited rows, chunked.
void create(hid_t group, const char* name, hid_t type, hsize_t ncols, hsize_t chunk_rows) {
  const int ndims = ncols == 0 ? 1 : 2;
  hsize_t dims[2] = {0, ncols}, maxdims[2] = {H5S_UNLIMITED, ncols}, chunk[2] = {chunk_rows, ncols};
  const hid_t space = checked(H5Screate_simple(ndims, dims, maxdims), std::string("dataspace ") + name);
  const hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_chunk(plist, ndims, chunk);
  const hid_t d = H5Dcreate2(group, name, type, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  H5Pclose(plist);
  H5Sclose(space);
  if (d < 0) fail(std::string("cannot create dataset ") + name);
  H5Dclose(d);
}

// Append `rows` rows (of the dataset's column count) at the end of a dataset.
void append(hid_t group, const char* name, hid_t mem_type, const void* data, hsize_t rows) {
  const hid_t d = checked(H5Dopen2(group, name, H5P_DEFAULT), std::string("cannot open dataset ") + name);
  hid_t space = H5Dget_space(d);
  const int ndims = H5Sget_simple_extent_ndims(space);
  hsize_t cur[2] = {0, 0};
  H5Sget_simple_extent_dims(space, cur, nullptr);
  H5Sclose(space);
  hsize_t next[2] = {cur[0] + rows, cur[1]};
  check(H5Dset_extent(d, next), std::string("cannot extend ") + name);
  space = H5Dget_space(d);
  hsize_t start[2] = {cur[0], 0}, count[2] = {rows, cur[1]};
  H5Sselect_hyperslab(space, H5S_SELECT_SET, start, nullptr, count, nullptr);
  const hid_t mem = H5Screate_simple(ndims, count, nullptr);
  const herr_t rc = H5Dwrite(d, mem_type, mem, space, H5P_DEFAULT, data);
  H5Sclose(mem);
  H5Sclose(space);
  H5Dclose(d);
  check(rc, std::string("cannot write ") + name);
}

void int_attr(hid_t obj, const char* name, const int* values, hsize_t n) {
  const hid_t space = H5Screate_simple(1, &n, nullptr);  // 1-D even for one value, as OpenSees
  const hid_t a = checked(H5Acreate2(obj, name, H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT), name);
  H5Awrite(a, H5T_NATIVE_INT, values);
  H5Aclose(a);
  H5Sclose(space);
}

}  // namespace

struct VtkhdfWriter::Impl {
  hid_t file = -1, root = -1, point_data = -1, cell_data = -1, steps = -1, point_offsets = -1,
        cell_offsets = -1;
  std::vector<std::string> fields, cell_fields;
  std::vector<int> field_offset;  // CurrentDispOffset, CurrentVelOffset, ...
  std::vector<int> cell_offset;   // current_Stress3D6Offset, current_Strain3D6Offset
  int num_steps = 0;
  int num_node = 0, num_element = 0, num_conn = 0;
  bool mesh = false;
};

VtkhdfWriter::VtkhdfWriter(const std::string& path, const std::vector<std::string>& fields,
                           const std::vector<std::string>& cell_fields, bool single_precision)
    : impl_(new Impl) {
  const hid_t response_type = single_precision ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
  auto& s = *impl_;
  s.fields = fields;
  s.field_offset.assign(fields.size(), 0);
  s.cell_fields = cell_fields;
  s.cell_offset.assign(cell_fields.size(), 0);
  // OpenSees: H5F_LIBVER_LATEST for both bounds (SWMR-compatible format).
  const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
  s.file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  H5Pclose(fapl);
  if (s.file < 0) {
    delete impl_;
    impl_ = nullptr;
    fail("cannot create " + path);
  }
  s.root = checked(H5Gcreate2(s.file, "/VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "/VTKHDF");
  const int version[2] = {2, 3};
  int_attr(s.root, "Version", version, 2);
  {
    // Fixed-length string of strlen("UnstructuredGrid") bytes, NULLTERM pad, as OpenSees.
    const char* type = "UnstructuredGrid";
    const hid_t space = H5Screate(H5S_SCALAR);
    const hid_t st = H5Tcopy(H5T_C_S1);
    H5Tset_size(st, std::strlen(type));
    H5Tset_strpad(st, H5T_STR_NULLTERM);
    const hid_t a = checked(H5Acreate2(s.root, "Type", st, space, H5P_DEFAULT, H5P_DEFAULT), "Type");
    H5Awrite(a, st, type);
    H5Aclose(a);
    H5Tclose(st);
    H5Sclose(space);
  }
  s.point_data = checked(H5Gcreate2(s.root, "/VTKHDF/PointData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "PointData");
  s.cell_data = checked(H5Gcreate2(s.root, "/VTKHDF/CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "CellData");
  create(s.root, "Points", H5T_NATIVE_DOUBLE, 3, 100);
  create(s.root, "Connectivity", H5T_NATIVE_INT, 0, 100);
  create(s.root, "Offsets", H5T_NATIVE_INT, 0, 100);
  create(s.root, "Types", H5T_NATIVE_UCHAR, 0, 100);

  s.steps = checked(H5Gcreate2(s.root, "/VTKHDF/Steps", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "Steps");
  const int zero = 0;
  int_attr(s.steps, "NSteps", &zero, 1);
  create(s.root, "NumberOfPoints", H5T_NATIVE_INT, 0, 1);
  create(s.root, "NumberOfCells", H5T_NATIVE_INT, 0, 1);
  create(s.root, "NumberOfConnectivityIds", H5T_NATIVE_INT, 0, 1);
  create(s.steps, "PointOffsets", H5T_NATIVE_INT, 0, 1);
  create(s.steps, "Values", H5T_NATIVE_DOUBLE, 0, 1);
  create(s.steps, "PartOffsets", H5T_NATIVE_INT, 0, 1);
  create(s.steps, "NumberOfParts", H5T_NATIVE_INT, 0, 100);
  create(s.steps, "CellOffsets", H5T_NATIVE_INT, 1, 1);
  create(s.steps, "ConnectivityIdOffsets", H5T_NATIVE_INT, 1, 1);
  s.point_offsets = checked(H5Gcreate2(s.steps, "PointDataOffsets", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "PointDataOffsets");
  s.cell_offsets = checked(H5Gcreate2(s.steps, "CellDataOffsets", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "CellDataOffsets");
  H5Gclose(checked(H5Gcreate2(s.steps, "FieldDataOffsets", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "FieldDataOffsets"));
  H5Gclose(checked(H5Gcreate2(s.steps, "FieldDataSizes", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "FieldDataSizes"));

  for (const auto& f : fields) {
    create(s.point_offsets, f.c_str(), H5T_NATIVE_INT, 0, 100);
    create(s.point_data, f.c_str(), response_type, 3, 100);
  }
  for (const auto& f : cell_fields) {  // stress3D6, strain3D6: [rows, 6], chunks of 500 rows
    create(s.cell_data, f.c_str(), response_type, 6, 500);
    create(s.cell_offsets, f.c_str(), H5T_NATIVE_INT, 0, 500);
  }
  create(s.point_data, "nodeTags", H5T_NATIVE_INT, 0, 500);
  create(s.point_offsets, "nodeTags", H5T_NATIVE_INT, 0, 500);
  create(s.cell_data, "eleTags", H5T_NATIVE_INT, 0, 500);
  create(s.cell_offsets, "eleTags", H5T_NATIVE_INT, 0, 500);
  H5Fflush(s.file, H5F_SCOPE_GLOBAL);
}

VtkhdfWriter::~VtkhdfWriter() {
  close();
  delete impl_;
}

void VtkhdfWriter::write_mesh(const VtkhdfMesh& m) {
  auto& s = *impl_;
  s.num_node = static_cast<int>(m.node_tags.size());
  s.num_element = static_cast<int>(m.ele_tags.size());
  s.num_conn = static_cast<int>(m.connectivity.size());
  s.mesh = true;
  if (s.num_node == 0 && s.num_element == 0) return;  // OpenSees writes nothing then
  append(s.root, "Points", H5T_NATIVE_DOUBLE, m.points.data(), static_cast<hsize_t>(s.num_node));
  append(s.root, "Connectivity", H5T_NATIVE_INT, m.connectivity.data(), static_cast<hsize_t>(s.num_conn));
  append(s.root, "Offsets", H5T_NATIVE_INT, m.offsets.data(), static_cast<hsize_t>(m.offsets.size()));
  append(s.root, "Types", H5T_NATIVE_UCHAR, m.types.data(), static_cast<hsize_t>(s.num_element));
  append(s.point_data, "nodeTags", H5T_NATIVE_INT, m.node_tags.data(), static_cast<hsize_t>(s.num_node));
  append(s.cell_data, "eleTags", H5T_NATIVE_INT, m.ele_tags.data(), static_cast<hsize_t>(s.num_element));
  H5Fflush(s.file, H5F_SCOPE_GLOBAL);
}

void VtkhdfWriter::write_step(double time, const std::vector<const double*>& values) {
  auto& s = *impl_;
  // writeStep
  ++s.num_steps;
  {
    const hid_t a = H5Aopen(s.steps, "NSteps", H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_INT, &s.num_steps);
    H5Aclose(a);
  }
  // The mesh never changes, so every step points at offset 0 (current_*Offset).
  const int one = 1, zero = 0;
  append(s.steps, "NumberOfParts", H5T_NATIVE_INT, &one, 1);
  append(s.steps, "PointOffsets", H5T_NATIVE_INT, &zero, 1);
  append(s.steps, "PartOffsets", H5T_NATIVE_INT, &zero, 1);
  append(s.steps, "Values", H5T_NATIVE_DOUBLE, &time, 1);
  append(s.root, "NumberOfPoints", H5T_NATIVE_INT, &s.num_node, 1);
  append(s.root, "NumberOfCells", H5T_NATIVE_INT, &s.num_element, 1);
  append(s.root, "NumberOfConnectivityIds", H5T_NATIVE_INT, &s.num_conn, 1);
  append(s.steps, "CellOffsets", H5T_NATIVE_INT, &zero, 1);
  append(s.steps, "ConnectivityIdOffsets", H5T_NATIVE_INT, &zero, 1);
  append(s.point_offsets, "nodeTags", H5T_NATIVE_INT, &zero, 1);
  append(s.cell_offsets, "eleTags", H5T_NATIVE_INT, &zero, 1);
  // writeDisp, writeVel, writeAccel
  for (std::size_t f = 0; f < s.fields.size(); ++f) {
    append(s.point_data, s.fields[f].c_str(), H5T_NATIVE_DOUBLE, values[f], static_cast<hsize_t>(s.num_node));
    append(s.point_offsets, s.fields[f].c_str(), H5T_NATIVE_INT, &s.field_offset[f], 1);
    s.field_offset[f] += s.num_node;
  }
  // writeStrees3D6, writeStrain3D6
  for (std::size_t f = 0; f < s.cell_fields.size(); ++f) {
    append(s.cell_data, s.cell_fields[f].c_str(), H5T_NATIVE_DOUBLE, values[s.fields.size() + f],
           static_cast<hsize_t>(s.num_element));
    append(s.cell_offsets, s.cell_fields[f].c_str(), H5T_NATIVE_INT, &s.cell_offset[f], 1);
    s.cell_offset[f] += s.num_element;
  }
  H5Fflush(s.file, H5F_SCOPE_GLOBAL);
}

void VtkhdfWriter::close() {
  if (!impl_ || impl_->file < 0) return;
  auto& s = *impl_;
  for (hid_t g : {s.point_offsets, s.cell_offsets, s.steps, s.point_data, s.cell_data, s.root})
    if (g >= 0) H5Gclose(g);
  H5Fclose(s.file);
  s.file = -1;
}

}  // namespace femorax::io
