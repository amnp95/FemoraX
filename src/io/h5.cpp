#include "io/h5.hpp"

#include <hdf5.h>

#include <algorithm>
#include <cstring>

static_assert(sizeof(hid_t) == sizeof(femorax::io::h5::hid), "hid_t is 64-bit since HDF5 1.10");

namespace femorax::io::h5 {

namespace {

// HDF5 prints its own error stack to stderr by default; the reader reports errors itself.
struct SilenceHdf5 {
  SilenceHdf5() { H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr); }
};
const SilenceHdf5 silence_hdf5;

enum class Kind { Int32, UInt8, Float64, String };

const char* kind_name(Kind k) {
  switch (k) {
    case Kind::Int32: return "int32";
    case Kind::UInt8: return "uint8";
    case Kind::Float64: return "float64";
    case Kind::String: return "string";
  }
  return "?";
}

std::string describe_type(hid_t t) {
  const H5T_class_t c = H5Tget_class(t);
  const std::size_t size = H5Tget_size(t);
  switch (c) {
    case H5T_INTEGER:
      return std::string(H5Tget_sign(t) == H5T_SGN_NONE ? "uint" : "int") +
             std::to_string(8 * size);
    case H5T_FLOAT: return "float" + std::to_string(8 * size);
    case H5T_STRING: return "string";
    default: return "type class " + std::to_string(static_cast<int>(c));
  }
}

bool type_matches(hid_t t, Kind k) {
  const H5T_class_t c = H5Tget_class(t);
  const std::size_t size = H5Tget_size(t);
  switch (k) {
    case Kind::Int32: return c == H5T_INTEGER && size == 4 && H5Tget_sign(t) == H5T_SGN_2;
    case Kind::UInt8: return c == H5T_INTEGER && size == 1 && H5Tget_sign(t) == H5T_SGN_NONE;
    case Kind::Float64: return c == H5T_FLOAT && size == 8;
    case Kind::String: return c == H5T_STRING;
  }
  return false;
}

void check_type(hid_t obj_type, Kind k, const std::string& what) {
  if (!type_matches(obj_type, k)) {
    throw ModelError(what + " must be " + kind_name(k) + ", found " + describe_type(obj_type));
  }
}

std::vector<std::int64_t> space_shape(hid_t space) {
  const int rank = H5Sget_simple_extent_ndims(space);
  std::vector<hsize_t> dims(static_cast<std::size_t>(std::max(rank, 0)));
  if (rank > 0) H5Sget_simple_extent_dims(space, dims.data(), nullptr);
  return {dims.begin(), dims.end()};
}

std::int64_t element_count(const std::vector<std::int64_t>& shape) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  return n;
}

// H5L_info_t / H5Literate map to the right versioned names on every HDF5 >= 1.10.
herr_t collect_link(hid_t, const char* name, const H5L_info_t*, void* out) {
  static_cast<std::vector<std::string>*>(out)->emplace_back(name);
  return 0;
}

herr_t collect_attr(hid_t, const char* name, const H5A_info_t*, void* out) {
  static_cast<std::vector<std::string>*>(out)->emplace_back(name);
  return 0;
}

std::string read_string_attr(hid_t attr, const std::string& what) {
  Handle type(H5Aget_type(attr));
  check_type(type.get(), Kind::String, what);
  Handle space(H5Aget_space(attr));
  if (element_count(space_shape(space.get())) != 1) throw ModelError(what + " must be a scalar");
  if (H5Tis_variable_str(type.get()) > 0) {
    Handle mem(H5Tcopy(H5T_C_S1));
    H5Tset_size(mem.get(), H5T_VARIABLE);
    H5Tset_cset(mem.get(), H5Tget_cset(type.get()));
    char* buf = nullptr;
    if (H5Aread(attr, mem.get(), &buf) < 0) throw ModelError("cannot read " + what);
    std::string s = buf ? buf : "";
    H5free_memory(buf);
    return s;
  }
  const std::size_t size = H5Tget_size(type.get());
  std::string s(size, '\0');
  Handle mem(H5Tcopy(H5T_C_S1));
  H5Tset_size(mem.get(), size);
  H5Tset_cset(mem.get(), H5Tget_cset(type.get()));
  H5Tset_strpad(mem.get(), H5T_STR_NULLPAD);
  if (H5Aread(attr, mem.get(), s.data()) < 0) throw ModelError("cannot read " + what);
  s.resize(std::strlen(s.c_str()));
  return s;
}

template <class T>
std::vector<T> read_attr_values(hid_t attr, Kind k, hid_t native, const std::string& what,
                                int expected_count) {
  Handle type(H5Aget_type(attr));
  check_type(type.get(), k, what);
  Handle space(H5Aget_space(attr));
  const auto shape = space_shape(space.get());
  const std::int64_t n = element_count(shape);
  if (expected_count == 0) {
    if (!shape.empty() && n != 1) throw ModelError(what + " must be a scalar");
  } else if (shape.size() != 1 || n != expected_count) {
    throw ModelError(what + " must be an array of " + std::to_string(expected_count) +
                     " values");
  }
  std::vector<T> v(static_cast<std::size_t>(n));
  if (H5Aread(attr, native, v.data()) < 0) throw ModelError("cannot read " + what);
  return v;
}

}  // namespace

// ---------------------------------------------------------------------------
Handle::~Handle() {
  if (id_ < 0) return;
  switch (H5Iget_type(id_)) {
    case H5I_FILE: H5Fclose(id_); break;
    case H5I_GROUP: H5Gclose(id_); break;
    case H5I_DATASET: H5Dclose(id_); break;
    case H5I_ATTR: H5Aclose(id_); break;
    case H5I_DATASPACE: H5Sclose(id_); break;
    case H5I_DATATYPE: H5Tclose(id_); break;
    default: break;
  }
}

Handle& Handle::operator=(Handle&& o) noexcept {
  if (this != &o) {
    Handle tmp(id_);  // closes the old id
    id_ = o.id_;
    o.id_ = -1;
  }
  return *this;
}

// ---------------------------------------------------------------------------
File::File(const std::string& filename) : filename_(filename) {
#if H5_VERSION_GE(1, 12, 0)
  const htri_t is_hdf5 = H5Fis_accessible(filename.c_str(), H5P_DEFAULT);
#else
  const htri_t is_hdf5 = H5Fis_hdf5(filename.c_str());
#endif
  if (is_hdf5 <= 0) {
    throw ModelError("cannot open '" + filename + "' as an HDF5 file");
  }
  h_ = Handle(H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
  if (!h_.valid()) throw ModelError("cannot open '" + filename + "'");
}

Group File::root() {
  Handle g(H5Gopen2(h_.get(), "/", H5P_DEFAULT));
  return Group(std::move(g), "");
}

// ---------------------------------------------------------------------------
Group::Group(Handle h, std::string path) : h_(std::move(h)), path_(std::move(path)) {}

std::string Group::item(const std::string& name) const { return path_ + "/" + name; }

std::string Group::attr_item(const std::string& name) const {
  return (path_.empty() ? "/" : path_) + " attribute '" + name + "'";
}

std::vector<std::string> Group::children() const {
  std::vector<std::string> names;
  hsize_t idx = 0;
  H5Literate(h_.get(), H5_INDEX_NAME, H5_ITER_INC, &idx, collect_link, &names);
  std::sort(names.begin(), names.end());
  return names;
}

bool Group::has_child(const std::string& name) const {
  return H5Lexists(h_.get(), name.c_str(), H5P_DEFAULT) > 0;
}

int Group::object_type(const std::string& name) const {
  // Opening the object and asking its identifier type works on every HDF5 version.
  Handle o(H5Oopen(h_.get(), name.c_str(), H5P_DEFAULT));
  return o.valid() ? static_cast<int>(H5Iget_type(o.get())) : -1;
}

bool Group::has_attr(const std::string& name) const {
  return H5Aexists(h_.get(), name.c_str()) > 0;
}

Group Group::group(const std::string& name) {
  if (!has_child(name)) throw ModelError("missing group " + item(name));
  if (object_type(name) != H5I_GROUP) throw ModelError(item(name) + " must be a group");
  read_children_.insert(name);
  Handle g(H5Gopen2(h_.get(), name.c_str(), H5P_DEFAULT));
  return Group(std::move(g), item(name));
}

std::optional<Group> Group::optional_group(const std::string& name) {
  if (!has_child(name)) return std::nullopt;
  return group(name);
}

Handle Group::open_attr(const std::string& name) {
  if (!has_attr(name)) throw ModelError("missing " + attr_item(name));
  read_attrs_.insert(name);
  return Handle(H5Aopen(h_.get(), name.c_str(), H5P_DEFAULT));
}

std::int32_t Group::attr_int(const std::string& name) {
  Handle a = open_attr(name);
  return read_attr_values<std::int32_t>(a.get(), Kind::Int32, H5T_NATIVE_INT32,
                                        attr_item(name), 0)[0];
}

double Group::attr_double(const std::string& name) {
  Handle a = open_attr(name);
  return read_attr_values<double>(a.get(), Kind::Float64, H5T_NATIVE_DOUBLE, attr_item(name),
                                  0)[0];
}

bool Group::attr_bool(const std::string& name) {
  Handle a = open_attr(name);
  const auto v = read_attr_values<std::uint8_t>(a.get(), Kind::UInt8, H5T_NATIVE_UINT8,
                                                attr_item(name), 0)[0];
  if (v > 1) throw ModelError(attr_item(name) + " must be 0 or 1, found " + std::to_string(v));
  return v == 1;
}

std::string Group::attr_string(const std::string& name) {
  Handle a = open_attr(name);
  return read_string_attr(a.get(), attr_item(name));
}

std::vector<std::int32_t> Group::attr_int_array(const std::string& name, int expected_size) {
  Handle a = open_attr(name);
  return read_attr_values<std::int32_t>(a.get(), Kind::Int32, H5T_NATIVE_INT32,
                                        attr_item(name), expected_size);
}

std::vector<double> Group::attr_double_array(const std::string& name, int expected_size) {
  Handle a = open_attr(name);
  return read_attr_values<double>(a.get(), Kind::Float64, H5T_NATIVE_DOUBLE, attr_item(name),
                                  expected_size);
}

std::optional<std::int32_t> Group::opt_attr_int(const std::string& name) {
  if (!has_attr(name)) return std::nullopt;
  return attr_int(name);
}
std::optional<double> Group::opt_attr_double(const std::string& name) {
  if (!has_attr(name)) return std::nullopt;
  return attr_double(name);
}
std::optional<bool> Group::opt_attr_bool(const std::string& name) {
  if (!has_attr(name)) return std::nullopt;
  return attr_bool(name);
}
std::optional<std::string> Group::opt_attr_string(const std::string& name) {
  if (!has_attr(name)) return std::nullopt;
  return attr_string(name);
}

Handle Group::open_dataset(const std::string& name, int cols, std::vector<std::int64_t>& shape) {
  if (!has_child(name)) throw ModelError("missing dataset " + item(name));
  if (object_type(name) != H5I_DATASET) throw ModelError(item(name) + " must be a dataset");
  read_children_.insert(name);
  Handle d(H5Dopen2(h_.get(), name.c_str(), H5P_DEFAULT));
  Handle space(H5Dget_space(d.get()));
  shape = space_shape(space.get());
  const bool ok = cols == 0 ? shape.size() == 1 : (shape.size() == 2 && shape[1] == cols);
  if (!ok) {
    std::string got = "[";
    for (std::size_t i = 0; i < shape.size(); ++i) got += (i ? ", " : "") + std::to_string(shape[i]);
    got += "]";
    throw ModelError(item(name) + " must have shape " +
                     (cols == 0 ? std::string("[n]") : "[n, " + std::to_string(cols) + "]") +
                     ", found " + got);
  }
  return d;
}

Array<std::int32_t> Group::dataset_int(const std::string& name, int cols) {
  Array<std::int32_t> a;
  Handle d = open_dataset(name, cols, a.shape);
  Handle type(H5Dget_type(d.get()));
  check_type(type.get(), Kind::Int32, item(name));
  a.data.resize(static_cast<std::size_t>(element_count(a.shape)));
  if (!a.data.empty() &&
      H5Dread(d.get(), H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.data.data()) < 0) {
    throw ModelError("cannot read " + item(name));
  }
  return a;
}

Array<double> Group::dataset_double(const std::string& name, int cols) {
  Array<double> a;
  Handle d = open_dataset(name, cols, a.shape);
  Handle type(H5Dget_type(d.get()));
  check_type(type.get(), Kind::Float64, item(name));
  a.data.resize(static_cast<std::size_t>(element_count(a.shape)));
  if (!a.data.empty() &&
      H5Dread(d.get(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, a.data.data()) < 0) {
    throw ModelError("cannot read " + item(name));
  }
  return a;
}

std::optional<Array<std::int32_t>> Group::opt_dataset_int(const std::string& name, int cols) {
  if (!has_child(name)) return std::nullopt;
  return dataset_int(name, cols);
}

std::optional<Array<double>> Group::opt_dataset_double(const std::string& name, int cols) {
  if (!has_child(name)) return std::nullopt;
  return dataset_double(name, cols);
}

void Group::finish() const {
  std::vector<std::string> unknown;
  std::vector<std::string> attrs;
  hsize_t idx = 0;
  H5Aiterate2(h_.get(), H5_INDEX_NAME, H5_ITER_INC, &idx, collect_attr, &attrs);
  for (const auto& a : attrs)
    if (!read_attrs_.count(a)) unknown.push_back("attribute '" + a + "'");
  for (const auto& c : children())
    if (!read_children_.count(c)) unknown.push_back("'" + c + "'");
  if (unknown.empty()) return;
  std::string msg = "unknown item(s) in " + (path_.empty() ? std::string("/") : path_) + ": ";
  for (std::size_t i = 0; i < unknown.size(); ++i) msg += (i ? ", " : "") + unknown[i];
  msg += " (not part of schema v0)";
  throw ModelError(msg);
}

}  // namespace femorax::io::h5
