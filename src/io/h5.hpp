#pragma once

// Thin, strict wrapper over the HDF5 C API for the model reader.
//
// A Group remembers every attribute and child it has been asked for; finish() throws if
// the file holds anything that was never read. That is how the reader rejects unknown
// items (schema v0, "Strictness") without listing them twice.
//
// Type rules (schema v0, "Conventions"): integers are 32-bit signed, booleans 8-bit
// unsigned (0 or 1), reals 64-bit floats, strings UTF-8 or ASCII, fixed or variable
// length. Anything else is an error that names the item and the type found.

#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace femorax::io {

// A model that breaks the schema (wrong type, missing item, bad reference, ...).
class ModelError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// A valid model that uses something FemoraX does not support (yet).
class UnsupportedComponent : public ModelError {
 public:
  using ModelError::ModelError;
};

namespace h5 {

using hid = std::int64_t;  // hid_t, without including hdf5.h in this header

// Owns one HDF5 identifier.
class Handle {
 public:
  Handle() = default;
  explicit Handle(hid id) : id_(id) {}
  ~Handle();
  Handle(Handle&& o) noexcept : id_(o.id_) { o.id_ = -1; }
  Handle& operator=(Handle&& o) noexcept;
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  hid get() const { return id_; }
  bool valid() const { return id_ >= 0; }

 private:
  hid id_ = -1;
};

// A 1-D or 2-D dataset read in full. shape is {n} or {n, k}.
template <class T>
struct Array {
  std::vector<T> data;
  std::vector<std::int64_t> shape;
  std::int64_t rows() const { return shape.empty() ? 0 : shape[0]; }
};

class Group {
 public:
  Group(Handle h, std::string path);
  Group(Group&&) = default;
  Group& operator=(Group&&) = default;

  const std::string& path() const { return path_; }

  // Names of child groups and datasets, sorted (marks nothing as read).
  std::vector<std::string> children() const;
  bool has_child(const std::string& name) const;
  bool has_attr(const std::string& name) const;

  // Child group. Marks it as read.
  Group group(const std::string& name);
  std::optional<Group> optional_group(const std::string& name);

  // Attributes. Each marks the attribute as read.
  std::int32_t attr_int(const std::string& name);
  double attr_double(const std::string& name);
  bool attr_bool(const std::string& name);
  std::string attr_string(const std::string& name);
  std::vector<std::int32_t> attr_int_array(const std::string& name, int expected_size);
  std::vector<double> attr_double_array(const std::string& name, int expected_size);

  std::optional<std::int32_t> opt_attr_int(const std::string& name);
  std::optional<double> opt_attr_double(const std::string& name);
  std::optional<bool> opt_attr_bool(const std::string& name);
  std::optional<std::string> opt_attr_string(const std::string& name);

  // Datasets, 1-D ([n]) or 2-D ([n, cols]); cols = 0 means 1-D. Mark as read.
  Array<std::int32_t> dataset_int(const std::string& name, int cols = 0);
  Array<double> dataset_double(const std::string& name, int cols = 0);
  std::optional<Array<std::int32_t>> opt_dataset_int(const std::string& name, int cols = 0);
  std::optional<Array<double>> opt_dataset_double(const std::string& name, int cols = 0);

  // Throws ModelError naming every attribute or child that was not read.
  void finish() const;

  // "<path>/<name>" or "<path> attribute '<name>'", for messages.
  std::string item(const std::string& name) const;
  std::string attr_item(const std::string& name) const;

 private:
  Handle open_attr(const std::string& name);
  int object_type(const std::string& name) const;  // H5I_type_t of a child, -1 if none
  Handle open_dataset(const std::string& name, int cols, std::vector<std::int64_t>& shape);

  Handle h_;
  std::string path_;
  std::set<std::string> read_attrs_;
  std::set<std::string> read_children_;
};

class File {
 public:
  explicit File(const std::string& filename);  // read-only
  Group root();

 private:
  Handle h_;
  std::string filename_;
};

}  // namespace h5
}  // namespace femorax::io
