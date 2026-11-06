#pragma once

// Reader for OpenSees reference data written by tests/reference/drivers
// (format: tests/reference/drivers/driver_common.hpp).

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace femorax::test {

// Commit the reference data must come from (docs/dependencies.md).
inline constexpr const char* opensees_pinned_commit = "955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea";

inline std::filesystem::path reference_data_dir() { return FEMORAX_REFERENCE_DATA; }

struct ReferenceFile {
  std::string path;
  std::map<std::string, std::string> header;  // first word -> rest of the line
  std::vector<std::string> columns;
  std::vector<std::vector<double>> rows;

  int column(const std::string& name) const {
    const auto it = std::find(columns.begin(), columns.end(), name);
    if (it == columns.end()) throw std::runtime_error(path + ": no column " + name);
    return static_cast<int>(it - columns.begin());
  }

  // Header "coords 0 0 0 1 0 0 ..." -> values("coords") == {0, 0, 0, 1, 0, 0, ...}.
  std::vector<double> values(const std::string& key) const {
    std::istringstream in(header.at(key));
    std::vector<double> v;
    for (std::string x; in >> x;) v.push_back(std::strtod(x.c_str(), nullptr));
    return v;
  }

  // Header "params E 1 nu 0.3" -> param("E") == 1.
  double param(const std::string& name) const {
    std::istringstream in(header.at("params"));
    for (std::string k, v; in >> k >> v;)
      if (k == name) return std::strtod(v.c_str(), nullptr);
    throw std::runtime_error(path + ": no parameter " + name);
  }
};

inline ReferenceFile read_reference(const std::filesystem::path& p) {
  ReferenceFile f;
  f.path = p.string();
  std::ifstream in(p);
  if (!in) throw std::runtime_error("cannot read " + f.path);
  for (std::string line; std::getline(in, line);) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      std::istringstream h(line.substr(1));
      std::string key;
      h >> key;
      std::string rest;
      std::getline(h, rest);
      rest.erase(0, rest.find_first_not_of(' '));
      if (key == "columns") {
        std::istringstream c(rest);
        for (std::string name; c >> name;) f.columns.push_back(name);
      } else {
        f.header[key] = rest;
      }
      continue;
    }
    std::istringstream r(line);
    std::vector<double> row;
    for (std::string v; r >> v;) row.push_back(std::strtod(v.c_str(), nullptr));
    if (row.size() != f.columns.size()) throw std::runtime_error(f.path + ": bad row width");
    f.rows.push_back(std::move(row));
  }
  return f;
}

inline std::vector<std::filesystem::path> reference_files(const std::string& subdir,
                                                          const std::string& prefix) {
  std::vector<std::filesystem::path> out;
  const auto dir = reference_data_dir() / subdir;
  if (!std::filesystem::exists(dir)) return out;
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    const auto name = e.path().filename().string();
    if (name.rfind(prefix, 0) == 0 && e.path().extension() == ".txt") out.push_back(e.path());
  }
  std::sort(out.begin(), out.end());
  return out;
}

// |got - want| <= rel * scale, where scale is the largest magnitude in the compared set
// (so entries that are zero in one result are compared at the set's own magnitude).
inline double max_abs(const double* v, int n) {
  double m = 0.0;
  for (int i = 0; i < n; ++i) m = std::max(m, std::abs(v[i]));
  return m;
}

}  // namespace femorax::test
