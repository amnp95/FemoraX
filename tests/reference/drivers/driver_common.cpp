#include "driver_common.hpp"

#include <cstdlib>

namespace refdrv {

Writer::Writer(const std::string& path, const std::vector<std::string>& header,
               const std::vector<std::string>& columns)
    : f_(std::fopen(path.c_str(), "w")), ncols_(columns.size()) {
  if (!f_) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    std::exit(1);
  }
  std::fprintf(f_, "# femorax-reference 1\n# opensees_commit %s\n", OPENSEES_COMMIT);
  for (const auto& h : header) std::fprintf(f_, "# %s\n", h.c_str());
  std::fprintf(f_, "# columns");
  for (const auto& c : columns) std::fprintf(f_, " %s", c.c_str());
  std::fprintf(f_, "\n");
}

Writer::~Writer() { std::fclose(f_); }

void Writer::row(const std::vector<double>& values) {
  if (values.size() != ncols_) {
    std::fprintf(stderr, "row has %zu values, expected %zu\n", values.size(), ncols_);
    std::exit(1);
  }
  for (std::size_t i = 0; i < values.size(); ++i)
    std::fprintf(f_, i ? " %.17g" : "%.17g", values[i]);
  std::fprintf(f_, "\n");
}

}  // namespace refdrv
