#pragma once

// Node and element renumbering for memory locality (docs/design.md, memory contract
// rule 5). Off keeps the file order: system tests against OpenSees run with it off at the
// tightest tolerance, and once with it on; benchmarks run with it on.

#include <cstdint>
#include <vector>

#include "io/model.hpp"

namespace femorax::compile {

enum class Renumbering {
  Off,  // file order
  RCM,  // reverse Cuthill-McKee on the node graph (elements and equalDOF ties as edges)
};

struct Ordering {
  // node_order[i] = model node index of internal node i; node_rank is its inverse.
  std::vector<std::int32_t> node_order, node_rank;
  // Global element index = position in the concatenation of model.elements groups in file
  // order. element_order lists them in the order kernels should visit them (by smallest
  // internal node index, ties by file position); element_rank is its inverse.
  std::vector<std::int32_t> element_order, element_rank;
};

Ordering compute_ordering(const io::Model& model, Renumbering mode);

// Largest |rank(a) - rank(b)| over node pairs that share an element (the node bandwidth);
// for tests and the compile report.
std::int32_t node_bandwidth(const io::Model& model, const std::vector<std::int32_t>& node_rank);

}  // namespace femorax::compile
