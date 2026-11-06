#pragma once

#include <string>

#include "io/h5.hpp"
#include "io/model.hpp"

namespace femorax::io {

// Reads and validates a schema v0 model file (schema/v0/README.md).
//
// Throws UnsupportedComponent when the file is valid but uses something FemoraX cannot
// run (the message names it and suggests the OpenSees backend), and ModelError for
// anything else wrong with the file: wrong types or shapes, missing or unknown items,
// broken references, duplicate tags. Non-fatal findings go to Model::warnings.
//
// This checks the file against the schema only. Whether the current build can run it
// (milestone limits, explicit-stage rules) is check_supported() in io/support.hpp.
Model read_model(const std::string& filename);

}  // namespace femorax::io
