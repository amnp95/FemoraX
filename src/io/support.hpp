#pragma once

#include <vector>

#include "io/model.hpp"

namespace femorax::io {

// What this build can run. The model reader checks a file against the schema; this
// checks a valid model against the build, so the schema can describe more than one
// milestone (schema v0 covers 1a and 1b).
struct Capabilities {
  std::vector<ElementType> elements = {ElementType::StdBrick, ElementType::SSPbrick};
  bool implicit = false;  // Newmark, Newton, load_const: milestone 1b
};

// Throws UnsupportedComponent naming the first thing the build cannot run, or ModelError
// for combinations OpenSees itself does not allow (an explicit integrator with an
// algorithm other than Linear). Rules: schema/v0/README.md, "Rules for explicit stages".
void check_supported(const Model& model, const Capabilities& caps);

}  // namespace femorax::io
