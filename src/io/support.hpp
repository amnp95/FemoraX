#pragma once

#include <vector>

#include "io/model.hpp"

namespace femorax::io {

// What this build can run. The model reader checks a file against the schema; this
// checks a valid model against the build, so the schema can describe more than one
// milestone (schema v0 covers 1a and 1b).
struct Capabilities {
  std::vector<ElementType> elements = {ElementType::StdBrick, ElementType::SSPbrick};
  bool implicit = true;  // Newmark, LoadControl, Newton, ModifiedNewton, load_const (1b)
  // Milestone 2 (schema 0.4), switched on as each part is ported and verified:
  bool pressure_depend_multi_yield = true;
  bool pressure_independ_multi_yield = true;
  bool material_stages = true;            // the update_material_stage action
  bool vtkhdf_element_responses = true;   // VTKHDF stress3D6, strain3D6
};

// Throws UnsupportedComponent naming the first thing the build cannot run, or ModelError
// for combinations OpenSees itself does not allow (an explicit integrator with an
// algorithm other than Linear). Rules: schema/v0/README.md, "Rules for explicit stages".
void check_supported(const Model& model, const Capabilities& caps);

}  // namespace femorax::io
