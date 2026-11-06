#pragma once

// DOF numbering (docs/design.md, "DOF numbering"): host code, once per analysis stage.
//
// - Any number of DOFs per node: node n's DOF d (0-based) is dof_offset[n] + d.
// - equalDOF rows (and the laminar boundary, which Femora writes as equalDOF) join DOFs;
//   fix marks DOFs; a joined group that contains a fixed DOF is fixed as a whole.
// - Each free group gets one equation; equations are numbered in the order of the group's
//   smallest DOF id, so the numbering follows the node order chosen by the caller.
//
// Constraint handling matches OpenSees' TransformationConstraintHandler (commit 955ceb0)
// wherever OpenSees resolves the constraints as written, and refuses the model where it
// silently does something else (docs/deviations.md, "Constraint handling"):
// - one retained node per constrained node (OpenSees applies only the first MP constraint
//   that names a node);
// - everything else is checked by emulating the handler (TransformationDOF_Group IDs,
//   addSP_Constraint, doneID in ascending tag order) and comparing with the numbering
//   above. Chains in ascending tag order with all DOFs, and fixed DOFs inside a fixed
//   group, are accepted; reversed or partial chains, and a fix on a constrained DOF whose
//   retained DOF is free, are refused.

#include <cstdint>
#include <vector>

#include "io/model.hpp"

namespace femorax::compile {

struct DofNumbering {
  // Internal node order: node i of the numbering is model node node_model_index[i].
  std::vector<std::int32_t> node_model_index;  // [N]
  std::vector<std::int32_t> node_tag;          // [N]
  std::vector<std::int32_t> ndf;               // [N]
  std::vector<std::int32_t> dof_offset;        // [N + 1]
  std::vector<std::int32_t> eq;                // [ndof]; -1 = fixed
  std::int32_t num_equations = 0;

  std::int32_t num_nodes() const { return static_cast<std::int32_t>(node_tag.size()); }
  std::int32_t num_dofs() const { return dof_offset.empty() ? 0 : dof_offset.back(); }
  // Equation of internal node i, DOF d (0-based); -1 if fixed.
  std::int32_t equation(std::int32_t i, std::int32_t d) const { return eq[dof_offset[i] + d]; }
};

// node_order[i] = index into model.nodes of the node that becomes internal node i
// (a permutation; identity = file order, renumbering off). Throws io::ModelError.
// check_opensees = false skips the comparison with the Transformation handler (tests).
DofNumbering number_dofs(const io::Model& model, const std::vector<std::int32_t>& node_order,
                         bool check_opensees = true);

// What OpenSees' Transformation handler makes of each DOF of the numbering: -1 if fixed,
// otherwise the global DOF id (dof_offset[node] + d) of the node's own free DOF it equals.
// Requires one retained node per constrained node.
std::vector<std::int64_t> emulate_transformation_handler(const io::Model& model,
                                                         const DofNumbering& numbering);

// Identity order (renumbering off).
std::vector<std::int32_t> file_order(const io::Model& model);

}  // namespace femorax::compile
