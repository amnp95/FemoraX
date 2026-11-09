#pragma once

// Element batches, assembly maps, lumped mass and (on request) element coloring: host data
// built once per analysis stage, laid out exactly as the device arrays (memory contract
// rule 2).
//
// One batch per registered (element, material) pair (compile/registry.hpp). Per-element
// arrays are structure of arrays, value[k * padded + e], with padded = number of elements
// rounded up to the SIMD width; padding elements are inert (equations -1, never scattered).
//
// Subdomains: the compile step sees the whole model, and everything after it works in the
// subdomain's local numbering (internal node and equation indices); global tags appear
// only in the maps kept here. For now the model is one subdomain: every node owned, no
// ghost nodes (docs/design.md, "Multi-GPU").

#include <cstdint>
#include <string>
#include <vector>

#include "compile/dof_numbering.hpp"
#include "compile/registry.hpp"
#include "compile/renumber.hpp"
#include "io/model.hpp"
#include "physics/contracts.hpp"

namespace femorax::compile {

struct ParamSet {  // one per distinct (element group, damping region) in a batch
  std::int32_t group = 0;     // index into model.elements
  std::int32_t material = 0;  // index into model.materials
  std::int32_t region = -1;   // index into model.rayleigh, -1 for none
  Rayleigh damping;
  registry::ParamBlob element;  // the element's typed Params

  template <class Elem>
  typename Elem::Params params() const {
    return element.as<typename Elem::Params>();
  }
};

struct Batch {
  int pair = -1;  // registry pair index (element entry, material entry)
  io::ElementType element = io::ElementType::StdBrick;
  int nnode = 0, ndof = 0;
  int nstate = 0;           // material history doubles per element (Elem::NSTATE), per buffer
  // Bytes per element of the compacted corrector path's buffers (exec/compacted_kernels.hpp:
  // stress, tangent and strain per point, a flag and a list entry), for split materials.
  int compact_bytes = 0;
  std::int32_t count = 0;   // real elements
  std::int32_t padded = 0;  // count rounded up to the SIMD width
  std::vector<std::int32_t> tag;    // [padded]; -1 for padding
  std::vector<std::int32_t> node;   // [nnode * padded]: internal node index, -1 padding
  std::vector<std::int32_t> eq;     // [ndof * padded]: equation, -1 fixed or padding
  std::vector<std::int32_t> param;  // [padded]: index into params (0 for padding)
  std::vector<ParamSet> params;
  // Coloring, only when CompileOptions::build_coloring: no two elements of one color share
  // an equation. color_start has num_colors + 1 entries; color_elements lists batch
  // positions of each color, in batch order. Batch order itself is never changed by it.
  std::vector<std::int32_t> color_start, color_elements;
  std::int32_t num_colors() const {
    return color_start.empty() ? 0 : static_cast<std::int32_t>(color_start.size()) - 1;
  }
  bool has_coloring() const { return !color_start.empty(); }
};

struct Subdomain {
  int id = 0, count = 1;
  std::vector<std::int32_t> owned_nodes;  // internal node indices this subdomain owns
  std::vector<std::int32_t> ghost_nodes;  // internal node indices owned elsewhere (none yet)
};

struct Compiled {
  DofNumbering dofs;
  Ordering ordering;
  Subdomain subdomain;
  std::vector<double> coords;       // [3 * N] structure of arrays: x[N] y[N] z[N]
  std::vector<std::int8_t> eq_dof;  // [num_equations]: the nodal DOF (0-based) of each equation
  std::vector<Batch> batches;
  std::vector<double> lumped_mass;  // [num_equations]: element lumped mass + nodal mass
  // [num_equations]: sum over elements of alphaM_e * lumped mass (mass-proportional
  // damping coefficient per equation, for CentralDifference's diagonal system).
  std::vector<double> lumped_alpha_mass;
  std::vector<double> nodal_mass;  // [num_equations]: the nodal masses alone (matrix-free M)
  std::vector<std::string> warnings;
};

struct CompileOptions {
  Renumbering renumbering = Renumbering::RCM;
  int simd_width = 1;           // pad batches to a multiple of this (exec::simd_width())
  bool build_coloring = false;  // only the colored scatter needs it
};

// Throws io::ModelError (constraint problems) or io::UnsupportedComponent (an element and
// material pair with no registered kernel).
Compiled compile_model(const io::Model& model, const CompileOptions& options);

// Bytes the device needs for the compiled model and an explicit stage: coordinates, maps,
// parameters, state, colorings, and the u, v, a, f, mass and external-force vectors.
std::int64_t memory_estimate_bytes(const Compiled& c);

}  // namespace femorax::compile
