#pragma once

// Element batches, assembly maps, lumped mass and element coloring: host data built once
// per analysis stage, laid out exactly as the device arrays (memory contract rule 2).
//
// One batch per (element type, material type). Per-element arrays are structure of arrays,
// value[k * padded + e], with padded = number of elements rounded up to the SIMD width;
// padding elements are inert (equations -1, zero mass, never scattered).

#include <cstdint>
#include <string>
#include <vector>

#include "compile/dof_numbering.hpp"
#include "compile/renumber.hpp"
#include "io/model.hpp"
#include "physics/contracts.hpp"

namespace femorax::compile {

enum class MaterialType { ElasticIsotropic };

struct ParamSet {  // one per distinct (element group, damping region) in a batch
  std::int32_t group = 0;    // index into model.elements
  std::int32_t material = 0; // index into model.materials
  double E = 0, nu = 0, rho = 0;
  double b[3] = {0, 0, 0};
  int massType = 0;          // 0 consistent, 1 lumped
  Rayleigh damping;
};

struct Batch {
  io::ElementType element = io::ElementType::StdBrick;
  MaterialType material = MaterialType::ElasticIsotropic;
  int nnode = 8, ndof = 24;
  std::int32_t count = 0;   // real elements
  std::int32_t padded = 0;  // count rounded up to the SIMD width
  std::vector<std::int32_t> tag;        // [padded]; -1 for padding
  std::vector<std::int32_t> node;       // [nnode * padded]: internal node index, -1 padding
  std::vector<std::int32_t> eq;         // [ndof * padded]: equation, -1 fixed or padding
  std::vector<std::int32_t> param;      // [padded]: index into params (0 for padding)
  std::vector<ParamSet> params;
  // Coloring: no two elements of one color share an equation. color_start has
  // num_colors + 1 entries; elements color_elements[color_start[c] .. color_start[c+1]) are
  // batch positions of color c, in batch order.
  std::vector<std::int32_t> color_start, color_elements;
  std::int32_t num_colors() const { return static_cast<std::int32_t>(color_start.size()) - 1; }
};

struct Compiled {
  DofNumbering dofs;
  Ordering ordering;
  std::vector<double> coords;       // [3 * N] structure of arrays: x[N] y[N] z[N]
  std::vector<Batch> batches;
  std::vector<double> lumped_mass;  // [num_equations]: element lumped mass + nodal mass
  std::vector<std::string> warnings;
};

struct CompileOptions {
  Renumbering renumbering = Renumbering::RCM;
  int simd_width = 1;  // pad batches to a multiple of this (exec::simd_width())
};

// Throws io::ModelError (constraint problems, a material/element pair without a kernel).
Compiled compile_model(const io::Model& model, const CompileOptions& options);

// Bytes the device needs for the compiled model and an explicit stage: coordinates, maps,
// parameters, state, colorings, and the u, v, a, f, mass and external-force vectors.
std::int64_t memory_estimate_bytes(const Compiled& c);

}  // namespace femorax::compile
