#pragma once

// Exact linear constraints u = T q (docs/design.md, "Exact constraints u = T q"), host code of
// the compile step, step 1: every constrained DOF is a fixed weighted sum of other DOFs.
//
//   - LinearConstraint: one constrained DOF and its terms (DOF ids: dof_offset[node] + d), from
//     the generators below (rigidLink beam / bar, rigidDiaphragm, the embedded node).
//   - resolve_constraints: chains resolved by substitution (a term that is itself constrained
//     is replaced by its own terms), duplicate terms merged; a cycle or a DOF constrained twice
//     refused (io::ModelError naming the constraints). The result expresses every constrained
//     DOF in DOFs that are not constrained; DOF numbering then maps those to equations (and a
//     fixed one drops out).
//
// Small rotations throughout (OpenSees' RigidBeam, RigidDiaphragm and ASDEmbeddedNodeElement are
// linear too). Nothing here uses Kokkos.

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace femorax::compile {

struct LinearConstraint {
  std::int32_t dof = -1;                                  // the constrained DOF
  std::vector<std::pair<std::int32_t, double>> terms;     // (DOF, weight)
  std::string source;                                     // for messages: "rigidLink beam 3 7", ...
};

// Every constrained DOF in terms of unconstrained DOFs (CSR: row r is constrained DOF dof[r],
// its terms term_dof / term_weight[start[r] .. start[r + 1])), rows in ascending DOF order.
struct ResolvedConstraints {
  std::vector<std::int32_t> dof;
  std::vector<std::int64_t> start{0};
  std::vector<std::int32_t> term_dof;
  std::vector<double> term_weight;
  std::size_t rows() const { return dof.size(); }
};

ResolvedConstraints resolve_constraints(std::int32_t num_dofs, const std::vector<LinearConstraint>& constraints);

// --- Generators. m, s: the DOF offsets of the master (retained) and slave (constrained) nodes,
// xm, xs their coordinates. Each appends its constraints to `out`.

// `rigidLink beam m s` (OpenSees RigidBeam): u_s = u_m + theta_m x (x_s - x_m), theta_s = theta_m;
// both nodes 6 DOFs.
void rigid_link_beam(std::int32_t m, std::int32_t s, const std::array<double, 3>& xm, const std::array<double, 3>& xs,
                     std::vector<LinearConstraint>& out, const std::string& source);
// `rigidLink bar m s` (OpenSees RigidRod): equal translations.
void rigid_link_bar(std::int32_t m, std::int32_t s, std::vector<LinearConstraint>& out, const std::string& source);
// `rigidDiaphragm perp m s` (OpenSees RigidDiaphragm, perp = 1, 2 or 3: the axis normal to the
// plane): the two in-plane translations and the rotation about perp follow the master rigidly.
// Both nodes 6 DOFs.
void rigid_diaphragm(int perp, std::int32_t m, std::int32_t s, const std::array<double, 3>& xm,
                     const std::array<double, 3>& xs, std::vector<LinearConstraint>& out, const std::string& source);

// --- The embedded node in an 8-node brick (stdBrick / SSPbrick node order): its natural
// coordinates by Newton (true: converged inside the brick, within tol), the trilinear shape
// functions there, and the constraints of the node's translations (and, with rot, its
// rotations: half the curl of the brick's displacement gradient at the node). host[a]: the DOF
// offset of the brick's node a (3 DOFs each); c: the embedded node's offset.
bool hex8_natural_coordinates(const double (&xe)[8][3], const std::array<double, 3>& x, std::array<double, 3>& xi,
                              double tol = 1e-12);
void hex8_shape(const std::array<double, 3>& xi, double N[8], double dN[8][3]);
void embedded_in_hex8(const std::int32_t (&host)[8], const double (&xe)[8][3], std::int32_t c,
                      const std::array<double, 3>& xc, bool rot, std::vector<LinearConstraint>& out,
                      const std::string& source);

}  // namespace femorax::compile
