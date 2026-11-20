#include "compile/linear_constraints.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <string>

#include "io/h5.hpp"

namespace femorax::compile {

ResolvedConstraints resolve_constraints(std::int32_t num_dofs, const std::vector<LinearConstraint>& constraints) {
  // The constraint of each DOF (a DOF constrained twice is refused).
  std::vector<std::int32_t> of(static_cast<std::size_t>(num_dofs), -1);
  for (std::size_t i = 0; i < constraints.size(); ++i) {
    const auto d = constraints[i].dof;
    if (d < 0 || d >= num_dofs) throw io::ModelError(constraints[i].source + ": constrained DOF out of range");
    if (of[static_cast<std::size_t>(d)] >= 0)
      throw io::ModelError("a DOF is constrained twice: by " + constraints[static_cast<std::size_t>(of[static_cast<std::size_t>(d)])].source +
                           " and by " + constraints[i].source);
    of[static_cast<std::size_t>(d)] = static_cast<std::int32_t>(i);
  }
  // Substitution, depth first, with the cycle check (0 unvisited, 1 on the path, 2 done).
  std::vector<int> state(constraints.size(), 0);
  std::vector<std::map<std::int32_t, double>> resolved(constraints.size());
  std::function<void(std::size_t)> resolve = [&](std::size_t i) {
    if (state[i] == 2) return;
    if (state[i] == 1) throw io::ModelError("the constraints form a cycle through " + constraints[i].source);
    state[i] = 1;
    auto& r = resolved[i];
    for (const auto& [d, w] : constraints[i].terms) {
      if (d < 0 || d >= num_dofs) throw io::ModelError(constraints[i].source + ": a term's DOF is out of range");
      const auto j = of[static_cast<std::size_t>(d)];
      if (j < 0) {
        r[d] += w;
      } else {
        resolve(static_cast<std::size_t>(j));
        for (const auto& [dd, ww] : resolved[static_cast<std::size_t>(j)]) r[dd] += w * ww;
      }
    }
    state[i] = 2;
  };
  std::vector<std::size_t> order(constraints.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return constraints[a].dof < constraints[b].dof; });
  ResolvedConstraints out;
  for (const auto i : order) {
    resolve(i);
    out.dof.push_back(constraints[i].dof);
    for (const auto& [d, w] : resolved[i]) {
      if (w == 0.0) continue;
      out.term_dof.push_back(d);
      out.term_weight.push_back(w);
    }
    out.start.push_back(static_cast<std::int64_t>(out.term_dof.size()));
  }
  return out;
}

void rigid_link_beam(std::int32_t m, std::int32_t s, const std::array<double, 3>& xm, const std::array<double, 3>& xs,
                     std::vector<LinearConstraint>& out, const std::string& source) {
  const double r[3] = {xs[0] - xm[0], xs[1] - xm[1], xs[2] - xm[2]};
  // theta x r: x: ty rz - tz ry; y: tz rx - tx rz; z: tx ry - ty rx (theta at m + 3, m + 4, m + 5).
  out.push_back({s + 0, {{m + 0, 1.0}, {m + 4, r[2]}, {m + 5, -r[1]}}, source});
  out.push_back({s + 1, {{m + 1, 1.0}, {m + 5, r[0]}, {m + 3, -r[2]}}, source});
  out.push_back({s + 2, {{m + 2, 1.0}, {m + 3, r[1]}, {m + 4, -r[0]}}, source});
  for (int d = 3; d < 6; ++d) out.push_back({s + d, {{m + d, 1.0}}, source});
}

void rigid_link_bar(std::int32_t m, std::int32_t s, std::vector<LinearConstraint>& out, const std::string& source) {
  for (int d = 0; d < 3; ++d) out.push_back({s + d, {{m + d, 1.0}}, source});
}

void rigid_diaphragm(int perp, std::int32_t m, std::int32_t s, const std::array<double, 3>& xm,
                     const std::array<double, 3>& xs, std::vector<LinearConstraint>& out, const std::string& source) {
  if (perp < 1 || perp > 3) throw io::ModelError(source + ": the diaphragm's perpendicular direction must be 1, 2 or 3");
  const int n = perp - 1;          // the normal axis
  const int a = (n + 1) % 3, b = (n + 2) % 3;  // the in-plane axes, (a, b, n) right-handed
  const double r[3] = {xs[0] - xm[0], xs[1] - xm[1], xs[2] - xm[2]};
  // theta_n e_n x r: a component -theta_n r_b, b component +theta_n r_a.
  out.push_back({s + a, {{m + a, 1.0}, {m + 3 + n, -r[b]}}, source});
  out.push_back({s + b, {{m + b, 1.0}, {m + 3 + n, r[a]}}, source});
  out.push_back({s + 3 + n, {{m + 3 + n, 1.0}}, source});
}

void hex8_shape(const std::array<double, 3>& xi, double N[8], double dN[8][3]) {
  // stdBrick's node order: the corners (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1), then z = +1.
  static const double c[8][3] = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
                                 {-1, -1, 1},  {1, -1, 1},  {1, 1, 1},  {-1, 1, 1}};
  for (int i = 0; i < 8; ++i) {
    const double a = 1.0 + c[i][0] * xi[0], b = 1.0 + c[i][1] * xi[1], d = 1.0 + c[i][2] * xi[2];
    N[i] = 0.125 * a * b * d;
    dN[i][0] = 0.125 * c[i][0] * b * d;
    dN[i][1] = 0.125 * a * c[i][1] * d;
    dN[i][2] = 0.125 * a * b * c[i][2];
  }
}

namespace {

// J[j][k] = d x_j / d xi_k at xi; returns det J and fills its inverse.
double jacobian(const double (&xe)[8][3], const double dN[8][3], double Jinv[3][3]) {
  double J[3][3] = {};
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) J[j][k] += xe[i][j] * dN[i][k];
  const double det = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                     J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
  Jinv[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) / det;
  Jinv[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) / det;
  Jinv[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / det;
  Jinv[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) / det;
  Jinv[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / det;
  Jinv[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) / det;
  Jinv[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) / det;
  Jinv[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) / det;
  Jinv[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) / det;
  return det;
}

}  // namespace

bool hex8_natural_coordinates(const double (&xe)[8][3], const std::array<double, 3>& x, std::array<double, 3>& xi,
                              double tol) {
  xi = {0.0, 0.0, 0.0};
  double scale = 0.0;
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 3; ++j) scale = std::max(scale, std::abs(xe[i][j] - xe[0][j]));
  for (int it = 0; it < 50; ++it) {
    double N[8], dN[8][3], Jinv[3][3];
    hex8_shape(xi, N, dN);
    double r[3] = {x[0], x[1], x[2]};
    for (int i = 0; i < 8; ++i)
      for (int j = 0; j < 3; ++j) r[j] -= N[i] * xe[i][j];
    if (std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) <= tol * scale)
      return std::abs(xi[0]) <= 1.0 + 1e-10 && std::abs(xi[1]) <= 1.0 + 1e-10 && std::abs(xi[2]) <= 1.0 + 1e-10;
    if (!(jacobian(xe, dN, Jinv) > 0.0)) return false;
    for (int k = 0; k < 3; ++k) xi[static_cast<std::size_t>(k)] += Jinv[k][0] * r[0] + Jinv[k][1] * r[1] + Jinv[k][2] * r[2];
  }
  return false;
}

void embedded_in_hex8(const std::int32_t (&host)[8], const double (&xe)[8][3], std::int32_t c,
                      const std::array<double, 3>& xc, bool rot, std::vector<LinearConstraint>& out,
                      const std::string& source) {
  std::array<double, 3> xi{};
  if (!hex8_natural_coordinates(xe, xc, xi))
    throw io::ModelError(source + ": the embedded node is not inside its host brick");
  double N[8], dN[8][3], Jinv[3][3];
  hex8_shape(xi, N, dN);
  jacobian(xe, dN, Jinv);
  for (int d = 0; d < 3; ++d) {
    LinearConstraint lc{c + d, {}, source};
    for (int i = 0; i < 8; ++i) lc.terms.push_back({host[i] + d, N[i]});
    out.push_back(std::move(lc));
  }
  if (!rot) return;
  // dN_i / dx_j = sum_k dN_i / dxi_k  dxi_k / dx_j (Jinv[k][j]).
  double g[8][3];
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 3; ++j) g[i][j] = dN[i][0] * Jinv[0][j] + dN[i][1] * Jinv[1][j] + dN[i][2] * Jinv[2][j];
  // theta = 1/2 curl u: x: (du_z/dy - du_y/dz) / 2, y: (du_x/dz - du_z/dx) / 2, z: (du_y/dx - du_x/dy) / 2.
  LinearConstraint tx{c + 3, {}, source}, ty{c + 4, {}, source}, tz{c + 5, {}, source};
  for (int i = 0; i < 8; ++i) {
    tx.terms.push_back({host[i] + 2, 0.5 * g[i][1]});
    tx.terms.push_back({host[i] + 1, -0.5 * g[i][2]});
    ty.terms.push_back({host[i] + 0, 0.5 * g[i][2]});
    ty.terms.push_back({host[i] + 2, -0.5 * g[i][0]});
    tz.terms.push_back({host[i] + 1, 0.5 * g[i][0]});
    tz.terms.push_back({host[i] + 0, -0.5 * g[i][1]});
  }
  out.push_back(std::move(tx));
  out.push_back(std::move(ty));
  out.push_back(std::move(tz));
}

}  // namespace femorax::compile
