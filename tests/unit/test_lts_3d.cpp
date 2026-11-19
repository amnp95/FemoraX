// Local time stepping in 3D (docs/design.md, "Local time stepping", step 3a): the stabilized
// LF-LTS reference step (compile/local_time_stepping.hpp) on a brick column with a stiff
// inclusion, its levels from the elements' own stable steps (exec::InternalForce, compile/
// lts_levels.hpp) and its stiffness from the host reference assembly (tests/common). The
// column: 4 x 4 x 8 stdBrick (lumped mass), the base fixed, the sides free (no laminar ties,
// so the fine region stays local), no damping; one interior element 16 times stiffer. The
// coarse step is 0.9 of the soil's stable step, which is unstable for the inclusion:
//   - the global leapfrog at the coarse step blows up (the reason for LTS);
//   - LTS at the coarse step follows the global leapfrog at the fine step (the inclusion's
//     level) to the schemes' accuracy;
//   - the scheme's modified energy is conserved.

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <variant>
#include <vector>

#include "../common/reference_assembly.hpp"
#include "../common/synthetic_models.hpp"
#include "compile/local_time_stepping.hpp"
#include "compile/lts_levels.hpp"
#include "exec/internal_force.hpp"

namespace {

using femorax::compile::Compiled;

struct Column {
  femorax::io::Model model;
  Compiled c;
  femorax::testing::ReferenceSystem ref;
  std::vector<double> m;                // lumped mass per equation
  std::vector<std::uint8_t> fine;       // equations above level 0
  int p = 1;                            // fine-to-coarse step ratio
  double dt = 0.0;                      // the coarse step
  double soil_dt = 0.0, inclusion_dt = 0.0;
  femorax::compile::LtsLevels levels;
};

Column make_column(double stiffness = 16.0) {
  femorax::testing::ColumnSpec s;
  s.nx = s.ny = 4;
  s.nz = 8;
  s.damping = false;
  Column r;
  r.model = femorax::testing::soil_column(s);
  r.model.equal_dof = {};
  // The inclusion: an interior element of the upper layer, in a group of its own with a
  // material 16 times stiffer (the same density).
  auto& upper = r.model.elements[1];
  const std::size_t e = 2 * 16 + 1 * 4 + 1;  // k = 2 within the layer, j = 1, i = 1
  femorax::io::ElementGroup inc = upper;
  inc.id = "inclusion";
  inc.material = 3;
  inc.tags = {upper.tags[e]};
  inc.connectivity.assign(upper.connectivity.begin() + static_cast<long>(8 * e),
                          upper.connectivity.begin() + static_cast<long>(8 * e + 8));
  upper.tags.erase(upper.tags.begin() + static_cast<long>(e));
  upper.connectivity.erase(upper.connectivity.begin() + static_cast<long>(8 * e),
                           upper.connectivity.begin() + static_cast<long>(8 * e + 8));
  r.model.elements.push_back(inc);
  femorax::io::NDMaterial stiff = r.model.materials[1];  // the upper layer's, 'stiffness' times stiffer
  stiff.tag = 3;
  std::get<femorax::io::ElasticIsotropicData>(stiff.data).E *= stiffness;
  r.model.materials.push_back(stiff);
  r.c = femorax::compile::compile_model(r.model, femorax::compile::CompileOptions{});
  r.ref = femorax::testing::assemble_reference(r.model, r.c);
  const auto n = static_cast<std::size_t>(r.c.dofs.num_equations);
  r.m.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i)
    r.m[i] = r.ref.M[static_cast<std::size_t>(r.ref.pattern.find(static_cast<std::int32_t>(i), static_cast<std::int32_t>(i)))];

  // Each element's own stable step from the device; the coarse step from the soil's.
  femorax::exec::InternalForce force(r.c, femorax::exec::GeometryCache::Auto);
  const auto dts = force.element_critical_dts(femorax::ExplicitScheme::CentralDifference);
  r.soil_dt = 1e300;
  r.inclusion_dt = 1e300;
  for (std::size_t bi = 0; bi < r.c.batches.size(); ++bi)
    for (std::size_t k = 0; k < dts[bi].size(); ++k) {
      const bool is_inc = r.c.batches[bi].tag[k] == inc.tags[0];
      (is_inc ? r.inclusion_dt : r.soil_dt) = std::min(is_inc ? r.inclusion_dt : r.soil_dt, dts[bi][k]);
    }
  r.dt = 0.9 * r.soil_dt;
  r.levels = femorax::compile::assign_lts_levels(r.c, dts, r.dt);
  r.p = 1 << (r.levels.levels - 1);  // two levels: everything above level 0 at the finest step
  r.fine.assign(n, 0);
  for (std::size_t i = 0; i < n; ++i) r.fine[i] = r.levels.equation_level[i] > 0 ? 1 : 0;
  return r;
}

// A x = M^-1 K x with the reference stiffness.
std::vector<double> apply_a(const Column& s, const std::vector<double>& x) {
  auto y = s.ref.multiply(s.ref.K, x);
  for (std::size_t i = 0; i < y.size(); ++i) y[i] /= s.m[i];
  return y;
}

// The initial displacement: the column's first bending shape in x (v0 = 0), plus a small
// short-wave part so the inclusion's modes are excited.
std::vector<double> initial(const Column& s) {
  const auto n = static_cast<std::size_t>(s.c.dofs.num_equations);
  std::vector<double> u(n, 0.0);
  const std::int32_t N = s.c.dofs.num_nodes();
  for (std::int32_t node = 0; node < N; ++node) {
    const double z = s.c.coords[static_cast<std::size_t>(2 * N + node)];  // -4 .. 0
    const double x = s.c.coords[static_cast<std::size_t>(node)];
    for (int d = 0; d < 3; ++d) {
      const std::int32_t q = s.c.dofs.equation(node, d);
      if (q < 0) continue;
      const double shape = std::sin(0.5 * M_PI * (z + 4.0) / 4.0);
      u[static_cast<std::size_t>(q)] = d == 0 ? 1e-3 * shape : 1e-5 * std::sin(3.0 * x + 5.0 * z + d);
    }
  }
  return u;
}

// The largest magnitude; a non-finite value (a blown-up run) counts as infinite.
double max_abs(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) {
    if (!std::isfinite(x)) return std::numeric_limits<double>::infinity();
    m = std::max(m, std::abs(x));
  }
  return m;
}

// Leapfrog with step h for `steps` steps from u0 (v0 = 0); returns u at the end.
std::vector<double> leapfrog(const Column& s, std::vector<double> u0, double h, int steps) {
  auto au = apply_a(s, u0);
  std::vector<double> u1(u0.size());
  for (std::size_t i = 0; i < u0.size(); ++i) u1[i] = u0[i] - 0.5 * h * h * au[i];
  for (int k = 1; k < steps; ++k) {
    au = apply_a(s, u1);
    for (std::size_t i = 0; i < u0.size(); ++i) {
      const double next = 2.0 * u1[i] - u0[i] - h * h * au[i];
      u0[i] = u1[i];
      u1[i] = next;
    }
  }
  return u1;
}

// LTS at the coarse step from u0 (v0 = 0) for `steps` coarse steps.
std::vector<double> lts_run(const Column& s, const std::vector<double>& u0, int steps) {
  const auto c = femorax::compile::lts_coefficients(s.p, 0.05);
  auto apply = [&](const std::vector<double>& x, std::vector<double>& y) { y = apply_a(s, x); };
  std::vector<double> a = u0, b(u0.size()), next;
  const auto au = apply_a(s, u0);
  for (std::size_t i = 0; i < u0.size(); ++i) b[i] = u0[i] - 0.5 * s.dt * s.dt * au[i];
  for (int k = 1; k < steps; ++k) {
    femorax::compile::lts_step(c, s.dt, s.fine, apply, a, b, next);
    a.swap(b);
    b.swap(next);
  }
  return b;
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double x = a[i] - b[i];
    if (!std::isfinite(x)) return std::numeric_limits<double>::infinity();
    d = std::max(d, std::abs(x));
  }
  return d;
}

// With the inclusion 16 times stiffer the assembled model still tolerates the soil's step (the
// element bound is conservative): there LTS must be as accurate as the global leapfrog at the
// same coarse step, against a global run at a 64 times smaller step (the reference). With it
// 256 times stiffer the global leapfrog at the coarse step blows up, and LTS stays stable with
// the same accuracy.
TEST(Lts3D, AsAccurateAsLeapfrogAndStableWhereLeapfrogIsNot) {
  const int steps = 400;
  double err16 = 0.0;
  for (const double stiffness : {16.0, 256.0}) {
    const Column s = make_column(stiffness);
    const auto u0 = initial(s);
    const double scale = max_abs(u0);
    const auto lts = lts_run(s, u0, steps);
    const auto coarse = leapfrog(s, u0, s.dt, steps);
    const auto truth = leapfrog(s, u0, s.dt / 64, steps * 64);
    const double e_lts = max_diff(lts, truth) / scale, e_coarse = max_diff(coarse, truth) / scale;
    std::printf("[ inclusion %gx: soil step %.4e, inclusion %.4e, coarse %.4e, %d levels (p = %d), %lld fine "
                "equations of %lld, ideal speedup %.2f; error after %d coarse steps: LTS %.2e, global leapfrog at "
                "the coarse step %.2e (of the initial peak) ]\n",
                stiffness, s.soil_dt, s.inclusion_dt, s.dt, s.levels.levels, s.p,
                static_cast<long long>(std::count(s.fine.begin(), s.fine.end(), std::uint8_t{1})),
                static_cast<long long>(s.fine.size()), s.levels.speedup(), steps, e_lts, e_coarse);
    ASSERT_GE(s.levels.levels, 2);
    EXPECT_LE(max_abs(lts), 2.0 * scale);  // stable
    if (stiffness == 16.0) {
      err16 = e_lts;
      EXPECT_LE(e_lts, 1.2 * e_coarse);  // LTS adds (almost) no error to the coarse step's
    } else {
      EXPECT_GT(max_abs(coarse), 1e3 * scale);  // the global coarse step is unstable here
      EXPECT_LE(e_lts, 2.0 * err16);            // LTS keeps the coarse step's accuracy
    }
  }
}

TEST(Lts3D, ConservesTheModifiedEnergy) {
  const Column s = make_column();
  const auto c = femorax::compile::lts_coefficients(s.p, 0.05);
  auto apply = [&](const std::vector<double>& x, std::vector<double>& y) { y = apply_a(s, x); };
  const std::vector<double> zero(s.m.size(), 0.0);
  auto energy = [&](const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> next;
    femorax::compile::lts_step(c, s.dt, s.fine, apply, zero, b, next);  // dt^2 A^{p,nu} b = 2 b - next
    double e = 0.0;
    for (std::size_t i = 0; i < b.size(); ++i) {
      const double v = (b[i] - a[i]) / s.dt;
      e += s.m[i] * (v * v + a[i] * (2.0 * b[i] - next[i]) / (s.dt * s.dt));
    }
    return 0.5 * e;
  };
  std::vector<double> a = initial(s), b(a.size()), next;
  const auto au = apply_a(s, a);
  for (std::size_t i = 0; i < a.size(); ++i) b[i] = a[i] - 0.5 * s.dt * s.dt * au[i];
  const double e0 = energy(a, b);
  double worst = 0.0;
  for (int k = 0; k < 5000; ++k) {
    femorax::compile::lts_step(c, s.dt, s.fine, apply, a, b, next);
    a.swap(b);
    b.swap(next);
    if (k % 250 == 0 || k == 4999) worst = std::max(worst, std::abs(energy(a, b) / e0 - 1.0));
  }
  std::printf("[ 5000 coarse steps: largest relative change of the modified energy %.2e ]\n", worst);
  EXPECT_LE(worst, 1e-10);
}

}  // namespace
