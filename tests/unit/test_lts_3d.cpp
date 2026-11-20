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
// Step 3b: the same scheme in the explicit time loop (exec::run_model with
// RunOptions::local_time_stepping, the real force kernels; the substeps re-evaluate only the
// fine elements and their ring, InternalForce::compute_subset), against the host reference.

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <variant>
#include <vector>

#include "../common/reference_assembly.hpp"
#include "../common/synthetic_models.hpp"
#include "compile/local_time_stepping.hpp"
#include "compile/lts_levels.hpp"
#include "exec/explicit_analysis.hpp"
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

Column make_column(double stiffness = 16.0,
                   femorax::compile::Renumbering renumbering = femorax::compile::Renumbering::RCM) {
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
  femorax::compile::CompileOptions co;
  co.renumbering = renumbering;
  r.c = femorax::compile::compile_model(r.model, co);
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

// --- Step 3b: the time loop ----------------------------------------------------------

namespace fs = std::filesystem;

std::string temp_file(const std::string& name) {
  const fs::path d = fs::temp_directory_path() / ("femorax_test_lts_" + std::to_string(::getpid()));
  fs::create_directories(d);
  return (d / name).string();
}

std::vector<std::vector<double>> read_rows(const std::string& file) {
  std::vector<std::vector<double>> rows;
  std::ifstream in(file);
  for (std::string line; std::getline(in, line);) {
    std::istringstream ss(line);
    std::vector<double> r;
    for (double x; ss >> x;) r.push_back(x);
    if (!r.empty()) rows.push_back(r);
  }
  return rows;
}

// Runs `steps` explicit steps of `dt` from rest under gravity (the body force), every node's
// displacement recorded after each step; returns the rows as equation-space vectors (the
// compile step's numbering with renumbering off).
std::vector<std::vector<double>> run(const Column& s, femorax::io::Integrator integrator, double dt, int steps,
                                     bool lts, femorax::exec::RunReport* report = nullptr,
                                     bool allow_unstable_dt = false) {
  femorax::io::Model m = s.model;
  femorax::io::NodeRecorder nr;
  nr.file = temp_file("disp.txt");
  nr.response = femorax::io::NodeResponse::Disp;
  nr.time = true;
  nr.precision = 17;
  nr.nodes = m.nodes.tags;
  nr.dofs = {1, 2, 3};
  m.recorders.push_back({"d", nr, {}});
  m.process.push_back({"0", "", femorax::io::RecorderStep{"d"}});
  femorax::io::AnalysisStep a;
  a.name = "run";
  a.dt = dt;
  a.num_steps = steps;
  a.integrator = integrator;
  a.allow_unstable_dt = allow_unstable_dt;
  m.process.push_back({"1", "", a});
  femorax::exec::RunOptions o;
  o.verbose = false;
  o.renumbering = femorax::compile::Renumbering::Off;
  o.local_time_stepping = lts;
  const auto rep = femorax::exec::run_model(m, o);
  if (report) *report = rep;
  const auto rows = read_rows(nr.file);
  std::vector<std::vector<double>> out;
  const auto n = static_cast<std::size_t>(s.c.dofs.num_equations);
  for (const auto& r : rows) {
    std::vector<double> u(n, 0.0);
    for (std::int32_t node = 0; node < s.c.dofs.num_nodes(); ++node)
      for (int d = 0; d < 3; ++d) {
        const std::int32_t q = s.c.dofs.equation(node, d);
        if (q >= 0) u[static_cast<std::size_t>(q)] = r[static_cast<std::size_t>(1 + 3 * node + d)];
      }
    out.push_back(u);
  }
  return out;
}

// The force of the subset after a change of the subset's inputs only equals the full force,
// bit for bit.
TEST(Lts3D, SubsetForceEqualsTheFullForce) {
  const Column s = make_column(256.0);
  femorax::exec::InternalForce force(s.c, femorax::exec::GeometryCache::Auto);
  std::vector<std::vector<std::uint8_t>> member(s.levels.touched_level.size());
  for (std::size_t b = 0; b < member.size(); ++b)
    for (const auto l : s.levels.touched_level[b]) member[b].push_back(l > 0 ? 1 : 0);
  const int ring = force.add_element_subset(member);
  ASSERT_GT(force.subset_elements(ring), 0);
  ASSERT_LT(force.subset_elements(ring), force.num_elements());
  const auto u0 = initial(s);
  auto hu = Kokkos::create_mirror_view(force.u);
  for (std::size_t i = 0; i < u0.size(); ++i) hu(i) = u0[i];
  Kokkos::deep_copy(force.u, hu);
  Kokkos::deep_copy(force.v, 0.0);
  femorax::exec::ForceSettings fs;
  force.compute(fs);
  for (std::size_t i = 0; i < u0.size(); ++i)
    if (s.fine[i]) hu(i) = u0[i] * 1.7 + 1e-4;
  Kokkos::deep_copy(force.u, hu);
  force.compute_subset(fs, ring);
  const auto sub = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), force.f);
  force.compute(fs);
  const auto full = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), force.f);
  for (std::size_t i = 0; i < u0.size(); ++i) ASSERT_EQ(sub(i), full(i)) << i;
}

// With one level (no inclusion) the run is the usual one, bit for bit, for both schemes.
TEST(Lts3D, OneLevelIsTheUsualRunBitForBit) {
  const Column s = make_column(1.0, femorax::compile::Renumbering::Off);
  ASSERT_EQ(s.levels.levels, 1);
  for (const auto integ : {femorax::io::Integrator::CentralDifference, femorax::io::Integrator::ExplicitDifference}) {
    femorax::exec::RunReport rep;
    const auto a = run(s, integ, s.dt, 50, true, &rep);
    const auto b = run(s, integ, s.dt, 50, false);
    ASSERT_EQ(rep.stages.at(0).lts_levels, 1);
    ASSERT_EQ(a.size(), 50u);
    for (std::size_t k = 0; k < a.size(); ++k) ASSERT_EQ(a[k], b[k]) << k;
  }
}

// The run with the 256 times stiffer inclusion at the soil's step: CentralDifference with LTS
// equals the host reference recurrence (lts_step on the reference assembly, the body force as
// an affine term) to rounding; ExplicitDifference is the same recurrence one step later
// (it starts from the committed acceleration 0); the global run at the fine step stays close.
TEST(Lts3D, TimeLoopMatchesTheHostReference) {
  const Column s = make_column(256.0, femorax::compile::Renumbering::Off);
  ASSERT_GE(s.levels.levels, 2);
  const int steps = 200;
  femorax::exec::RunReport rep;
  const auto cd = run(s, femorax::io::Integrator::CentralDifference, s.dt, steps, true, &rep);
  const auto& st = rep.stages.at(0);
  std::printf("[ LTS run: %d levels, %d substeps, %lld fine equations, %lld of %lld elements in the substeps, "
              "estimated gain %.2fx ]\n",
              st.lts_levels, st.lts_substeps, static_cast<long long>(st.lts_fine_equations),
              static_cast<long long>(st.lts_substep_elements), static_cast<long long>(rep.elements), st.lts_speedup);
  ASSERT_EQ(st.lts_substeps, s.p);
  ASSERT_EQ(cd.size(), static_cast<std::size_t>(steps));

  // The host reference: f(u) = K u + f0 (f0 = f(0): the body force), so the mixed-field force
  // is A Pi_c u + A Pi_f z + M^-1 f0, split in two halves over the two applications.
  femorax::exec::InternalForce force(s.c, femorax::exec::GeometryCache::Off);
  force.fill(0.0, 0.0);
  force.compute(femorax::exec::ForceSettings{});
  const auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), force.f);
  const auto n = s.m.size();
  auto apply = [&](const std::vector<double>& x, std::vector<double>& y) {
    y = s.ref.multiply(s.ref.K, x);
    for (std::size_t i = 0; i < n; ++i) y[i] = (y[i] + 0.5 * hf(i)) / s.m[i];
  };
  const auto c = femorax::compile::lts_coefficients(s.p, 0.05);
  std::vector<double> a(n, 0.0), b(n, 0.0), next;  // CentralDifference: u(-1) = u(0) = 0
  double peak = 0.0, err = 0.0;
  for (int k = 0; k < steps; ++k) {
    femorax::compile::lts_step(c, s.dt, s.fine, apply, a, b, next);
    a.swap(b);
    b.swap(next);
    peak = std::max(peak, max_abs(b));
    err = std::max(err, max_diff(b, cd[static_cast<std::size_t>(k)]));
  }
  std::printf("[ CentralDifference with LTS against the host reference: %.2e of the peak %.3e ]\n", err / peak, peak);
  EXPECT_LE(err, 1e-10 * peak);

  const auto ed = run(s, femorax::io::Integrator::ExplicitDifference, s.dt, steps, true);
  double err_ed = 0.0;
  for (int k = 0; k + 1 < steps; ++k)
    err_ed = std::max(err_ed, max_diff(ed[static_cast<std::size_t>(k + 1)], cd[static_cast<std::size_t>(k)]));
  std::printf("[ ExplicitDifference with LTS against CentralDifference one step earlier: %.2e of the peak ]\n",
              err_ed / peak);
  EXPECT_LE(err_ed, 1e-10 * peak);

}

// Accuracy in the time loop: with the inclusion 16 times stiffer the global run at the soil's
// step is still stable in the assembled model (allowed past the element check here), so LTS
// must be as accurate as it, both measured against a global run at a 64 times smaller step.
// The comparison is at the same scheme and start (CentralDifference from rest), so the start's
// own dependence on dt is in both errors.
TEST(Lts3D, TimeLoopIsAsAccurateAsTheGlobalCoarseStep) {
  const Column s = make_column(16.0, femorax::compile::Renumbering::Off);
  ASSERT_GE(s.levels.levels, 2);
  const int steps = 100;
  const auto cd = femorax::io::Integrator::CentralDifference;
  const auto lts = run(s, cd, s.dt, steps, true);
  const auto coarse = run(s, cd, s.dt, steps, false, nullptr, true);
  const auto truth = run(s, cd, s.dt / 64, steps * 64, false);
  double peak = 0.0, e_lts = 0.0, e_coarse = 0.0;
  for (int k = 0; k < steps; ++k) {
    const auto& t = truth[static_cast<std::size_t>((k + 1) * 64 - 1)];
    peak = std::max(peak, max_abs(t));
    e_lts = std::max(e_lts, max_diff(lts[static_cast<std::size_t>(k)], t));
    e_coarse = std::max(e_coarse, max_diff(coarse[static_cast<std::size_t>(k)], t));
  }
  std::printf("[ against a global run at dt/64: LTS %.2e, global at the coarse step %.2e (of the peak) ]\n",
              e_lts / peak, e_coarse / peak);
  EXPECT_LE(e_lts, 1.2 * e_coarse);
}

}  // namespace
