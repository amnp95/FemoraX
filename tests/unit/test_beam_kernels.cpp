// DispBeam3d through the real kernels (docs/design.md, "Beams", step 4a: explicit stages): a
// vertical cantilever of six dispBeamColumn elements (6-DOF nodes, base fixed), the section
// elastic, Linear and PDelta, 3 integration points.
//   - the force kernel equals the host reference assembly's K u (tests/common), and every force
//     kernel variant gives the same bits (atomic: to rounding);
//   - the lumped mass: rho A L on each translation, the rotary inertia on the rotations;
//   - an explicit CentralDifference run under base shaking (run_model) follows the host
//     recurrence with the reference stiffness and the compile step's lumped mass.

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "../common/reference_assembly.hpp"
#include "compile/batches.hpp"
#include "physics/elements/disp_beam_3d.hpp"
#include "physics/sections/elastic_section_3d.hpp"
#include "exec/explicit_analysis.hpp"
#include "exec/internal_force.hpp"

namespace {

namespace fs = std::filesystem;
using femorax::io::Model;

constexpr int kElems = 6;
constexpr double kH = 6.0, E = 2.5e7, A = 0.5, Iz = 0.02, Iy = 0.015, G = 1.0e7, J = 0.03, rho = 2.4;

// Nodes 1..7 along z (x = 1, y = 2), base node 1 fixed in all six DOFs.
Model cantilever(int transf) {
  Model m;
  for (int k = 0; k <= kElems; ++k) {
    m.nodes.tags.push_back(1 + k);
    m.nodes.ndf.push_back(6);
    m.nodes.coords.insert(m.nodes.coords.end(), {1.0, 2.0, kH * k / kElems});
  }
  m.materials.push_back({1, femorax::io::ElasticSection3dData{E, A, Iz, Iy, G, J}, {}});
  femorax::io::ElementGroup g;
  g.id = "column";
  g.type = femorax::io::ElementType::DispBeamColumn;
  g.material = 1;
  g.beam.transf = transf;
  g.beam.vecxz = {1.0, 0.0, 0.0};
  g.beam.points = 3;
  g.beam.density = rho;
  for (int e = 0; e < kElems; ++e) {
    g.tags.push_back(1 + e);
    g.connectivity.insert(g.connectivity.end(), {1 + e, 2 + e});
  }
  m.elements.push_back(g);
  for (int d = 1; d <= 6; ++d) {
    m.fix.node.push_back(1);
    m.fix.dof.push_back(d);
  }
  return m;
}

std::vector<double> host(const femorax::exec::Vec& v) {
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v);
  return std::vector<double>(h.data(), h.data() + h.extent(0));
}

TEST(BeamKernels, ForceEqualsTheReferenceAssemblyInEveryVariant) {
  for (const int transf : {0, 1}) {
    const Model m = cantilever(transf);
    femorax::compile::CompileOptions co;
    co.build_coloring = true;
    const auto c = femorax::compile::compile_model(m, co);
    ASSERT_EQ(c.dofs.num_equations, 6 * kElems);
    const auto ref = femorax::testing::assemble_reference(m, c);
    femorax::exec::InternalForce force(c, femorax::exec::GeometryCache::Auto);
    const auto n = static_cast<std::size_t>(c.dofs.num_equations);
    std::vector<double> u(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = 1e-4 * std::sin(0.37 * static_cast<double>(i) + 0.1);
    auto hu = Kokkos::create_mirror_view(force.u);
    for (std::size_t i = 0; i < n; ++i) hu(i) = u[i];
    Kokkos::deep_copy(force.u, hu);
    Kokkos::deep_copy(force.v, 0.0);
    // The expected force: K u for Linear; for PDelta (not linear: q0 times the chord) every
    // element's own host compute, added through the batch's equation map.
    auto ku = ref.multiply(ref.K, u);
    if (transf == 1) {
      using Beam = femorax::DispBeam3d<femorax::ElasticSection3d, 3>;
      std::fill(ku.begin(), ku.end(), 0.0);
      const auto& b = c.batches[0];
      for (std::int32_t e = 0; e < b.count; ++e) {
        const auto p = b.params[static_cast<std::size_t>(b.param[e])].params<Beam>();
        double xe[2][3], ue[12], fe[12];
        for (int a = 0; a < 2; ++a) {
          const std::int32_t node = b.node[static_cast<std::size_t>(a) * b.padded + e];
          for (int dd = 0; dd < 3; ++dd) xe[a][dd] = c.coords[static_cast<std::size_t>(dd * c.dofs.num_nodes() + node)];
        }
        for (int k = 0; k < 12; ++k) {
          const std::int32_t q = b.eq[static_cast<std::size_t>(k) * b.padded + e];
          ue[k] = q >= 0 ? u[static_cast<std::size_t>(q)] : 0.0;
        }
        Beam::compute<false>(p, xe, ue, {nullptr, 1, 0}, {nullptr, 1, 0}, fe, nullptr);
        for (int k = 0; k < 12; ++k) {
          const std::int32_t q = b.eq[static_cast<std::size_t>(k) * b.padded + e];
          if (q >= 0) ku[static_cast<std::size_t>(q)] += fe[k];
        }
      }
    }
    double peak = 0.0;
    for (const double x : ku) peak = std::max(peak, std::abs(x));
    std::vector<double> first;
    // The cached settings too: a beam has no geometry cache and must recompute (a run asks for
    // cached geometry whenever the model has a cache).
    for (const char* name : {"flat_gather", "flat_gather_cached", "flat_colored", "team_gather", "team_gather_cached",
                             "flat_atomic"}) {
      femorax::exec::ForceSettings s;
      ASSERT_TRUE(femorax::exec::parse_settings(name, s));
      force.compute(s);
      const auto f = host(force.f);
      double d = 0.0;
      for (std::size_t i = 0; i < n; ++i) d = std::max(d, std::abs(f[i] - ku[i]));
      EXPECT_LE(d, 1e-12 * peak) << name;
      if (first.empty()) {
        first = f;
      } else if (std::string(name) != "flat_atomic") {
        EXPECT_EQ(f, first) << name;
      }
    }
  }
}

TEST(BeamKernels, LumpedMassHasTheTranslationsAndTheRotaryInertia) {
  const Model m = cantilever(0);
  const auto c = femorax::compile::compile_model(m, femorax::compile::CompileOptions{});
  double trans[3] = {0, 0, 0}, rot[3] = {0, 0, 0};
  for (std::int32_t node = 0; node < c.dofs.num_nodes(); ++node)
    for (int d = 0; d < 6; ++d) {
      const std::int32_t q = c.dofs.equation(node, d);
      if (q < 0) continue;
      (d < 3 ? trans[d] : rot[d - 3]) += c.lumped_mass[static_cast<std::size_t>(q)];
    }
  // The free nodes carry the whole member's mass but the base's half element.
  const double L = kH / kElems, free_length = kH - 0.5 * L;
  for (int d = 0; d < 3; ++d) EXPECT_NEAR(trans[d], rho * A * free_length, 1e-12);
  // Member along z with vecxz = x: local x = z, y = -y... local axes (x, y, z) = (Z, -Y, X).
  EXPECT_NEAR(rot[2], rho * (Iy + Iz) * free_length, 1e-12);  // torsion about global z
  EXPECT_NEAR(rot[0], rho * Iz * free_length, 1e-12);         // local z = global x
  EXPECT_NEAR(rot[1], rho * Iy * free_length, 1e-12);         // local y = -global y
}

std::string temp_file(const std::string& name) {
  const fs::path d = fs::temp_directory_path() / ("femorax_test_beam_" + std::to_string(::getpid()));
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

TEST(BeamKernels, ExplicitRunFollowsTheHostRecurrence) {
  Model m = cantilever(0);
  femorax::compile::CompileOptions co;
  co.renumbering = femorax::compile::Renumbering::Off;  // the run's numbering (RunOptions below)
  const auto c = femorax::compile::compile_model(m, co);
  const auto ref = femorax::testing::assemble_reference(m, c);
  double dt_crit = 0.0;
  {
    femorax::exec::InternalForce force(c, femorax::exec::GeometryCache::Auto);
    dt_crit = force.min_critical_dt(femorax::ExplicitScheme::CentralDifference);
  }
  const double dt = 0.5 * dt_crit;
  const int steps = 400;
  std::printf("[ cantilever: stable step %.4e (L / c = %.4e) ]\n", dt_crit, (kH / kElems) / std::sqrt(E / rho));
  // A base acceleration pulse in x, one sample per step.
  femorax::io::PathSeries ps;
  ps.tag = 1;
  ps.dt = dt;
  for (int k = 0; k <= steps + 2; ++k) {
    const double t = k * dt, T = 60 * dt;
    ps.values.push_back(t < T ? std::sin(M_PI * t / T) * std::sin(M_PI * t / T) : 0.0);
  }
  m.time_series.push_back(ps);
  m.patterns.push_back({2, 1, 1, 0.0, 1.0, {}});
  femorax::io::NodeRecorder nr;
  nr.file = temp_file("disp.txt");
  nr.response = femorax::io::NodeResponse::Disp;
  nr.time = true;
  nr.precision = 17;
  nr.nodes = m.nodes.tags;
  nr.dofs = {1, 2, 3, 4, 5, 6};
  m.recorders.push_back({"d", nr, {}});
  m.process.push_back({"0", "", femorax::io::RecorderStep{"d"}});
  m.process.push_back({"1", "", femorax::io::PatternStep{2}});
  femorax::io::AnalysisStep a;
  a.name = "shake";
  a.dt = dt;
  a.num_steps = steps;
  a.integrator = femorax::io::Integrator::CentralDifference;
  m.process.push_back({"2", "", a});
  femorax::exec::RunOptions o;
  o.verbose = false;
  o.renumbering = femorax::compile::Renumbering::Off;
  femorax::exec::run_model(m, o);
  const auto rows = read_rows(nr.file);
  ASSERT_EQ(rows.size(), static_cast<std::size_t>(steps));

  // Host CentralDifference (undamped, lumped): u(n+1) = 2 u(n) - u(n-1) + dt^2 (R / m), with
  // R = -K u(n) - m ag(t_n) on the x translations; from rest with u(-1) = u(0).
  const auto nq = static_cast<std::size_t>(c.dofs.num_equations);
  std::vector<double> u(nq, 0.0), um(nq, 0.0);
  double peak = 0.0, err = 0.0;
  for (int k = 0; k < steps; ++k) {
    const auto ku = ref.multiply(ref.K, u);
    std::vector<double> next(nq);
    for (std::size_t q = 0; q < nq; ++q) {
      const double mq = c.lumped_mass[q];
      double R = -ku[q];
      if (c.eq_dof[q] == 0) R -= mq * ps.values[static_cast<std::size_t>(k)];
      next[q] = 2.0 * u[q] - um[q] + dt * dt * R / mq;
    }
    um = u;
    u = next;
    for (std::int32_t node = 0; node < c.dofs.num_nodes(); ++node)
      for (int d = 0; d < 6; ++d) {
        const std::int32_t q = c.dofs.equation(node, d);
        if (q < 0) continue;
        const double run = rows[static_cast<std::size_t>(k)][static_cast<std::size_t>(1 + 6 * node + d)];
        peak = std::max(peak, std::abs(u[static_cast<std::size_t>(q)]));
        err = std::max(err, std::abs(run - u[static_cast<std::size_t>(q)]));
      }
  }
  std::printf("[ explicit cantilever against the host recurrence: %.2e of the peak %.3e ]\n", err / peak, peak);
  EXPECT_GT(peak, 0.0);
  EXPECT_LE(err, 1e-10 * peak);
}

// What beams refuse for now: implicit stages, Rayleigh stiffness damping in explicit stages, and
// beams without a mass of their own (their nodal masses do not reach the stable step yet).
TEST(BeamKernels, RefusesWhatBeamsCannotDoYet) {
  auto stage = [](femorax::io::Integrator integ) {
    femorax::io::AnalysisStep a;
    a.name = "s";
    a.dt = 1e-5;
    a.num_steps = 1;
    a.integrator = integ;
    if (integ == femorax::io::Integrator::Newmark) {
      a.type = femorax::io::AnalysisType::Transient;
      a.gamma = 0.5;
      a.beta = 0.25;
      a.algorithm = femorax::io::Algorithm::Newton;
    }
    return a;
  };
  femorax::exec::RunOptions o;
  o.verbose = false;
  {
    Model m = cantilever(0);
    m.process.push_back({"1", "", stage(femorax::io::Integrator::Newmark)});
    EXPECT_THROW(femorax::exec::run_model(m, o), femorax::io::UnsupportedComponent);
  }
  {
    Model m = cantilever(0);
    femorax::io::RayleighRegion r;
    r.betaK = 1e-3;
    r.elements = m.elements[0].tags;
    m.rayleigh.push_back(r);
    m.process.push_back({"1", "", stage(femorax::io::Integrator::CentralDifference)});
    EXPECT_THROW(femorax::exec::run_model(m, o), femorax::io::ModelError);
  }
  {
    Model m = cantilever(0);
    m.elements[0].beam.density = 0.0;
    m.process.push_back({"1", "", stage(femorax::io::Integrator::CentralDifference)});
    EXPECT_THROW(femorax::exec::run_model(m, o), femorax::io::ModelError);
  }
}

}  // namespace
