// DispBeam3d through the real kernels (docs/design.md, "Beams", step 4a: explicit stages): a
// vertical cantilever of six dispBeamColumn elements (6-DOF nodes, base fixed), the section
// elastic, Linear and PDelta, 3 integration points.
//   - the force kernel equals the host reference assembly's K u (tests/common), and every force
//     kernel variant gives the same bits (atomic: to rounding);
//   - the lumped mass: rho A L on each translation, the rotary inertia on the rotations;
//   - an explicit CentralDifference run under base shaking (run_model) follows the host
//     recurrence with the reference stiffness and the compile step's lumped mass;
//   - step 4b: the matrix-free operator cK K_t + cC C + cM M, its diagonal and the ground-motion
//     mass term equal the host reference assembly (Linear and PDelta at a displaced state, lumped
//     and consistent mass, all four Rayleigh terms, the initial-stiffness variant); a Newmark run
//     with Newton under base shaking (run_model) equals host Newmark on the reference matrices.

#include <Eigen/SparseCholesky>
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
#include "../common/synthetic_models.hpp"
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

// What beams refuse in explicit stages: Rayleigh stiffness damping, and beams without a mass of
// their own (their nodal masses do not reach the stable step yet). Implicit stages run.
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
    EXPECT_NO_THROW(femorax::exec::run_model(m, o));
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
  {  // density 0 and no nodal masses: free DOFs without mass
    Model m = cantilever(0);
    m.elements[0].beam.density = 0.0;
    m.process.push_back({"1", "", stage(femorax::io::Integrator::CentralDifference)});
    EXPECT_THROW(femorax::exec::run_model(m, o), femorax::io::ModelError);
  }
}

TEST(BeamKernels, OperatorEqualsTheReferenceAssembly) {
  using femorax::exec::Vec;
  for (const int transf : {0, 1})
    for (const bool consistent : {false, true}) {
      Model m = cantilever(transf);
      m.elements[0].beam.consistent_mass = consistent;
      femorax::io::RayleighRegion r;
      r.alphaM = 0.3, r.betaK = 1e-3, r.betaKinit = 5e-4, r.betaKcomm = 2e-4;
      r.elements = m.elements[0].tags;
      m.rayleigh.push_back(r);
      const auto c = femorax::compile::compile_model(m, femorax::compile::CompileOptions{});
      const auto n = static_cast<std::size_t>(c.dofs.num_equations);
      std::vector<double> u(n), x(n);
      for (std::size_t i = 0; i < n; ++i) {
        u[i] = 2e-4 * std::sin(0.31 * static_cast<double>(i) + 0.4);
        x[i] = std::cos(0.53 * static_cast<double>(i));
      }
      const auto ref = femorax::testing::assemble_reference(m, c, u);  // K_t at u
      femorax::exec::InternalForce force(c, femorax::exec::GeometryCache::Auto);
      auto upload = [&](const std::vector<double>& h) {
        Vec d("v", n);
        auto hm = Kokkos::create_mirror_view(d);
        for (std::size_t i = 0; i < n; ++i) hm(i) = h[i];
        Kokkos::deep_copy(d, hm);
        return d;
      };
      const Vec du = upload(u), dx = upload(x), dy("y", n);
      for (const bool initial : {false, true}) {
        femorax::OperatorCoefficients k;
        k.stiffness = 1.3, k.damping = 0.7, k.mass = 2.1, k.initial = initial;
        const auto Kx = ref.multiply(initial ? ref.K0 : ref.K, x), Cx = ref.multiply(ref.C, x),
                   Mx = ref.multiply(ref.M, x);
        femorax::exec::OperatorSettings s;
        force.apply_operator(k, s, du, dx, dy);
        const auto y = host(dy);
        double d = 0.0, mm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          const double want = k.stiffness * Kx[i] + k.damping * Cx[i] + k.mass * Mx[i];
          d = std::max(d, std::abs(y[i] - want));
          mm = std::max(mm, std::abs(want));
        }
        EXPECT_LE(d, 1e-12 * mm) << "transf " << transf << " consistent " << consistent << " initial " << initial;
        // The diagonal.
        force.operator_diagonal(k, s, du, dy);
        const auto dg = host(dy);
        double dd = 0.0, dm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          const auto pos = static_cast<std::size_t>(ref.pattern.find(static_cast<std::int32_t>(i), static_cast<std::int32_t>(i)));
          const double want = k.stiffness * (initial ? ref.K0 : ref.K)[pos] + k.damping * ref.C[pos] + k.mass * ref.M[pos];
          dd = std::max(dd, std::abs(dg[i] - want));
          dm = std::max(dm, std::abs(want));
        }
        EXPECT_LE(dd, 1e-12 * dm) << "diagonal: transf " << transf << " consistent " << consistent;
      }
      // M r_d, the UniformExcitation load (the base's DOFs included in r).
      for (int dir = 0; dir < 3; ++dir) {
        force.mass_times_ground(dir, femorax::exec::OperatorSettings{}, du, dy);
        const auto mr = host(dy);
        double d = 0.0, mm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          d = std::max(d, std::abs(mr[i] - ref.Mr[dir][i]));
          mm = std::max(mm, std::abs(ref.Mr[dir][i]));
        }
        EXPECT_LE(d, 1e-12 * mm) << "M r, direction " << dir;
      }
    }
}

// Newmark (average acceleration) with Newton, Rayleigh damping (alphaM and betaK) and a base
// pulse in x, through run_model; the host does Newmark on the reference K, C, M (Eigen LDLT).
TEST(BeamKernels, NewmarkRunFollowsTheHostRecurrence) {
  for (const bool consistent : {false, true}) {
    Model m = cantilever(0);
    m.elements[0].beam.consistent_mass = consistent;
    femorax::io::RayleighRegion r;
    r.alphaM = 0.5, r.betaK = 2e-4;
    r.elements = m.elements[0].tags;
    m.rayleigh.push_back(r);
    femorax::compile::CompileOptions co;
    co.renumbering = femorax::compile::Renumbering::Off;
    const auto c = femorax::compile::compile_model(m, co);
    const auto ref = femorax::testing::assemble_reference(m, c);
    const double dt = 1e-3;
    const int steps = 200;
    femorax::io::PathSeries ps;
    ps.tag = 1;
    ps.dt = dt;
    for (int k = 0; k <= steps + 2; ++k) {
      const double t = k * dt, T = 0.05;
      ps.values.push_back(t < T ? std::sin(M_PI * t / T) * std::sin(M_PI * t / T) : 0.0);
    }
    m.time_series.push_back(ps);
    m.patterns.push_back({2, 1, 1, 0.0, 1.0, {}});
    femorax::io::NodeRecorder nr;
    nr.file = temp_file(consistent ? "newmark_c.txt" : "newmark_l.txt");
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
    a.type = femorax::io::AnalysisType::Transient;
    a.dt = dt;
    a.num_steps = steps;
    a.integrator = femorax::io::Integrator::Newmark;
    a.gamma = 0.5;
    a.beta = 0.25;
    a.algorithm = femorax::io::Algorithm::Newton;
    a.numberer = "Plain";
    a.system = "BandGeneral";
    a.test_type = femorax::io::TestType::NormDispIncr;
    a.test_tol = 1e-14;
    a.test_max_iter = 10;
    m.process.push_back({"2", "", a});
    femorax::exec::RunOptions o;
    o.verbose = false;
    o.renumbering = femorax::compile::Renumbering::Off;
    femorax::exec::run_model(m, o);
    const auto rows = read_rows(nr.file);
    ASSERT_EQ(rows.size(), static_cast<std::size_t>(steps));

    // Host Newmark: Khat u(n+1) = p(n+1) + M (a0 u + a2 v + a3 a) + C (a1 u + a4 v + a5 a).
    const double g = 0.5, b = 0.25;
    const double a0 = 1.0 / (b * dt * dt), a1 = g / (b * dt), a2 = 1.0 / (b * dt), a3 = 1.0 / (2 * b) - 1.0,
                 a4 = g / b - 1.0, a5 = dt / 2.0 * (g / b - 2.0);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(ref.to_eigen(ref.effective(1.0, a1, a0)));
    const auto nq = static_cast<std::size_t>(c.dofs.num_equations);
    std::vector<double> u(nq, 0.0), v(nq, 0.0), acc(nq, 0.0);
    double peak = 0.0, err = 0.0;
    for (int k = 0; k < steps; ++k) {
      std::vector<double> wm(nq), wc(nq);
      for (std::size_t q = 0; q < nq; ++q) {
        wm[q] = a0 * u[q] + a2 * v[q] + a3 * acc[q];
        wc[q] = a1 * u[q] + a4 * v[q] + a5 * acc[q];
      }
      const auto Mw = ref.multiply(ref.M, wm), Cw = ref.multiply(ref.C, wc);
      Eigen::VectorXd rhs(static_cast<Eigen::Index>(nq));
      for (std::size_t q = 0; q < nq; ++q)
        rhs[static_cast<Eigen::Index>(q)] = -ps.values[static_cast<std::size_t>(k + 1)] * ref.Mr[0][q] + Mw[q] + Cw[q];
      const Eigen::VectorXd un = ldlt.solve(rhs);
      for (std::size_t q = 0; q < nq; ++q) {
        const double unq = un[static_cast<Eigen::Index>(q)];
        const double an = a0 * (unq - u[q]) - a2 * v[q] - a3 * acc[q];
        v[q] += dt * ((1.0 - g) * acc[q] + g * an);
        acc[q] = an;
        u[q] = unq;
      }
      for (std::int32_t node = 0; node < c.dofs.num_nodes(); ++node)
        for (int d = 0; d < 6; ++d) {
          const std::int32_t q = c.dofs.equation(node, d);
          if (q < 0) continue;
          const double run = rows[static_cast<std::size_t>(k)][static_cast<std::size_t>(1 + 6 * node + d)];
          peak = std::max(peak, std::abs(u[static_cast<std::size_t>(q)]));
          err = std::max(err, std::abs(run - u[static_cast<std::size_t>(q)]));
        }
    }
    std::printf("[ Newmark cantilever (%s mass) against the host recurrence: %.2e of the peak %.3e ]\n",
                consistent ? "consistent" : "lumped", err / peak, peak);
    EXPECT_GT(peak, 0.0);
    EXPECT_LE(err, 1e-9 * peak);
  }
}

// docs/todo.md item 22: a beam whose mass sits on its nodes (density 0, Femora's export today)
// gets the same stable step as the beam with the density, when the nodal masses are the
// element masses lumped to the nodes (each element's share is then exactly its own mass).
TEST(BeamKernels, NodalMassesEnterTheStableStep) {
  const Model dense = cantilever(0);
  femorax::compile::CompileOptions co;
  co.renumbering = femorax::compile::Renumbering::Off;
  const auto cd = femorax::compile::compile_model(dense, co);
  Model light = dense;
  light.elements[0].beam.density = 0.0;
  // The element masses as nodal masses, DOF by DOF (the compile step's lumped mass of `dense`,
  // free DOFs only; the base node's are fixed).
  for (std::int32_t node = 0; node < cd.dofs.num_nodes(); ++node)
    for (int d = 0; d < 6; ++d) {
      const std::int32_t q = cd.dofs.equation(node, d);
      if (q < 0) continue;
      light.mass.node.push_back(cd.dofs.node_tag[static_cast<std::size_t>(node)]);
      light.mass.dof.push_back(d + 1);
      light.mass.value.push_back(cd.lumped_mass[static_cast<std::size_t>(q)]);
    }
  const auto cl = femorax::compile::compile_model(light, co);
  double dt_dense = 0.0, dt_light = 0.0;
  {
    femorax::exec::InternalForce f(cd, femorax::exec::GeometryCache::Auto);
    dt_dense = f.min_critical_dt(femorax::ExplicitScheme::CentralDifference);
  }
  {
    femorax::exec::InternalForce f(cl, femorax::exec::GeometryCache::Auto);
    dt_light = f.min_critical_dt(femorax::ExplicitScheme::CentralDifference);
  }
  std::printf("[ stable step: density %.6e, nodal masses %.6e ]\n", dt_dense, dt_light);
  ASSERT_TRUE(std::isfinite(dt_light));
  // The two partitions differ only at the base element (its base node is fixed: its share of
  // that node is not there in either model) and in rounding.
  EXPECT_NEAR(dt_light, dt_dense, 1e-6 * dt_dense);
  // And the run itself goes through (an explicit stage with only nodal masses on the beams).
  Model run = light;
  femorax::io::AnalysisStep a;
  a.name = "s";
  a.dt = 0.5 * dt_light;
  a.num_steps = 5;
  a.integrator = femorax::io::Integrator::CentralDifference;
  run.process.push_back({"1", "", a});
  femorax::exec::RunOptions o;
  o.verbose = false;
  EXPECT_NO_THROW(femorax::exec::run_model(run, o));
}

// The local time stepping checkpoint for beams (docs/design.md, "Beams", step 6): a steel pile
// along the centre line of an undamped 4 x 4 x 8 soil column, its 6-DOF nodes tied to the soil
// nodes in x, y, z (equalDOF), base shaking with a smooth pulse. The pile's stable step is far
// below the soil's, so LTS puts the pile and the soil around it on fine levels; LTS at the soil's
// step must follow a global run at the finest level's step.
TEST(BeamKernels, LocalTimeSteppingWithAPileInSoil) {
  femorax::testing::ColumnSpec cs;
  cs.nx = cs.ny = 4;
  cs.nz = 8;
  cs.damping = false;
  Model m = femorax::testing::soil_column(cs);
  m.equal_dof = {};
  for (auto& g : m.elements) g.body_force = {0.0, 0.0, 0.0};
  const int NX = cs.nx + 1, NY = cs.ny + 1;
  auto soil_tag = [&](int i, int j, int k) { return 1 + i + NX * (j + NY * k); };
  // The pile: nodes 10001.. at the soil's centre-line nodes, one beam per soil layer of nodes.
  m.materials.push_back({1, femorax::io::ElasticSection3dData{2.0e8, 0.05, 4.0e-4, 4.0e-4, 7.7e7, 8.0e-4}, {}});
  femorax::io::ElementGroup pile;
  pile.id = "pile";
  pile.type = femorax::io::ElementType::DispBeamColumn;
  pile.material = 1;
  pile.beam.vecxz = {1.0, 0.0, 0.0};
  pile.beam.points = 3;
  pile.beam.density = 7.8;
  for (int k = 0; k <= cs.nz; ++k) {
    const int ts = soil_tag(2, 2, k);
    std::size_t idx = 0;
    while (m.nodes.tags[idx] != ts) ++idx;
    m.nodes.tags.push_back(10001 + k);
    m.nodes.ndf.push_back(6);
    m.nodes.coords.insert(m.nodes.coords.end(), {m.nodes.coords[3 * idx], m.nodes.coords[3 * idx + 1],
                                                 m.nodes.coords[3 * idx + 2]});
    for (int d = 1; d <= 3; ++d) {
      m.equal_dof.retained.push_back(ts);
      m.equal_dof.constrained.push_back(10001 + k);
      m.equal_dof.dof.push_back(d);
    }
    if (k < cs.nz) {
      pile.tags.push_back(50001 + k);
      pile.connectivity.insert(pile.connectivity.end(), {10001 + k, 10002 + k});
    }
  }
  m.elements.push_back(pile);
  // The stable steps, and the coarse step (0.9 of the soil's).
  femorax::compile::CompileOptions co;
  co.renumbering = femorax::compile::Renumbering::Off;
  const auto c = femorax::compile::compile_model(m, co);
  double soil_dt = 1e300, pile_dt = 1e300;
  {
    femorax::exec::InternalForce force(c, femorax::exec::GeometryCache::Auto);
    const auto dts = force.element_critical_dts(femorax::ExplicitScheme::CentralDifference);
    for (std::size_t b = 0; b < dts.size(); ++b)
      for (const double x : dts[b])
        (c.batches[b].element == femorax::io::ElementType::DispBeamColumn ? pile_dt : soil_dt) =
            std::min(c.batches[b].element == femorax::io::ElementType::DispBeamColumn ? pile_dt : soil_dt, x);
  }
  const double dt = 0.9 * soil_dt;
  const int steps = 150;
  // A smooth base pulse (zero value and slope at t = 0): the start from rest is step-independent.
  auto with_motion = [&](Model mm, double step, int n, const std::string& file, bool lts,
                         femorax::exec::RunReport* rep) {
    femorax::io::PathSeries ps;
    ps.tag = 1;
    ps.dt = step;
    const double T = 40 * dt;
    for (int k = 0; k <= n + 2; ++k) {
      const double t = k * step;
      ps.values.push_back(t < T ? std::pow(std::sin(M_PI * t / T), 2) : 0.0);
    }
    mm.time_series.push_back(ps);
    mm.patterns.push_back({2, 1, 1, 0.0, 1.0, {}});
    femorax::io::NodeRecorder nr;
    nr.file = temp_file(file);
    nr.response = femorax::io::NodeResponse::Disp;
    nr.time = true;
    nr.precision = 17;
    for (int k = 0; k <= cs.nz; ++k) nr.nodes.push_back(10001 + k);  // the pile
    nr.dofs = {1, 5};                                                 // x and the rotation about y
    mm.recorders.push_back({"d", nr, {}});
    mm.process.push_back({"0", "", femorax::io::RecorderStep{"d"}});
    mm.process.push_back({"1", "", femorax::io::PatternStep{2}});
    femorax::io::AnalysisStep a;
    a.name = "shake";
    a.dt = step;
    a.num_steps = n;
    a.integrator = femorax::io::Integrator::CentralDifference;
    mm.process.push_back({"2", "", a});
    femorax::exec::RunOptions o;
    o.verbose = false;
    o.renumbering = femorax::compile::Renumbering::Off;
    o.local_time_stepping = lts;
    const auto r = femorax::exec::run_model(mm, o);
    if (rep) *rep = r;
    return read_rows(nr.file);
  };
  femorax::exec::RunReport rep;
  const auto lts = with_motion(m, dt, steps, "lts.txt", true, &rep);
  const auto& st = rep.stages.at(0);
  const int p = st.lts_substeps;
  std::printf("[ pile in soil: soil step %.4e, pile step %.4e (ratio %.1f); %d levels, %d substeps, %lld fine "
              "equations, %lld of %lld elements in the substeps, estimated gain %.2fx ]\n",
              soil_dt, pile_dt, soil_dt / pile_dt, st.lts_levels, p, static_cast<long long>(st.lts_fine_equations),
              static_cast<long long>(st.lts_substep_elements), static_cast<long long>(rep.elements), st.lts_speedup);
  ASSERT_GE(st.lts_levels, 2);
  const auto fine = with_motion(m, dt / p, steps * p, "fine.txt", false, nullptr);
  ASSERT_EQ(lts.size(), static_cast<std::size_t>(steps));
  double peak = 0.0, err = 0.0;
  for (int k = 0; k < steps; ++k) {
    const auto& a = lts[static_cast<std::size_t>(k)];
    const auto& b = fine[static_cast<std::size_t>((k + 1) * p - 1)];
    for (std::size_t col = 1; col < a.size(); col += 2) {  // x displacements
      peak = std::max(peak, std::abs(b[col]));
      err = std::max(err, std::abs(a[col] - b[col]));
    }
  }
  std::printf("[ pile x displacement, LTS at the soil's step against a global run at dt/%d: %.2e of the peak %.3e ]\n",
              p, err / peak, peak);
  EXPECT_GT(peak, 0.0);
  EXPECT_LE(err, 1e-2 * peak);  // measured 3.05e-3 (the coarse step's time discretization)
}

// Constraints step 2 (docs/design.md, "Exact constraints u = T q"): the pile tied into the soil
// column by embedded nodes (each pile node in a brick of which its soil node is a corner, so the
// weights are one at that corner and zero elsewhere) runs as the same pile tied by equalDOF, to
// rounding: an explicit CentralDifference run under base shaking, the pile's x displacements.
TEST(BeamKernels, EmbeddedPileRunsAsTheEqualDofPile) {
  auto pile_model = [](bool embedded) {
    femorax::testing::ColumnSpec cs;
    cs.nx = cs.ny = 4;
    cs.nz = 8;
    cs.damping = false;
    Model m = femorax::testing::soil_column(cs);
    m.equal_dof = {};
    for (auto& g : m.elements) g.body_force = {0.0, 0.0, 0.0};
    const int NX = cs.nx + 1, NY = cs.ny + 1;
    auto soil_tag = [&](int i, int j, int k) { return 1 + i + NX * (j + NY * k); };
    m.materials.push_back({1, femorax::io::ElasticSection3dData{2.0e8, 0.05, 4.0e-4, 4.0e-4, 7.7e7, 8.0e-4}, {}});
    femorax::io::ElementGroup pile;
    pile.id = "pile";
    pile.type = femorax::io::ElementType::DispBeamColumn;
    pile.material = 1;
    pile.beam.vecxz = {1.0, 0.0, 0.0};
    pile.beam.points = 3;
    pile.beam.density = 7.8;
    for (int k = 1; k <= cs.nz; ++k) {
      const int ts = soil_tag(2, 2, k);
      std::size_t idx = 0;
      while (m.nodes.tags[idx] != ts) ++idx;
      m.nodes.tags.push_back(10000 + k);
      m.nodes.ndf.push_back(6);
      m.nodes.coords.insert(m.nodes.coords.end(), {m.nodes.coords[3 * idx], m.nodes.coords[3 * idx + 1],
                                                   m.nodes.coords[3 * idx + 2]});
      if (embedded) {
        // A brick that has the soil node as a corner.
        int host = -1;
        for (const auto& g : m.elements) {
          if (g.type != femorax::io::ElementType::StdBrick) continue;
          for (std::size_t e = 0; e < g.size() && host < 0; ++e)
            for (int a = 0; a < 8; ++a)
              if (g.connectivity[8 * e + static_cast<std::size_t>(a)] == ts) host = g.tags[e];
          if (host >= 0) break;
        }
        m.embedded_nodes.push_back({10000 + k, host, false});
      } else {
        for (int d = 1; d <= 3; ++d) {
          m.equal_dof.retained.push_back(ts);
          m.equal_dof.constrained.push_back(10000 + k);
          m.equal_dof.dof.push_back(d);
        }
      }
      if (k < cs.nz) {
        pile.tags.push_back(50000 + k);
        pile.connectivity.insert(pile.connectivity.end(), {10000 + k, 10001 + k});
      }
    }
    m.elements.push_back(pile);
    return m;
  };
  auto run_it = [&](Model mm, const std::string& file, double dt, int n) {
    femorax::io::PathSeries ps;
    ps.tag = 1;
    ps.dt = dt;
    const double T = 40 * dt;
    for (int k = 0; k <= n + 2; ++k) {
      const double t = k * dt;
      ps.values.push_back(t < T ? std::pow(std::sin(M_PI * t / T), 2) : 0.0);
    }
    mm.time_series.push_back(ps);
    mm.patterns.push_back({2, 1, 1, 0.0, 1.0, {}});
    femorax::io::NodeRecorder nr;
    nr.file = temp_file(file);
    nr.response = femorax::io::NodeResponse::Disp;
    nr.time = true;
    nr.precision = 17;
    for (int k = 1; k <= 8; ++k) nr.nodes.push_back(10000 + k);
    nr.dofs = {1, 5};
    mm.recorders.push_back({"d", nr, {}});
    mm.process.push_back({"0", "", femorax::io::RecorderStep{"d"}});
    mm.process.push_back({"1", "", femorax::io::PatternStep{2}});
    femorax::io::AnalysisStep a;
    a.name = "shake";
    a.dt = dt;
    a.num_steps = n;
    a.integrator = femorax::io::Integrator::CentralDifference;
    mm.process.push_back({"2", "", a});
    femorax::exec::RunOptions o;
    o.verbose = false;
    femorax::exec::run_model(mm, o);
    return read_rows(nr.file);
  };
  const Model tied = pile_model(false), emb = pile_model(true);
  ASSERT_EQ(emb.embedded_nodes.size(), 8u);
  double dt = 0.0;
  {
    const auto c = femorax::compile::compile_model(tied, femorax::compile::CompileOptions{});
    femorax::exec::InternalForce f(c, femorax::exec::GeometryCache::Auto);
    dt = 0.5 * f.min_critical_dt(femorax::ExplicitScheme::CentralDifference);
  }
  const int steps = 300;
  const auto a = run_it(tied, "tied.txt", dt, steps), b = run_it(emb, "embedded.txt", dt, steps);
  ASSERT_EQ(a.size(), b.size());
  double peak = 0.0, err = 0.0;
  for (std::size_t k = 0; k < a.size(); ++k)
    for (std::size_t col = 1; col < a[k].size(); col += 2) {
      peak = std::max(peak, std::abs(a[k][col]));
      err = std::max(err, std::abs(a[k][col] - b[k][col]));
    }
  std::printf("[ embedded pile against the equalDOF pile (explicit, %d steps): %.2e of the peak %.3e ]\n", steps,
              err / peak, peak);
  EXPECT_GT(peak, 0.0);
  EXPECT_LE(err, 1e-10 * peak);
}

}  // namespace
