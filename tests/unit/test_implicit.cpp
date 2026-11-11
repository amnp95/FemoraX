// Implicit stages (exec/implicit_stage.hpp) through femorax's run_model on the laminar soil
// column, against references computed on the host from the assembled matrices
// (tests/common/reference_assembly.hpp) with Eigen direct solves:
//   - Static LoadControl gravity: the exact solution K u = F (and Linear = Newton).
//   - Newmark (form D) with a base motion: the Newmark recurrence of OpenSees, step by step,
//     with exact solves; Linear, Newton and ModifiedNewton, every convergence test;
//     consistent and lumped mass, all Rayleigh terms.
//   - A step that cannot converge stops the run with a message.
//   - Femora's retry and substep recovery (schema 0.5): a step that fails at full size is
//     retried, then split into substeps, as the host recurrence does with the same rule.

#include <Eigen/SparseCholesky>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "../common/reference_assembly.hpp"
#include "../common/synthetic_models.hpp"
#include "compile/batches.hpp"
#include "compile/registry.hpp"
#include "exec/explicit_analysis.hpp"
#include "exec/force_settings.hpp"
#include "exec/internal_force.hpp"
#include "io/h5.hpp"

namespace {

namespace fs = std::filesystem;
using namespace femorax;

const int kNx = 2, kNz = 8;

io::Model column(bool lumped, io::ElementType type = io::ElementType::StdBrick) {
  femorax::testing::ColumnSpec s;
  s.nx = s.ny = kNx;
  s.nz = kNz;
  s.lumped = lumped;
  s.element = type;
  auto m = femorax::testing::soil_column(s);
  m.rayleigh[1].betaKcomm = 2e-5;  // every Rayleigh term somewhere
  return m;
}


void add_recorder(io::Model& m, const std::string& file) {
  io::NodeRecorder nr;
  nr.file = file;
  nr.response = io::NodeResponse::Disp;
  nr.time = true;
  nr.precision = 17;
  for (int k = 0; k <= kNz; ++k) nr.nodes.push_back(1 + (kNx + 1) * (kNx + 1) * k);
  nr.dofs = {1, 2, 3};
  m.recorders.push_back({"disp", nr, {}});
}

io::AnalysisStep newmark_step(io::Algorithm alg, double dt, int steps) {
  io::AnalysisStep a;
  a.name = "shake";
  a.type = io::AnalysisType::Transient;
  a.dt = dt;
  a.num_steps = steps;
  a.integrator = io::Integrator::Newmark;
  a.gamma = 0.5;
  a.beta = 0.25;
  a.algorithm = alg;
  a.numberer = "Plain";
  a.system = "BandGeneral";
  a.test_type = io::TestType::NormUnbalance;
  a.test_tol = 1e-9;
  a.test_max_iter = 20;
  return a;
}

io::AnalysisStep static_step(io::Algorithm alg) {
  io::AnalysisStep a;
  a.name = "gravity";
  a.type = io::AnalysisType::Static;
  a.num_steps = 3;
  a.integrator = io::Integrator::LoadControl;
  a.load_increment = 1.0 / 3.0;
  a.algorithm = alg;
  a.numberer = "Plain";
  a.system = "BandGeneral";
  a.test_type = io::TestType::NormDispIncr;
  a.test_tol = 1e-12;
  a.test_max_iter = 10;
  return a;
}

// A base motion: 0.2 g sine at 3 Hz, uniform dt 0.005.
void add_motion(io::Model& m) {
  io::PathSeries s;
  s.tag = 1;
  s.factor = 9.81;
  s.dt = 0.005;
  for (int i = 0; i <= 200; ++i) s.values.push_back(0.2 * std::sin(2 * M_PI * 3.0 * i * 0.005));
  m.time_series.push_back(s);
  m.patterns.push_back({2, 1, 1, 0.0, 1.0, {}});
}

double ag_at(const io::PathSeries& s, double t) {  // PathSeries::getFactor
  if (t < 0.0) return 0.0;
  const double incr = t / *s.dt;
  const auto i1 = static_cast<long long>(std::floor(incr));
  if (i1 + 1 >= static_cast<long long>(s.values.size())) return 0.0;
  return s.factor * (s.values[i1] + (s.values[i1 + 1] - s.values[i1]) * (incr - static_cast<double>(i1)));
}

std::vector<std::vector<double>> read_rows(const std::string& file) {
  std::vector<std::vector<double>> rows;
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::vector<double> r;
    double x;
    while (s >> x) r.push_back(x);
    rows.push_back(r);
  }
  return rows;
}

// Per process: ctest runs the cases in parallel, and the whole binary at the same time.
std::string temp_file(const std::string& name) {
  const fs::path d = fs::temp_directory_path() / ("femorax_test_implicit_" + std::to_string(::getpid()));
  fs::create_directories(d);
  return (d / name).string();
}

std::vector<std::vector<double>> run(io::Model m, const std::string& name) {
  const std::string file = temp_file(name);
  add_recorder(m, file);
  m.process.insert(m.process.begin(), {"recorder", "", io::RecorderStep{"disp"}});
  exec::RunOptions o;
  o.verbose = false;
  o.renumbering = compile::Renumbering::Off;
  exec::run_model(m, o);
  return read_rows(file);
}

// The recorded line (equation space) from a displacement vector.
std::vector<double> recorded(const compile::Compiled& c, const Eigen::VectorXd& u) {
  std::vector<double> r;
  for (int k = 0; k <= kNz; ++k) {
    const std::int32_t tag = 1 + (kNx + 1) * (kNx + 1) * k;
    std::int32_t i = 0;
    while (c.dofs.node_tag[i] != tag) ++i;
    for (int d = 0; d < 3; ++d) {
      const std::int32_t q = c.dofs.equation(i, d);
      r.push_back(q >= 0 ? u[q] : 0.0);
    }
  }
  return r;
}

double max_rel(const std::vector<double>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::abs(a[i] - b[i]));
    den = std::max(den, std::abs(b[i]));
  }
  return den > 0 ? num / den : num;
}

// The body force load P = -f(0, 0) in equation space.
Eigen::VectorXd body_force(const compile::Compiled& c) {
  exec::InternalForce k(c, exec::GeometryCache::Off);
  k.fill(0.0, 0.0);
  k.compute(exec::default_force_settings());
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k.f);
  Eigen::VectorXd p(static_cast<Eigen::Index>(h.extent(0)));
  for (std::size_t i = 0; i < h.extent(0); ++i) p[static_cast<Eigen::Index>(i)] = -h(i);
  return p;
}

TEST(Implicit, StaticLoadControlGravityIsTheExactSolution) {
  for (auto alg : {io::Algorithm::Linear, io::Algorithm::Newton, io::Algorithm::ModifiedNewton}) {
    SCOPED_TRACE(static_cast<int>(alg));
    io::Model m = column(false);
    m.process.push_back({"000000", "", static_step(alg)});
    compile::CompileOptions co;
    co.renumbering = compile::Renumbering::Off;
    const compile::Compiled c = compile::compile_model(m, co);
    const auto ref = femorax::testing::assemble_reference(m, c);
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(ref.to_eigen(ref.K));
    const Eigen::VectorXd u = ldlt.solve(body_force(c));  // the body force is constant: full at step 1
    const auto rows = run(m, "static.txt");
    ASSERT_EQ(rows.size(), 3u);
    EXPECT_NEAR(rows[0][0], 1.0 / 3.0, 1e-15);  // time = lambda
    EXPECT_NEAR(rows[2][0], 1.0, 1e-15);
    const auto want = recorded(c, u);
    for (const auto& r : rows) {
      const std::vector<double> got(r.begin() + 1, r.end());
      EXPECT_LE(max_rel(got, want), 1e-9);
    }
  }
}

// OpenSees Newmark (form D) on the host with exact solves: the reference history.
std::vector<std::vector<double>> host_newmark(const io::Model& m, const compile::Compiled& c, double dt, int steps) {
  const auto ref = femorax::testing::assemble_reference(m, c);
  const double gamma = 0.5, beta = 0.25;
  const double c2 = gamma / (beta * dt), c3 = 1.0 / (beta * dt * dt);
  const auto K = ref.to_eigen(ref.K), C = ref.to_eigen(ref.C), M = ref.to_eigen(ref.M);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> keff(ref.to_eigen(ref.effective(1.0, c2, c3)));
  const Eigen::Index n = K.rows();
  // The ground load couples the free DOFs to the moving base (consistent mass): M r over
  // every element DOF, fixed ones included.
  const Eigen::VectorXd Mr = Eigen::Map<const Eigen::VectorXd>(ref.Mr[0].data(), n), P = body_force(c);
  Eigen::VectorXd U = Eigen::VectorXd::Zero(n), V = U, A = U;
  std::vector<std::vector<double>> out;
  double t = 0.0;
  for (int k = 0; k < steps; ++k) {
    const Eigen::VectorXd Vt = V, At = A;
    V = V * (1.0 - gamma / beta) + At * (dt * (1.0 - 0.5 * gamma / beta));
    A = A * (1.0 - 0.5 / beta) + Vt * (-1.0 / (beta * dt));
    t += dt;
    const double ag = ag_at(m.time_series[0], t);
    // Linear problem: one exact solve converges.
    const Eigen::VectorXd B = P - K * U - C * V - M * A - ag * Mr;
    const Eigen::VectorXd dU = keff.solve(B);
    U += dU;
    V += c2 * dU;
    A += c3 * dU;
    std::vector<double> row = {t};
    const auto rec = recorded(c, U);
    row.insert(row.end(), rec.begin(), rec.end());
    out.push_back(row);
  }
  return out;
}

TEST(Implicit, NewmarkMatchesTheHostRecurrence) {
  const double dt = 2e-3;
  const int steps = 40;
  // Each algorithm and each test once, consistent and lumped mass (kept small: the
  // unit tests also run as a whole with 8 threads next to the per-case runs).
  struct Case {
    bool lumped;
    io::Algorithm alg;
    io::TestType test;
  };
  const Case cases[] = {{false, io::Algorithm::Linear, io::TestType::NormUnbalance},
                        {false, io::Algorithm::Newton, io::TestType::NormUnbalance},
                        {true, io::Algorithm::Newton, io::TestType::NormDispIncr},
                        {false, io::Algorithm::ModifiedNewton, io::TestType::EnergyIncr},
                        {true, io::Algorithm::Linear, io::TestType::NormUnbalance}};
  for (const auto& cs : cases) {
    const bool lumped = cs.lumped;
    const auto alg = cs.alg;
    const auto test = cs.test;
    {
      {
        io::Model m = column(lumped);
        add_motion(m);
        m.process.push_back({"000000", "", io::PatternStep{2}});
        auto a = newmark_step(alg, dt, steps);
        a.test_type = test;
        a.test_tol = test == io::TestType::NormUnbalance ? 1e-8 : test == io::TestType::NormDispIncr ? 1e-14 : 1e-20;
        m.process.push_back({"000001", "", a});
        compile::CompileOptions co;
        co.renumbering = compile::Renumbering::Off;
        const compile::Compiled c = compile::compile_model(m, co);
        const auto want = host_newmark(m, c, dt, steps);
        const auto got = run(m, "newmark.txt");
        ASSERT_EQ(got.size(), want.size());
        double worst = 0.0, scale = 0.0;
        for (std::size_t k = 0; k < got.size(); ++k) {
          ASSERT_EQ(got[k][0], want[k][0]) << "time of step " << k;  // t += dt, as OpenSees
          for (std::size_t j = 1; j < got[k].size(); ++j) {
            worst = std::max(worst, std::abs(got[k][j] - want[k][j]));
            scale = std::max(scale, std::abs(want[k][j]));
          }
        }
        EXPECT_GT(scale, 1e-5);
        EXPECT_LE(worst, 1e-8 * scale);
      }
    }
  }
}

// Uniform damping in an implicit stage (docs/design.md, "Frequency-independent damping",
// step 3). The damped material's tangent is D (km - kcorr), the exact derivative of its
// stress, so the stored-tangent operator is Newton's consistent tangent. Reference: the
// Newmark recurrence on the host with the residual from the production force kernel (which
// carries the damping) and Newton to a tight tolerance with an approximate matrix, the
// elastic K_eff with K scaled by km - kcorr: the converged history does not depend on the
// matrix used.
TEST(Implicit, UniformDampingNewmarkMatchesTheHostRecurrence) {
  for (const auto type : {io::ElementType::StdBrick, io::ElementType::SSPbrick}) {
  SCOPED_TRACE(io::to_string(type));
  const double dt = 2e-3;
  const int steps = 40;
  io::Model m = column(false, type);
  io::UniformDampingRegion r;
  r.id = "soil";
  r.ratio = 0.05;
  r.freq1 = 0.1;
  r.freq2 = 10.0;
  for (const auto& g : m.elements) r.elements.insert(r.elements.end(), g.tags.begin(), g.tags.end());
  m.uniform.push_back(r);
  add_motion(m);
  m.process.push_back({"000000", "", io::PatternStep{2}});
  auto a = newmark_step(io::Algorithm::Newton, dt, steps);
  a.test_tol = 1e-10;
  m.process.push_back({"000001", "", a});
  compile::CompileOptions co;
  co.renumbering = compile::Renumbering::Off;
  const compile::Compiled c = compile::compile_model(m, co);
  ASSERT_GT(c.batches[0].nstate, 0);  // damped batches

  // the host recurrence
  const auto ref = femorax::testing::assemble_reference(m, c);
  const auto fl = registry::uniform_damping_filters(r);
  UniformDampingStep st;
  st.dynamic = true;
  st.dT = dt;
  st.t = dt;
  const double km = uniform_damping_stiffness_multiplier(fl, uniform_damping_eta(r.ratio), st) - fl.kcorr;
  const double gamma = 0.5, beta = 0.25, c2 = gamma / (beta * dt), c3 = 1.0 / (beta * dt * dt);
  const auto M = ref.to_eigen(ref.M);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> keff(ref.to_eigen(ref.effective(km, c2 * km, c3)));
  const Eigen::Index n = M.rows();
  const Eigen::VectorXd Mr = Eigen::Map<const Eigen::VectorXd>(ref.Mr[0].data(), n);
  exec::InternalForce k(c, exec::GeometryCache::Off);
  const auto settings = exec::default_force_settings();
  auto hu = Kokkos::create_mirror_view(k.u);
  auto hv = Kokkos::create_mirror_view(k.v);
  auto force = [&](const Eigen::VectorXd& U, const Eigen::VectorXd& V, double t) {
    for (Eigen::Index q = 0; q < n; ++q) {
      hu(q) = U[q];
      hv(q) = V[q];
    }
    Kokkos::deep_copy(k.u, hu);
    Kokkos::deep_copy(k.v, hv);
    k.step = StepContext{t, dt, 1};
    k.compute(settings);
    auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k.f);
    Eigen::VectorXd F(n);
    for (Eigen::Index q = 0; q < n; ++q) F[q] = hf(q);
    return F;
  };
  Eigen::VectorXd U = Eigen::VectorXd::Zero(n), V = U, A = U;
  std::vector<std::vector<double>> want;
  double t = 0.0;
  int host_iterations = 0;
  for (int s = 0; s < steps; ++s) {
    const Eigen::VectorXd Vt = V, At = A;
    V = V * (1.0 - gamma / beta) + At * (dt * (1.0 - 0.5 * gamma / beta));
    A = A * (1.0 - 0.5 / beta) + Vt * (-1.0 / (beta * dt));
    t += dt;
    const double ag = ag_at(m.time_series[0], t);
    for (int it = 0;; ++it) {
      ASSERT_LT(it, 100) << "host Newton, step " << s;
      const Eigen::VectorXd R = -force(U, V, t) - M * A - ag * Mr;
      if (R.lpNorm<Eigen::Infinity>() <= 1e-11) break;
      const Eigen::VectorXd dU = keff.solve(R);
      U += dU;
      V += c2 * dU;
      A += c3 * dU;
      ++host_iterations;
    }
    k.commit_state();
    std::vector<double> row = {t};
    const auto rec = recorded(c, U);
    row.insert(row.end(), rec.begin(), rec.end());
    want.push_back(row);
  }

  // the production run
  const std::string file = temp_file("uniform.txt");
  io::Model mr = m;
  add_recorder(mr, file);
  mr.process.insert(mr.process.begin(), {"recorder", "", io::RecorderStep{"disp"}});
  exec::RunOptions o;
  o.verbose = false;
  o.renumbering = compile::Renumbering::Off;
  const auto rep = exec::run_model(mr, o);
  const auto got = read_rows(file);
  ASSERT_EQ(got.size(), want.size());
  double worst = 0.0, scale = 0.0;
  for (std::size_t q = 0; q < got.size(); ++q) {
    ASSERT_EQ(got[q][0], want[q][0]);
    for (std::size_t j = 1; j < got[q].size(); ++j) {
      worst = std::max(worst, std::abs(got[q][j] - want[q][j]));
      scale = std::max(scale, std::abs(want[q][j]));
    }
  }
  const auto& stage = rep.stages.back();
  std::printf("[ %s, Uniform damping, Newmark + Newton: within %.2e of the host recurrence (peak %.3e); Newton %.2f "
              "iterations per step (host, approximate matrix: %.2f) ]\n",
              io::to_string(type), worst / scale, scale, static_cast<double>(stage.newton_iterations) / steps,
              static_cast<double>(host_iterations) / steps);
  EXPECT_GT(scale, 1e-5);
  EXPECT_LE(worst, 1e-8 * scale);
  // the consistent tangent: Newton converges in a few iterations per step
  EXPECT_LE(stage.newton_iterations, 3 * steps);
  }
}

TEST(Implicit, GravityThenShakingAsFemoraWritesIt) {
  // Femora's layered column: implicit-dynamic gravity (dt = 1, Newmark 0.6 / 0.3025), then the
  // pattern, setTime 0, and the shaking stage. The gravity stage settles to the static
  // solution; the run completes.
  io::Model m = column(false);
  add_motion(m);
  auto g = newmark_step(io::Algorithm::Newton, 1.0, 30);
  g.name = "gravity";
  g.gamma = 0.6;
  g.beta = 0.3025;
  m.process.push_back({"000000", "", g});
  m.process.push_back({"000001", "", io::PatternStep{2}});
  m.process.push_back({"000002", "", io::SetTimeStep{0.0}});
  m.process.push_back({"000003", "", newmark_step(io::Algorithm::Newton, 2e-3, 50)});
  compile::CompileOptions co;
  co.renumbering = compile::Renumbering::Off;
  const compile::Compiled c = compile::compile_model(m, co);
  const auto ref = femorax::testing::assemble_reference(m, c);
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt(ref.to_eigen(ref.K));
  const auto settled = recorded(c, ldlt.solve(body_force(c)));
  const auto rows = run(m, "gravity_shake.txt");
  ASSERT_EQ(rows.size(), 80u);
  const std::vector<double> after_gravity(rows[29].begin() + 1, rows[29].end());
  EXPECT_LE(max_rel(after_gravity, settled), 1e-3);  // damped out after 30 large steps
  EXPECT_NEAR(rows[30][0], 2e-3, 1e-15);           // setTime 0, then t = dt
}

// Femora's recovery on the host: Newton with max_iter 1 and NormDispIncr on a linear
// problem converges iff the first (exact) increment satisfies |dU| <= tol, so a step of size
// h succeeds iff that increment is small enough. A failed step is retried (the same result),
// then split into `substeps` steps down to `sublevels` levels. Returns the recorded rows and
// counts the retries and the steps split.
struct HostRecovery {
  std::vector<std::vector<double>> rows;
  long retries = 0, substepped = 0;
  std::vector<double> full_increments;  // |dU| of each full-size attempt, in order
};

HostRecovery host_recovery(const io::Model& m, const compile::Compiled& c, double dt, int steps, double tol,
                           int max_retries, int sublevels, int substeps) {
  const auto ref = femorax::testing::assemble_reference(m, c);
  const double gamma = 0.5, beta = 0.25;
  const auto K = ref.to_eigen(ref.K), C = ref.to_eigen(ref.C), M = ref.to_eigen(ref.M);
  const Eigen::Index n = K.rows();
  const Eigen::VectorXd Mr = Eigen::Map<const Eigen::VectorXd>(ref.Mr[0].data(), n), P = body_force(c);
  Eigen::VectorXd U = Eigen::VectorXd::Zero(n), V = U, A = U;
  std::map<double, std::unique_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>> keff;
  HostRecovery out;
  double t = 0.0;
  auto attempt = [&](double h) {
    const double c2 = gamma / (beta * h), c3 = 1.0 / (beta * h * h);
    auto& solver = keff[h];
    if (!solver)
      solver = std::make_unique<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>(ref.to_eigen(ref.effective(1.0, c2, c3)));
    const Eigen::VectorXd Vt = V, At = A;
    Eigen::VectorXd Vp = V * (1.0 - gamma / beta) + At * (h * (1.0 - 0.5 * gamma / beta));
    Eigen::VectorXd Ap = A * (1.0 - 0.5 / beta) + Vt * (-1.0 / (beta * h));
    const double tn = t + h;
    const Eigen::VectorXd B = P - K * U - C * Vp - M * Ap - ag_at(m.time_series[0], tn) * Mr;
    const Eigen::VectorXd dU = solver->solve(B);
    if (h == dt) out.full_increments.push_back(dU.norm());
    if (!(dU.norm() <= tol)) return false;  // reverted: U, V, A, t untouched
    U += dU;
    V = Vp + c2 * dU;
    A = Ap + c3 * dU;
    t = tn;
    std::vector<double> row = {t};
    const auto rec = recorded(c, U);
    row.insert(row.end(), rec.begin(), rec.end());
    out.rows.push_back(row);
    return true;
  };
  std::function<bool(double, int)> recover = [&](double h, int level) {
    if (attempt(h)) return true;
    for (int r = 0; r < max_retries; ++r) {
      ++out.retries;
      if (attempt(h)) return true;
    }
    if (level >= sublevels) return false;
    ++out.substepped;
    for (int i = 0; i < substeps; ++i)
      if (!recover(h / substeps, level + 1)) return false;
    return true;
  };
  for (int k = 0; k < steps; ++k)
    if (!recover(dt, 0)) break;
  return out;
}

TEST(Implicit, RetriesAndSubstepsRecoverStepsThatFailAtFullSize) {
  const double dt = 4e-3;
  const int steps = 30;
  io::Model m = column(false);
  add_motion(m);
  m.process.push_back({"000000", "", io::PatternStep{2}});
  compile::CompileOptions co;
  co.renumbering = compile::Renumbering::Off;
  const compile::Compiled c = compile::compile_model(m, co);
  // The tolerance: between the full-size increments, so some steps fail at full size; the
  // halves and quarters then pass. Away from every increment (CG solves to 1e-12, not exactly).
  auto probe = host_recovery(m, c, dt, steps, 1e300, 0, 0, 2).full_increments;
  std::sort(probe.begin(), probe.end());
  const double tol = 0.5 * (probe[steps / 3] + probe[steps / 3 + 1]);
  const auto want = host_recovery(m, c, dt, steps, tol, 1, 2, 2);
  ASSERT_GT(want.substepped, 0);
  // Both sublevels are used: some commits are quarter steps.
  bool quarter = false;
  for (std::size_t k = 1; k < want.rows.size(); ++k)
    quarter = quarter || std::abs(want.rows[k][0] - want.rows[k - 1][0] - dt / 4) < 1e-12;
  ASSERT_TRUE(quarter);
  ASSERT_EQ(want.rows.back()[0], [&] {  // every step completed
    double t = 0.0;
    for (const auto& r : want.rows) t = r[0];
    return t;
  }());
  for (double x : want.full_increments) ASSERT_GT(std::abs(x - tol), 1e-6 * tol);

  auto a = newmark_step(io::Algorithm::Newton, dt, steps);
  a.test_type = io::TestType::NormDispIncr;
  a.test_tol = tol;
  a.test_max_iter = 1;
  a.max_retries = 1;
  a.num_sublevels = 2;
  a.num_substeps = 2;
  m.process.push_back({"000001", "", a});
  const std::string file = temp_file("recovery.txt");
  add_recorder(m, file);
  m.process.insert(m.process.begin(), {"recorder", "", io::RecorderStep{"disp"}});
  exec::RunOptions o;
  o.verbose = false;
  o.renumbering = compile::Renumbering::Off;
  const auto rep = exec::run_model(m, o);
  const auto got = read_rows(file);
  ASSERT_EQ(got.size(), want.rows.size());  // one record per commit, substeps included
  double worst = 0.0, scale = 0.0;
  for (std::size_t k = 0; k < got.size(); ++k) {
    ASSERT_EQ(got[k][0], want.rows[k][0]) << "time of commit " << k;  // t += h, as OpenSees
    for (std::size_t j = 1; j < got[k].size(); ++j) {
      worst = std::max(worst, std::abs(got[k][j] - want.rows[k][j]));
      scale = std::max(scale, std::abs(want.rows[k][j]));
    }
  }
  EXPECT_LE(worst, 1e-8 * scale);
  const auto& st = rep.stages.at(0);
  EXPECT_EQ(st.steps, steps);
  EXPECT_EQ(st.commits, static_cast<std::int64_t>(want.rows.size()));
  EXPECT_EQ(st.retries, want.retries);
  EXPECT_EQ(st.substepped, want.substepped);
  std::printf("[ recovery: %d steps, %zu commits, %ld retries, %ld steps split ]\n", steps, want.rows.size(),
              want.retries, want.substepped);
}

TEST(Implicit, RecoveryExhaustedStopsTheRun) {
  io::Model m = column(false);
  add_motion(m);
  m.process.push_back({"000000", "", io::PatternStep{2}});
  auto a = newmark_step(io::Algorithm::ModifiedNewton, 2e-3, 5);
  a.modified_newton_factor_once = true;
  a.test_tol = 0.0;  // unreachable
  a.test_max_iter = 2;
  a.max_retries = 2;
  a.num_sublevels = 1;
  a.num_substeps = 3;
  m.process.push_back({"000001", "", a});
  exec::RunOptions o;
  o.verbose = false;
  try {
    exec::run_model(m, o);
    FAIL() << "expected a convergence failure";
  } catch (const io::ModelError& e) {
    const std::string w = e.what();
    EXPECT_NE(w.find("did not converge in 2 iterations"), std::string::npos) << w;
    EXPECT_NE(w.find("recovery exhausted (2 retries, 1 sublevels of 3 substeps)"), std::string::npos) << w;
  }
}

TEST(Implicit, ANonConvergingStepStopsTheRun) {
  io::Model m = column(false);
  add_motion(m);
  m.process.push_back({"000000", "", io::PatternStep{2}});
  auto a = newmark_step(io::Algorithm::Newton, 2e-3, 5);
  a.test_type = io::TestType::NormUnbalance;
  a.test_tol = 0.0;  // unreachable
  a.test_max_iter = 2;
  m.process.push_back({"000001", "", a});
  exec::RunOptions o;
  o.verbose = false;
  try {
    exec::run_model(m, o);
    FAIL() << "expected a convergence failure";
  } catch (const io::ModelError& e) {
    EXPECT_NE(std::string(e.what()).find("did not converge in 2 iterations"), std::string::npos) << e.what();
  }
}

}  // namespace
