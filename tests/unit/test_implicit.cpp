// Implicit stages (exec/implicit_stage.hpp) through femorax's run_model on the laminar soil
// column, against references computed on the host from the assembled matrices
// (tests/common/reference_assembly.hpp) with Eigen direct solves:
//   - Static LoadControl gravity: the exact solution K u = F (and Linear = Newton).
//   - Newmark (form D) with a base motion: the Newmark recurrence of OpenSees, step by step,
//     with exact solves; Linear, Newton and ModifiedNewton, every convergence test;
//     consistent and lumped mass, all Rayleigh terms.
//   - A step that cannot converge stops the run with a message.

#include <Eigen/SparseCholesky>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <unistd.h>

#include "../common/reference_assembly.hpp"
#include "../common/synthetic_models.hpp"
#include "compile/batches.hpp"
#include "exec/explicit_analysis.hpp"
#include "exec/force_settings.hpp"
#include "exec/internal_force.hpp"
#include "io/h5.hpp"

namespace {

namespace fs = std::filesystem;
using namespace femorax;

const int kNx = 2, kNz = 8;

io::Model column(bool lumped) {
  femorax::testing::ColumnSpec s;
  s.nx = s.ny = kNx;
  s.nz = kNz;
  s.lumped = lumped;
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
