// Material history state in the kernels (exec/internal_force.hpp): a soil column of
// PressureDependMultiYield (stdBrick and SSPbrick), driven through a sequence of
// displacement fields with the switch to the plastic stage, against a host reference that
// runs the same element physics with its own committed and trial state per point.
//   - forces bit-identical to the reference on host backends at every step (gather), to
//     rounding on GPUs; every layout and geometry mode gives the same bits;
//   - a force evaluation without a commit changes nothing (trial work is redone from the
//     committed state), so Newton iterations and the auto-tune may call it freely;
//   - set_material_stage changes only the materials named;
//   - the stored tangents and the committed tangent use the state.

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <type_traits>
#include <vector>

#include "../common/synthetic_models.hpp"
#include "compile/batches.hpp"
#include "compile/registry.hpp"
#include "exec/force_settings.hpp"
#include "exec/internal_force.hpp"
#include "exec/runtime.hpp"

namespace {

using namespace femorax;
using compile::Compiled;
using compile::CompileOptions;
using exec::InternalForce;

constexpr bool kHost = std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;

io::PressureDependMultiYieldData soil(bool loose) {
  io::PressureDependMultiYieldData d;
  d.rho = loose ? 19.1 / 9.81 : 19.9 / 9.81;
  d.refShearModul = loose ? 75.0e3 : 145.0e3;
  d.refBulkModul = 2.0 * d.refShearModul * 1.3 / 1.2;
  d.frictionAng = loose ? 29.0 : 40.0;
  d.peakShearStra = 0.1;
  d.refPress = 80.0;
  d.pressDependCoe = 0.5;
  d.PTAng = loose ? 29.0 : 27.0;
  d.contrac = loose ? 0.21 : 0.03;
  d.dilat1 = loose ? 0.0 : 0.8;
  d.dilat2 = loose ? 0.0 : 5.0;
  d.liquefac1 = loose ? 10.0 : 0.0;
  d.liquefac2 = loose ? 0.02 : 0.0;
  d.liquefac3 = loose ? 1.0 : 0.0;
  d.noYieldSurf = loose ? 20 : 25;  // two surface buckets (20 and 40)
  d.e = loose ? 0.85 : 0.45;
  return d;
}

io::Model column(io::ElementType type) {
  femorax::testing::ColumnSpec s;
  s.nx = 2;
  s.ny = 2;
  s.nz = 6;
  s.element = type;
  auto m = femorax::testing::soil_column(s);
  m.materials[0].data = soil(false);
  m.materials[1].data = soil(true);
  return m;
}

// The displacement field of step k: oedometric compression growing to step 4, then
// cyclic shear in x with a random part.
std::vector<double> field(const Compiled& c, int k, double scale) {
  std::vector<double> u(static_cast<std::size_t>(c.dofs.num_equations), 0.0);
  const std::int32_t N = c.dofs.num_nodes();
  std::uint64_t seed = 77 + k;
  for (std::int32_t i = 0; i < N; ++i) {
    const double z = c.coords[2 * N + i] + 3.0;  // height above the base
    for (int d = 0; d < 3; ++d) {
      const auto q = c.dofs.equation(i, d);
      if (q < 0) continue;
      seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
      const double r = static_cast<double>(seed >> 11) * (2.0 / 9007199254740992.0) - 1.0;
      double v = 0.0;
      if (d == 2) v = -6e-4 * z * std::min(k, 4) / 4.0;
      if (d == 0 && k > 4) v = 3e-3 * z * std::sin(0.7 * (k - 4));
      u[q] = scale * (v + 2e-6 * r);
    }
  }
  return u;
}

// Host reference: every element with its own committed and trial state.
struct Reference {
  const Compiled& c;
  std::map<std::pair<int, std::int32_t>, std::vector<double>> committed, trial;  // (batch, element)

  explicit Reference(const Compiled& cc) : c(cc) {}

  template <class Elem>
  void element(std::size_t bi, const compile::Batch& b, std::int32_t e, const std::vector<double>& u,
               const std::vector<double>& v, int stage, double* fe) {
    using Mat = typename Elem::Material;
    const auto P = static_cast<std::size_t>(b.padded);
    const std::int32_t N = c.dofs.num_nodes();
    double xe[8][3], ue[24], ve[24];
    for (int a = 0; a < 8; ++a)
      for (int d = 0; d < 3; ++d) xe[a][d] = c.coords[d * N + b.node[a * P + e]];
    for (int k = 0; k < 24; ++k) {
      const auto q = b.eq[k * P + e];
      ue[k] = q >= 0 ? u[q] : 0.0;
      ve[k] = q >= 0 ? v[q] : 0.0;
    }
    const auto& ps = b.params[b.param[e]];
    auto p = ps.params<Elem>();
    if constexpr (requires { p.mat.loadStage; }) p.mat.loadStage = stage;
    auto& co = committed[{static_cast<int>(bi), e}];
    auto& tr = trial[{static_cast<int>(bi), e}];
    if (co.empty()) {
      co.assign(Elem::NSTATE > 0 ? Elem::NSTATE : 1, 0.0);
      tr = co;
      if constexpr (Mat::NSTATE > 0)
        for (int gp = 0; gp < Elem::NGP; ++gp) Mat::initial_state(p.mat, {co.data() + gp * Mat::NSTATE, 1});
    }
    const PointArray<const double> cv{co.data(), 1, Mat::NSTATE};
    const PointArray<double> tv{tr.data(), 1, Mat::NSTATE};
    Elem::template compute<false>(p, ps.damping, xe, ue, ve, cv, tv, fe, nullptr);
  }

  // The gather order: each equation adds its contributions in (batch, element, DOF) order.
  std::vector<double> force(const std::vector<double>& u, const std::vector<double>& v, int stage) {
    std::vector<std::vector<double>> parts(static_cast<std::size_t>(c.dofs.num_equations));
    for (std::size_t bi = 0; bi < c.batches.size(); ++bi) {
      const auto& b = c.batches[bi];
      const auto P = static_cast<std::size_t>(b.padded);
      for (std::int32_t e = 0; e < b.count; ++e) {
        double fe[24];
        registry::for_each_pair([&]<class E, class M>(int index) {
          if (index == b.pair) element<typename E::template Element<typename M::Material>>(bi, b, e, u, v, stage, fe);
        });
        for (int k = 0; k < 24; ++k) {
          const auto q = b.eq[k * P + e];
          if (q >= 0) parts[q].push_back(fe[k]);
        }
      }
    }
    std::vector<double> f(parts.size(), 0.0);
    for (std::size_t q = 0; q < parts.size(); ++q) {
      double s = 0.0;
      for (double x : parts[q]) s += x;
      f[q] = s;
    }
    return f;
  }
  void commit() { std::swap(committed, trial); }
};

std::vector<double> run(InternalForce& k, const std::vector<double>& u, const std::vector<double>& v,
                        const exec::ForceSettings& s) {
  auto hu = Kokkos::create_mirror_view(k.u);
  auto hv = Kokkos::create_mirror_view(k.v);
  for (std::size_t i = 0; i < u.size(); ++i) {
    hu(i) = u[i];
    hv(i) = v[i];
  }
  Kokkos::deep_copy(k.u, hu);
  Kokkos::deep_copy(k.v, hv);
  k.compute(s);
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k.f);
  return std::vector<double>(hf.data(), hf.data() + hf.extent(0));
}

double max_abs(const std::vector<double>& a) {
  double m = 0;
  for (double x : a) m = std::max(m, std::abs(x));
  return m;
}

TEST(MaterialState, PressureDependMultiYieldColumnMatchesTheHostReference) {
  for (auto type : {io::ElementType::StdBrick, io::ElementType::SSPbrick}) {
    SCOPED_TRACE(io::to_string(type));
    const io::Model m = column(type);
    CompileOptions co;
    co.simd_width = femorax::simd_width();
    const Compiled c = compile_model(m, co);
    ASSERT_EQ(c.batches.size(), 2u);  // NS 40 (dense, 25 surfaces) and NS 20 (loose)
    InternalForce k(c, exec::GeometryCache::On);
    ASSERT_TRUE(k.has_state());
    Reference ref(c);
    const auto variants = exec::tunable_variants(k);
    int stage = 0;
    double worst = 0.0;
    for (int step = 1; step <= 14; ++step) {
      SCOPED_TRACE("step " + std::to_string(step));
      if (step == 5) {  // updateMaterialStage -stage 1 for both materials
        stage = 1;
        EXPECT_TRUE(k.set_material_stage(0, 1));
        EXPECT_TRUE(k.set_material_stage(1, 1));
      }
      const auto u = field(c, step, 1.0), v = field(c, step, 0.01);
      const auto want = ref.force(u, v, stage);
      const auto got = run(k, u, v, exec::default_force_settings());
      if (kHost) {
        ASSERT_EQ(std::memcmp(got.data(), want.data(), got.size() * sizeof(double)), 0);
      } else {
        for (std::size_t q = 0; q < got.size(); ++q) EXPECT_NEAR(got[q], want[q], 1e-9 * max_abs(want));
      }
      for (std::size_t q = 0; q < got.size(); ++q) worst = std::max(worst, std::abs(got[q] - want[q]) / max_abs(want));
      // Without a commit, any number of evaluations (other variants, a trial far away) leave
      // the next one unchanged.
      for (const auto& s : variants) {
        const auto again = run(k, u, v, s);
        if (exec::layouts_share_arithmetic() || s.layout == exec::default_force_settings().layout) {
          EXPECT_EQ(again, got) << exec::to_string(s);
        }
      }
      run(k, field(c, step + 7, 3.0), v, exec::default_force_settings());
      EXPECT_EQ(run(k, u, v, exec::default_force_settings()), got);
      k.commit_state();
      ref.commit();
    }
    std::printf("[ %s PDMY column, 14 steps with the stage switch: largest difference from the host reference "
                "%.2e ]\n", io::to_string(type), worst);
  }
}

TEST(MaterialState, StageChangesOnlyTheNamedMaterial) {
  const io::Model m = column(io::ElementType::SSPbrick);
  CompileOptions co;
  co.simd_width = femorax::simd_width();
  const Compiled c = compile_model(m, co);
  InternalForce k(c, exec::GeometryCache::On);
  const auto u0 = field(c, 4, 1.0), v = field(c, 4, 0.0);
  run(k, u0, v, exec::default_force_settings());
  k.commit_state();
  const auto u = field(c, 9, 1.0);
  const auto elastic = run(k, u, v, exec::default_force_settings());
  EXPECT_TRUE(k.set_material_stage(1, 1));       // the loose sand only
  EXPECT_FALSE(k.set_material_stage(5, 1));      // no such material in the model's batches
  const auto partly = run(k, u, v, exec::default_force_settings());
  EXPECT_TRUE(k.set_material_stage(0, 1));
  const auto plastic = run(k, u, v, exec::default_force_settings());
  EXPECT_NE(partly, elastic);
  EXPECT_NE(plastic, partly);
  EXPECT_TRUE(k.set_material_stage(0, 0));
  EXPECT_TRUE(k.set_material_stage(1, 0));
  EXPECT_EQ(run(k, u, v, exec::default_force_settings()), elastic);  // nothing was committed
}

TEST(MaterialState, OperatorUsesTheStateAndRecomputedFallsBackToStored) {
  const io::Model m = column(io::ElementType::StdBrick);
  CompileOptions co;
  co.simd_width = femorax::simd_width();
  const Compiled c = compile_model(m, co);
  InternalForce k(c, exec::GeometryCache::On);
  EXPECT_FALSE(k.constant_tangents());
  EXPECT_EQ(k.tangent_symmetry(), TangentSymmetry::SymmetricPositiveSemidefinite);
  const auto r = k.resolve(exec::OperatorSettings{exec::Scatter::Gather, exec::Geometry::Cached,
                                                  exec::TangentSource::Recomputed});
  EXPECT_EQ(r.tangent, exec::TangentSource::Stored);
  EXPECT_EQ(k.resolve(exec::OperatorSettings{}).tangent, exec::TangentSource::Stored);
  k.enable_stored_tangents();
  // Plastic state: consolidate elastically, switch, shear; then the tangent at the
  // committed state differs from the elastic one, and the operator sees it.
  const std::size_t n = static_cast<std::size_t>(k.num_equations());
  exec::Vec uv("u", n), x("x", n), y0("y0", n), y1("y1", n);
  auto set = [&](const std::vector<double>& h) {
    auto m_ = Kokkos::create_mirror_view(uv);
    for (std::size_t i = 0; i < n; ++i) m_(i) = h[i];
    Kokkos::deep_copy(uv, m_);
  };
  Kokkos::deep_copy(x, 1e-3);
  set(field(c, 0, 0.0));
  k.store_tangents(uv, {});
  k.apply_operator({1.0, 0.0, 0.0}, {}, uv, x, y0);  // elastic, zero state
  for (int step = 1; step <= 4; ++step) {
    run(k, field(c, step, 1.0), field(c, step, 0.0), exec::default_force_settings());
    k.commit_state();
  }
  k.set_material_stage(0, 1);
  k.set_material_stage(1, 1);
  const auto us = field(c, 8, 1.0);
  run(k, us, field(c, 8, 0.0), exec::default_force_settings());
  set(us);
  k.store_tangents(uv, {});
  k.apply_operator({1.0, 0.0, 0.0}, {}, uv, x, y1);
  auto h0 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y0);
  auto h1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y1);
  double diff = 0, scale = 0;
  for (std::size_t i = 0; i < n; ++i) {
    diff = std::max(diff, std::abs(h1(i) - h0(i)));
    scale = std::max(scale, std::abs(h0(i)));
  }
  EXPECT_GT(diff, 1e-3 * scale);  // the pressure-dependent, elastoplastic tangent
  // The stable step reads the committed state too (finite and positive).
  const double dt = k.min_critical_dt(ExplicitScheme::ExplicitDifference);
  EXPECT_GT(dt, 0.0);
  EXPECT_TRUE(std::isfinite(dt));
}

}  // namespace
