// Compile step (batches, maps, lumped mass, coloring, padding, memory estimate) and the
// explicit force kernels in every layout and scatter mode, against a host reference that
// runs the same physics element by element.

#include "compile/batches.hpp"
#include "exec/internal_force.hpp"
#include "exec/force_settings.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
#include "../common/synthetic_models.hpp"
#include "../reference/drivers/driver_common.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <type_traits>

namespace {

using femorax::compile::Compiled;
using femorax::compile::CompileOptions;
using femorax::compile::compile_model;
using femorax::compile::Renumbering;
using femorax::exec::InternalForce;
using femorax::exec::Layout;
using femorax::exec::Scatter;
using Brick = femorax::StdBrick<femorax::ElasticIsotropic>;

std::vector<double> random_vector(std::size_t n, double amp, std::uint64_t seed) {
  std::vector<double> v(n);
  refdrv::Lcg rng(seed);
  for (auto& x : v) x = amp * rng.symmetric();
  return v;
}

// Host reference: same physics per element, scattered in color order (the colored
// kernels' order), or element order when by_color is false.
std::vector<double> reference_force(const Compiled& c, const std::vector<double>& u,
                                    const std::vector<double>& v) {
  std::vector<double> f(static_cast<std::size_t>(c.dofs.num_equations), 0.0);
  const std::int32_t N = c.dofs.num_nodes();
  for (const auto& b : c.batches) {
    const auto P = static_cast<std::size_t>(b.padded);
    for (std::int32_t col = 0; col < b.num_colors(); ++col)
      for (std::int32_t i = b.color_start[col]; i < b.color_start[col + 1]; ++i) {
        const std::int32_t e = b.color_elements[i];
        double xe[8][3], ue[24], ve[24], fe[24];
        for (int a = 0; a < 8; ++a)
          for (int d = 0; d < 3; ++d) xe[a][d] = c.coords[d * N + b.node[a * P + e]];
        for (int k = 0; k < 24; ++k) {
          const auto q = b.eq[k * P + e];
          ue[k] = q >= 0 ? u[q] : 0.0;
          ve[k] = q >= 0 ? v[q] : 0.0;
        }
        const auto& ps = b.params[b.param[e]];
        const Brick::Params p = ps.params<Brick>();
        Brick::compute<false>(p, ps.damping, xe, ue, ve, {}, {}, fe, nullptr);
        for (int k = 0; k < 24; ++k) {
          const auto q = b.eq[k * P + e];
          if (q >= 0) f[q] += fe[k];
        }
      }
  }
  return f;
}

// Host reference for the gather scatter: each equation adds its contributions in
// (batch, element, DOF) order, the order InternalForce builds its gather lists in.
std::vector<double> reference_force_gather(const Compiled& c, const std::vector<double>& u,
                                           const std::vector<double>& v) {
  std::vector<std::vector<double>> parts(static_cast<std::size_t>(c.dofs.num_equations));
  const std::int32_t N = c.dofs.num_nodes();
  for (const auto& b : c.batches) {
    const auto P = static_cast<std::size_t>(b.padded);
    for (std::int32_t e = 0; e < b.count; ++e) {
      double xe[8][3], ue[24], ve[24], fe[24];
      for (int a = 0; a < 8; ++a)
        for (int d = 0; d < 3; ++d) xe[a][d] = c.coords[d * N + b.node[a * P + e]];
      for (int k = 0; k < 24; ++k) {
        const auto q = b.eq[k * P + e];
        ue[k] = q >= 0 ? u[q] : 0.0;
        ve[k] = q >= 0 ? v[q] : 0.0;
      }
      const auto& ps = b.params[b.param[e]];
      const Brick::Params p = ps.params<Brick>();
      Brick::compute<false>(p, ps.damping, xe, ue, ve, {}, {}, fe, nullptr);
      for (int k = 0; k < 24; ++k) {
        const auto q = b.eq[k * P + e];
        if (q >= 0) parts[q].push_back(fe[k]);
      }
    }
  }
  std::vector<double> f(parts.size(), 0.0);
  for (std::size_t q = 0; q < parts.size(); ++q) {
    double sum = 0.0;
    for (double x : parts[q]) sum += x;
    f[q] = sum;
  }
  return f;
}

// Host backends compute like the host reference, bit for bit. A GPU compiler fuses
// multiply-adds (nvcc --fmad=true), so there the reference holds to rounding only.
constexpr bool kHostBackend =
    std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;

std::vector<double> run(InternalForce& k, const std::vector<double>& u, const std::vector<double>& v,
                        Layout l, Scatter s, int ept = 0,
                        femorax::exec::Geometry g = femorax::exec::Geometry::Recompute) {
  auto hu = Kokkos::create_mirror_view(k.u);
  auto hv = Kokkos::create_mirror_view(k.v);
  for (std::size_t i = 0; i < u.size(); ++i) {
    hu(i) = u[i];
    hv(i) = v[i];
  }
  Kokkos::deep_copy(k.u, hu);
  Kokkos::deep_copy(k.v, hv);
  k.compute(femorax::exec::ForceSettings{l, s, ept, g});
  auto hf = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), k.f);
  return std::vector<double>(hf.data(), hf.data() + hf.extent(0));
}

femorax::io::Model column() {
  femorax::testing::ColumnSpec s;
  s.nx = 3;
  s.ny = 2;
  s.nz = 10;
  return femorax::testing::soil_column(s);
}

TEST(Compile, BatchesMapsAndColors) {
  const auto m = column();
  CompileOptions with_colors;
  with_colors.build_coloring = true;
  const Compiled c = compile_model(m, with_colors);
  ASSERT_EQ(c.batches.size(), 1u);  // one (stdBrick, ElasticIsotropic) batch
  const auto& b = c.batches[0];
  EXPECT_EQ(b.count, 3 * 2 * 10);
  // Param sets: (lower group, region 0), (upper group, region 0), (upper group, region 1).
  EXPECT_EQ(b.params.size(), 3u);
  // Each element in exactly one color; no two elements of a color share an equation.
  std::set<std::int32_t> seen;
  for (std::int32_t col = 0; col < b.num_colors(); ++col) {
    std::set<std::int32_t> eqs;  // equations used by earlier elements of this color
    for (std::int32_t i = b.color_start[col]; i < b.color_start[col + 1]; ++i) {
      const auto e = b.color_elements[i];
      EXPECT_TRUE(seen.insert(e).second);
      // Tied nodes of one element may share an equation; elements of a color may not.
      std::set<std::int32_t> mine;
      for (int k = 0; k < 24; ++k) {
        const auto q = b.eq[k * b.padded + e];
        if (q >= 0) mine.insert(q);
      }
      for (const auto q : mine) EXPECT_TRUE(eqs.insert(q).second) << "color " << col << " shares eq " << q;
    }
  }
  EXPECT_EQ(seen.size(), static_cast<std::size_t>(b.count));
  // A brick grid with laminar ties needs few colors (8 for the grid, more for the ties).
  EXPECT_LE(b.num_colors(), 16);
}

TEST(Compile, LumpedMassCollectsTiedAndDropsFixedDofs) {
  const auto m = column();
  const Compiled c = compile_model(m, CompileOptions{});
  // Total element mass, and the part on the fixed base nodes.
  double total = 0, base = 0;
  const auto& b = c.batches[0];
  const std::int32_t N = c.dofs.num_nodes();
  for (std::int32_t e = 0; e < b.count; ++e) {
    double xe[8][3], mj[8];
    for (int a = 0; a < 8; ++a)
      for (int d = 0; d < 3; ++d) xe[a][d] = c.coords[d * N + b.node[a * b.padded + e]];
    const auto& ps = b.params[b.param[e]];
    Brick::lumped_mass(ps.params<Brick>(), xe, mj);
    for (int a = 0; a < 8; ++a) {
      total += mj[a];
      if (b.eq[(3 * a) * b.padded + e] < 0) base += mj[a];
    }
  }
  double sum = 0;
  for (double x : c.lumped_mass) sum += x;
  EXPECT_NEAR(sum, 3.0 * (total - base), 1e-12 * sum);
  for (double x : c.lumped_mass) EXPECT_GT(x, 0.0);
}

TEST(InternalForce, EveryLayoutAndScatterMatchesTheHostReference) {
  const auto m = column();
  for (Renumbering r : {Renumbering::Off, Renumbering::RCM}) {
    SCOPED_TRACE(r == Renumbering::Off ? "renumbering off" : "renumbering on");
    const Compiled c = compile_model(m, CompileOptions{r, 1, true});
    InternalForce k(c, femorax::exec::GeometryCache::On);
    const auto u = random_vector(static_cast<std::size_t>(k.num_equations()), 1e-3, 1);
    const auto v = random_vector(static_cast<std::size_t>(k.num_equations()), 1e-2, 2);
    const auto ref = reference_force(c, u, v);
    double scale = 0;
    for (double x : ref) scale = std::max(scale, std::abs(x));
    ASSERT_GT(scale, 0.0);

    // The deterministic results on this backend: flat, recomputed geometry, colored and
    // gather scatter. Each against its host reference: bitwise on host backends, to
    // rounding where the device compiler fuses multiply-adds.
    const auto base = run(k, u, v, Layout::Flat, Scatter::Colored);
    const auto gbase = run(k, u, v, Layout::Flat, Scatter::Gather);
    const auto gref = reference_force_gather(c, u, v);
    if (kHostBackend) {
      EXPECT_EQ(std::memcmp(base.data(), ref.data(), ref.size() * sizeof(double)), 0);
      EXPECT_EQ(std::memcmp(gbase.data(), gref.data(), gref.size() * sizeof(double)), 0);
    } else {
      for (std::size_t q = 0; q < ref.size(); ++q) EXPECT_NEAR(base[q], ref[q], 1e-13 * scale);
      for (std::size_t q = 0; q < ref.size(); ++q) EXPECT_NEAR(gbase[q], gref[q], 1e-13 * scale);
    }
    // Gather and colored are the same sums in different (fixed) orders.
    for (std::size_t q = 0; q < ref.size(); ++q) EXPECT_NEAR(gbase[q], base[q], 1e-13 * scale);
    for (auto g : {femorax::exec::Geometry::Recompute, femorax::exec::Geometry::Cached}) {
      for (Layout l : {Layout::Flat, Layout::Team}) {
        for (int ept : {1, 3, 4}) {
          if (l == Layout::Flat && ept != 1) continue;
          SCOPED_TRACE(std::string(femorax::exec::to_string(l)) + " ept " + std::to_string(ept) +
                       " " + femorax::exec::to_string(g));
          // Colored: bit-identical to the base result, run after run and in every layout
          // of the base's arithmetic family (every layout on host backends; on GPU the
          // compiler fuses multiply-adds differently per layout: rounding only).
          const bool same_family = femorax::exec::layouts_share_arithmetic() || l == Layout::Flat;
          auto expect_same = [&](const std::vector<double>& a, const std::vector<double>& b) {
            if (same_family) {
              EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)), 0);
            } else {
              for (std::size_t q = 0; q < a.size(); ++q) EXPECT_NEAR(a[q], b[q], 1e-14 * scale);
            }
          };
          const auto fc = run(k, u, v, l, Scatter::Colored, ept, g);
          expect_same(fc, base);
          const auto again = run(k, u, v, l, Scatter::Colored, ept, g);
          EXPECT_EQ(std::memcmp(again.data(), fc.data(), fc.size() * sizeof(double)), 0);
          // Gather: bit-identical to its base, run after run and in every layout.
          const auto fg = run(k, u, v, l, Scatter::Gather, ept, g);
          expect_same(fg, gbase);
          const auto gagain = run(k, u, v, l, Scatter::Gather, ept, g);
          EXPECT_EQ(std::memcmp(gagain.data(), fg.data(), fg.size() * sizeof(double)), 0);
          // Atomic: the same sums in another order.
          const auto fa = run(k, u, v, l, Scatter::Atomic, ept, g);
          for (std::size_t q = 0; q < ref.size(); ++q) EXPECT_NEAR(fa[q], base[q], 1e-13 * scale);
        }
      }
    }
  }
}

TEST(InternalForce, PaddingIsInert) {
  const auto m = column();
  const Compiled c1 = compile_model(m, CompileOptions{Renumbering::RCM, 1, true});
  const Compiled c32 = compile_model(m, CompileOptions{Renumbering::RCM, 32, true});
  EXPECT_EQ(c32.batches[0].padded % 32, 0);
  EXPECT_GT(c32.batches[0].padded, c32.batches[0].count);
  InternalForce k1(c1, femorax::exec::GeometryCache::On), k32(c32, femorax::exec::GeometryCache::On);
  const auto u = random_vector(static_cast<std::size_t>(k1.num_equations()), 1e-3, 3);
  const auto v = random_vector(static_cast<std::size_t>(k1.num_equations()), 1e-2, 4);
  for (Layout l : {Layout::Flat, Layout::Team}) {
    for (Scatter sc : {Scatter::Colored, Scatter::Gather}) {
      const auto a = run(k1, u, v, l, sc, 2), b = run(k32, u, v, l, sc, 2);
      EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)), 0);
    }
  }
}

TEST(InternalForce, RenumberingChangesOnlyTheRounding) {
  const auto m = column();
  const Compiled off = compile_model(m, CompileOptions{Renumbering::Off, 1});
  const Compiled on = compile_model(m, CompileOptions{Renumbering::RCM, 1, true});
  ASSERT_EQ(off.dofs.num_equations, on.dofs.num_equations);
  // The same field per (node tag, dof) in both numberings.
  std::map<std::pair<int, int>, double> field_u, field_v;
  refdrv::Lcg rng(8);
  for (std::int32_t i = 0; i < off.dofs.num_nodes(); ++i)
    for (int d = 0; d < 3; ++d) {
      field_u[{off.dofs.node_tag[i], d}] = 1e-3 * rng.symmetric();
      field_v[{off.dofs.node_tag[i], d}] = 1e-2 * rng.symmetric();
    }
  // A tied group takes the value of its smallest node tag, so both numberings agree.
  auto canonical = [&](const Compiled& c, const std::map<std::pair<int, int>, double>& fld) {
    std::vector<double> x(static_cast<std::size_t>(c.dofs.num_equations), 0.0);
    std::vector<int> best(x.size(), INT32_MAX);
    for (std::int32_t i = 0; i < c.dofs.num_nodes(); ++i)
      for (int d = 0; d < 3; ++d) {
        const auto q = c.dofs.equation(i, d);
        if (q >= 0 && c.dofs.node_tag[i] < best[q]) {
          best[q] = c.dofs.node_tag[i];
          x[q] = fld.at({c.dofs.node_tag[i], d});
        }
      }
    return x;
  };
  InternalForce koff(off), kon(on);
  const auto foff = run(koff, canonical(off, field_u), canonical(off, field_v), Layout::Flat, Scatter::Gather);
  const auto fon = run(kon, canonical(on, field_u), canonical(on, field_v), Layout::Flat, Scatter::Gather);
  double scale = 0;
  for (double x : foff) scale = std::max(scale, std::abs(x));
  for (std::int32_t i = 0; i < off.dofs.num_nodes(); ++i) {
    const int t = off.dofs.node_tag[i];
    std::int32_t j = -1;
    for (std::int32_t k = 0; k < on.dofs.num_nodes(); ++k)
      if (on.dofs.node_tag[k] == t) j = k;
    for (int d = 0; d < 3; ++d) {
      const auto a = off.dofs.equation(i, d), b = on.dofs.equation(j, d);
      ASSERT_EQ(a < 0, b < 0);
      if (a >= 0) {
        EXPECT_NEAR(foff[a], fon[b], 1e-13 * scale);
      }
    }
  }
}

TEST(ForceSettings, EveryTunableVariantGivesIdenticalResults) {
  // The auto-tune may pick any of these, so none may change an answer: bit for bit.
  const Compiled c = compile_model(column(), CompileOptions{});
  InternalForce k(c, femorax::exec::GeometryCache::On);
  const auto u = random_vector(static_cast<std::size_t>(k.num_equations()), 1e-3, 5);
  const auto v = random_vector(static_cast<std::size_t>(k.num_equations()), 1e-2, 6);
  const auto variants = femorax::exec::tunable_variants(k);
  ASSERT_GE(variants.size(), 4u);
  const auto first = run(k, u, v, variants[0].layout, variants[0].scatter,
                         variants[0].elements_per_team, variants[0].geometry);
  for (const auto& s : variants) {
    SCOPED_TRACE(femorax::exec::to_string(s));
    EXPECT_TRUE(s.deterministic());
    const auto f = run(k, u, v, s.layout, s.scatter, s.elements_per_team, s.geometry);
    int ndiff = 0;
    double maxrel = 0, scale = 0;
    for (double x : first) scale = std::max(scale, std::abs(x));
    for (std::size_t q = 0; q < f.size(); ++q)
      if (f[q] != first[q]) {
        ++ndiff;
        maxrel = std::max(maxrel, std::abs(f[q] - first[q]) / scale);
      }
    EXPECT_EQ(ndiff, 0) << ndiff << " of " << f.size() << " differ, max " << maxrel << " of the largest";
  }
  // The default is one of them, and so is a cached choice on a force object without a
  // cache (it falls back to recomputing, with the same bits).
  InternalForce none(c, femorax::exec::GeometryCache::Off);
  const auto d = none.resolve(femorax::exec::default_force_settings());
  EXPECT_EQ(d.scatter, Scatter::Gather);
  EXPECT_EQ(d.geometry, femorax::exec::Geometry::Recompute);
  const auto fd = run(none, u, v, d.layout, d.scatter, d.elements_per_team, d.geometry);
  EXPECT_EQ(std::memcmp(fd.data(), first.data(), fd.size() * sizeof(double)), 0);
}

TEST(ForceSettings, AutotuneTimesEveryVariantAndPicksTheFastest) {
  const Compiled c = compile_model(column(), CompileOptions{});
  InternalForce k(c, femorax::exec::GeometryCache::On);
  k.fill(1e-4, 1e-3);
  const auto variants = femorax::exec::tunable_variants(k);
  const auto r = femorax::exec::autotune(k, variants, 1);
  ASSERT_EQ(r.seconds.size(), variants.size());
  double best = 1e300;
  for (const auto& [s, t] : r.seconds) {
    EXPECT_GT(t, 0.0);
    best = std::min(best, t);
  }
  for (const auto& [s, t] : r.seconds) {
    if (t == best) {
      EXPECT_EQ(femorax::exec::to_string(s), femorax::exec::to_string(r.best));
    }
  }
}

TEST(ForceSettings, ChoiceOrderOverrideThenSavedThenTuned) {
  const Compiled c = compile_model(column(), CompileOptions{});
  InternalForce k(c, femorax::exec::GeometryCache::On);
  const auto cache = std::filesystem::temp_directory_path() / "femorax_test_autotune.txt";
  std::filesystem::remove(cache);
  setenv("FEMORAX_TUNE_CACHE", cache.c_str(), 1);
  unsetenv("FEMORAX_FORCE_SETTINGS");
  const auto first = femorax::exec::choose_force_settings(k, c);
  EXPECT_EQ(first.source, "tuned");
  EXPECT_EQ(first.settings.scatter, Scatter::Gather);
  const auto again = femorax::exec::choose_force_settings(k, c);
  EXPECT_EQ(again.source, "saved");
  EXPECT_EQ(femorax::exec::to_string(again.settings), femorax::exec::to_string(first.settings));
  setenv("FEMORAX_FORCE_SETTINGS", "flat_atomic", 1);
  const auto over = femorax::exec::choose_force_settings(k, c);
  EXPECT_EQ(over.source, "override");
  EXPECT_EQ(over.settings.scatter, Scatter::Atomic);
  unsetenv("FEMORAX_FORCE_SETTINGS");
  unsetenv("FEMORAX_TUNE_CACHE");
  std::filesystem::remove(cache);
}

TEST(ForceSettings, SettingsStringsRoundTrip) {
  for (const auto* t : {"flat_gather", "flat_gather_cached", "team_gather_ept4_cached", "team_atomic_ept2",
                        "flat_colored"}) {
    femorax::exec::ForceSettings s;
    ASSERT_TRUE(femorax::exec::parse_settings(t, s)) << t;
    EXPECT_EQ(femorax::exec::to_string(s), t);
  }
  femorax::exec::ForceSettings s;
  EXPECT_FALSE(femorax::exec::parse_settings("diagonal_magic", s));
}

TEST(Compile, ColoringOnlyWhenRequested) {
  const Compiled plain = compile_model(column(), CompileOptions{});
  EXPECT_FALSE(plain.batches[0].has_coloring());
  InternalForce k(plain, femorax::exec::GeometryCache::Off);
  EXPECT_FALSE(k.has_coloring());
  for (const auto& s : femorax::exec::all_variants(k)) EXPECT_NE(s.scatter, Scatter::Colored);
}

TEST(Compile, MemoryEstimateCoversTheArrays) {
  const auto small = compile_model(column(), CompileOptions{});
  femorax::testing::ColumnSpec s;
  s.nx = 10;
  s.ny = 10;
  s.nz = 20;
  const auto big = compile_model(femorax::testing::soil_column(s), CompileOptions{});
  const auto a = femorax::compile::memory_estimate_bytes(small);
  const auto b = femorax::compile::memory_estimate_bytes(big);
  EXPECT_GT(a, 0);
  // At least the maps: 24 equations + 8 nodes per element, 4 bytes each.
  EXPECT_GE(b, static_cast<std::int64_t>(big.batches[0].count) * 32 * 4);
  EXPECT_GT(b, 20 * a);
}

}  // namespace
