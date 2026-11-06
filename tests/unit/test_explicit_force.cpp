// Compile step (batches, maps, lumped mass, coloring, padding, memory estimate) and the
// explicit force kernels in every layout and scatter mode, against a host reference that
// runs the same physics element by element.

#include "compile/batches.hpp"
#include "exec/explicit_force.hpp"
#include "../common/synthetic_models.hpp"
#include "../reference/drivers/driver_common.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <set>

namespace {

using femorax::compile::Compiled;
using femorax::compile::CompileOptions;
using femorax::compile::compile_model;
using femorax::compile::Renumbering;
using femorax::exec::ExplicitForce;
using femorax::exec::Layout;
using femorax::exec::Scatter;
using Brick = ExplicitForce::Brick;

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
        const Brick::Params p{{ps.E, ps.nu, ps.rho}, {ps.b[0], ps.b[1], ps.b[2]}, ps.massType};
        Brick::compute<false>(p, ps.damping, xe, ue, ve, {}, {}, fe, nullptr);
        for (int k = 0; k < 24; ++k) {
          const auto q = b.eq[k * P + e];
          if (q >= 0) f[q] += fe[k];
        }
      }
  }
  return f;
}

std::vector<double> run(ExplicitForce& k, const std::vector<double>& u, const std::vector<double>& v,
                        Layout l, Scatter s, int ept = 0) {
  auto hu = Kokkos::create_mirror_view(k.u);
  auto hv = Kokkos::create_mirror_view(k.v);
  for (std::size_t i = 0; i < u.size(); ++i) {
    hu(i) = u[i];
    hv(i) = v[i];
  }
  Kokkos::deep_copy(k.u, hu);
  Kokkos::deep_copy(k.v, hv);
  k.compute(l, s, ept);
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
  const Compiled c = compile_model(m, CompileOptions{});
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
    Brick::lumped_mass(Brick::Params{{ps.E, ps.nu, ps.rho}, {0, 0, 0}, 1}, xe, mj);
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

TEST(ExplicitForce, EveryLayoutAndScatterMatchesTheHostReference) {
  const auto m = column();
  for (Renumbering r : {Renumbering::Off, Renumbering::RCM}) {
    SCOPED_TRACE(r == Renumbering::Off ? "renumbering off" : "renumbering on");
    const Compiled c = compile_model(m, CompileOptions{r, 1});
    ExplicitForce k(c);
    const auto u = random_vector(static_cast<std::size_t>(k.num_equations()), 1e-3, 1);
    const auto v = random_vector(static_cast<std::size_t>(k.num_equations()), 1e-2, 2);
    const auto ref = reference_force(c, u, v);
    double scale = 0;
    for (double x : ref) scale = std::max(scale, std::abs(x));
    ASSERT_GT(scale, 0.0);

    for (Layout l : {Layout::Flat, Layout::Team}) {
      for (int ept : {1, 3, 4}) {
        if (l == Layout::Flat && ept != 1) continue;
        SCOPED_TRACE(std::string(femorax::exec::to_string(l)) + " ept " + std::to_string(ept));
        // Colored: bit-identical to the host reference, whatever the thread count.
        const auto fc = run(k, u, v, l, Scatter::Colored, ept);
        EXPECT_EQ(std::memcmp(fc.data(), ref.data(), ref.size() * sizeof(double)), 0);
        // Atomic: the same sums in another order.
        const auto fa = run(k, u, v, l, Scatter::Atomic, ept);
        for (std::size_t q = 0; q < ref.size(); ++q) EXPECT_NEAR(fa[q], ref[q], 1e-13 * scale);
      }
    }
  }
}

TEST(ExplicitForce, PaddingIsInert) {
  const auto m = column();
  const Compiled c1 = compile_model(m, CompileOptions{Renumbering::RCM, 1});
  const Compiled c32 = compile_model(m, CompileOptions{Renumbering::RCM, 32});
  EXPECT_EQ(c32.batches[0].padded % 32, 0);
  EXPECT_GT(c32.batches[0].padded, c32.batches[0].count);
  ExplicitForce k1(c1), k32(c32);
  const auto u = random_vector(static_cast<std::size_t>(k1.num_equations()), 1e-3, 3);
  const auto v = random_vector(static_cast<std::size_t>(k1.num_equations()), 1e-2, 4);
  for (Layout l : {Layout::Flat, Layout::Team}) {
    const auto a = run(k1, u, v, l, Scatter::Colored, 2), b = run(k32, u, v, l, Scatter::Colored, 2);
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)), 0);
  }
}

TEST(ExplicitForce, RenumberingChangesOnlyTheRounding) {
  const auto m = column();
  const Compiled off = compile_model(m, CompileOptions{Renumbering::Off, 1});
  const Compiled on = compile_model(m, CompileOptions{Renumbering::RCM, 1});
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
  ExplicitForce koff(off), kon(on);
  const auto foff = run(koff, canonical(off, field_u), canonical(off, field_v), Layout::Flat, Scatter::Colored);
  const auto fon = run(kon, canonical(on, field_u), canonical(on, field_v), Layout::Flat, Scatter::Colored);
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
