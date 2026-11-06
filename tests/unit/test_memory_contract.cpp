// The memory and parallelism contract (docs/design.md): physics results are bit-identical
//  1. whatever the state layout (structure of arrays, contiguous per point, padded stride),
//  2. whatever the split of inner loops across vector lanes (emulated exactly: each lane is
//     a separate call with private locals and shared outputs, as on a GPU), and
//  3. under the Kokkos lane mapping of src/exec/inner_kokkos.hpp.

#include "exec/inner_kokkos.hpp"
#include "physics/common/inner.hpp"
#include "physics/common/strided.hpp"
#include "physics/elements/std_brick.hpp"
#include "physics/materials/elastic_isotropic.hpp"
#include "../reference/drivers/driver_common.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <vector>

namespace {

using femorax::PointArray;
using femorax::Rayleigh;
using femorax::Strided;

// A material with history, for layout tests only: three state values per point that
// depend on the committed state and the strain, and a stress that depends on them.
struct ToyHistory {
  static constexpr femorax::MaterialKind KIND = femorax::MaterialKind::NDMaterial3D;
  static constexpr int NSTRAIN = 6;
  static constexpr int NSTATE = 3;
  struct Params { double E, rho; };
  static void update(const Params& p, Strided<const double> c, Strided<double> t,
                     const double* eps, double* sig, double (*D)[NSTRAIN]) {
    t[0] = std::fmax(c[0], std::abs(eps[0]));
    t[1] = c[1] + eps[3];
    t[2] = 0.5 * c[2] + eps[1];
    for (int i = 0; i < NSTRAIN; ++i) {
      sig[i] = p.E * eps[i] + 1e-3 * p.E * t[i % 3];
      for (int j = 0; j < NSTRAIN; ++j) D[i][j] = i == j ? p.E * (1.0 + c[i % 3]) : 0.0;
    }
  }
  static void initial_tangent(const Params& p, double (*D)[NSTRAIN]) {
    for (int i = 0; i < NSTRAIN; ++i)
      for (int j = 0; j < NSTRAIN; ++j) D[i][j] = i == j ? p.E : 0.0;
  }
  static void committed_tangent(const Params& p, Strided<const double> c, double (*D)[NSTRAIN]) {
    for (int i = 0; i < NSTRAIN; ++i)
      for (int j = 0; j < NSTRAIN; ++j) D[i][j] = i == j ? p.E * (1.0 + c[i % 3]) : 0.0;
  }
  static double density(const Params& p) { return p.rho; }
};
static_assert(femorax::Material<ToyHistory>);

using ToyBrick = femorax::StdBrick<ToyHistory>;
using Brick = femorax::StdBrick<femorax::ElasticIsotropic>;
constexpr int NDOF = 24, NGP = 8;

struct Element {
  double xe[8][3];
  double u[NDOF], v[NDOF];
};

std::vector<Element> elements(int n) {
  std::vector<Element> out(n);
  refdrv::Lcg rng(77);
  for (int e = 0; e < n; ++e) {
    const double base[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                               {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
    for (int a = 0; a < 8; ++a)
      for (int c = 0; c < 3; ++c) out[e].xe[a][c] = base[a][c] + e + 0.1 * rng.symmetric();
    for (double& x : out[e].u) x = 1e-3 * rng.symmetric();
    for (double& x : out[e].v) x = 1e-2 * rng.symmetric();
  }
  return out;
}

// One state layout: where value k of point (e, gp) lives.
struct Layout {
  const char* name;
  int nElements;
  int stride;      // distance between values of one point
  int point_step;  // distance between points of one element
  int elem_step;   // distance between elements
  std::size_t size;
  std::size_t index(int e, int gp, int k) const {
    return static_cast<std::size_t>(e) * elem_step + static_cast<std::size_t>(gp) * point_step +
           static_cast<std::size_t>(k) * stride;
  }
};

std::vector<Layout> layouts(int nE) {
  const int nP = nE * NGP, S = ToyHistory::NSTATE;
  const int padded = ((nP + 63) / 64) * 64 + 5;  // any stride works, not only nPoints
  return {
      {"struct_of_arrays", nE, nP, 1, NGP, static_cast<std::size_t>(S) * nP},
      {"contiguous_per_point", nE, 1, S, NGP * S, static_cast<std::size_t>(S) * nP},
      {"padded_stride", nE, padded, 1, NGP, static_cast<std::size_t>(S) * padded},
  };
}

struct Result {
  std::vector<double> fe, Ke, trial;  // trial in logical order (e, gp, k)
};

Result run_layout(const Layout& L, const std::vector<Element>& els) {
  const int S = ToyHistory::NSTATE;
  std::vector<double> committed(L.size, -1.0), trial(L.size, -2.0);
  refdrv::Lcg rng(5);
  for (int e = 0; e < L.nElements; ++e)
    for (int gp = 0; gp < NGP; ++gp)
      for (int k = 0; k < S; ++k) committed[L.index(e, gp, k)] = 0.01 * (e + gp + k) + 1e-3 * (k + 1);
  Result r;
  const ToyBrick::Params p{{1000.0, 2.0}, {0.0, 0.0, -9.81}, 0};
  const Rayleigh d{0.3, 1e-3, 2e-4, 3e-4};
  for (int e = 0; e < L.nElements; ++e) {
    const PointArray<const double> c{committed.data() + L.index(e, 0, 0), L.stride, L.point_step};
    const PointArray<double> t{trial.data() + L.index(e, 0, 0), L.stride, L.point_step};
    double fe[NDOF], Ke[NDOF][NDOF];
    ToyBrick::compute<true>(p, d, els[e].xe, els[e].u, els[e].v, c, t, fe, Ke);
    r.fe.insert(r.fe.end(), fe, fe + NDOF);
    r.Ke.insert(r.Ke.end(), &Ke[0][0], &Ke[0][0] + NDOF * NDOF);
    r.trial.push_back(ToyBrick::max_omega_sq(p, els[e].xe, c));
    for (int gp = 0; gp < NGP; ++gp)
      for (int k = 0; k < S; ++k) r.trial.push_back(trial[L.index(e, gp, k)]);
  }
  return r;
}

void expect_bitwise(const std::vector<double>& a, const std::vector<double>& b, const char* what) {
  ASSERT_EQ(a.size(), b.size()) << what;
  EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)), 0) << what;
}

TEST(MemoryContract, StateLayoutDoesNotChangeResults) {
  const int nE = 5;
  const auto els = elements(nE);
  const auto Ls = layouts(nE);
  const Result ref = run_layout(Ls[0], els);
  for (std::size_t i = 1; i < Ls.size(); ++i) {
    SCOPED_TRACE(Ls[i].name);
    const Result r = run_layout(Ls[i], els);
    expect_bitwise(r.fe, ref.fe, "fe");
    expect_bitwise(r.Ke, ref.Ke, "Ke");
    expect_bitwise(r.trial, ref.trial, "trial state and max_omega_sq");
  }
}

// ---------------------------------------------------------------------------
// Lane emulation: lane `lane` of `lanes` runs the iterations i = lane, lane + lanes, ...
// Each lane is a separate call to the physics function, so locals are private to it, as
// on a GPU; only fe, Ke, work and the state are shared.
struct LaneSplit {
  int lane, lanes;
};

template <class F>
void inner_for(const LaneSplit& c, int n, F&& f) {
  for (int i = c.lane; i < n; i += c.lanes) f(i);
}

template <class F>
auto inner_reduce(const LaneSplit&, int n, F&& f) -> decltype(f(0)) {
  // The contract: terms added in index order. Every lane gets the full ordered sum.
  decltype(f(0)) acc = 0;
  for (int i = 0; i < n; ++i) acc += f(i);
  return acc;
}

// Lanes run one after another, forward or reverse: a lane that reads what another lane
// wrote sees stale data in one of the two orders.
template <class Elem, class Params>
void lane_vs_serial(const Params& p, const Element& el, int lanes, bool reverse) {
  const Rayleigh d{0.3, 1e-3, 2e-4, 3e-4};
  const int S = Elem::NSTATE > 0 ? Elem::NSTATE / NGP : 0;
  std::vector<double> committed(NGP * S + 1, 0.02), trial_serial(NGP * S + 1), trial_lanes(NGP * S + 1);
  const PointArray<const double> c{committed.data(), 1, S};

  double fe_ref[NDOF], Ke_ref[NDOF][NDOF];
  Elem::template compute<true>(p, d, el.xe, el.u, el.v, c, PointArray<double>{trial_serial.data(), 1, S},
                               fe_ref, Ke_ref);

  double fe[NDOF], Ke[NDOF][NDOF];
  typename Elem::Work work;
  std::memset(fe, 0xff, sizeof(fe));  // garbage: every entry must be written
  std::memset(Ke, 0xff, sizeof(Ke));
  std::memset(&work, 0xff, sizeof(work));
  for (int k = 0; k < lanes; ++k) {
    const int lane = reverse ? lanes - 1 - k : k;
    Elem::template compute<true>(LaneSplit{lane, lanes}, p, d, el.xe, el.u, el.v, c,
                                 PointArray<double>{trial_lanes.data(), 1, S}, fe, Ke, work);
  }
  EXPECT_EQ(std::memcmp(fe, fe_ref, sizeof(fe)), 0) << lanes << " lanes: fe";
  EXPECT_EQ(std::memcmp(Ke, Ke_ref, sizeof(Ke)), 0) << lanes << " lanes: Ke";
  EXPECT_EQ(std::memcmp(trial_lanes.data(), trial_serial.data(), trial_serial.size() * sizeof(double)), 0)
      << lanes << " lanes: trial state";
}

TEST(MemoryContract, InnerLoopSplitAcrossLanesDoesNotChangeResults) {
  const auto els = elements(3);
  for (int lanes : {1, 2, 3, 7, 8, 32, 64}) {
    for (const auto& el : els) {
      for (bool reverse : {false, true}) {
        lane_vs_serial<ToyBrick>(ToyBrick::Params{{1000.0, 2.0}, {0.0, 0.0, -9.81}, 0}, el, lanes,
                                 reverse);
        lane_vs_serial<Brick>(Brick::Params{{377000.0, 0.3, 2.0285}, {0.5, -1.0, -19.9}, 1}, el,
                              lanes, reverse);
      }
    }
  }
}

TEST(MemoryContract, InnerReduceAddsInIndexOrder) {
  // Terms chosen so that any other order gives a different double.
  std::vector<double> terms = {1e16, 1.0, -1e16, 3.0, 1e-3, 7e15, -7e15, 0.5};
  double serial = 0;
  for (double t : terms) serial += t;
  for (int lanes : {1, 2, 3, 8}) {
    const double r =
        inner_reduce(LaneSplit{0, lanes}, static_cast<int>(terms.size()), [&](int i) { return terms[i]; });
    EXPECT_EQ(r, serial);
  }
  EXPECT_EQ(femorax::inner_reduce(femorax::SerialInner{}, static_cast<int>(terms.size()),
                                  [&](int i) { return terms[i]; }),
            serial);
}

// ---------------------------------------------------------------------------
// Kokkos mapping (exec layer): the same physics inside a TeamPolicy kernel, with the
// shared buffers in team scratch, gives the serial result bit for bit.
// The kernel lives in a free function: nvcc does not allow device lambdas inside a test
// body (a private member function of the generated test class).
struct KokkosLaneResult {
  Kokkos::View<double**> fe, Ke;
  Kokkos::View<double*> reduced;
};

KokkosLaneResult run_kokkos_lanes(const std::vector<Element>& els, const Brick::Params& p,
                                  const Rayleigh& d, int vl) {
  using Space = Kokkos::DefaultExecutionSpace;
  using Policy = Kokkos::TeamPolicy<Space>;
  using Member = Policy::member_type;
  using Scratch = Kokkos::View<double*, Space::scratch_memory_space, Kokkos::MemoryUnmanaged>;

  const int nE = static_cast<int>(els.size());

  Kokkos::View<Element*, Space> dev_els("els", nE);
  auto h_els = Kokkos::create_mirror_view(dev_els);
  for (int e = 0; e < nE; ++e) h_els(e) = els[e];
  Kokkos::deep_copy(dev_els, h_els);
  Kokkos::View<double**, Space> fe("fe", nE, NDOF), Ke("Ke", nE, NDOF * NDOF);
  Kokkos::View<double*, Space> reduced("reduced", nE);

  constexpr int work_doubles = 2 * NDOF + NDOF + NDOF * NDOF + 64;  // Work, fe, Ke, reduce
  Policy policy(nE, 1, vl);
  policy.set_scratch_size(0, Kokkos::PerTeam(work_doubles * sizeof(double)));
  Kokkos::parallel_for(
      "contract_kokkos_lanes", policy, KOKKOS_LAMBDA(const Member& m) {
        const int e = m.league_rank();
        Scratch buf(m.team_scratch(0), work_doubles);
        auto* work = reinterpret_cast<Brick::Work*>(buf.data());
        double* f = buf.data() + 2 * NDOF;
        auto* K = reinterpret_cast<double(*)[NDOF]>(f + NDOF);
        double* red = f + NDOF + NDOF * NDOF;
        femorax::exec::VectorLanes<Member> ctx{m, red, 64};
        const Element& el = dev_els(e);
        Brick::compute<true>(ctx, p, d, el.xe, el.u, el.v, {}, {}, f, K, *work);
        const double s = inner_reduce(ctx, NDOF, [&](int i) { return f[i] * (i + 1); });
        Kokkos::single(Kokkos::PerThread(m), [&]() {
          for (int i = 0; i < NDOF; ++i) fe(e, i) = f[i];
          for (int i = 0; i < NDOF * NDOF; ++i) Ke(e, i) = (&K[0][0])[i];
          reduced(e) = s;
        });
      });
  return {fe, Ke, reduced};
}

TEST(MemoryContract, KokkosLaneMappingMatchesSerial) {
  const auto els = elements(4);
  const int nE = static_cast<int>(els.size());
  const Brick::Params p{{377000.0, 0.3, 2.0285}, {0.5, -1.0, -19.9}, 0};
  const Rayleigh d{0.3, 1e-3, 2e-4, 3e-4};
  // Lanes per thread: one, and the backend's maximum (a full warp on a GPU).
  using Policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
  const int vmax = std::min(Policy::vector_length_max(), 32);
  const auto one = run_kokkos_lanes(els, p, d, 1);
  const auto [fe, Ke, reduced] = run_kokkos_lanes(els, p, d, vmax);
  // The lane split never changes a bit, on any backend.
  auto bits = [](const auto& a, const auto& b) {
    auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), a);
    auto hb = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
    return std::memcmp(ha.data(), hb.data(), ha.size() * sizeof(double)) == 0;
  };
  EXPECT_TRUE(bits(one.fe, fe)) << "fe differs between 1 and " << vmax << " lanes";
  EXPECT_TRUE(bits(one.Ke, Ke)) << "Ke differs between 1 and " << vmax << " lanes";
  EXPECT_TRUE(bits(one.reduced, reduced)) << "inner_reduce differs between 1 and " << vmax << " lanes";
  // Against the host serial result: bit for bit on host backends; to rounding where the
  // device compiler fuses multiply-adds (nvcc --fmad=true).
  constexpr bool host = std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;
  auto h_fe = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fe);
  auto h_Ke = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Ke);
  auto h_red = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reduced);
  for (int e = 0; e < nE; ++e) {
    double f[NDOF], K[NDOF][NDOF];
    Brick::compute<true>(p, d, els[e].xe, els[e].u, els[e].v, {}, {}, f, K);
    double s = 0;
    for (int i = 0; i < NDOF; ++i) s += f[i] * (i + 1);
    double fs = 0, ks = 0;
    for (int i = 0; i < NDOF; ++i) fs = std::max(fs, std::abs(f[i]));
    for (int i = 0; i < NDOF * NDOF; ++i) ks = std::max(ks, std::abs((&K[0][0])[i]));
    const double tf = host ? 0.0 : 1e-13 * fs, tk = host ? 0.0 : 1e-13 * ks;
    for (int i = 0; i < NDOF; ++i) EXPECT_NEAR(h_fe(e, i), f[i], tf) << "element " << e;
    for (int i = 0; i < NDOF * NDOF; ++i) EXPECT_NEAR(h_Ke(e, i), (&K[0][0])[i], tk) << "element " << e;
    EXPECT_NEAR(h_red(e), s, host ? 0.0 : 1e-13 * std::abs(s) * NDOF) << "element " << e;
  }
}

}  // namespace
