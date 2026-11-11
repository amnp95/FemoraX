// PressureDependMultiYield (3D) against OpenSees along the strain paths of
// tests/reference/drivers/soil_material.cpp (data in tests/reference/data/nd): the same
// calls in the same order (stage switches, discarded trials, commits), stress and tangent
// compared at every step. Material tests' default tolerance, relative 1e-14 of the file's
// largest value (docs/deviations.md); the port keeps OpenSees' operation order, so the test
// also reports how many values match bit for bit, and the first step that does not.
//
// On the device backend (one thread per file, the whole path in one kernel): the same as
// the host on host backends; on GPUs the compiler fuses multiply-adds, and the test reports
// how far each file drifts and whether a path takes a different plastic branch.
//
// Also: the predictor/corrector split (predict, then correct for the points that need it)
// gives the bits of update at every step, and the surface bucket (NS = 20 or 40) changes
// nothing.

#include "physics/materials/pressure_depend_multi_yield.hpp"
#include "reference_data.hpp"

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace femorax::test;
constexpr int N = 6;
// Intel's compilers call Intel's own math library (pow, exp, ...), whose last bits differ
// from glibc's: up to 5.8e-14 along these paths, as the Intel-built OpenSees shows against
// the stored (GCC) data (docs/deviations.md, "Tolerances of other compilers and GPUs").
#if defined(__INTEL_LLVM_COMPILER)
constexpr double kRelTol = 1e-13;
#else
constexpr double kRelTol = 1e-14;
#endif

template <int NS>
typename femorax::PressureDependMultiYield<NS>::Params params_of(const ReferenceFile& ref, std::vector<double>& pairs) {
  using M = femorax::PressureDependMultiYield<NS>;
  pairs = ref.values("pairs");
  typename M::Input in{};
  in.rho = ref.param("rho");
  in.refShearModul = ref.param("refShearModul");
  in.refBulkModul = ref.param("refBulkModul");
  in.frictionAng = ref.param("frictionAng");
  in.peakShearStra = ref.param("peakShearStra");
  in.refPress = ref.param("refPress");
  in.pressDependCoe = ref.param("pressDependCoe");
  in.phaseTransformAng = ref.param("PTAng");
  in.contractionParam1 = ref.param("contrac");
  in.dilationParam1 = ref.param("dilat1");
  in.dilationParam2 = ref.param("dilat2");
  in.liquefactionParam1 = ref.param("liquefac1");
  in.liquefactionParam2 = ref.param("liquefac2");
  in.liquefactionParam4 = ref.param("liquefac3");
  in.numberOfYieldSurf = static_cast<int>(ref.param("noYieldSurf"));
  in.gredu = pairs.empty() ? nullptr : pairs.data();
  in.e = ref.param("e");
  in.volLimit1 = ref.param("cs1");
  in.volLimit2 = ref.param("cs2");
  in.volLimit3 = ref.param("cs3");
  in.atm = ref.param("pa");
  in.cohesi = ref.param("c");
  in.hv = 0.0;
  in.pv = 1.0;
  typename M::Params p{};
  EXPECT_EQ(M::setup(in, p), M::Ok);
  return p;
}

struct FileStats {
  long values = 0, bitwise = 0;
  double max_rel = 0.0;
  int first_diff = -1;  // first step with a value that is not bit-identical
};

// Replays one file; returns the stress and tangent of every step.
template <int NS>
std::vector<std::vector<double>> replay(const ReferenceFile& ref, bool split) {
  using M = femorax::PressureDependMultiYield<NS>;
  std::vector<double> pairs;
  auto p = params_of<NS>(ref, pairs);
  std::vector<double> C(M::NSTATE), T(M::NSTATE);
  M::initial_state(p, {C.data(), 1});
  M::initial_state(p, {T.data(), 1});
  const int c_stage = ref.column("stage"), c_pre = ref.column("pre"), c_pre0 = ref.column("pre0"),
            c_eps = ref.column("eps0");
  std::vector<std::vector<double>> out;
  for (const auto& row : ref.rows) {
    p.loadStage = static_cast<int>(row[c_stage]);
    double sig[N], D[N][N];
    const femorax::Strided<const double> c{C.data(), 1};
    const femorax::Strided<double> t{T.data(), 1};
    auto step = [&](const double* eps) {
      if (split) {
        if (M::predict(femorax::SerialInner{}, p, c, t, eps, sig, D))
          M::correct(femorax::SerialInner{}, p, c, t, eps, sig, D);
      } else {
        M::update(p, c, t, eps, sig, D);
      }
    };
    if (row[c_pre] != 0.0) step(&row[c_pre0]);  // the discarded trial: no commit
    step(&row[c_eps]);
    std::vector<double> v(sig, sig + N);
    v.insert(v.end(), &D[0][0], &D[0][0] + N * N);
    out.push_back(std::move(v));
    std::swap(C, T);  // commitState
  }
  return out;
}

TEST(PressureDependMultiYieldReference, MatchesOpenSeesAlongStrainPaths) {
  const auto files = reference_files("nd", "PressureDependMultiYield_");
  ASSERT_GE(files.size(), 20u) << "reference data missing in " << reference_data_dir()
                               << "; see tests/reference/README.md";
  long values = 0, bitwise = 0, files_bitwise = 0;
  double worst = 0.0;
  for (const auto& path : files) {
    SCOPED_TRACE(path.filename().string());
    const ReferenceFile ref = read_reference(path);
    ASSERT_EQ(ref.header.at("opensees_commit"), opensees_pinned_commit);
    const auto got = replay<20>(ref, false);
    const int c_sig = ref.column("sig0");
    double s_scale = 0, d_scale = 0;
    for (const auto& row : ref.rows) {
      s_scale = std::max(s_scale, max_abs(&row[c_sig], N));
      d_scale = std::max(d_scale, max_abs(&row[c_sig + N], N * N));
    }
    FileStats st;
    for (std::size_t k = 0; k < ref.rows.size(); ++k) {
      const double* want = &ref.rows[k][c_sig];
      for (int i = 0; i < N + N * N; ++i) {
        const double scale = i < N ? s_scale : d_scale;
        const double err = std::abs(got[k][i] - want[i]) / scale;
        st.max_rel = std::max(st.max_rel, err);
        EXPECT_LE(err, kRelTol) << "step " << k << (i < N ? " sig" : " D") << (i < N ? i : i - N);
        const bool same = got[k][i] == want[i];
        st.bitwise += same;
        ++st.values;
        if (!same && st.first_diff < 0) st.first_diff = static_cast<int>(k);
      }
    }
    values += st.values;
    bitwise += st.bitwise;
    files_bitwise += st.bitwise == st.values;
    worst = std::max(worst, st.max_rel);
    if (st.bitwise != st.values) {
      std::printf("  %-60s %6ld/%6ld bit-identical, first difference at step %d, max %.2e\n",
                  path.filename().string().c_str(), st.bitwise, st.values, st.first_diff, st.max_rel);
    }
  }
  std::printf("PressureDependMultiYield: %ld of %ld values bit-identical to OpenSees (%ld of %zu files "
              "entirely), largest difference %.2e of the file's largest value\n",
              bitwise, values, files_bitwise, files.size(), worst);
}

// The whole path of one file on the default execution space. A free function: nvcc does
// not allow device lambdas in a test body.
std::vector<std::vector<double>> replay_on_device(const ReferenceFile& ref) {
  using M = femorax::PressureDependMultiYield<20>;
  using Space = Kokkos::DefaultExecutionSpace;
  std::vector<double> pairs;
  const auto p = params_of<20>(ref, pairs);
  const int n = static_cast<int>(ref.rows.size());
  const int c_stage = ref.column("stage"), c_pre = ref.column("pre"), c_pre0 = ref.column("pre0"),
            c_eps = ref.column("eps0");
  Kokkos::View<double**, Kokkos::LayoutRight, Space> in("in", n, 14), out("out", n, N + N * N);
  auto h_in = Kokkos::create_mirror_view(in);
  for (int k = 0; k < n; ++k) {
    h_in(k, 0) = ref.rows[k][c_stage];
    h_in(k, 1) = ref.rows[k][c_pre];
    for (int i = 0; i < 6; ++i) h_in(k, 2 + i) = ref.rows[k][c_pre0 + i];
    for (int i = 0; i < 6; ++i) h_in(k, 8 + i) = ref.rows[k][c_eps + i];
  }
  Kokkos::deep_copy(in, h_in);
  Kokkos::View<double*, Space> state("state", 2 * M::NSTATE);
  Kokkos::parallel_for(
      "pdmy_path", Kokkos::RangePolicy<Space>(0, 1), KOKKOS_LAMBDA(const int) {
        M::Params q = p;
        double* buf[2] = {state.data(), state.data() + M::NSTATE};
        M::initial_state(q, {buf[0], 1});
        M::initial_state(q, {buf[1], 1});
        int c = 0;
        for (int k = 0; k < n; ++k) {
          q.loadStage = static_cast<int>(in(k, 0));
          double eps[6], sig[6], D[6][6];
          if (in(k, 1) != 0.0) {
            for (int i = 0; i < 6; ++i) eps[i] = in(k, 2 + i);
            M::update(q, {buf[c], 1}, {buf[1 - c], 1}, eps, sig, D);
          }
          for (int i = 0; i < 6; ++i) eps[i] = in(k, 8 + i);
          M::update(q, {buf[c], 1}, {buf[1 - c], 1}, eps, sig, D);
          for (int i = 0; i < 6; ++i) out(k, i) = sig[i];
          for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) out(k, N + 6 * i + j) = D[i][j];
          c = 1 - c;
        }
      });
  auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), out);
  std::vector<std::vector<double>> r(static_cast<std::size_t>(n));
  for (int k = 0; k < n; ++k) r[k].assign(&h(k, 0), &h(k, 0) + N + N * N);
  return r;
}

TEST(PressureDependMultiYieldReference, DeviceBackend) {
  constexpr bool host = std::is_same_v<Kokkos::DefaultExecutionSpace::memory_space, Kokkos::HostSpace>;
  const auto files = reference_files("nd", "PressureDependMultiYield_");
  ASSERT_GE(files.size(), 20u);
  int identical = 0, rounding = 0, flipped = 0;
  for (const auto& path : files) {
    SCOPED_TRACE(path.filename().string());
    const ReferenceFile ref = read_reference(path);
    const auto dev = replay_on_device(ref);
    const auto want = replay<20>(ref, false);  // the host result, bit-identical to OpenSees
    if (host) {
      EXPECT_EQ(dev, want);
      continue;
    }
    double s_scale = 0, d_scale = 0, worst = 0;
    for (const auto& v : want) {
      s_scale = std::max(s_scale, max_abs(v.data(), N));
      d_scale = std::max(d_scale, max_abs(v.data() + N, N * N));
    }
    for (std::size_t k = 0; k < want.size(); ++k)
      for (int i = 0; i < N + N * N; ++i)
        worst = std::max(worst, std::abs(dev[k][i] - want[k][i]) / (i < N ? s_scale : d_scale));
    if (dev == want) ++identical;
    else if (worst < 1e-10) ++rounding;
    else ++flipped;
    std::printf("  GPU %-60s largest difference %.2e of the largest value\n", path.filename().string().c_str(),
                worst);
  }
  if (!host) {
    std::printf("PressureDependMultiYield on the GPU: %d files bit-identical to the host, %d within 1e-10 "
                "(rounding), %d beyond (a different plastic path)\n",
                identical, rounding, flipped);
  }
}

TEST(PressureDependMultiYieldReference, PredictorCorrectorSplitAndBucketChangeNoBits) {
  const auto files = reference_files("nd", "PressureDependMultiYield_");
  ASSERT_GE(files.size(), 20u);
  for (const auto& path : files) {
    SCOPED_TRACE(path.filename().string());
    const ReferenceFile ref = read_reference(path);
    const auto whole = replay<20>(ref, false);
    EXPECT_EQ(replay<20>(ref, true), whole);
    EXPECT_EQ(replay<40>(ref, false), whole);
  }
}

}  // namespace
