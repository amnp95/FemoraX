// The beams with a fiber section (docs/design.md, "Fiber sections and uniaxial materials",
// step 3): ForceBeam3d (5 Lobatto points) and DispBeam3d (3 Gauss-Legendre points) with a
// reinforced-concrete FiberSection3d (Concrete02 core and cover, Steel01 bars), against OpenSees'
// ForceBeamColumn3d and DispBeamColumn3d in a Domain (tests/reference/drivers/
// fiber_beam_column.cpp): a column driven through growing lateral cycles to 2.5 % drift with
// axial shortening, one update and one commit per step; the force and the tangent at every step.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "physics/elements/disp_beam_3d.hpp"
#include "physics/elements/force_beam_3d.hpp"
#include "physics/sections/fiber_section_3d.hpp"
#include "reference_data.hpp"

namespace {

using namespace femorax;
using namespace femorax::test;
using Sec = FiberSection3d<32>;

void section_from(const ReferenceFile& ref, Sec::Params& p) {
  const auto mv = ref.values("materials"), fv = ref.values("fibers");
  for (std::size_t m = 0; m < mv.size() / 8; ++m) {
    p.materials[m].type = static_cast<int>(mv[8 * m]);
    for (int k = 0; k < 7; ++k) p.materials[m].v[k] = mv[8 * m + 1 + static_cast<std::size_t>(k)];
  }
  p.numFibers = static_cast<int>(fv.size() / 4);
  ASSERT_LE(p.numFibers, Sec::MAX_FIBERS);
  for (int i = 0; i < p.numFibers; ++i) {
    const std::size_t o = 4 * static_cast<std::size_t>(i);
    p.matData[3 * i] = fv[o];
    p.matData[3 * i + 1] = fv[o + 1];
    p.matData[3 * i + 2] = fv[o + 2];
    p.material[i] = static_cast<int>(fv[o + 3]);
  }
  p.GJ = ref.values("GJ")[0];
  Sec::compute_centroid(p);
}

struct Worst {
  double f = 0.0, K = 0.0, K0 = 0.0;
  long values = 0, bitwise = 0;
};

template <class Beam, class Converged>
Worst run(const std::string& file, int point_state, const Converged& converged) {
  Worst w;
  const ReferenceFile ref = read_reference(reference_data_dir() / "element" / file);
  EXPECT_EQ(ref.header.at("opensees_commit"), opensees_pinned_commit);
  EXPECT_EQ(static_cast<int>(ref.values("nip")[0]), Beam::NGP);
  auto p = std::make_unique<typename Beam::Params>();
  section_from(ref, p->mat);
  const auto xi = ref.values("xi"), xj = ref.values("xj"), vz = ref.values("vecxz");
  const double xe[2][3] = {{xi[0], xi[1], xi[2]}, {xj[0], xj[1], xj[2]}};
  for (int k = 0; k < 3; ++k) p->transf.vecxz[k] = vz[static_cast<std::size_t>(k)];
  p->transf.type = file.find("PDelta") != std::string::npos ? CrdTransfType::PDelta : CrdTransfType::Linear;
  std::vector<double> a(static_cast<std::size_t>(Beam::NSTATE), 0.0), b(a);
  double* committed = a.data();
  double* trial = b.data();
  const int c_d = ref.column("dj0"), c_f = ref.column("f0"), c_K = ref.column("K0_0");
  double f_scale = 0.0, K_scale = 0.0;
  for (const auto& row : ref.rows) {
    f_scale = std::max(f_scale, max_abs(&row[c_f], 12));
    K_scale = std::max(K_scale, max_abs(&row[c_K], 144));
  }
  double u[12] = {}, f[12], K[12][12];
  for (const auto& row : ref.rows) {
    for (int k = 0; k < 6; ++k) u[6 + k] = row[c_d + k];
    Beam::template compute<true>(*p, xe, u, PointArray<const double>{committed, 1, point_state},
                                 PointArray<double>{trial, 1, point_state}, f, K);
    EXPECT_TRUE(converged(PointArray<const double>{trial, 1, point_state})) << "step " << row[0];
    const bool initial = row[0] < 0.0;
    double df = 0.0, dK = 0.0;
    for (int k = 0; k < 12; ++k) {
      if (!initial) {
        df = std::max(df, std::fabs(f[k] - row[c_f + k]) / f_scale);
        w.bitwise += (f[k] == row[c_f + k]);
        ++w.values;
      }
      for (int l = 0; l < 12; ++l) {
        dK = std::max(dK, std::fabs(K[k][l] - row[c_K + 12 * k + l]) / K_scale);
        w.bitwise += (K[k][l] == row[c_K + 12 * k + l]);
        ++w.values;
      }
    }
    if (initial) {
      w.K0 = dK;  // the first row: getInitialStiff, here the tangent of the fresh state at rest
    } else {
      w.f = std::max(w.f, df);
      w.K = std::max(w.K, dK);
      std::swap(committed, trial);  // commit
    }
  }
  std::printf("[ %s against OpenSees: f %.2e, K %.2e, K0 %.2e of the peak; %ld of %ld values bit-identical ]\n",
              file.c_str(), w.f, w.K, w.K0, w.bitwise, w.values);
  return w;
}

TEST(FiberBeam3d, ForceBeamMatchesOpenSeesThroughYieldingCycles) {
  using Beam = ForceBeam3d<Sec, 5>;
  static_assert(Element<Beam>);
  for (const std::string type : {"Linear", "PDelta"}) {
    const Worst w = run<Beam>("FiberBeam3d_force_" + type + ".txt", Beam::POINT_STATE,
                              [](PointArray<const double> t) { return Beam::converged(t); });
    EXPECT_LT(w.f, 1e-12) << type;
    EXPECT_LT(w.K, 1e-12) << type;
    EXPECT_LT(w.K0, 1e-14) << type;
  }
}

TEST(FiberBeam3d, DisplacementBeamMatchesOpenSeesThroughYieldingCycles) {
  using Beam = DispBeam3d<Sec, 3>;
  static_assert(Element<Beam>);
  for (const std::string type : {"Linear", "PDelta"}) {
    const Worst w = run<Beam>("FiberBeam3d_disp_" + type + ".txt", Sec::NSTATE,
                              [](PointArray<const double>) { return true; });
    EXPECT_LT(w.f, 1e-13) << type;
    EXPECT_LT(w.K, 1e-13) << type;
    EXPECT_LT(w.K0, 1e-13) << type;
  }
}

}  // namespace
