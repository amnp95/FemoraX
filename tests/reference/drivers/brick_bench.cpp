// OpenSees speed reference for the explicit force: the element loop over a soil-column mesh
// of Brick (stdBrick) + ElasticIsotropic elements, single thread, at the pinned commit.
//
//   brick_bench <n> [reps]
//
// The mesh, materials, body forces, lumped mass and Rayleigh regions are those of
// tests/common/synthetic_models.hpp (soil_column with nx = ny = nz = n): the model the
// FemoraX benchmark explicit_force/<n>^3 uses. Every node gets trial displacement 1e-4 and
// trial velocity 1e-3 in all DOFs (as ExplicitForce::fill in the FemoraX benchmark).
//
// Timed (minimum over reps), each over all elements:
//   internal   ele->update(); ele->getResistingForce();            (elements without damping)
//   damped     ele->update(); ele->getResistingForceIncInertia();  (with the Rayleigh factors;
//              what an explicit step asks of each element; trial accelerations are zero)
// Element vectors only: OpenSees assembles them into the system afterwards, which is not
// timed here (FemoraX's times include its scatter).

#include "driver_common.hpp"

#include <Brick.h>
#include <Domain.h>
#include <ElasticIsotropicMaterial.h>
#include <Node.h>
#include <Vector.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: brick_bench <n> [reps]\n");
    return 1;
  }
  const int n = std::atoi(argv[1]);
  const int reps = argc > 2 ? std::atoi(argv[2]) : 5;
  const int NX = n + 1, NY = n + 1, NZ = n + 1;
  const double dx = 1.0, dy = 1.0, dz = 0.5;
  auto tag = [&](int i, int j, int k) { return 1 + i + NX * (j + NY * k); };

  Domain domain;
  for (int k = 0; k < NZ; ++k)
    for (int j = 0; j < NY; ++j)
      for (int i = 0; i < NX; ++i)
        domain.addNode(new Node(tag(i, j, k), 3, i * dx, j * dy, -n * dz + k * dz));

  ElasticIsotropicMaterial lower(1, 377000.0, 0.3, 2.0285), upper(2, 109200.0, 0.3, 2.0183);
  std::vector<Brick*> elements;
  int etag = 1;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const bool low = k < n / 2;
        auto* e = new Brick(etag++, tag(i, j, k), tag(i + 1, j, k), tag(i + 1, j + 1, k),
                            tag(i, j + 1, k), tag(i, j, k + 1), tag(i + 1, j, k + 1),
                            tag(i + 1, j + 1, k + 1), tag(i, j + 1, k + 1), low ? lower : upper,
                            0.0, 0.0, low ? -19.9 : -19.8, nullptr, 1);
        domain.addElement(e);
        elements.push_back(e);
      }

  Vector u(3), v(3);
  u(0) = u(1) = u(2) = 1e-4;
  v(0) = v(1) = v(2) = 1e-3;
  for (int t = 1; t <= NX * NY * NZ; ++t) {
    domain.getNode(t)->setTrialDisp(u);
    domain.getNode(t)->setTrialVel(v);
  }

  double checksum = 0;
  auto time_loop = [&](bool damped) {
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (Brick* e : elements) {
        e->update();
        const Vector& f = damped ? e->getResistingForceIncInertia() : e->getResistingForce();
        checksum += f(0);
      }
      const auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    return best;
  };

  const double t_internal = time_loop(false);
  // Rayleigh regions as in synthetic_models.hpp: all elements, then the top layer.
  const int nel = static_cast<int>(elements.size()), top = n * n;
  for (int e = 0; e < nel; ++e) {
    if (e >= nel - top) elements[e]->setRayleighDampingFactors(0.9, 3e-4, 1e-4, 0.0);
    else elements[e]->setRayleighDampingFactors(0.6, 5.8e-4, 0.0, 0.0);
  }
  const double t_damped = time_loop(true);

  std::printf("opensees_commit %s\n", OPENSEES_COMMIT);
  std::printf("mesh %d^3: %d elements, %d nodes, %d reps (minimum)\n", n, nel, NX * NY * NZ, reps);
  std::printf("internal %.6f s  (%.3f us/element)\n", t_internal, 1e6 * t_internal / nel);
  std::printf("damped   %.6f s  (%.3f us/element)\n", t_damped, 1e6 * t_damped / nel);
  std::printf("checksum %.6e\n", checksum);
  return 0;
}
