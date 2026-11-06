#pragma once

// Exact floating-point operation counts of the stdBrick + ElasticIsotropic explicit force,
// per element, from the physics source itself (benchmarks/flop_count.cpp). Used for the
// benchmarks' GFLOP/s. An add, subtract, multiply, divide or square root counts as one
// FLOP (a fused multiply-add as two, which is what a GPU executes for a * b + c).

namespace femorax::bench {

struct BrickFlopCase {
  int massType;  // 0 consistent, 1 lumped
  double alphaM, betaK, betaKinit, betaKcomm;
};

// One element's force in the flat layout (StdBrick::compute / compute_cached) or the team
// layout (per Gauss point, as launch_team does it), with recomputed or cached geometry.
double brick_flops_per_element(const BrickFlopCase& c, bool team, bool cached_geometry);

}  // namespace femorax::bench
