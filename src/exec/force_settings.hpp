#pragma once

// Force-kernel settings per device: defaults chosen by benchmark, and a startup auto-tune.
// Nothing above the kernels assumes a variant; it asks for settings here. Every variant
// stays in the code, the tests and the benchmarks.

#include <vector>

#include "exec/explicit_force.hpp"

namespace femorax::exec {

enum class Purpose {
  Production,     // fastest
  Deterministic,  // bit-identical run to run (validation): colored or gather scatter
};

// Measured defaults for the default execution space (benchmarks/bench_explicit.cpp). An
// entry changes only with benchmark numbers in the commit message.
ForceSettings default_force_settings(Purpose purpose);

// Every variant valid for the purpose on this build (team sizes that fit the scratch,
// cached geometry only if the force object has a cache).
std::vector<ForceSettings> force_variants(Purpose purpose, const ExplicitForce& force);

struct TuneResult {
  ForceSettings best;
  std::vector<std::pair<ForceSettings, double>> seconds;  // minimum per variant
};

// Time every variant on the actual model (one warm-up, then reps timed runs, minimum
// kept) and return the fastest. Leaves f as computed by the last variant.
TuneResult autotune(ExplicitForce& force, Purpose purpose, int reps = 3);

}  // namespace femorax::exec
