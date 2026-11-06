#pragma once

// Force-kernel settings. Users never choose them: choose_force_settings() picks them at
// startup, and every choice it can make gives the same bits (gather scatter, any layout,
// any geometry mode, any elements per team: tested in test_internal_force.cpp), so the
// choice never changes answers.
//
// Order of precedence:
//   1. FEMORAX_FORCE_SETTINGS (developer only, not in user docs or Femora): a settings
//      string such as "flat_atomic_cached" or "team_colored_ept4"; atomic and colored are
//      reachable only this way.
//   2. A saved auto-tune result for this machine and model type (tune cache file).
//   3. Auto-tune now (times every gather variant on the actual model), then save it.
// The tune cache is $FEMORAX_TUNE_CACHE, else $XDG_CACHE_HOME/femorax/autotune.txt, else
// ~/.cache/femorax/autotune.txt.

#include <string>
#include <utility>
#include <vector>

#include "compile/batches.hpp"
#include "exec/internal_force.hpp"

namespace femorax::exec {

// Measured per-device defaults (benchmarks/bench_explicit.cpp); used when auto-tuning is
// off (tests) and as the first candidate. Gather everywhere: the same model always gives
// identical results.
ForceSettings default_force_settings();

// The variants the auto-tune chooses among: gather scatter, every elements-per-team that
// fits, cached geometry if the force object has a cache; every layout on host backends,
// the default's layout on GPU backends (layouts_share_arithmetic). All bit-identical.
std::vector<ForceSettings> tunable_variants(const InternalForce& force);

// Whether the flat and team layouts compile to the same arithmetic (host backends: yes;
// GPU backends: no, the device compiler fuses multiply-adds differently in each).
bool layouts_share_arithmetic();

// Every variant, including atomic and colored (colored only with a coloring): for tests,
// benchmarks and the developer override.
std::vector<ForceSettings> all_variants(const InternalForce& force);

struct TuneResult {
  ForceSettings best;
  std::vector<std::pair<ForceSettings, double>> seconds;  // minimum per variant
};

// Time each variant (one warm-up, then reps timed runs, minimum kept) on the actual model.
TuneResult autotune(InternalForce& force, const std::vector<ForceSettings>& variants, int reps = 3);

// What the model looks like to the tuner: element-material pairs, damping kind, size class
// (powers of two of the element count).
std::string tune_key(const compile::Compiled& c);
// This machine and build: host, backend, concurrency, SIMD width, FemoraX version.
std::string machine_key();

struct Choice {
  ForceSettings settings;
  std::string source;  // "override", "saved", "tuned"
};

// The precedence above. Saves a new auto-tune result to the tune cache.
Choice choose_force_settings(InternalForce& force, const compile::Compiled& c);

}  // namespace femorax::exec
