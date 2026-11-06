#!/usr/bin/env bash
# Pre-commit check: build every CPU preset, run its tests, then the benchmark check.
# Stops at the first failure, so a failed build is never hidden by stale test binaries.
#
#   scripts/check.sh            # cpu-debug, cpu-release, cpu-asan, then benchmarks
set -euo pipefail
cd "$(dirname "$0")/.."
for preset in cpu-debug cpu-release cpu-asan; do
  echo "== ${preset}: build"
  [ -d "build/${preset}" ] || cmake --preset "${preset}" > /dev/null
  cmake --build --preset "${preset}" -j"$(nproc)" > "build/${preset}.build.log" 2>&1 || {
    grep -E "error|FAILED" "build/${preset}.build.log" | head -20
    echo "build failed: ${preset} (full log: build/${preset}.build.log)"
    exit 1
  }
  echo "== ${preset}: test"
  ctest --preset "${preset}" -j"$(nproc)" 2>&1 | tail -3
done
echo "== benchmarks"
python3 benchmarks/check.py --no-build
