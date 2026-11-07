#!/usr/bin/env bash
# Pre-commit check: build every preset and run its correctness tests (unit tests, OpenSees
# comparisons, small models). Stops at the first failure, so a failed build is never hidden
# by stale test binaries. Benchmarks are not part of it: performance is measured on TACC
# (benchmarks/README.md).
#
#   scripts/check.sh            # cpu-debug, cpu-release, cpu-asan
#                               # (plus cuda-ampere86 when nvidia-smi sees a GPU)
set -euo pipefail
cd "$(dirname "$0")/.."
presets="cpu-debug cpu-release cpu-asan"
if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 && [ -x /usr/local/cuda-13.1/bin/nvcc ]; then
  cmake -P cmake/FetchKokkosSource.cmake > /dev/null  # nvcc_wrapper for the CUDA preset
  presets="${presets} cuda-ampere86"
fi
for preset in ${presets}; do
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
