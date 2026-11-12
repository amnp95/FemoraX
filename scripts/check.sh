#!/usr/bin/env bash
# Pre-commit check: build every preset and run its correctness tests (unit tests, OpenSees
# comparisons, small models). Stops at the first failure, so a failed build is never hidden
# by stale test binaries. Benchmarks are not part of it: performance is measured on TACC
# (benchmarks/README.md).
#
#   scripts/check.sh            # laptop: cpu-debug, cpu-release, cpu-asan
#                               # (plus cuda-ampere86 when nvidia-smi sees a GPU)
#                               # Stampede3: s3-cpu-debug, s3-cpu-release, s3-cpu-ubsan
#                               # (ASan cannot run on its compute nodes: CMakePresets.json)
#                               # (source scripts/stampede3_env.sh first; on a compute node)
#                               # Vista: vista-cpu-debug, vista-cpu-release, vista-cpu-ubsan
#                               # (source scripts/vista_env.sh first; on a compute node)
set -euo pipefail
cd "$(dirname "$0")/.."
# Build jobs: every core. TACC sets OMP_NUM_THREADS=1, which nproc obeys.
jobs=$(env -u OMP_NUM_THREADS nproc)
presets="cpu-debug cpu-release cpu-asan"
host="$(hostname -f 2>/dev/null || hostname)"
if [[ "${host}" == *.stampede3.tacc.utexas.edu ]]; then
  presets="s3-cpu-debug s3-cpu-release s3-cpu-ubsan"
elif [[ "${host}" == *.vista.tacc.utexas.edu ]]; then
  presets="vista-cpu-debug vista-cpu-release vista-cpu-ubsan"
elif command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 && [ -x /usr/local/cuda-13.1/bin/nvcc ]; then
  cmake -P cmake/FetchKokkosSource.cmake > /dev/null  # nvcc_wrapper for the CUDA preset
  presets="${presets} cuda-ampere86"
fi
for preset in ${presets}; do
  echo "== ${preset}: build"
  [ -d "build/${preset}" ] || cmake --preset "${preset}" > /dev/null
  cmake --build --preset "${preset}" -j"${jobs}" > "build/${preset}.build.log" 2>&1 || {
    grep -E "error|FAILED" "build/${preset}.build.log" | head -20
    echo "build failed: ${preset} (full log: build/${preset}.build.log)"
    exit 1
  }
  echo "== ${preset}: test"
  ctest --preset "${preset}" -j"${jobs}" 2>&1 | tail -3
done
