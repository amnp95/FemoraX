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
#                               # (plus vista-cuda-hopper90 on gh nodes)
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
  # gh nodes: the H200 too
  if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
    presets="${presets} vista-cuda-hopper90"
  fi
elif command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 && [ -x /usr/local/cuda-13.1/bin/nvcc ]; then
  cmake -P cmake/FetchKokkosSource.cmake > /dev/null  # nvcc_wrapper for the CUDA preset
  presets="${presets} cuda-ampere86"
fi
# Build each preset in turn (each build uses every core), then run the test suites of all
# presets at the same time, the cores split between them (each case runs 2 OpenMP threads).
for preset in ${presets}; do
  echo "== ${preset}: build"
  [ -d "build/${preset}" ] || cmake --preset "${preset}" > /dev/null
  cmake --build --preset "${preset}" -j"${jobs}" > "build/${preset}.build.log" 2>&1 || {
    grep -E "error|FAILED" "build/${preset}.build.log" | head -20
    echo "build failed: ${preset} (full log: build/${preset}.build.log)"
    exit 1
  }
done
n=$(echo ${presets} | wc -w)
test_jobs=$(( jobs / n > 1 ? jobs / n : 1 ))
declare -A pids
for preset in ${presets}; do
  echo "== ${preset}: test (log: build/${preset}.test.log)"
  ctest --preset "${preset}" -j"${test_jobs}" > "build/${preset}.test.log" 2>&1 &
  pids[${preset}]=$!
done
failed=0
for preset in ${presets}; do
  if wait "${pids[${preset}]}"; then status=passed; else status=FAILED; failed=1; fi
  echo "== ${preset}: tests ${status}: $(grep -E "tests passed|tests failed" "build/${preset}.test.log" | tail -1)"
  [ "${status}" = passed ] || grep -E "Failed|\*\*\*" "build/${preset}.test.log" | head -20
done
exit ${failed}
