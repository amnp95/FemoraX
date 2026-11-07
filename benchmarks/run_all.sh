#!/usr/bin/env bash
# Build FemoraX and run the full benchmark suite on a TACC node (Vista or Stampede3).
# Benchmarks never run in the commit check; performance is measured on TACC only
# (CLAUDE.md, "Performance"). This script submits no jobs: run it inside idev or your own
# job script. See benchmarks/README.md for the exact commands.
#
#   benchmarks/run_all.sh [--build-only] [--no-build] [--sizes "40 64 96"] [--reps N]
#
#   --build-only   configure and build, then stop (use on a login node: Kokkos and
#                  GoogleTest are downloaded at configure time, compute nodes may lack
#                  internet access)
#   --no-build     run with the existing build
#   --sizes        soil-column sizes n (n x n x n elements); default "40 64 96"
#   --reps         timed repetitions per benchmark (default 20)
#
# Environment overrides: FEMORAX_MODULES (modules to load instead of the defaults below),
# FEMORAX_BUILD_DIR (default build/tacc-<machine>-<node type>).
#
# Output: benchmarks/results/<machine>-<node>-<date>-<commit>/ with meta.json (commit,
# machine, node, CPU, GPU, modules, date) and one femorax_bench JSON per size. Compare
# with a baseline recorded on the same machine and node type:
#   python3 benchmarks/check.py --no-build --preset <build dir name> [--update-baseline]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

BUILD=1 RUN=1 SIZES="40 64 96" REPS=20
while [ $# -gt 0 ]; do
  case "$1" in
    --build-only) RUN=0 ;;
    --no-build) BUILD=0 ;;
    --sizes) SIZES="$2"; shift ;;
    --reps) REPS="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

# --- Which machine and node --------------------------------------------------------------
MACHINE="${TACC_SYSTEM:-}"
if [ -z "$MACHINE" ]; then
  case "$(hostname -f 2>/dev/null || hostname)" in
    *vista*) MACHINE=vista ;;
    *stampede3*) MACHINE=stampede3 ;;
    *) echo "not a TACC Vista or Stampede3 node (TACC_SYSTEM unset, hostname $(hostname))" >&2; exit 2 ;;
  esac
fi
HAS_NVIDIA=0
if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then HAS_NVIDIA=1; fi
CPU_ARCH=$(uname -m)

# Kokkos backend and architectures per node type.
#   Vista GH  (Grace + H200):        CUDA, HOPPER90 + ARMV9_GRACE
#   Vista GG  (Grace-Grace):         OpenMP, ARMV9_GRACE
#   Stampede3 h100 (SPR + 4x H100):  CUDA, HOPPER90 + SPR
#   Stampede3 spr / icx / skx:       OpenMP, SPR / ICX / SKX
#   Stampede3 pvc (Intel Max GPU):   not supported yet (no SYCL build)
CMAKE_ARGS=()
case "$MACHINE" in
  vista)
    if [ "$HAS_NVIDIA" = 1 ]; then NODE=gh; else NODE=gg; fi
    DEFAULT_MODULES="gcc cuda cmake hdf5"
    CMAKE_ARGS+=(-DKokkos_ARCH_ARMV9_GRACE=ON)
    ;;
  stampede3)
    if [ "$HAS_NVIDIA" = 1 ]; then
      NODE=h100
      CMAKE_ARGS+=(-DKokkos_ARCH_SPR=ON)
    else
      model=$(grep -m1 "model name" /proc/cpuinfo || true)
      case "$model" in
        *"CPU Max 9480"*|*"8468"*|*"8480"*) NODE=spr; CMAKE_ARGS+=(-DKokkos_ARCH_SPR=ON) ;;
        *"8380"*) NODE=icx; CMAKE_ARGS+=(-DKokkos_ARCH_ICX=ON) ;;
        *"8160"*) NODE=skx; CMAKE_ARGS+=(-DKokkos_ARCH_SKX=ON) ;;
        *) echo "unknown Stampede3 CPU: $model (a pvc node? Intel GPUs are not supported yet)" >&2; exit 2 ;;
      esac
    fi
    DEFAULT_MODULES="gcc cuda cmake hdf5"
    [ "$HAS_NVIDIA" = 1 ] || DEFAULT_MODULES="gcc cmake hdf5"
    ;;
  *) echo "unsupported machine: $MACHINE" >&2; exit 2 ;;
esac

# --- Modules ------------------------------------------------------------------------------
MODULES="${FEMORAX_MODULES:-$DEFAULT_MODULES}"
if command -v module > /dev/null 2>&1 || type module > /dev/null 2>&1; then
  module reset > /dev/null 2>&1 || true
  # shellcheck disable=SC2086
  module load $MODULES
else
  echo "warning: no 'module' command; using the current environment" >&2
fi
echo "machine $MACHINE, node $NODE ($CPU_ARCH), modules: $MODULES"

BUILD_DIR="${FEMORAX_BUILD_DIR:-build/tacc-$MACHINE-$NODE}"
PRESET_NAME=$(basename "$BUILD_DIR")

# --- Build --------------------------------------------------------------------------------
if [ "$BUILD" = 1 ]; then
  if [ "$HAS_NVIDIA" = 1 ] || [ "$NODE" = gh ] || [ "$NODE" = h100 ]; then
    cmake -P cmake/FetchKokkosSource.cmake > /dev/null   # nvcc_wrapper
    KSRC="$ROOT/build/_kokkos/kokkos-5.1.0"
    CMAKE_ARGS+=(-DCMAKE_CXX_COMPILER="$KSRC/bin/nvcc_wrapper" -DFETCHCONTENT_SOURCE_DIR_KOKKOS="$KSRC"
                 -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_HOPPER90=ON -DKokkos_ENABLE_SERIAL=ON
                 -DKokkos_ENABLE_OPENMP=OFF)
  else
    CMAKE_ARGS+=(-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON)
  fi
  cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DFEMORAX_BUILD_TESTS=OFF \
        -DFEMORAX_BUILD_BENCHMARKS=ON "${CMAKE_ARGS[@]}"
  cmake --build "$BUILD_DIR" --target femorax_bench femorax -j "$(nproc)"
fi
[ "$RUN" = 1 ] || { echo "built: $BUILD_DIR"; exit 0; }

EXE="$BUILD_DIR/benchmarks/femorax_bench"
[ -x "$EXE" ] || { echo "$EXE not found; build first" >&2; exit 2; }

# --- Run ----------------------------------------------------------------------------------
COMMIT=$(git rev-parse --short=12 HEAD)
DIRTY=$(git status --porcelain --untracked-files=no | head -1)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT="benchmarks/results/$MACHINE-$NODE-$STAMP-$COMMIT"
mkdir -p "$OUT"
GPU="none"
[ "$HAS_NVIDIA" = 1 ] && GPU=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)
CPU=$(grep -m1 "model name" /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//' || true)
[ -n "$CPU" ] || CPU=$(lscpu 2>/dev/null | grep -m1 "Model name" | cut -d: -f2- | sed 's/^ *//' || echo unknown)
cat > "$OUT/meta.json" <<EOF
{
  "commit": "$COMMIT",
  "dirty": $([ -n "$DIRTY" ] && echo true || echo false),
  "date_utc": "$STAMP",
  "machine": "$MACHINE",
  "node": "$NODE",
  "hostname": "$(hostname)",
  "cpu": "$CPU",
  "cpu_arch": "$CPU_ARCH",
  "cores": $(nproc),
  "gpu": "$GPU",
  "modules": "$MODULES",
  "build_dir": "$BUILD_DIR",
  "slurm_job": "${SLURM_JOB_ID:-}"
}
EOF
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" OMP_PLACES="${OMP_PLACES:-threads}"
for n in $SIZES; do
  echo "== size $n^3"
  "$EXE" --size "$n" --reps "$REPS" --json "$OUT/size$n.json" | tee "$OUT/size$n.txt"
done
echo "results: $OUT"
echo "compare the 40^3 run with a baseline of this machine and node:"
echo "  python3 benchmarks/check.py --no-build --preset $PRESET_NAME"
