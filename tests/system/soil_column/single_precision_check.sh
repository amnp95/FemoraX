#!/usr/bin/env bash
# Single-precision VTKHDF output (femorax run --vtkhdf-precision single) against the double
# output of the same run: a 1 x 1 x 4 layered PressureDependMultiYield column with explicit
# shaking (nodes and elements recorded every 0.01 s; a few seconds). No OpenSees.
#   tests/system/soil_column/single_precision_check.sh <femorax> <python> <work dir>
set -euo pipefail
FX=$(realpath "$1"); PY=$(realpath -s "$2"); W=$3
HERE=$(cd "$(dirname "$0")" && pwd)
rm -rf "$W"
"$PY" "$HERE/make_nonlinear.py" "$W" 1 1 4 0.5 5e-4 --element SSPbrick --soil layered --shaking explicit \
  --vtkhdf 0.01 > /dev/null
cd "$W"
OMP_NUM_THREADS=2 "$FX" run model.femorax.h5 --quiet 2> double.log
mv femorax.vtkhdf double.vtkhdf
OMP_NUM_THREADS=2 "$FX" run model.femorax.h5 --quiet --vtkhdf-precision single 2> single.log
mv femorax.vtkhdf single.vtkhdf
"$PY" "$HERE/compare_single_precision.py" double.vtkhdf single.vtkhdf
