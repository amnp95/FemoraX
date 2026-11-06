#!/usr/bin/env bash
# VTKHDF output against OpenSees through femora.results, on a 2x2x4 column (a few seconds).
#   tests/system/soil_column/vtkhdf_check.sh <femorax> <OpenSees> <python> <work dir>
# Every step recorded (-dT 0) plus a run at -dT 0.0025, renumbering off and on.
set -euo pipefail
FX=$1; OS=$2; PY=$3; W=$4
HERE=$(cd "$(dirname "$0")" && pwd)
for dT in 0 0.0025; do
  for renum in off on; do
    D=$W/dT$dT-$renum
    rm -rf "$D"
    "$PY" "$HERE/make_models.py" "$D" 2 2 4 0.01 2.5e-4 --vtkhdf $dT --no-node-recorders > /dev/null
    (cd "$D" && "$OS" model.tcl > opensees.log 2>&1)
    (cd "$D" && OMP_NUM_THREADS=2 "$FX" run model.femorax.h5 --renumbering $renum --quiet 2> femorax.log)
    echo "== dT $dT, renumbering $renum"
    "$PY" "$HERE/compare_vtkhdf.py" "$D/opensees.vtkhdf" "$D/femorax.vtkhdf"
  done
done
