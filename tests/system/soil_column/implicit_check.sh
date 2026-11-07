#!/usr/bin/env bash
# Implicit stages against OpenSees on a 2x2x10 column with consistent mass (a few seconds):
# implicit-dynamic and static gravity, Linear and Newton, then 0.2 s of shaking; every
# recorded value within the system-test tolerance for implicit stages (docs/deviations.md).
#   tests/system/soil_column/implicit_check.sh <femorax> <OpenSees> <python> <work dir>
set -euo pipefail
FX=$1; OS=$2; PY=$3; W=$4
HERE=$(cd "$(dirname "$0")" && pwd)
for gravity in dynamic static; do
  for alg in Linear Newton; do
    D=$W/$gravity-$alg
    rm -rf "$D"
    "$PY" "$HERE/make_implicit.py" "$D" 2 2 10 0.2 1e-3 --gravity $gravity --algorithm $alg > /dev/null
    (cd "$D" && "$OS" model.tcl > opensees.log 2>&1)
    (cd "$D" && OMP_NUM_THREADS=2 "$FX" run model.femorax.h5 --quiet 2> femorax.log)
    echo "== gravity $gravity, $alg"
    "$PY" "$HERE/compare.py" "$D" --tol 1e-8
  done
done
