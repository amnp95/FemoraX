#!/usr/bin/env bash
# FemoraX against the stored OpenSees output of the nonlinear reference cases
# (nonlinear_cases.sh, stored by nonlinear_references.sh). OpenSees does not run here.
#   tests/system/soil_column/nonlinear_check.sh <femorax> <python> <work dir> [host|gpu]
set -euo pipefail
FX=$1; PY=$2; W=$3; BACKEND=${4:-host}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$HERE/../../reference/data/system
source "$HERE/nonlinear_cases.sh"
status=0
for c in "${NONLINEAR_CASES[@]}"; do
  IFS='|' read -r name args tol gtol <<< "$c"
  [ "$BACKEND" = gpu ] && tol=$gtol
  D=$W/$name
  rm -rf "$D"
  # shellcheck disable=SC2086
  "$PY" "$HERE/make_nonlinear.py" "$D" $args > /dev/null
  if [ "$(sha256sum "$D/model.tcl" | cut -d' ' -f1)" != "$(cat "$DATA/$name/model.tcl.sha256")" ]; then
    echo "$name: the stored OpenSees reference is out of date (model.tcl changed); rerun"
    echo "  tests/system/soil_column/nonlinear_references.sh <OpenSees> <python>"
    status=1
    continue
  fi
  cp "$DATA/$name"/opensees_*.txt "$D/"
  (cd "$D" && OMP_NUM_THREADS=2 "$FX" run model.femorax.h5 --quiet 2> femorax.log)
  echo "== $name (tolerance $tol)"
  "$PY" "$HERE/compare.py" "$D" --tol "$tol" || status=1
  if [ -f "$DATA/$name/opensees.vtkhdf" ]; then  # VTKHDF with element output, through femora.results
    "$PY" "$HERE/compare_vtkhdf.py" "$DATA/$name/opensees.vtkhdf" "$D/femorax.vtkhdf" --tol "$tol" || status=1
  fi
done
exit $status
