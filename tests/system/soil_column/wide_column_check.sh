#!/usr/bin/env bash
# A wide column (n x n x 25 SSPbrick, laminar sides) against the stored OpenSees output of the
# single 1 x 1 x 25 column (column25_ssp_explicit): with laminar sides and a uniform base
# motion every vertical line of nodes must move as the single column does. FemoraX only; no
# new OpenSees run. The re-equilibration uses NormDispIncr 1e-9 (an absolute force
# tolerance does not carry over to n^2 columns) and the shaking stops at 0.5 s of the same
# 2 s motion. Tolerance: docs/deviations.md, "Wide columns".
#   tests/system/soil_column/wide_column_check.sh <femorax> <python> <work dir> <n> <tol>
set -euo pipefail
FX=$1; PY=$2; W=$3; N=$4; TOL=$5
HERE=$(cd "$(dirname "$0")" && pwd)
REF=$HERE/../../reference/data/system/column25_ssp_explicit
rm -rf "$W"
"$PY" "$HERE/make_nonlinear.py" "$W" "$N" "$N" 25 2.0 2.5e-4 --element SSPbrick --soil layered --shaking explicit \
  --vtkhdf 0.05 --record-dt 0.01 --equilibration newton --equilibration-test NormDispIncr:1e-9 --until 0.5 > /dev/null
(cd "$W" && "$FX" run model.femorax.h5 > femorax.log 2>&1) || { cat "$W/femorax.log"; exit 1; }
grep -E "stage|done" "$W/femorax.log" || true
"$PY" "$HERE/compare_wide_column.py" "$REF/opensees.vtkhdf" "$W/femorax.vtkhdf" --tol "$TOL" --nodes "$REF" "$W"
