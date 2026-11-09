#!/usr/bin/env bash
# FemoraX on Femora's nonlinear soil column window against the stored OpenSees output
# (femora_example_reference.sh). OpenSees does not run here. Checks:
#   - the retry and substep events, in order (stage, retry number, limit, step size);
#   - displacement, stress3D6, strain3D6 of the VTKHDF output (velocity and acceleration
#     reported only);
#   - the surface PGA and 5% response spectrum.
#   tests/system/femora_example/femora_example_check.sh <femorax> <python> <work dir> [host|gpu]
set -euo pipefail
FX=$1; PY=$2; W=$3; BACKEND=${4:-host}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$HERE/../../reference/data/system/femora_example
source "$HERE/femora_example_cases.sh"
TOL=$FEMORA_EXAMPLE_TOL_HOST
[ "$BACKEND" = gpu ] && TOL=$FEMORA_EXAMPLE_TOL_GPU
rm -rf "$W"
# shellcheck disable=SC2086
"$PY" "$HERE/make_femora_example.py" "$W" $FEMORA_EXAMPLE_ARGS > /dev/null
if [ "$(sha256sum "$W/model.tcl" | cut -d' ' -f1)" != "$(cat "$DATA/model.tcl.sha256")" ]; then
  echo "femora_example: the stored OpenSees reference is out of date (model.tcl changed); rerun"
  echo "  tests/system/femora_example/femora_example_reference.sh <OpenSees> <python>"
  exit 1
fi
(cd "$W" && OMP_NUM_THREADS=2 FEMORAX_TRACE_EVENTS=1 "$FX" run model.femorax.h5 > femorax.log 2>&1)
status=0
grep "^femorax-event|RETRY|" "$W/femorax.log" | cut -d"|" -f3-6 > "$W/femorax_events.txt" || true
cut -d'|' -f1-4 "$DATA/opensees_events.txt" > "$W/opensees_events.txt"
echo "== events: OpenSees $(wc -l < "$W/opensees_events.txt"), FemoraX $(wc -l < "$W/femorax_events.txt")"
if ! diff "$W/opensees_events.txt" "$W/femorax_events.txt"; then echo "  FAIL retry events differ"; status=1; fi
echo "== fields (tolerance $TOL)"
"$PY" "$HERE/../soil_column/compare_vtkhdf.py" "$DATA/opensees.vtkhdf" "$W/femorax.vtkhdf" --tol "$TOL" \
  --only displacement,stress3D6,strain3D6 || status=1
echo "== surface (tolerance $TOL)"
"$PY" "$HERE/compare_surface.py" "$DATA/opensees_surface_accel.txt" "$W/femorax_surface_accel.txt" \
  "$W/sweep_window.acc" "$W/sweep_window.time" 9.81 --tol "$TOL" || status=1
exit $status
