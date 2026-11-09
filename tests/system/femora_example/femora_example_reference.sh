#!/usr/bin/env bash
# Run OpenSees once on Femora's nonlinear soil column window and store its output in
# tests/reference/data/system/femora_example/ (with the SHA-256 of the Tcl model it ran and
# the retry events Femora's Tcl printed), so the normal check never runs OpenSees. Rerun
# only when the case, the generator or Femora's example changes.
#   tests/system/femora_example/femora_example_reference.sh <OpenSees> <python>
set -euo pipefail
OS=$1; PY=$2
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$HERE/../../reference/data/system/femora_example
source "$HERE/femora_example_cases.sh"
W=$(mktemp -d)
# shellcheck disable=SC2086
"$PY" "$HERE/make_femora_example.py" "$W" $FEMORA_EXAMPLE_ARGS > /dev/null
(cd "$W" && "$OS" model.tcl > opensees.log 2>&1) || { echo "OpenSees failed (see $W/opensees.log)"; exit 1; }
mkdir -p "$DATA"
cp "$W/opensees.vtkhdf" "$W/opensees_surface_accel.txt" "$DATA/"
{ grep "FEMORA_PROGRESS|RETRY" "$W/opensees.log" || true; } | cut -d'|' -f3- > "$DATA/opensees_events.txt"
sha256sum "$W/model.tcl" | cut -d' ' -f1 > "$DATA/model.tcl.sha256"
echo "femora_example: stored ($(wc -l < "$DATA/opensees_events.txt") retry events)"
rm -rf "$W"
