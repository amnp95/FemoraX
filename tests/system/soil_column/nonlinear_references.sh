#!/usr/bin/env bash
# Run OpenSees once per nonlinear reference case and store its output in
# tests/reference/data/system/<case>/ (with the SHA-256 of the Tcl model it ran), so the
# normal check never runs OpenSees (CLAUDE.md: nonlinear OpenSees references stay tiny and
# are stored). Rerun only when a case or make_nonlinear.py changes; name cases to run only those.
#   tests/system/soil_column/nonlinear_references.sh <OpenSees> <python> [case ...]
set -euo pipefail
OS=$1; PY=$2; shift 2
ONLY=" $* "
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$HERE/../../reference/data/system
source "$HERE/nonlinear_cases.sh"
W=$(mktemp -d)
for c in "${NONLINEAR_CASES[@]}"; do
  IFS='|' read -r name args _tol _gtol <<< "$c"
  if [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $name "* ]]; then continue; fi
  D=$W/$name
  # shellcheck disable=SC2086
  "$PY" "$HERE/make_nonlinear.py" "$D" $args > /dev/null
  (cd "$D" && "$OS" model.tcl > opensees.log 2>&1) || { echo "$name: OpenSees failed (see $D/opensees.log)"; exit 1; }
  mkdir -p "$DATA/$name"
  cp "$D"/opensees_{disp,vel,accel}.txt "$DATA/$name/"
  if [ -f "$D/opensees.vtkhdf" ]; then cp "$D/opensees.vtkhdf" "$DATA/$name/"; fi
  sha256sum "$D/model.tcl" | cut -d' ' -f1 > "$DATA/$name/model.tcl.sha256"
  echo "$name: stored"
done
rm -rf "$W"
