#!/usr/bin/env bash
# Run the issue examples with an OpenSees executable and report which issues it still has.
#   tests/reference/opensees_issues/run.sh [path/to/OpenSees]
# Exit 0 when the results are as recorded in README.md (issues 1-4 reproduce, the control is
# correct). If an issue no longer reproduces, OpenSees has fixed it: FemoraX can then relax
# the matching refusal in src/compile/dof_numbering.cpp (1-4) or revisit the deviation in
# docs/deviations.md (5).
set -uo pipefail
cd "$(dirname "$0")"
OS="${1:-../../../build/reference/opensees/OpenSees}"
status=0
for f in 1_*.tcl 2_*.tcl 3_*.tcl 4_*.tcl 5_*.tcl; do
  if "$OS" "$f" 2>&1 | grep -q "^WRONG"; then echo "reproduces: $f"
  else echo "FIXED in this OpenSees: $f"; status=1; fi
done
if "$OS" 0_control_correct_patterns.tcl 2>&1 | grep -q "^correct"; then echo "control correct"
else echo "control FAILED: 0_control_correct_patterns.tcl"; status=1; fi
exit $status
