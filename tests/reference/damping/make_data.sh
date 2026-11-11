#!/usr/bin/env bash
# OpenSees records of single-mode free vibration with Uniform damping (5 % over 0.1-10 Hz),
# for tests/unit/test_uniform_damping.cpp: a mass of 1 on a zeroLength spring at 0.1, 1 and
# 10 Hz, released from u = 1, 6 cycles at 400 steps per period (sdof_uniform.tcl).
#   tests/reference/damping/make_data.sh <OpenSees>
set -euo pipefail
OS=$1
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/../data/damping
W=$(mktemp -d)
COMMIT=$(git -C "$HERE/../../../../OpenSees" rev-parse HEAD)
for f in 0.1 1 10; do
  (cd "$W" && "$OS" "$HERE/sdof_uniform.tcl" uniform $f 0.05 0.1 10 6 > /dev/null 2>&1)
  {
    echo "# femorax-reference damping 1"
    echo "# opensees_commit $COMMIT"
    echo "# case zeroLength spring, mass 1, frequency $f Hz, u0 = 1; damping Uniform 1 0.05 0.1 10;"
    echo "#      Newmark 0.5 0.25, dt = period / 400, 6 cycles (sdof_uniform.tcl)"
    echo "# columns time disp"
    cat "$W/sdof_uniform_$f.txt"
  } > "$OUT/sdof_uniform_$f.txt"
done
rm -rf "$W"
