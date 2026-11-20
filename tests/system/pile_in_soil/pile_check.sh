#!/usr/bin/env bash
# A pile tied into a soil column (make_pile.py) through the converter and FemoraX, against the
# stored OpenSees output (OpenSees 955ceb0, Stampede3 Intel build; regenerate with
#   python make_pile.py <dir> && cd <dir> && OpenSees pile.tcl
# and copy opensees_pile.txt and the Tcl's sha256 to tests/reference/data/system/pile_in_soil/).
#   tests/system/pile_in_soil/pile_check.sh <femorax> <python> <work dir> [tolerance]
set -euo pipefail
FX=$1; PY=$2; W=$3; TOL=${4:-1e-12}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$HERE/../../reference/data/system/pile_in_soil
rm -rf "$W"
"$PY" "$HERE/make_pile.py" "$W"
if [ "$(sha256sum "$W/pile.tcl" | cut -d' ' -f1)" != "$(cat "$DATA/pile.tcl.sha256")" ]; then
  echo "pile_in_soil: the stored OpenSees reference is out of date (pile.tcl changed); regenerate it"
  exit 1
fi
cd "$W"
"$PY" "$HERE/../femora_example/tcl_to_femorax.py" pile.tcl pile.femorax.h5
OMP_NUM_THREADS=2 "$FX" run pile.femorax.h5 --quiet > femorax.log 2>&1 || { cat femorax.log; exit 1; }
"$PY" - "$DATA/opensees_pile.txt" "$TOL" <<'PY'
import sys
import numpy as np
o = np.loadtxt(sys.argv[1]); f = np.loadtxt("femorax_pile.txt"); tol = float(sys.argv[2])
assert o.shape == f.shape, (o.shape, f.shape)
assert np.abs(o[:, 0] - f[:, 0]).max() < 1e-12
d = np.abs(o[:, 1:] - f[:, 1:]).max() / np.abs(o[:, 1:]).max()
print(f"== pile x displacement, FemoraX against OpenSees: {d:.2e} of the peak")
sys.exit(0 if d <= tol else 1)
PY
echo "pile_in_soil: ok"
