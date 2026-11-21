#!/usr/bin/env bash
# A two-storey 3D frame of force-based beams (make_frame.py), one element per member, through the
# converter and FemoraX, against the stored output of OpenSees' own nonlinearBeamColumn run
# (OpenSees 955ceb0, Vista build, scripts/vista_opensees.sh; regenerate with
#   python make_frame.py <dir> && cd <dir> && build/reference-vista/OpenSees.sh force_frame.tcl
# and copy opensees_*.txt and the Tcl's sha256 to tests/reference/data/system/force_beam_frame/).
#   tests/system/force_beam_frame/frame_check.sh <femorax> <python> <work dir> [tolerance]
set -euo pipefail
FX=$1; PY=$2; W=$3; TOL=${4:-1e-12}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$HERE/../../reference/data/system/force_beam_frame
rm -rf "$W"
"$PY" "$HERE/make_frame.py" "$W"
cd "$W"
if ! sha256sum --quiet -c "$DATA/tcl.sha256"; then
  echo "force beam frame: the stored OpenSees reference is out of date (the Tcl changed); regenerate it"
  exit 1
fi
"$PY" "$HERE/../femora_example/tcl_to_femorax.py" force_frame.tcl frame.femorax.h5
OMP_NUM_THREADS=2 "$FX" run frame.femorax.h5 --quiet > frame.log 2>&1 || { cat frame.log; exit 1; }
"$PY" - "$DATA" "$TOL" <<'PY'
import sys
import numpy as np
data, tol = sys.argv[1], float(sys.argv[2])
ok = True
for name in ("floor1", "floor2", "joints"):
    o, f = np.loadtxt(f"{data}/opensees_{name}.txt"), np.loadtxt(f"femorax_{name}.txt")
    assert o.shape == f.shape, (o.shape, f.shape)
    assert np.abs(o[:, 0] - f[:, 0]).max() < 1e-12
    d = np.abs(o[:, 1:] - f[:, 1:]).max() / np.abs(o[:, 1:]).max()
    print(f"== force beam frame {name}, FemoraX against OpenSees: {d:.2e} of the peak")
    ok &= d <= tol
sys.exit(0 if ok else 1)
PY
echo "force beam frame: ok"
