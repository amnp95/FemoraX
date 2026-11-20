#!/usr/bin/env bash
# Rigid links, a rigid diaphragm and embedded nodes (make_constraints.py) through the converter
# and FemoraX, against the stored OpenSees output (OpenSees 955ceb0, Stampede3 Intel build;
# regenerate with
#   python make_constraints.py <dir> && cd <dir> && for f in *.tcl; do OpenSees $f; done
# (each in its own directory) and copy the opensees_*.txt files (the embedded ones with their K
# appended) and the Tcls' sha256 to tests/reference/data/system/constraints/).
#   - the frame: FemoraX's exact constraints against OpenSees' Transformation handler, to rounding;
#   - the embedded pile: FemoraX's exact tie is the limit of OpenSees' penalty: the difference
#     falls as 1 / K (K = 1e8, 1e10, 1e12) and is small at the largest K.
#   tests/system/constraints/constraints_check.sh <femorax> <python> <work dir> [frame tolerance]
set -euo pipefail
FX=$1; PY=$2; W=$3; TOL=${4:-1e-12}
HERE=$(cd "$(dirname "$0")" && pwd)
DATA=$HERE/../../reference/data/system/constraints
rm -rf "$W"
"$PY" "$HERE/make_constraints.py" "$W"
cd "$W"
if ! sha256sum --quiet -c "$DATA/tcl.sha256"; then
  echo "constraints: the stored OpenSees references are out of date (a Tcl changed); regenerate them"
  exit 1
fi
CONV=$HERE/../femora_example/tcl_to_femorax.py
"$PY" "$CONV" frame.tcl frame.femorax.h5
OMP_NUM_THREADS=2 "$FX" run frame.femorax.h5 --quiet > frame.log 2>&1 || { cat frame.log; exit 1; }
# FemoraX ignores -K: one run serves every K.
"$PY" "$CONV" embedded_1.0e12.tcl embedded.femorax.h5 2> convert.log
OMP_NUM_THREADS=2 "$FX" run embedded.femorax.h5 --quiet > embedded.log 2>&1 || { cat embedded.log; exit 1; }
"$PY" - "$DATA" "$TOL" <<'PY'
import sys
import numpy as np
data, tol = sys.argv[1], float(sys.argv[2])
ok = True
def diff(o, f):
    assert o.shape == f.shape, (o.shape, f.shape)
    assert np.abs(o[:, 0] - f[:, 0]).max() < 1e-12
    return np.abs(o[:, 1:] - f[:, 1:]).max() / np.abs(o[:, 1:]).max()
for name in ("master", "mass", "column"):
    d = diff(np.loadtxt(f"{data}/opensees_{name}.txt"), np.loadtxt(f"femorax_{name}.txt"))
    print(f"== frame {name}, FemoraX against OpenSees: {d:.2e} of the peak")
    ok &= d <= tol
for resp in ("ux", "ry"):
    f = np.loadtxt(f"femorax_pile_{resp}.txt")
    d = [diff(np.loadtxt(f"{data}/opensees_pile_{resp}_{K}.txt"), f) for K in ("1.0e8", "1.0e10", "1.0e12")]
    print(f"== embedded pile {resp}, FemoraX against OpenSees with K = 1e8, 1e10, 1e12: "
          + ", ".join(f"{x:.2e}" for x in d))
    ok &= 70 < d[0] / d[1] < 130 and 70 < d[1] / d[2] < 130 and d[2] < 1e-6
sys.exit(0 if ok else 1)
PY
echo "constraints: ok"
