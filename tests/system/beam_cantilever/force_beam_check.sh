#!/usr/bin/env bash
# The converter's force-beam command (schema 1.3) end to end: cantilever.tcl with its
# dispBeamColumn elements rewritten as `element nonlinearBeamColumn ... -mass m -iter 10 1e-12`
# (Femora's force-based beam) -> tcl_to_femorax.py -> the model file (its forceBeamColumn group
# checked) -> `femorax check` -> `femorax run`, implicit and explicit. The member is elastic and
# prismatic, where the force beam and the displacement beam have the same stiffness and mass, so the
# tip history must equal the displacement-beam run of the same Tcl.
#   tests/system/beam_cantilever/force_beam_check.sh <femorax> <python> <work dir> [tolerance]
set -euo pipefail
FX=$1; PY=$2; W=$3; TOL=${4:-1e-9}
HERE=$(cd "$(dirname "$0")" && pwd)
CONV=$HERE/../femora_example/tcl_to_femorax.py
rm -rf "$W"; mkdir -p "$W"
cp "$HERE/cantilever.tcl" "$HERE/accel.txt" "$HERE/time.txt" "$W/"
cd "$W"
sed -E 's/^element dispBeamColumn (.*)$/element nonlinearBeamColumn \1 -iter 10 1e-12/' cantilever.tcl > force.tcl
grep -c "^element nonlinearBeamColumn" force.tcl > /dev/null
for k in disp force; do
  src=cantilever.tcl; [ $k = force ] && src=force.tcl
  "$PY" "$CONV" $src ${k}_implicit.femorax.h5
  "$PY" "$CONV" $src ${k}_explicit.femorax.h5 --explicit 1e-4
done
"$PY" - <<'PY'
import h5py
f = h5py.File("force_implicit.femorax.h5", "r")
assert tuple(f.attrs["schema_version"]) == (1, 3), f.attrs["schema_version"]
g = [f["elements"][k] for k in f["elements"] if f["elements"][k].attrs["type"] == "forceBeamColumn"]
assert len(g) == 1 and g[0].attrs["integration_points"] == 5 and g[0].attrs["mass_per_length"] == 1.2
assert g[0].attrs["max_iterations"] == 10 and g[0].attrs["tolerance"] == 1e-12
assert g[0]["connectivity"].shape == (6, 2)
assert not [k for k in f["elements"] if f["elements"][k].attrs["type"] == "dispBeamColumn"]
print("== the model file has the forceBeamColumn group")
PY
for k in disp force; do
  for s in implicit explicit; do
    "$FX" check ${k}_$s.femorax.h5
    OMP_NUM_THREADS=2 "$FX" run ${k}_$s.femorax.h5 --quiet > ${k}_$s.log 2>&1 || { cat ${k}_$s.log; exit 1; }
    mv femorax_tip.txt ${k}_${s}_tip.txt
  done
done
"$PY" - "$TOL" <<'PY'
import sys
import numpy as np
tol = float(sys.argv[1])
ok = True
for s in ("implicit", "explicit"):
    d = np.loadtxt(f"disp_{s}_tip.txt"); f = np.loadtxt(f"force_{s}_tip.txt")
    assert d.shape == f.shape and d.shape[0] > 0 and np.all(np.isfinite(f)), s
    peak = np.abs(d[:, 1:]).max()
    err = np.abs(d[:, 1:] - f[:, 1:]).max() / peak
    print(f"== {s}: force beam against displacement beam, {err:.2e} of the peak {peak:.4e}")
    ok &= peak > 1e-7 and err <= tol
sys.exit(0 if ok else 1)
PY
echo "force_beam_cantilever: ok"
