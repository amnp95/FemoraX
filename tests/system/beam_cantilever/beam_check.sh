#!/usr/bin/env bash
# The converter's beam commands (schema 1.1) end to end: cantilever.tcl -> tcl_to_femorax.py ->
# the model file (its beam groups checked) -> `femorax check` -> `femorax run` (implicit, then
# explicit with --explicit); the tip's history must be finite and move. The beam physics is
# verified in tests/unit (test_disp_beam_3d, test_beam_kernels).
#   tests/system/beam_cantilever/beam_check.sh <femorax> <python> <work dir>
set -euo pipefail
FX=$1; PY=$2; W=$3
HERE=$(cd "$(dirname "$0")" && pwd)
CONV=$HERE/../femora_example/tcl_to_femorax.py
rm -rf "$W"; mkdir -p "$W"
cp "$HERE/cantilever.tcl" "$HERE/accel.txt" "$HERE/time.txt" "$W/"
cd "$W"
"$PY" "$CONV" cantilever.tcl implicit.femorax.h5
"$PY" "$CONV" cantilever.tcl explicit.femorax.h5 --explicit 1e-4
"$PY" - <<'PY'
import h5py, numpy as np
f = h5py.File("implicit.femorax.h5", "r")
assert tuple(f.attrs["schema_version"]) == (1, 2), f.attrs["schema_version"]
assert set(f["nodes"]["ndf"][...]) == {6}
s = f["sections"]["1"].attrs
assert s["type"] == "Elastic" and abs(s["Iz"] - 0.02) < 1e-15
t = f["transformations"]["1"].attrs
assert t["type"] == "PDelta" and list(t["vecxz"]) == [1.0, 0.0, 0.0] and list(t["joint_offset_j"]) == [0, 0, 0.05]
g = [f["elements"][k] for k in f["elements"] if f["elements"][k].attrs["type"] == "dispBeamColumn"]
assert len(g) == 1 and g[0].attrs["integration_points"] == 5 and g[0].attrs["mass_per_length"] == 1.2
assert g[0]["connectivity"].shape == (6, 2)
m = f["mass"]
assert list(m["node"][...]) == [7, 7, 7] and list(m["dof"][...]) == [1, 2, 3]
print("== the model file has the beam groups")
PY
"$FX" check implicit.femorax.h5
"$FX" check explicit.femorax.h5
for k in implicit explicit; do
  OMP_NUM_THREADS=2 "$FX" run $k.femorax.h5 --quiet > $k.log 2>&1 || { cat $k.log; exit 1; }
  mv femorax_tip.txt $k\_tip.txt
done
"$PY" - <<'PY'
import numpy as np
for k, steps in (("implicit", 200), ("explicit", 200)):
    d = np.loadtxt(f"{k}_tip.txt")
    assert d.shape[0] > 0 and np.all(np.isfinite(d)), k
    ux = d[:, 1]
    print(f"== {k}: {d.shape[0]} records, tip ux peak {np.abs(ux).max():.4e}")
    assert np.abs(ux).max() > 1e-7, k
# the two schemes on the same model: the same motion (explicit at 1e-4 s against Newmark at 1e-3 s)
i = np.loadtxt("implicit_tip.txt"); e = np.loadtxt("explicit_tip.txt")
ti, te = i[:, 0], e[:, 0]
ue = np.interp(ti, te, e[:, 1])
mask = ti <= te[-1]
err = np.abs(ue[mask] - i[mask, 1]).max() / np.abs(i[:, 1]).max()
print(f"== explicit against implicit at the implicit times: {err:.3e} of the peak")
assert err < 0.05, err
PY
echo "beam_cantilever: ok"
