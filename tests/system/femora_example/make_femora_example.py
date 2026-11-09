#!/usr/bin/env python3
"""Femora's nonlinear layered soil column for both programs, on a window of its motion.

    .venv/bin/python tests/system/femora_example/make_femora_example.py <out_dir> [--start S] [--duration D]

1. Runs Femora's own example, examples/site_response/nonlinear_layered_soil_column.py
   (25 stdBricks of PressureDependMultiYield, 18 m in five layers, Femora's staging and
   retry and substep recovery), unchanged, in <out_dir>; it exports its Tcl model.
2. Trims its motion, FrequencySweep (zero before 1.9 s), to the window [S, S + D]
   (default 1.9 s and 2 s: the first two seconds of shaking) and writes it to
   sweep_window.acc / sweep_window.time, with the window's start at time 0.
3. Writes model.tcl: Femora's Tcl with the trimmed motion, the shaking stage ending at D,
   the VTKHDF recorder writing opensees.vtkhdf, one added node recorder (the x acceleration
   of the top corner node at every step, opensees_surface_accel.txt), and without the
   `file mkdir` of Femora's absolute results folder. Nothing else changes.
4. Converts model.tcl to model.femorax.h5 (tcl_to_femorax.py), recording femorax.vtkhdf.

Femora is a test-only dependency: the .venv's femora package locates the example.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tcl_to_femorax as t2f  # noqa: E402


def femora_root() -> Path:
    import femora

    return Path(femora.__file__).resolve().parents[2]


def main() -> int:
    argv = sys.argv[1:]
    opts = {"--start": "1.9", "--duration": "2.0"}
    for k in list(opts):
        if k in argv:
            i = argv.index(k)
            opts[k] = argv[i + 1]
            del argv[i:i + 2]
    if len(argv) != 1:
        print(__doc__)
        return 1
    out = Path(argv[0]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start, duration = float(opts["--start"]), float(opts["--duration"])

    root = femora_root()
    example = root / "examples" / "site_response" / "nonlinear_layered_soil_column.py"
    env = {k: v for k, v in os.environ.items() if k != "FEMORA_OPENSEES"}  # export only
    subprocess.run([sys.executable, str(example)], cwd=out, env=env, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    tcl = out / "example_outputs" / "nonlinear_layered_soil_column" / "nonlinear_layered_soil_column.tcl"
    text = tcl.read_text()

    motions = root / "examples" / "inputs" / "motions"
    acc = np.loadtxt(motions / "FrequencySweep.acc").ravel()
    time = np.loadtxt(motions / "FrequencySweep.time").ravel()
    i0 = int(np.argmin(np.abs(time - start)))
    i1 = int(np.argmin(np.abs(time - (start + duration))))
    np.savetxt(out / "sweep_window.acc", acc[i0:i1 + 1], fmt="%.17g")
    np.savetxt(out / "sweep_window.time", np.round(time[i0:i1 + 1] - time[i0], 9), fmt="%.17g")

    def sub(pattern: str, repl: str, s: str) -> str:
        new, n = re.subn(pattern, repl, s)
        if n == 0:
            raise SystemExit(f"make_femora_example: Femora's Tcl no longer matches {pattern!r}")
        return new

    text = sub(r"-filePath \S+ -fileTime \S+", "-filePath sweep_window.acc -fileTime sweep_window.time", text)
    text = sub(r"recorder vtkhdf \S+ ", "recorder vtkhdf opensees.vtkhdf ", text)
    # The surface acceleration at every step (for the peak and the response spectrum): a
    # node recorder next to Femora's, on the column's top corner node at (0, 0, 0).
    top = min(int(m.group(1)) for m in re.finditer(r"node (\d+) 0\.0 0\.0 0\.0 -ndf 3", text))
    text = sub(r"(recorder vtkhdf opensees.vtkhdf [^\n]*\n)",
               rf"\1recorder Node -file opensees_surface_accel.txt -precision 17 -time -node {top} -dof 1 accel\n",
               text)
    # Femora's results folder (an absolute path): not needed, and it would make the model
    # depend on where it is generated.
    text = sub(r"if \{\$pid == 0\} \{file mkdir [^\n]*\n", "", text)
    # The shaking stage (Femora's final_time loop): end at the window's length.
    final = re.search(r"while \{\[getTime\] < ([0-9.]+)\}", text).group(1)
    final_short = repr(float(final))
    text = sub(re.escape(f"while {{[getTime] < {final}}}"), f"while {{[getTime] < {duration:f}}}", text)
    text = sub(re.escape(f"(({final_short} - [getTime])"), f"(({duration!r} - [getTime])", text)
    text = text.replace(f"|{final_short}|s", f"|{duration!r}|s")
    (out / "model.tcl").write_text(text)
    t2f.write_model(t2f.convert(out / "model.tcl"), out / "model.femorax.h5")
    print(f"Femora's nonlinear layered soil column, motion window {start} s + {duration} s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
