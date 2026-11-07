#!/usr/bin/env python3
"""Krylov iteration study (milestone 1b step 10): CG + Jacobi iterations per solve.

    .venv/bin/python tests/system/soil_column/krylov_study.py <femorax> <work dir> [sizes...] [--no-damping]

For the soil columns n x n x 40 (default n = 10, 20, 40; consistent mass, Rayleigh damping
as in make_models.py), runs FemoraX on:
  - static gravity (LoadControl, one step, Linear): the stiffness-dominated case;
  - implicit dynamic steps (Newmark 0.5 / 0.25, Linear, 10 steps of the base motion) at
    dt = 0.5, 1, 4, 16 x the explicit stable step of the mesh (ExplicitDifference with the
    same damping, as FemoraX reports it),
and prints the average CG iterations per solve (relative tolerance 1e-12, Jacobi). The
point: when do the iteration counts grow enough that a better preconditioner (Chebyshev,
then multigrid) pays off.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_implicit as mi  # noqa: E402
import make_models as mm  # noqa: E402

MULTIPLES = (0.5, 1.0, 4.0, 16.0)


def run(fx: str, model: Path) -> list[tuple[str, float]]:
    out = subprocess.run([fx, "run", str(model)], cwd=model.parent, capture_output=True, text=True)
    if out.returncode != 0:
        sys.stderr.write(out.stdout + out.stderr)
        raise SystemExit(f"femorax failed on {model}")
    stages = []
    for line in out.stdout.splitlines():
        m = re.search(r"stage (\w+): .* ([0-9.]+) (cg|gmres) iterations per step", line)
        if m:
            stages.append((m.group(1), float(m.group(2))))
    return stages


def stable_dt(fx: str, work: Path, n: int) -> float:
    d = work / f"explicit_{n}_{'damped' if mm.RAYLEIGH[0][1] else 'undamped'}"
    subprocess.run([sys.executable, str(HERE / "make_models.py"), str(d), str(n), str(n), "40", "0.001", "1e-6",
                    "--no-node-recorders"], check=True, capture_output=True)
    out = subprocess.run([fx, "run", "model.femorax.h5"], cwd=d, capture_output=True, text=True)
    m = re.search(r"stable step ([0-9.eE+-]+)", out.stdout)
    if not m:
        raise SystemExit(out.stdout + out.stderr)
    return float(m.group(1))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    args = sys.argv[1:]
    damping = "--no-damping" not in args
    args = [a for a in args if a != "--no-damping"]
    if not damping:
        mm.RAYLEIGH = [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0)]
    fx, work = args[0], Path(args[1])
    sizes = [int(x) for x in args[2:]] or [10, 20, 40]
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in sizes:
        dtc = stable_dt(fx, work, n)
        row = {"n": n, "dt_crit": dtc}
        for k, mult in enumerate(MULTIPLES):
            dt = mult * dtc
            d = work / f"col{n}_dt{mult:g}_{'damped' if damping else 'undamped'}"
            m = mm.build(n, n, 40, 1.0)
            m["node_recorders"] = False
            g, s = mi.stages("static", "Linear", 10 * dt, dt)
            g["num_steps"] = 1
            g["load_increment"] = 1.0
            s["final_time"] = 10 * dt * (1 + 1e-9)
            d.mkdir(parents=True, exist_ok=True)
            mi.write_femorax(m, d / "model.femorax.h5", g, s)
            st = dict(run(fx, d / "model.femorax.h5"))
            if k == 0:
                row["static"] = st["gravity"]
            row[mult] = st["shaking"]
        rows.append(row)
        print(f"n = {n} done", file=sys.stderr)
    eqs = {n: 3 * (n + 1) * (n + 1) * 40 for n in sizes}
    print("| column | ~equations | explicit dt_crit (damped) | static gravity | " +
          " | ".join(f"dynamic {m:g} dt_crit" for m in MULTIPLES) + " |")
    print("|---|---|---|---|" + "---|" * len(MULTIPLES))
    for r in rows:
        print(f"| {r['n']}x{r['n']}x40 | {eqs[r['n']]} | {r['dt_crit']:.3g} | {r['static']:.0f} | " +
              " | ".join(f"{r[m]:.1f}" for m in MULTIPLES) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
