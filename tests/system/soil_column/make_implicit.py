#!/usr/bin/env python3
"""Write the implicit soil-column model for both programs: model.femorax.h5 and model.tcl.

    .venv/bin/python tests/system/soil_column/make_implicit.py <out_dir> [nx ny nz] [duration] [dt]
        [--gravity dynamic|static] [--algorithm Linear|Newton]

The column of make_models.py (two layers, laminar boundary, fixed base, Rayleigh damping
per region, the sine-sweep base motion) with consistent mass, run as Femora's layered
elastic example does:

  1. gravity: "dynamic" (default) is Femora's implicit-dynamic gravity stage, Newmark
     gamma 0.6, beta 0.3025, dt 1.0, 30 steps; "static" is a Static LoadControl stage
     (increment 0.5, 2 steps);
  2. setTime 0, the UniformExcitation pattern, the recorders;
  3. shaking: Newmark 0.5 / 0.25, dt (default 1e-3), while getTime < duration (default
     0.5 s).

Both stages use the Linear algorithm by default (OpenSees: `algorithm Linear -factorOnce`,
the same mathematics, a direct solver factorized once per stage), or Newton with
NormUnbalance 1e-6 (OpenSees: ProfileSPD factorized at every iteration).
Recorders: displacement, velocity and acceleration of three vertical node lines, all
DOFs, every step of the shaking stage, 17 significant digits.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_models as mm  # noqa: E402

w = mm.w


def stages(gravity: str, algorithm: str, duration: float, dt: float):
    test = {"test_type": "NormUnbalance", "test_tol": 1e-6, "test_max_iter": 25}
    g = {"name": "gravity", "constraint_handler": "Transformation", "numberer": "RCM",
         "system": "ProfileSPD", "algorithm": algorithm, **test}
    if gravity == "dynamic":
        g.update(analysis_type="Transient", integrator="Newmark", gamma=0.6, beta=0.3025, dt=1.0, num_steps=30)
    else:
        g.update(analysis_type="Static", integrator="LoadControl", load_increment=0.5, num_steps=2)
    s = {"name": "shaking", "constraint_handler": "Transformation", "numberer": "RCM", "system": "ProfileSPD",
         "algorithm": algorithm, "analysis_type": "Transient", "integrator": "Newmark", "gamma": 0.5,
         "beta": 0.25, "dt": dt, "final_time": duration, **test}
    return g, s


def tcl_stage(a: dict, loop: str) -> list[str]:
    alg = "Linear -factorOnce" if a["algorithm"] == "Linear" else "Newton"
    integ = (f"Newmark {a['gamma']!r} {a['beta']!r}" if a["integrator"] == "Newmark"
             else f"LoadControl {a['load_increment']!r}")
    return ["wipeAnalysis", "constraints Transformation", "numberer RCM", "system ProfileSPD",
            f"test NormUnbalance {a['test_tol']!r} {a['test_max_iter']}", f"algorithm {alg}",
            f"integrator {integ}", f"analysis {a['analysis_type']}", loop]


def write_tcl(m, path: Path, g: dict, s: dict):
    out = ["wipe", "model BasicBuilder -ndm 3 -ndf 3"]
    for i, (E, nu, rho) in enumerate(mm.MATS):
        out.append(f"nDMaterial ElasticIsotropic {i + 1} {E!r} {nu!r} {rho!r}")
    for (t, x, y, z) in m["nodes"]:
        out.append(f"node {t} {x!r} {y!r} {z!r}")
    for (e, conn, mi) in m["elements"]:
        b3 = -mm.GRAVITY * mm.MATS[mi][2]
        out.append(f"element stdBrick {e} {' '.join(map(str, conn))} {mi + 1} 0.0 0.0 {b3!r}")  # consistent mass
    for n in m["fixes"]:
        out.append(f"fix {n} 1 1 1")
    for (r, c) in m["ties"]:
        out.append(f"equalDOF {r} {c} 1 2 3")
    for i, (els, ray) in enumerate(zip(m["regions"], mm.RAYLEIGH)):
        out.append(f"region {i + 1} -ele {' '.join(map(str, els))} -rayleigh {' '.join(repr(x) for x in ray)}")
    out.append("timeSeries Path 1 -time {" + " ".join(repr(float(x)) for x in m["t"]) + "} -values {" +
               " ".join(repr(float(x)) for x in m["acc"]) + f"}} -factor {mm.GRAVITY!r}")
    gloop = (f"analyze {g['num_steps']} {g['dt']!r}" if g["analysis_type"] == "Transient"
             else f"analyze {g['num_steps']}")
    out += tcl_stage(g, gloop)
    out += ["setTime 0.0", "pattern UniformExcitation 2 1 -accel 1"]
    nodes = " ".join(map(str, m["rec_nodes"]))
    for resp in ("disp", "vel", "accel"):
        out.append(f"recorder Node -file opensees_{resp}.txt -precision 17 -time -node {nodes} -dof 1 2 3 {resp}")
    out += tcl_stage(s, f"while {{[getTime] < {s['final_time']:f}}} {{ analyze 1 {s['dt']!r} }}")
    out.append("wipe")
    path.write_text("\n".join(out) + "\n")


def write_femorax(m, path: Path, g: dict, s: dict):
    nodes = m["nodes"]
    materials = {str(i + 1): w.node(attrs={"type": "ElasticIsotropic", "E": w.f64(E), "nu": w.f64(nu), "rho": w.f64(rho)})
                 for i, (E, nu, rho) in enumerate(mm.MATS)}
    groups = {}
    for mi in range(len(mm.MATS)):
        els = [(e, c) for (e, c, k) in m["elements"] if k == mi]
        groups[f"{mi:06d}"] = w.node(
            attrs={"type": "stdBrick", "material": w.i32(mi + 1),
                   "body_force": w.f64([0.0, 0.0, -mm.GRAVITY * mm.MATS[mi][2]]), "lumped": w.u8(False)},
            data={"tags": w.i32([e for e, _ in els]), "connectivity": w.i32([c for _, c in els])})
    regions = {f"{i:06d}": w.node(attrs={"alphaM": w.f64(r[0]), "betaK": w.f64(r[1]), "betaKinit": w.f64(r[2]),
                                         "betaKcomm": w.f64(r[3])}, data={"elements": w.i32(els)})
               for i, (els, r) in enumerate(zip(m["regions"], mm.RAYLEIGH))}
    recorders = {resp: w.node(attrs={"type": "Node", "file": f"femorax_{resp}.txt", "response": resp,
                                     "time": w.u8(True), "precision": w.i32(17)},
                              data={"nodes": w.i32(m["rec_nodes"]), "dofs": w.i32([1, 2, 3])})
                 for resp in ("disp", "vel", "accel")}

    def attrs(a: dict) -> dict:
        typed = {}
        for k, v in a.items():
            if isinstance(v, str):
                typed[k] = v
            elif isinstance(v, int):
                typed[k] = w.i32(v)
            else:
                typed[k] = w.f64(v)
        return typed

    steps = [("analysis", attrs(g)), ("set_time", {"time": w.f64(0.0)}), ("pattern", {"tag": w.i32(2)})]
    steps += [("recorder", {"name": r}) for r in ("disp", "vel", "accel")]
    steps += [("analysis", attrs(s))]
    tree = w.node(attrs=w.root_attrs("system_soil_column_implicit"), groups={
        "nodes": w.node(data={"tags": w.i32([n[0] for n in nodes]), "coords": w.f64([n[1:] for n in nodes]),
                              "ndf": w.i32([3] * len(nodes))}),
        "materials": w.node(groups={"nd": w.node(groups=materials)}),
        "elements": w.node(groups=groups),
        "constraints": w.node(groups={
            "sp": w.node(groups={"fix": w.node(data={"node": w.i32([n for n in m["fixes"] for _ in range(3)]),
                                                     "dof": w.i32([d for _ in m["fixes"] for d in (1, 2, 3)])})}),
            "mp": w.node(groups={"equal_dof": w.node(data={
                "retained": w.i32([r for (r, c) in m["ties"] for _ in range(3)]),
                "constrained": w.i32([c for (r, c) in m["ties"] for _ in range(3)]),
                "dof": w.i32([d for _ in m["ties"] for d in (1, 2, 3)])})})}),
        "damping": w.node(groups={"rayleigh": w.node(groups=regions)}),
        "time_series": w.node(groups={"1": w.node(attrs={"type": "Path", "factor": w.f64(mm.GRAVITY)},
                                                  data={"time": w.f64(m["t"]), "values": w.f64(m["acc"])})}),
        "patterns": w.node(groups={"2": w.node(attrs={"type": "UniformExcitation", "dof": w.i32(1),
                                                     "accel": w.i32(1)})}),
        "recorders": w.node(groups=recorders),
        "process": w.node(groups={f"{i:06d}": w.node(attrs={"kind": k, **a}) for i, (k, a) in enumerate(steps)}),
    })
    w.write(path, tree)


def main() -> int:
    argv = sys.argv[1:]
    opts = {"--gravity": "dynamic", "--algorithm": "Linear"}
    for k in list(opts):
        if k in argv:
            i = argv.index(k)
            opts[k] = argv[i + 1]
            del argv[i:i + 2]
    if not argv:
        print(__doc__)
        return 1
    out = Path(argv[0])
    out.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = (int(x) for x in argv[1:4]) if len(argv) >= 4 else (10, 10, 40)
    duration = float(argv[4]) if len(argv) >= 5 else 0.5
    dt = float(argv[5]) if len(argv) >= 6 else 1e-3
    m = mm.build(nx, ny, nz, max(duration, 1.0))
    g, s = stages(opts["--gravity"], opts["--algorithm"], duration, dt)
    write_tcl(m, out / "model.tcl", g, s)
    write_femorax(m, out / "model.femorax.h5", g, s)
    print(f"{nx}x{ny}x{nz} column (consistent mass), gravity {opts['--gravity']}, {opts['--algorithm']}, "
          f"shaking {duration} s at dt {dt} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
