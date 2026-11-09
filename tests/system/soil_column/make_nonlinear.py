#!/usr/bin/env python3
"""Write a nonlinear soil column (PressureDependMultiYield or PressureIndependMultiYield) for both programs.

    .venv/bin/python tests/system/soil_column/make_nonlinear.py <out_dir> [nx ny nz] [duration] [dt]
        [--element stdBrick|SSPbrick] [--soil dense|loose|layered|clay|frictional|layered_clay] [--amp A] [--tol T]
        [--shaking implicit|explicit]
        [--vtkhdf DT]  (also a VTKHDF recorder: disp vel accel stress3D6 strain3D6 every DT)
        [--record-dt DT]  (the node recorders sample every DT of time instead of every step)
        [--equilibration modified|newton]  (stage 4's algorithm; newton is not Femora's: its
            ModifiedNewton -factoronce with NormUnbalance 1e-7 fails in OpenSees on columns of
            25 elements, so longer reference columns use Newton there)
        [--equilibration-test TYPE:TOL]  (stage 4's convergence test, e.g. NormDispIncr:1e-9;
            default NormUnbalance:1e-7, Femora's; an absolute force tolerance does not carry
            over to wide columns)
        [--lumped]  (stdBrick with a lumped mass, -lumped: what explicit shaking needs)
        [--until T]  (end the shaking stage at T < duration; the motion stays that of duration)

The mesh of make_models.py (base fixed, the laminar boundary as Femora writes it), with
Femora's soils of examples/site_response/nonlinear_layered_soil_column.py (dense Ottawa,
loose Ottawa with the liquefaction parameters; layered: dense below, loose above), or with
PressureIndependMultiYield (clay: Femora's basin clay; frictional; layered_clay: frictional
below, clay above),
Femora's frequency-Rayleigh damping (2.76 and 13.84 Hz, 3%), stdBrick with consistent mass
(or SSPbrick), and the example's staging:

  1. updateMaterialStage -stage 0 for every soil;
  2. gravity: Newmark 0.6 / 0.3025, dt 1, 30 steps, Newton, NormUnbalance 1e-7 (30);
  3. updateMaterialStage -stage 1;
  4. re-equilibration: the same integrator, dt 0.01, 100 steps, ModifiedNewton -factoronce
     (or Newton with --equilibration newton);
  5. the UniformExcitation pattern (make_models.py's sine sweep times A), the recorders,
     setTime 0;
  6. shaking: Newmark 0.5 / 0.25 at dt, Newton, NormDispIncr T (default 1e-10; 25 iterations), while
     getTime < duration.

Recorders: displacement, velocity and acceleration of the three vertical node lines of
make_models.py, all DOFs, every shaking step, 17 significant digits.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_implicit as mi  # noqa: E402
import make_models as mm  # noqa: E402

w = mm.w
LUMPED = False  # --lumped: stdBrick with a lumped mass

# Femora's example (kN, m, s): name -> parameters in the order of the Tcl command.
SOILS = {
    "dense": dict(G=145.0e3, gamma=19.9, phi=40.0, pt=27.0, contrac=0.03, dil1=0.8, dil2=5.0,
                  liq1=0.0, liq2=0.0, liq3=0.0, e=0.45),
    "loose": dict(G=75.0e3, gamma=19.1, phi=29.0, pt=29.0, contrac=0.21, dil1=0.0, dil2=0.0,
                  liq1=10.0, liq2=0.02, liq3=1.0, e=0.85),
}


# PressureIndependMultiYield: Femora's basin clay (examples/domain_reduction_method/basin.py:
# rho 1.8, Vs 150, Vp 300, cohesion 8, friction 0) and a frictional material (friction 30,
# pressure-dependent moduli).
CLAYS = {
    "clay": dict(nd=3, rho=1.8, refShearModul=1.8 * 150.0 ** 2, refBulkModul=1.8 * (300.0 ** 2 - 4.0 * 150.0 ** 2 / 3.0),
                 cohesi=8.0, peakShearStra=0.1, frictionAng=0.0, refPress=100.0, pressDependCoe=0.0, noYieldSurf=20),
    "frictional": dict(nd=3, rho=2.0, refShearModul=60.0e3, refBulkModul=150.0e3, cohesi=5.0, peakShearStra=0.1,
                       frictionAng=30.0, refPress=80.0, pressDependCoe=0.5, noYieldSurf=20),
}
PIMY_ORDER = ["nd", "rho", "refShearModul", "refBulkModul", "cohesi", "peakShearStra", "frictionAng", "refPress",
              "pressDependCoe", "noYieldSurf"]


def material_type(name: str) -> str:
    return "PressureIndependMultiYield" if name in CLAYS else "PressureDependMultiYield"


def material_order(name: str) -> list:
    return PIMY_ORDER if name in CLAYS else ORDER


def soil_params(name: str) -> dict:
    if name in CLAYS:
        return CLAYS[name]
    s = SOILS[name]
    G, nu = s["G"], 0.3
    return dict(nd=3, rho=s["gamma"] / mm.GRAVITY, refShearModul=G,
                refBulkModul=2.0 * G * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu)), frictionAng=s["phi"],
                peakShearStra=0.1, refPress=80.0, pressDependCoe=0.5, PTAng=s["pt"], contrac=s["contrac"],
                dilat1=s["dil1"], dilat2=s["dil2"], liquefac1=s["liq1"], liquefac2=s["liq2"],
                liquefac3=s["liq3"], noYieldSurf=20, e=s["e"], cs1=0.9, cs2=0.02, cs3=0.7, pa=101.0, c=0.3)


ORDER = ["nd", "rho", "refShearModul", "refBulkModul", "frictionAng", "peakShearStra", "refPress",
         "pressDependCoe", "PTAng", "contrac", "dilat1", "dilat2", "liquefac1", "liquefac2", "liquefac3",
         "noYieldSurf", "e", "cs1", "cs2", "cs3", "pa", "c"]


def rayleigh():  # Femora's frequency_rayleigh(f1=2.76, f2=13.84, damping_factor=0.03)
    w1, w2, xi = 2 * math.pi * 2.76, 2 * math.pi * 13.84, 0.03
    return (2 * xi * w1 * w2 / (w1 + w2), 2 * xi / (w1 + w2), 0.0, 0.0)


def stages(duration: float, dt: float, tol: float = 1e-10, explicit: bool = False, equilibration: str = "modified",
           equilibration_test: str = ""):
    base = {"constraint_handler": "Transformation", "numberer": "RCM", "system": "BandGeneral",
            "analysis_type": "Transient", "integrator": "Newmark"}
    g = {**base, "name": "elastic_gravity", "algorithm": "Newton", "test_type": "NormUnbalance",
         "test_tol": 1e-7, "test_max_iter": 30, "gamma": 0.6, "beta": 0.3025, "dt": 1.0, "num_steps": 30}
    q = {**base, "name": "plastic_equilibration", "algorithm": "ModifiedNewton", "modified_newton_factor_once": 1,
         "test_type": "NormUnbalance", "test_tol": 1e-7, "test_max_iter": 30, "gamma": 0.6, "beta": 0.3025,
         "dt": 0.01, "num_steps": 100}
    if equilibration == "newton":
        q = {k: v for k, v in q.items() if k != "modified_newton_factor_once"}
        q["algorithm"] = "Newton"
    elif equilibration != "modified":
        raise SystemExit(f"--equilibration {equilibration}: modified or newton")
    if equilibration_test:
        kind, value = equilibration_test.split(":")
        q["test_type"], q["test_tol"] = kind, float(value)
    s = {**base, "name": "shaking", "algorithm": "Newton", "test_type": "NormDispIncr", "test_tol": tol,
         "test_max_iter": 25, "gamma": 0.5, "beta": 0.25, "dt": dt, "final_time": duration}
    if explicit:  # ExplicitDifference shaking (lumped mass: SSPbrick)
        s = {**base, "name": "shaking", "algorithm": "Linear", "integrator": "ExplicitDifference", "dt": dt,
             "final_time": duration}
    return g, q, s


def tcl_stage(a: dict) -> list[str]:
    alg = "ModifiedNewton -factoronce" if a["algorithm"] == "ModifiedNewton" else a["algorithm"]
    if a["integrator"] == "ExplicitDifference":
        return ["wipeAnalysis", "constraints Transformation", "numberer RCM", "system Diagonal", "algorithm Linear",
                "integrator ExplicitDifference", "analysis Transient",
                f"while {{[getTime] < {a['final_time']:f}}} {{ if {{[analyze 1 {a['dt']!r}] != 0}} {{ exit 2 }} }}"]
    # A failed step stops the script (the bare while loop would repeat it forever).
    loop = (f"while {{[getTime] < {a['final_time']:f}}} {{ if {{[analyze 1 {a['dt']!r}] != 0}} {{ exit 2 }} }}"
            if "final_time" in a else f"if {{[analyze {a['num_steps']} {a['dt']!r}] != 0}} {{ exit 2 }}")
    return ["wipeAnalysis", "constraints Transformation", "numberer RCM", "system BandGeneral",
            f"test {a['test_type']} {a['test_tol']!r} {a['test_max_iter']}", f"algorithm {alg}",
            f"integrator Newmark {a['gamma']!r} {a['beta']!r}", "analysis Transient", loop]


def write_tcl(m, soils, path: Path, g, q, s, amp, vtkhdf=None, record_dt=None):
    out = ["wipe", "model BasicBuilder -ndm 3 -ndf 3"]
    for i, name in enumerate(soils):
        p = soil_params(name)
        out.append(f"nDMaterial {material_type(name)} {i + 1} " + " ".join(repr(p[k]) for k in material_order(name)))
    for (t, x, y, z) in m["nodes"]:
        out.append(f"node {t} {x!r} {y!r} {z!r}")
    for (e, conn, mi_) in m["elements"]:
        b3 = -mm.GRAVITY * soil_params(soils[mi_])["rho"]
        kind = mm.ELEMENT
        lumped = " -lumped" if LUMPED and kind == "stdBrick" else ""
        out.append(f"element {kind} {e} {' '.join(map(str, conn))} {mi_ + 1} 0.0 0.0 {b3!r}{lumped}")
    for n in m["fixes"]:
        out.append(f"fix {n} 1 1 1")
    for (r, c) in m["ties"]:
        out.append(f"equalDOF {r} {c} 1 2 3")
    ray = rayleigh()
    out.append(f"region 1 -ele {' '.join(str(e) for (e, _c, _m) in m['elements'])} -rayleigh "
               f"{' '.join(repr(x) for x in ray)}")
    out.append("timeSeries Path 1 -time {" + " ".join(repr(float(x)) for x in m["t"]) + "} -values {" +
               " ".join(repr(float(x)) for x in m["acc"]) + f"}} -factor {mm.GRAVITY * amp!r}")
    stage = lambda k: [f"updateMaterialStage -material {i + 1} -stage {k}" for i in range(len(soils))]  # noqa
    out += stage(0) + tcl_stage(g) + stage(1) + tcl_stage(q)
    out += ["pattern UniformExcitation 2 1 -accel 1"]
    nodes = " ".join(map(str, m["rec_nodes"]))
    for resp in ("disp", "vel", "accel"):
        every = f" -dT {record_dt!r}" if record_dt is not None else ""
        out.append(f"recorder Node -file opensees_{resp}.txt -precision 17 -time{every} -node {nodes} -dof 1 2 3 {resp}")
    if vtkhdf is not None:
        out.append(f"recorder vtkhdf opensees.vtkhdf -dT {vtkhdf!r} disp vel accel stress3D6 strain3D6")
    out += ["setTime 0.0"] + tcl_stage(s) + ["wipe"]
    path.write_text("\n".join(out) + "\n")


def write_femorax(m, soils, path: Path, g, q, s, amp, vtkhdf=None, record_dt=None):
    nodes = m["nodes"]
    materials = {}
    for i, name in enumerate(soils):
        p = soil_params(name)
        attrs = {"type": material_type(name)}
        for k in material_order(name):
            attrs[k] = w.i32(p[k]) if k in ("nd", "noYieldSurf") else w.f64(p[k])
        materials[str(i + 1)] = w.node(attrs=attrs)
    groups = {}
    for i, name in enumerate(soils):
        els = [(e, c) for (e, c, k) in m["elements"] if k == i]
        if not els:
            continue
        a = {"type": mm.ELEMENT, "material": w.i32(i + 1),
             "body_force": w.f64([0.0, 0.0, -mm.GRAVITY * soil_params(name)["rho"]])}
        if mm.ELEMENT == "stdBrick":
            a["lumped"] = w.u8(LUMPED)
        groups[f"{i:06d}"] = w.node(attrs=a, data={"tags": w.i32([e for e, _ in els]),
                                                   "connectivity": w.i32([c for _, c in els])})
    ray = rayleigh()
    regions = {"000000": w.node(attrs={"alphaM": w.f64(ray[0]), "betaK": w.f64(ray[1]), "betaKinit": w.f64(0.0),
                                       "betaKcomm": w.f64(0.0)},
                                data={"elements": w.i32([e for (e, _c, _m) in m["elements"]])})}
    every = {"dT": w.f64(record_dt)} if record_dt is not None else {}
    recorders = {resp: w.node(attrs={"type": "Node", "file": f"femorax_{resp}.txt", "response": resp,
                                     "time": w.u8(True), "precision": w.i32(17), **every},
                              data={"nodes": w.i32(m["rec_nodes"]), "dofs": w.i32([1, 2, 3])})
                 for resp in ("disp", "vel", "accel")}
    if vtkhdf is not None:
        recorders["vtkhdf"] = w.node(attrs={"type": "VTKHDF", "file": "femorax.vtkhdf", "dT": w.f64(vtkhdf),
                                            "responses": "disp vel accel stress3D6 strain3D6"})

    def attrs(a: dict) -> dict:
        typed = {}
        for k, v in a.items():
            if isinstance(v, str):
                typed[k] = v
            elif k == "modified_newton_factor_once":
                typed[k] = w.u8(bool(v))
            elif isinstance(v, int):
                typed[k] = w.i32(v)
            else:
                typed[k] = w.f64(v)
        return typed

    mats = w.i32(list(range(1, len(soils) + 1)))
    steps = [("update_material_stage", {"stage": w.i32(0)}, {"materials": mats}), ("analysis", attrs(g), {}),
             ("update_material_stage", {"stage": w.i32(1)}, {"materials": mats}), ("analysis", attrs(q), {}),
             ("pattern", {"tag": w.i32(2)}, {})]
    steps += [("recorder", {"name": r}, {}) for r in recorders]
    steps += [("set_time", {"time": w.f64(0.0)}, {}), ("analysis", attrs(s), {})]
    tree = w.node(attrs=w.root_attrs("system_soil_column_nonlinear"), groups={
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
        "time_series": w.node(groups={"1": w.node(attrs={"type": "Path", "factor": w.f64(mm.GRAVITY * amp)},
                                                  data={"time": w.f64(m["t"]), "values": w.f64(m["acc"])})}),
        "patterns": w.node(groups={"2": w.node(attrs={"type": "UniformExcitation", "dof": w.i32(1),
                                                     "accel": w.i32(1)})}),
        "recorders": w.node(groups=recorders),
        "process": w.node(groups={f"{i:06d}": w.node(attrs={"kind": k, **a}, data=d)
                                  for i, (k, a, d) in enumerate(steps)}),
    })
    w.write(path, tree)


def main() -> int:
    argv = sys.argv[1:]
    mm.take_element_option(argv)
    global LUMPED
    if "--lumped" in argv:
        argv.remove("--lumped")
        LUMPED = True
    opts = {"--equilibration-test": "", "--soil": "layered", "--amp": "3.0", "--tol": "1e-10", "--shaking": "implicit",
            "--vtkhdf": "", "--until": "",
            "--record-dt": "", "--equilibration": "modified"}
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
    nx, ny, nz = (int(x) for x in argv[1:4]) if len(argv) >= 4 else (1, 1, 1)
    duration = float(argv[4]) if len(argv) >= 5 else 0.5
    dt = float(argv[5]) if len(argv) >= 6 else 5e-3
    amp = float(opts["--amp"])
    m = mm.build(nx, ny, nz, max(duration, 1.0))
    soil = opts["--soil"]
    soils = {"layered": ["dense", "loose"], "layered_clay": ["frictional", "clay"]}.get(soil, [soil])
    if len(soils) == 1:
        m["elements"] = [(e, c, 0) for (e, c, _k) in m["elements"]]
    g, q, s = stages(float(opts["--until"]) if opts["--until"] else duration, dt, float(opts["--tol"]), opts["--shaking"] == "explicit", opts["--equilibration"],
                     opts["--equilibration-test"])
    vt = float(opts["--vtkhdf"]) if opts["--vtkhdf"] else None
    rdt = float(opts["--record-dt"]) if opts["--record-dt"] else None
    write_tcl(m, soils, out / "model.tcl", g, q, s, amp, vt, rdt)
    write_femorax(m, soils, out / "model.femorax.h5", g, q, s, amp, vt, rdt)
    print(f"{nx}x{ny}x{nz} {mm.ELEMENT} column, {material_type(soils[0])} ({soil}), shaking {duration} s at dt {dt}, "
          f"motion x{amp} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
