#!/usr/bin/env python3
"""Write one explicit soil-column model for both programs: model.femorax.h5 and model.tcl.

    .venv/bin/python tests/system/soil_column/make_models.py <out_dir> [nx ny nz] [duration] [dt]
        [--integrator ExplicitDifference|CentralDifference] [--mass-damping-only]
        [--vtkhdf DT] [--no-node-recorders] [--element stdBrick|SSPbrick]

CentralDifference keeps only the mass-proportional damping (alphaM), the case FemoraX
solves without a linear solver; --mass-damping-only does the same for ExplicitDifference.
--vtkhdf DT adds a whole-model VTKHDF recorder (disp vel accel every DT; DT = 0: every
step) writing opensees.vtkhdf and femorax.vtkhdf; --no-node-recorders drops the text ones.

The column follows Femora's layered soil column (examples/site_response): stdBrick +
ElasticIsotropic with lumped mass (or SSPbrick, which always lumps) and body force, two materials (lower and upper half),
the base fixed, the laminar boundary as Femora writes it (at every elevation above the base
the side nodes tied to the first one in DOFs 1 2 3), Rayleigh damping per region (all
elements, then the top layer with other coefficients, the later region winning), and a
UniformExcitation in x from a Path series with explicit times (Femora's -fileTime form).
ExplicitDifference, Linear, Transformation; the step loop is Femora's
`while {[getTime] < %f} { analyze 1 dt }` (getTime prints with %f, so the loop compares
times rounded to six decimals; FemoraX does the same).

Recorders (both programs, every step, 17 significant digits): displacement, velocity and
acceleration of three vertical node lines (corner, middle of an edge, centre), all DOFs.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "schema" / "v0"))
import writer as w  # noqa: E402

GRAVITY = 9.81
MATS = [  # (E kPa, nu, rho Mg/m^3), lower and upper half
    (377000.0, 0.3, 2.0285),
    (109200.0, 0.3, 2.0183),
]
ELEMENT = "stdBrick"  # or "SSPbrick" (--element)


def tcl_element(e, conn, mi, lumped: bool) -> str:
    b3 = -GRAVITY * MATS[mi][2]
    if ELEMENT == "SSPbrick":
        return f"element SSPbrick {e} {' '.join(map(str, conn))} {mi + 1} 0.0 0.0 {b3!r}"
    return f"element stdBrick {e} {' '.join(map(str, conn))} {mi + 1} 0.0 0.0 {b3!r}" + (" -lumped" if lumped else "")


def femorax_element_attrs(mi, lumped: bool) -> dict:
    a = {"type": ELEMENT, "material": w.i32(mi + 1), "body_force": w.f64([0.0, 0.0, -GRAVITY * MATS[mi][2]])}
    if ELEMENT == "stdBrick":
        a["lumped"] = w.u8(lumped)
    return a


def take_element_option(argv: list) -> None:
    """Removes --element X from argv and sets ELEMENT."""
    global ELEMENT
    if "--element" in argv:
        k = argv.index("--element")
        ELEMENT = argv[k + 1]
        del argv[k:k + 2]
        if ELEMENT not in ("stdBrick", "SSPbrick"):
            raise SystemExit(f"unknown element {ELEMENT}")


RAYLEIGH = [  # (alphaM, betaK, betaKinit, betaKcomm): all elements, then the top layer
    (0.6, 5.8e-4, 0.0, 0.0),
    (0.9, 3.0e-4, 1.0e-4, 0.0),
]


def build(nx=10, ny=10, nz=40, duration=1.0, dt=None, dx=1.0, dz=0.5):
    NX, NY, NZ = nx + 1, ny + 1, nz + 1
    tag = lambda i, j, k: 1 + i + NX * (j + NY * k)  # noqa: E731
    nodes = [(tag(i, j, k), i * dx, j * dx, -nz * dz + k * dz)
             for k in range(NZ) for j in range(NY) for i in range(NX)]
    elements = []  # (tag, conn, mat index)
    et = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                conn = [tag(i, j, k), tag(i + 1, j, k), tag(i + 1, j + 1, k), tag(i, j + 1, k),
                        tag(i, j, k + 1), tag(i + 1, j, k + 1), tag(i + 1, j + 1, k + 1), tag(i, j + 1, k + 1)]
                elements.append((et, conn, 0 if k < nz // 2 else 1))
                et += 1
    fixes = [tag(i, j, 0) for j in range(NY) for i in range(NX)]
    ties = []  # (retained, constrained)
    for k in range(1, NZ):
        side = [tag(i, j, k) for j in range(NY) for i in range(NX) if i in (0, nx) or j in (0, ny)]
        ties += [(side[0], s) for s in side[1:]]
    top_layer = [e for (e, _c, _m) in elements if e > len(elements) - nx * ny]
    regions = [[e for (e, _c, _m) in elements], top_layer]

    # Base motion: a tapered sine sweep with explicit times.
    t = np.arange(0.0, duration + 1e-12, 0.005)
    f0, f1 = 1.0, 12.0
    acc = 0.1 * np.sin(2 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t * t))
    acc *= np.minimum(1.0, np.minimum(t, duration - t) / 0.1)

    lines = [[tag(0, 0, k) for k in range(NZ)], [tag(nx // 2, 0, k) for k in range(NZ)],
             [tag(nx // 2, ny // 2, k) for k in range(NZ)]]
    rec_nodes = [n for line in lines for n in line]
    return dict(nodes=nodes, elements=elements, fixes=fixes, ties=ties, regions=regions, t=t, acc=acc,
                rec_nodes=rec_nodes, duration=duration, dt=dt)


def write_tcl(m, path: Path, dt: float, integrator: str):
    out = ["wipe", "model BasicBuilder -ndm 3 -ndf 3"]
    for i, (E, nu, rho) in enumerate(MATS):
        out.append(f"nDMaterial ElasticIsotropic {i + 1} {E!r} {nu!r} {rho!r}")
    for (t, x, y, z) in m["nodes"]:
        out.append(f"node {t} {x!r} {y!r} {z!r}")
    for (e, conn, mi) in m["elements"]:
        out.append(tcl_element(e, conn, mi, True))
    for n in m["fixes"]:
        out.append(f"fix {n} 1 1 1")
    for (r, c) in m["ties"]:
        out.append(f"equalDOF {r} {c} 1 2 3")
    for i, (els, ray) in enumerate(zip(m["regions"], m["rayleigh"])):
        out.append(f"region {i + 1} -ele {' '.join(map(str, els))} -rayleigh {' '.join(repr(x) for x in ray)}")
    out.append("timeSeries Path 1 -time {" + " ".join(repr(float(x)) for x in m["t"]) + "} -values {" +
               " ".join(repr(float(x)) for x in m["acc"]) + f"}} -factor {GRAVITY!r}")
    out.append("pattern UniformExcitation 2 1 -accel 1")
    nodes = " ".join(map(str, m["rec_nodes"]))
    if m["node_recorders"]:
        for resp in ("disp", "vel", "accel"):
            out.append(f"recorder Node -file opensees_{resp}.txt -precision 17 -time -node {nodes} -dof 1 2 3 {resp}")
    if m["vtkhdf"] is not None:
        out.append(f"recorder vtkhdf opensees.vtkhdf -dT {m['vtkhdf']!r} disp vel accel")
    out += ["constraints Transformation", "numberer Plain", "system Diagonal", "algorithm Linear",
            f"integrator {integrator}", "analysis Transient",
            f"while {{[getTime] < {m['duration']:f}}} {{ analyze 1 {dt!r} }}", "wipe"]
    path.write_text("\n".join(out) + "\n")


def write_femorax(m, path: Path, dt: float, prefix: str, integrator: str):
    nodes = m["nodes"]
    materials = {str(i + 1): w.node(attrs={"type": "ElasticIsotropic", "E": w.f64(E), "nu": w.f64(nu), "rho": w.f64(rho)})
                 for i, (E, nu, rho) in enumerate(MATS)}
    groups = {}
    for mi in range(len(MATS)):
        els = [(e, c) for (e, c, k) in m["elements"] if k == mi]
        groups[f"{mi:06d}"] = w.node(
            attrs=femorax_element_attrs(mi, True),
            data={"tags": w.i32([e for e, _ in els]), "connectivity": w.i32([c for _, c in els])})
    regions = {f"{i:06d}": w.node(attrs={"alphaM": w.f64(r[0]), "betaK": w.f64(r[1]), "betaKinit": w.f64(r[2]),
                                         "betaKcomm": w.f64(r[3])}, data={"elements": w.i32(els)})
               for i, (els, r) in enumerate(zip(m["regions"], m["rayleigh"]))}
    recorders = {}
    for resp in (("disp", "vel", "accel") if m["node_recorders"] else ()):
        recorders[resp] = w.node(attrs={"type": "Node", "file": f"{prefix}_{resp}.txt", "response": resp,
                                        "time": w.u8(True), "precision": w.i32(17)},
                                 data={"nodes": w.i32(m["rec_nodes"]), "dofs": w.i32([1, 2, 3])})
    if m["vtkhdf"] is not None:
        recorders["vtkhdf"] = w.node(attrs={"type": "VTKHDF", "file": f"{prefix}.vtkhdf", "dT": w.f64(m["vtkhdf"]),
                                            "responses": "disp vel accel"})
    steps = [("pattern", {"tag": w.i32(2)})] + [("recorder", {"name": r}) for r in recorders] + [
        ("analysis", {"name": "shaking", "analysis_type": "Transient", "dt": w.f64(dt),
                      "final_time": w.f64(m["duration"]), "integrator": integrator,
                      "algorithm": "Linear", "constraint_handler": "Transformation", "numberer": "Plain",
                      "system": "Diagonal"})]
    tree = w.node(attrs=w.root_attrs("system_soil_column"), groups={
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
        "time_series": w.node(groups={"1": w.node(attrs={"type": "Path", "factor": w.f64(GRAVITY)},
                                                  data={"time": w.f64(m["t"]), "values": w.f64(m["acc"])})}),
        "patterns": w.node(groups={"2": w.node(attrs={"type": "UniformExcitation", "dof": w.i32(1),
                                                     "accel": w.i32(1)})}),
        "recorders": w.node(groups=recorders),
        "process": w.node(groups={f"{i:06d}": w.node(attrs={"kind": k, **a}) for i, (k, a) in enumerate(steps)}),
    })
    w.write(path, tree)


def main() -> int:
    argv = sys.argv[1:]
    take_element_option(argv)
    integrator = "ExplicitDifference"
    if "--integrator" in argv:
        k = argv.index("--integrator")
        integrator = argv[k + 1]
        del argv[k:k + 2]
    vtkhdf = None
    if "--vtkhdf" in argv:
        k = argv.index("--vtkhdf")
        vtkhdf = float(argv[k + 1])
        del argv[k:k + 2]
    node_recorders = "--no-node-recorders" not in argv
    if not node_recorders:
        argv.remove("--no-node-recorders")
    alpha_only = "--mass-damping-only" in argv
    if alpha_only:
        argv.remove("--mass-damping-only")
    if integrator not in ("ExplicitDifference", "CentralDifference"):
        print(f"unknown integrator {integrator}")
        return 1
    sys.argv = [sys.argv[0]] + argv
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = (int(x) for x in sys.argv[2:5]) if len(sys.argv) >= 5 else (10, 10, 40)
    duration = float(sys.argv[5]) if len(sys.argv) >= 6 else 1.0
    dt = float(sys.argv[6]) if len(sys.argv) >= 7 else 2.5e-4
    m = build(nx, ny, nz, duration)
    m["vtkhdf"], m["node_recorders"] = vtkhdf, node_recorders
    m["rayleigh"] = RAYLEIGH if integrator == "ExplicitDifference" and not alpha_only else [(r[0], 0.0, 0.0, 0.0) for r in RAYLEIGH]
    write_tcl(m, out / "model.tcl", dt, integrator)
    write_femorax(m, out / "model.femorax.h5", dt, "femorax", integrator)
    print(f"{nx}x{ny}x{nz} {ELEMENT} column, {len(m['elements'])} elements, {len(m['nodes'])} nodes, "
          f"{duration} s at dt {dt}, {integrator} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
