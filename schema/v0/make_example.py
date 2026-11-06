#!/usr/bin/env python3
"""Write example schema v0 models: Femora's layered elastic soil column.

    .venv/bin/python schema/v0/make_example.py out_dir

writes
    soil_column_explicit.femorax.h5   milestone 1a: lumped stdBrick, ExplicitDifference,
                                      no gravity stage (schema/v0/README.md, section 5)
    soil_column_implicit.femorax.h5   milestone 1b: consistent mass, gravity stage, Newmark

The geometry, layers, damping, constraints and process follow
Femora/examples/site_response/layered_elastic_soil_column.py: a 1 m x 1 m x 18 m column of
five mesh parts, laminar boundaries (equalDOF at every elevation above the base, first node
retained) and a fixed base. The base motion is a synthetic frequency sweep, so this script
needs no files from Femora.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import writer as w  # noqa: E402

GRAVITY = 9.81
WIDTH = 1.0
DEPTH = 18.0

LAYERS = [  # from the Femora example, base upward
    {"name": "dense_ottawa_lower", "G": 145.0e6, "gamma": 19.9, "nu": 0.3, "h": 2.6, "size": 1.3},
    {"name": "dense_ottawa_middle", "G": 145.0e6, "gamma": 19.9, "nu": 0.3, "h": 2.4, "size": 1.2},
    {"name": "dense_ottawa_upper", "G": 145.0e6, "gamma": 19.9, "nu": 0.3, "h": 5.0, "size": 1.0},
    {"name": "loose_ottawa", "G": 75.0e6, "gamma": 19.1, "nu": 0.3, "h": 6.0, "size": 0.5},
    {"name": "dense_monterey", "G": 42.0e6, "gamma": 19.8, "nu": 0.3, "h": 2.0, "size": 0.5},
]


def frequency_rayleigh(f1: float, f2: float, xi: float) -> tuple[float, float]:
    """alphaM, betaK giving damping ratio xi at frequencies f1 and f2 (Hz)."""
    w1, w2 = 2.0 * math.pi * f1, 2.0 * math.pi * f2
    return 2.0 * xi * w1 * w2 / (w1 + w2), 2.0 * xi / (w1 + w2)


def column_mesh():
    """Nodes, per-layer element connectivity (OpenSees brick order) and layer data."""
    levels = [-DEPTH]
    layer_ranges = []
    for layer in LAYERS:
        n = round(layer["h"] / layer["size"])
        z0 = levels[-1]
        first = len(levels) - 1
        levels += [z0 + layer["h"] * (k + 1) / n for k in range(n)]
        layer_ranges.append((first, first + n))
    levels[-1] = 0.0
    # Four nodes per level, counter-clockwise seen from above; tag = 1 + 4*level + corner.
    corners = [(0.0, 0.0), (WIDTH, 0.0), (WIDTH, WIDTH), (0.0, WIDTH)]
    coords = np.array([[x, y, z] for z in levels for (x, y) in corners], dtype=np.float64)
    tags = np.arange(1, len(coords) + 1, dtype=np.int32)

    def brick(level: int) -> list[int]:
        b = [1 + 4 * level + c for c in range(4)]
        t = [1 + 4 * (level + 1) + c for c in range(4)]
        return b + t  # bottom face then top face, as stdBrick and SSPbrick expect

    conn = [[brick(k) for k in range(a, b)] for (a, b) in layer_ranges]
    return tags, coords, levels, conn


def sweep(duration: float, dt: float):
    """Unit-amplitude sine sweep from 0.5 to 15 Hz, stored with explicit times."""
    t = np.arange(0.0, duration + 0.5 * dt, dt)
    f0, f1 = 0.5, 15.0
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) / duration * t * t)
    taper = np.minimum(1.0, np.minimum(t, duration - t) / 1.0)
    return t, 0.1 * np.sin(phase) * taper


def soil_column(variant: str) -> dict:
    """Model tree for variant "explicit" (milestone 1a) or "implicit" (1b)."""
    if variant not in ("explicit", "implicit"):
        raise ValueError(variant)
    explicit = variant == "explicit"
    tags, coords, levels, conn = column_mesh()

    materials, elements = {}, {}
    next_ele = 1
    for i, (layer, cells) in enumerate(zip(LAYERS, conn)):
        rho_si = layer["gamma"] * 1000.0 / GRAVITY
        E_si = 2.0 * layer["G"] * (1.0 + layer["nu"])
        E, rho = E_si / 1000.0, rho_si / 1000.0          # kN, m, s, as in the example
        mat_tag = i + 1
        materials[str(mat_tag)] = w.node(attrs={
            "type": "ElasticIsotropic", "E": w.f64(E), "nu": w.f64(layer["nu"]),
            "rho": w.f64(rho), "femora_name": f'{layer["name"]}_material'})
        attrs = {"type": "stdBrick", "material": w.i32(mat_tag),
                 "body_force": w.f64([0.0, 0.0, -GRAVITY * rho]),
                 "femora_name": layer["name"]}
        if explicit:
            attrs["lumped"] = w.u8(True)
        elements[f"{i:06d}"] = w.node(attrs=attrs, data={
            "tags": w.i32(np.arange(next_ele, next_ele + len(cells))),
            "connectivity": w.i32(cells)})
        next_ele += len(cells)
    n_elements = next_ele - 1

    # Fixed base: z = -DEPTH, all three DOFs.
    base = [int(t) for t, c in zip(tags, coords) if abs(c[2] + DEPTH) < 1e-9]
    fix_node = [n for n in base for _ in range(3)]
    fix_dof = [d for _ in base for d in (1, 2, 3)]

    # Laminar boundary as Femora writes it: at each elevation above the base, the first node
    # is retained and the others are tied to it in DOFs 1 2 3.
    retained, constrained, dof = [], [], []
    for k in range(1, len(levels)):
        level_nodes = [1 + 4 * k + c for c in range(4)]
        for slave in level_nodes[1:]:
            for d in (1, 2, 3):
                retained.append(level_nodes[0])
                constrained.append(slave)
                dof.append(d)

    alphaM, betaK = frequency_rayleigh(2.76, 13.84, 0.03)
    t, a = sweep(duration=10.0, dt=0.005)
    surface = [int(x) for x in tags[-4:]]

    root = w.node(attrs=w.root_attrs(f"layered_elastic_soil_column_{variant}"), groups={
        "nodes": w.node(data={"tags": tags, "coords": coords,
                              "ndf": w.i32(np.full(len(tags), 3))}),
        "materials": w.node(groups={"nd": w.node(groups=materials)}),
        "elements": w.node(groups=elements),
        "constraints": w.node(groups={
            "sp": w.node(groups={"fix": w.node(data={"node": w.i32(fix_node),
                                                     "dof": w.i32(fix_dof)})}),
            "mp": w.node(groups={"equal_dof": w.node(data={
                "retained": w.i32(retained), "constrained": w.i32(constrained),
                "dof": w.i32(dof)})}),
        }),
        "damping": w.node(groups={"rayleigh": w.node(groups={"000000": w.node(
            attrs={"alphaM": w.f64(alphaM), "betaK": w.f64(betaK),
                   "betaKinit": w.f64(0.0), "betaKcomm": w.f64(0.0),
                   "femora_tag": w.i32(1), "femora_name": "soil_damping"},
            data={"elements": w.i32(np.arange(1, n_elements + 1))})})}),
        "time_series": w.node(groups={"1": w.node(
            attrs={"type": "Path", "factor": w.f64(GRAVITY)},
            data={"time": w.f64(t), "values": w.f64(a)})}),
        "patterns": w.node(groups={"2": w.node(attrs={
            "type": "UniformExcitation", "dof": w.i32(1), "accel": w.i32(1)})}),
        "recorders": w.node(groups={
            "site": w.node(attrs={"type": "VTKHDF", "file": "results/site_response$pid.vtkhdf",
                                  "dT": w.f64(0.01), "responses": "accel disp vel"}),
            "surface_accel": w.node(
                attrs={"type": "Node", "file": "results/surface_accel.txt",
                       "response": "accel", "time": w.u8(True)},
                data={"nodes": w.i32(surface), "dofs": w.i32([1, 2, 3])}),
        }),
    })

    common = {"constraint_handler": "Transformation", "numberer": "RCM"}
    steps = []
    if not explicit:
        gamma = 0.6
        steps.append(("analysis", "Establish gravity", {
            "name": "gravity", "analysis_type": "Transient", "dt": w.f64(1.0),
            "num_steps": w.i32(30), "integrator": "Newmark", "gamma": w.f64(gamma),
            "beta": w.f64((gamma + 0.5) ** 2 / 4.0), "algorithm": "Linear",
            "system": "BandGeneral", "test_type": "NormUnbalance", "test_tol": w.f64(1e-8),
            "test_max_iter": w.i32(10), **common}))
    steps += [
        ("pattern", "Apply the base excitation", {"tag": w.i32(2)}),
        ("recorder", "Record the column response", {"name": "site"}),
        ("recorder", "Record the surface acceleration", {"name": "surface_accel"}),
        ("set_time", "Reset pseudo-time", {"time": w.f64(0.0)}),
    ]
    if explicit:
        steps.append(("analysis", "Run the site-response analysis", {
            "name": "frequency_sweep", "analysis_type": "Transient", "dt": w.f64(1e-4),
            "final_time": w.f64(10.0), "integrator": "ExplicitDifference",
            "algorithm": "Linear", "system": "Diagonal", **common}))
    else:
        gamma = 0.6
        steps.append(("analysis", "Run the site-response analysis", {
            "name": "frequency_sweep", "analysis_type": "Transient", "dt": w.f64(0.001),
            "final_time": w.f64(10.0), "integrator": "Newmark", "gamma": w.f64(gamma),
            "beta": w.f64((gamma + 0.5) ** 2 / 4.0), "algorithm": "Linear",
            "system": "BandGeneral", "test_type": "NormUnbalance", "test_tol": w.f64(1e-8),
            "test_max_iter": w.i32(10), **common}))

    root["groups"]["process"] = w.node(groups={
        f"{i:06d}": w.node(attrs={"kind": kind, "label": label, **attrs})
        for i, (kind, label, attrs) in enumerate(steps)})
    return root


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for variant in ("explicit", "implicit"):
        path = out / f"soil_column_{variant}.femorax.h5"
        w.write(path, soil_column(variant))
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
