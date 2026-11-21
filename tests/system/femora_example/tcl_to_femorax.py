#!/usr/bin/env python3
"""Convert the Tcl that Femora exports for a soil column into a FemoraX model file.

    .venv/bin/python tests/system/femora_example/tcl_to_femorax.py <model.tcl> <model.femorax.h5>
        [--explicit DT] [--recorder-dt DT] [--recorder FILE] [--responses "disp vel ..."]

A test tool, not the Femora exporter (Femora owns that): it reads the subset of Femora's
Tcl export that its nonlinear layered soil column and its DRM basin use, so OpenSees and
FemoraX run the same model, and stops on anything else. Partitioned (OpenSeesMP) exports
are merged into one model: the `if {$pid == k}` guards are skipped and nodes repeated in
several partitions are kept once (their coordinates must agree). Recognised:

  set NAME VALUE                     (only to resolve $NAME in fixX / fixY / fixZ)
  nDMaterial PressureDependMultiYield tag nd rho ... c   (Femora's 22 values)
  nDMaterial PressureIndependMultiYield tag nd rho G K cohesi peakShearStra frictionAng
             refPress pressDependCoe noYieldSurf             (Femora's 10 values, no pairs)
  nDMaterial ElasticIsotropic tag E nu rho
  node tag x y z -ndf 3 (or -ndf 6: beam nodes)
  element stdBrick tag n1 .. n8 mat b1 b2 b3
  section Elastic tag E A Iz Iy G J                                  (schema 1.1)
  geomTransf Linear|PDelta|Corotational tag vx vy vz [-jntOffset dXi dYi dZi dXj dYj dZj]
  element dispBeamColumn tag i j nIP secTag transfTag [-mass m] [-cMass]
  element nonlinearBeamColumn|forceBeamColumn tag i j nIP secTag transfTag [-mass m]
          [-iter maxIters tol]                                       (schema 1.3: Lobatto points)
  mass node m1 .. m_ndf                                              (nodal masses)
  equalDOF retained constrained dofs...
  rigidLink beam|bar master slave                                    (schema 1.2)
  rigidDiaphragm perp master slaves...                               (schema 1.2)
  element ASDEmbeddedNodeElement tag C R1 R2 R3 R4 [-rot] [-K K] [-KP KP]   (schema 1.2: the
      node embedded in the tetrahedron of its four corners, an exact tie; -K and -KP are
      OpenSees' penalty and have no counterpart; -p and the triangle form are refused)
  fixX / fixY / fixZ coordinate dofs... -tol tol
  fix node flags...
  region tag -ele ... -rayleigh alphaM betaK betaKinit betaKcomm    (inside eval "...")
  damping Uniform tag ratio freq1 freq2 [-activateTime Ta] [-deactivateTime Td] [-fact ts]
  region tag -ele ... -damp dampingTag                               (inside eval "...")
  timeSeries Path tag -filePath f -fileTime f -factor x
  pattern UniformExcitation tag dof -accel series
  pattern H5DRM tag "file" factor crdScale distanceTolerance doTransform T00..T22 x0 x1 x2
  recorder vtkhdf file -dT dt responses...
  recorder Node -file f -precision p -time -node tags -dof dofs response   (opensees_* -> femorax_*)
  updateMaterialStage -material tag -stage s
  setTime t
  the analysis blocks of Femora's Analysis.to_tcl (test, algorithm, integrator Newmark or
  LoadControl, its FemoraAnalyzeTransientStep procedure: max retries, sublevels, substeps;
  the loop: a number of steps or a final time)

Options (FemoraX studies; each changes the model and says so):
  --explicit DT     every Transient stage runs ExplicitDifference (algorithm Linear) at step
                    DT, and every stdBrick gets lumped mass (ExplicitDifference needs it);
                    static stages are unchanged
  --recorder-dt DT  the VTKHDF recorder's interval
  --recorder FILE   the VTKHDF file FemoraX writes (default femorax.vtkhdf)
  --responses R     the VTKHDF responses (default: the Tcl's)

Femora's helper procedures, progress output and MPI guards are skipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "schema" / "v0"))
import writer as w  # noqa: E402

PDMY_ORDER = ["nd", "rho", "refShearModul", "refBulkModul", "frictionAng", "peakShearStra", "refPress",
              "pressDependCoe", "PTAng", "contrac", "dilat1", "dilat2", "liquefac1", "liquefac2", "liquefac3",
              "noYieldSurf", "e", "cs1", "cs2", "cs3", "pa", "c"]
PIMY_ORDER = ["nd", "rho", "refShearModul", "refBulkModul", "cohesi", "peakShearStra", "frictionAng", "refPress",
              "pressDependCoe", "noYieldSurf"]


class Unsupported(Exception):
    pass


def logical_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            continue
        out.append(line)
    return out


def parse_analysis(lines: list[str], i: int) -> tuple[dict, int]:
    """An analysis block from 'Starting analysis : name' to its wipeAnalysis."""
    name = re.search(r'Starting analysis : ([^"]+)"', lines[i]).group(1)
    a: dict = {"name": name, "analysis_type": "Transient", "max_retries": 0}
    level_limit = substeps = None
    while not lines[i].startswith("wipeAnalysis"):
        s = lines[i]
        tok = s.split()
        if tok[0] == "constraints":
            a["constraint_handler"] = tok[1]
        elif tok[0] == "numberer":
            a["numberer"] = tok[1]
        elif tok[0] == "system":
            a["system"] = tok[1]
        elif tok[0] == "test":
            a["test_type"], a["test_tol"], a["test_max_iter"] = tok[1], float(tok[2]), int(tok[3])
        elif tok[0] == "algorithm":
            a["algorithm"] = tok[1]
            if a["algorithm"] not in ("Newton", "ModifiedNewton", "Linear"):
                raise Unsupported(f"algorithm {tok[1]}")
            if "-factoronce" in tok:
                a["modified_newton_factor_once"] = 1
            if "-initial" in tok:
                a["modified_newton_initial"] = 1
        elif tok[0] == "integrator":
            if tok[1] == "Newmark" and len(tok) == 4:
                a["integrator"], a["gamma"], a["beta"] = "Newmark", float(tok[2]), float(tok[3])
            elif tok[1] == "LoadControl" and len(tok) == 6:
                a["integrator"], a["load_increment"] = "LoadControl", float(tok[2])
                a["num_iter"], a["min_increment"], a["max_increment"] = int(tok[3]), float(tok[4]), float(tok[5])
            else:
                raise Unsupported(f"integrator {' '.join(tok[1:])}")
        elif tok[0] == "analysis":
            if tok[1] not in ("Transient", "Static"):
                raise Unsupported(f"analysis {tok[1]}")
            a["analysis_type"] = tok[1]
        elif tok[0] == "initialize":
            a["initialize"] = 1
        elif m := re.search(r"\$FemoraRetry < (\d+)\}", s):
            a["max_retries"] = int(m.group(1))
        elif m := re.match(r"if \{\$level >= (\d+)\}", s):
            level_limit = int(m.group(1))
        elif m := re.search(r"\$dt / double\((\d+)\)", s):
            substeps = int(m.group(1))
        elif m := re.match(r"set FemoraAnalysisTotal (\d+)$", s):
            a["num_steps"] = int(m.group(1))
        elif m := re.match(r"while \{\[getTime\] < ([0-9.eE+-]+)\}", s):
            a["final_time"] = float(m.group(1))
        elif m := re.search(r"\[FemoraAnalyzeTransientStep ([0-9.eE+-]+) 0\]", s):
            a["dt"] = float(m.group(1))
        i += 1
    if level_limit:
        a["num_sublevels"], a["num_substeps"] = level_limit, substeps
    for k in ("algorithm", "integrator", "test_type") + (("dt",) if a["analysis_type"] == "Transient" else ("num_steps",)):
        if k not in a:
            raise Unsupported(f"analysis {name}: no {k}")
    return a, i + 1


def convert(tcl: Path) -> dict:
    lines = logical_lines(tcl.read_text())
    var: dict[str, str] = {}
    materials: dict[int, dict] = {}
    nodes: dict[int, tuple[float, float, float]] = {}
    node_ndf: dict[int, int] = {}
    sections: dict[int, dict] = {}
    transfs: dict[int, dict] = {}
    # (tag, [i, j], (nip, sec, transf, mass, cmass, kind, maxIters, tol)); kind: "dispBeamColumn" or
    # "forceBeamColumn" (maxIters and tol: the force beam's -iter, None for the defaults)
    beams: list[tuple[int, list[int], tuple]] = []
    masses: list[tuple[int, int, float]] = []       # (node, dof 1-based, value)
    elements: list[tuple[int, list[int], int, tuple[float, float, float]]] = []
    ties: list[tuple[int, int, list[int]]] = []
    rigid_links: list[tuple[int, int, int]] = []        # (master, slave, 1 beam / 2 bar)
    diaphragms: list[tuple[int, int, list[int]]] = []   # (perp, master, slaves)
    embedded: list[tuple[int, list[int], int]] = []     # (node, four corners, rot)
    fixes: list[tuple[int, float, list[int], float]] = []  # (axis, coordinate, dofs, tol)
    node_fixes: list[tuple[int, list[int]]] = []           # fix node flags...
    regions: list[tuple[list[int], list[float]]] = []
    uniform_defs: dict[int, dict] = {}
    uniform_regions: list[tuple[list[int], int]] = []
    drm = None
    series = None
    pattern = None
    recorder = None
    process: list[tuple[str, dict, dict]] = []
    node_recorders: dict[str, tuple] = {}
    skip_depth = 0
    i = 0
    while i < len(lines):
        s = lines[i].split(";")[0].strip()
        tok = s.split()
        # Femora's helper procedures (getFemoraMax): skipped whole.
        if s.startswith("proc getFemoraMax"):
            skip_depth = 1
            i += 1
            while skip_depth:
                skip_depth += lines[i].count("{") - lines[i].count("}")
                i += 1
            continue
        if "Starting analysis :" in s:
            a, i = parse_analysis(lines, i)
            process.append(("analysis", a, {}))
            continue
        if not tok:
            i += 1
            continue
        head = tok[0]
        if head == "set" and len(tok) == 3 and not tok[2].startswith("["):
            var[tok[1]] = tok[2]
        elif head == "nDMaterial":
            if tok[1] == "PressureDependMultiYield" and len(tok) == 3 + len(PDMY_ORDER):
                materials[int(tok[2])] = dict(zip(PDMY_ORDER, (float(x) for x in tok[3:])))
            elif tok[1] == "PressureIndependMultiYield" and len(tok) == 3 + len(PIMY_ORDER):
                materials[int(tok[2])] = {"type": "PressureIndependMultiYield",
                                          **dict(zip(PIMY_ORDER, (float(x) for x in tok[3:])))}
            elif tok[1] == "ElasticIsotropic" and len(tok) == 6:
                materials[int(tok[2])] = {"type": "ElasticIsotropic", "E": float(tok[3]), "nu": float(tok[4]),
                                          "rho": float(tok[5])}
            else:
                raise Unsupported(s)
        elif head == "node":
            if tok[5:] not in (["-ndf", "3"], ["-ndf", "6"]):
                raise Unsupported(s)
            x = (float(tok[2]), float(tok[3]), float(tok[4]))
            if nodes.setdefault(int(tok[1]), x) != x:
                raise Unsupported(f"node {tok[1]} appears with different coordinates")
            node_ndf[int(tok[1])] = int(tok[6])
        elif head == "section":
            if tok[1] != "Elastic" or len(tok) != 9:
                raise Unsupported(s)
            sections[int(tok[2])] = dict(zip(["E", "A", "Iz", "Iy", "G", "J"], (float(x) for x in tok[3:9])))
        elif head == "geomTransf":
            if tok[1] not in ("Linear", "PDelta", "Corotational") or len(tok) not in (6, 13) or \
                    (len(tok) == 13 and tok[6] != "-jntOffset"):
                raise Unsupported(s)
            t = {"type": tok[1], "vecxz": [float(x) for x in tok[3:6]]}
            if len(tok) == 13:
                t["joint_offset_i"] = [float(x) for x in tok[7:10]]
                t["joint_offset_j"] = [float(x) for x in tok[10:13]]
            transfs[int(tok[2])] = t
        elif head == "element" and tok[1] == "dispBeamColumn":
            if len(tok) < 8:
                raise Unsupported(s)
            rest = tok[8:]
            mass, cmass = None, False
            while rest:
                if rest[0] == "-mass" and len(rest) >= 2:
                    mass, rest = float(rest[1]), rest[2:]
                elif rest[0] == "-cMass":
                    cmass, rest = True, rest[1:]
                else:
                    raise Unsupported(s)
            beams.append((int(tok[2]), [int(tok[3]), int(tok[4])],
                          (int(tok[5]), int(tok[6]), int(tok[7]), mass, cmass, "dispBeamColumn", None, None)))
        elif head == "element" and tok[1] in ("nonlinearBeamColumn", "forceBeamColumn"):
            if len(tok) < 8:
                raise Unsupported(s)
            rest = tok[8:]
            mass, iters, tol = None, None, None
            while rest:
                if rest[0] == "-mass" and len(rest) >= 2:
                    mass, rest = float(rest[1]), rest[2:]
                elif rest[0] == "-iter" and len(rest) >= 3:
                    iters, tol, rest = int(rest[1]), float(rest[2]), rest[3:]
                else:
                    raise Unsupported(s)
            beams.append((int(tok[2]), [int(tok[3]), int(tok[4])],
                          (int(tok[5]), int(tok[6]), int(tok[7]), mass, False, "forceBeamColumn", iters, tol)))
        elif head == "element" and tok[1] == "ASDEmbeddedNodeElement":
            rest = tok[3:]
            nodes_ = []
            while rest and not rest[0].startswith("-"):
                nodes_.append(int(rest[0]))
                rest = rest[1:]
            rot = 0
            while rest:
                if rest[0] == "-rot":
                    rot, rest = 1, rest[1:]
                elif rest[0] in ("-K", "-KP") and len(rest) >= 2:
                    print(f"warning: {s}: {rest[0]} {rest[1]} ignored (FemoraX ties the node exactly; "
                          "docs/deviations.md)", file=sys.stderr)
                    rest = rest[2:]
                else:
                    raise Unsupported(s)
            if len(nodes_) != 5:
                raise Unsupported(s + " (FemoraX: a node and four corners)")
            embedded.append((nodes_[0], nodes_[1:], rot))
        elif head == "rigidLink":
            if tok[1] not in ("beam", "bar") or len(tok) != 4:
                raise Unsupported(s)
            rigid_links.append((int(tok[2]), int(tok[3]), 1 if tok[1] == "beam" else 2))
        elif head == "rigidDiaphragm":
            if len(tok) < 4:
                raise Unsupported(s)
            diaphragms.append((int(tok[1]), int(tok[2]), [int(x) for x in tok[3:]]))
        elif head == "mass":
            for d, v in enumerate(tok[2:], start=1):
                if float(v) != 0.0:
                    masses.append((int(tok[1]), d, float(v)))
        elif head == "element":
            if tok[1] != "stdBrick" or len(tok) != 15:
                raise Unsupported(s)
            elements.append((int(tok[2]), [int(x) for x in tok[3:11]], int(tok[11]),
                             (float(tok[12]), float(tok[13]), float(tok[14]))))
        elif head == "equalDOF":
            ties.append((int(tok[1]), int(tok[2]), [int(x) for x in tok[3:]]))
        elif head == "fix":
            node_fixes.append((int(tok[1]), [int(x) for x in tok[2:]]))
        elif head in ("fixX", "fixY", "fixZ"):
            z = float(var[tok[1][1:]]) if tok[1].startswith("$") else float(tok[1])
            t = tok.index("-tol")
            fixes.append(("XYZ".index(head[-1]), z, [int(x) for x in tok[2:t]], float(tok[t + 1])))
        elif head == "damping":
            if tok[1] != "Uniform" or len(tok) < 6:
                raise Unsupported(s)
            o = dict(zip(tok[6::2], tok[7::2]))
            if set(o) - {"-activateTime", "-deactivateTime", "-fact"}:
                raise Unsupported(s)
            d = dict(ratio=float(tok[3]), freq1=float(tok[4]), freq2=float(tok[5]))
            if "-activateTime" in o:
                d["activate_time"] = float(o["-activateTime"])
            if "-deactivateTime" in o:
                d["deactivate_time"] = float(o["-deactivateTime"])
            if "-fact" in o:
                d["factor_series"] = int(o["-fact"])
            if uniform_defs.setdefault(int(tok[2]), d) != d:
                raise Unsupported(f"damping {tok[2]} appears with different values")
        elif head == "eval" and "region" in s and "-damp " in s:
            m = re.search(r"region (\d+) -ele ([\d ]+) -damp (\d+)\"", s)
            if not m:
                raise Unsupported(s)
            uniform_regions.append(([int(x) for x in m.group(2).split()], int(m.group(3))))
        elif head == "eval" and "region" in s:
            m = re.search(r"region (\d+) -ele ([\d ]+) -rayleigh ([^\"]+)\"", s)
            if not m:
                raise Unsupported(s)
            regions.append(([int(x) for x in m.group(2).split()], [float(x) for x in m.group(3).split()]))
        elif head == "timeSeries":
            if tok[1] != "Path" or series is not None:
                raise Unsupported(s)
            o = dict(zip(tok[3::2], tok[4::2]))
            base = tcl.parent
            vals = np.loadtxt(base / o["-filePath"]).ravel()
            times = np.loadtxt(base / o["-fileTime"]).ravel()
            series = (int(tok[2]), times, vals, float(o.get("-factor", 1.0)))
        elif head == "pattern" and tok[1] == "H5DRM":
            m = re.match(r'pattern H5DRM (\d+) "([^"]+)" (.*)$', s)
            v = [float(x) for x in m.group(3).split()] if m else []
            if len(v) != 16 or drm is not None:
                raise Unsupported(s)
            drm = dict(tag=int(m.group(1)), file=m.group(2), factor=v[0], crd_scale=v[1], distance_tolerance=v[2],
                       do_coordinate_transformation=int(v[3]), transform=v[4:13], origin=v[13:16])
            process.append(("pattern", {"tag": w.i32(drm["tag"])}, {}))
            print(f"warning: pattern H5DRM {drm['tag']}: FemoraX (schema 1.0) reads the DRM file "
                  f"{drm['file']} as it is, its stations and motions in one frame; OpenSees negates the "
                  "z motion it reads, so a file made for OpenSees needs its z motion negated first",
                  file=sys.stderr)
        elif head == "pattern":
            if tok[1] != "UniformExcitation" or tok[4] != "-accel":
                raise Unsupported(s)
            pattern = (int(tok[2]), int(tok[3]), int(tok[5]))
            process.append(("pattern", {"tag": w.i32(pattern[0])}, {}))
        elif head == "recorder" and tok[1] == "Node":
            o = {"-precision": tok[tok.index("-precision") + 1], "-file": tok[tok.index("-file") + 1]}
            n0, d0 = tok.index("-node"), tok.index("-dof")
            name = f"node{len(node_recorders)}"
            node_recorders[name] = (o["-file"].replace("opensees", "femorax"), int(o["-precision"]), "-time" in tok,
                                    [int(x) for x in tok[n0 + 1:d0]], [int(x) for x in tok[d0 + 1:-1]], tok[-1])
            process.append(("recorder", {"name": name}, {}))
        elif head == "recorder":
            if tok[1] != "vtkhdf" or tok[3] != "-dT":
                raise Unsupported(s)
            recorder = (tok[2], float(tok[4]), " ".join(tok[5:]))
            process.append(("recorder", {"name": "vtkhdf"}, {}))
        elif head == "updateMaterialStage":
            mat, stage = int(tok[2]), int(tok[4])
            if process and process[-1][0] == "update_material_stage" and process[-1][1]["stage"] == stage:
                process[-1][2]["materials"].append(mat)
            else:
                process.append(("update_material_stage", {"stage": stage}, {"materials": [mat]}))
        elif head == "setTime":
            process.append(("set_time", {"time": w.f64(float(tok[1]))}, {}))
        elif head in ("wipe", "model", "if", "}", "exit", "wipeAnalysis", "puts", "flush") or head.startswith("set"):
            pass
        else:
            raise Unsupported(s)
        i += 1
    return dict(materials=materials, nodes=[(t, *x) for t, x in nodes.items()], node_ndf=node_ndf, sections=sections,
                transfs=transfs, beams=beams, masses=masses, elements=elements, ties=ties,
                rigid_links=rigid_links, diaphragms=diaphragms, embedded=embedded,
                fixes=fixes, node_fixes=node_fixes, regions=regions, uniform_defs=uniform_defs, uniform_regions=uniform_regions, series=series, pattern=pattern, drm=drm, recorder=recorder,
                node_recorders=node_recorders, process=process)


def mp_constraints(m: dict) -> dict:
    """Schema 1.2's rigid links, rigid diaphragms and embedded nodes (those present)."""
    g = {}
    if m["rigid_links"]:
        g["rigid_link"] = w.node(data={"master": w.i32([x[0] for x in m["rigid_links"]]),
                                       "slave": w.i32([x[1] for x in m["rigid_links"]]),
                                       "type": w.i32([x[2] for x in m["rigid_links"]])})
    if m["diaphragms"]:
        g["rigid_diaphragm"] = w.node(groups={f"{k:06d}": w.node(attrs={"perp": w.i32(p), "master": w.i32(mm)},
                                                                 data={"slaves": w.i32(sl)})
                                              for k, (p, mm, sl) in enumerate(m["diaphragms"])})
    if m["embedded"]:
        g["embedded_node"] = w.node(data={"node": w.i32([x[0] for x in m["embedded"]]),
                                          "host": w.i32([-1] * len(m["embedded"])),
                                          "rot": w.i32([x[2] for x in m["embedded"]]),
                                          "corners": w.i32([x[1] for x in m["embedded"]])})
    return g


def uniform_group(m: dict) -> dict:
    """/damping/uniform (schema 0.7): one child per `region ... -damp` command."""
    if not m["uniform_regions"]:
        return {}
    children = {}
    for k, (els, tag) in enumerate(m["uniform_regions"]):
        d = m["uniform_defs"].get(tag)
        if d is None:
            raise Unsupported(f"region -damp {tag}: no damping Uniform {tag}")
        attrs = {key: (w.i32(v) if key == "factor_series" else w.f64(v)) for key, v in d.items()}
        attrs["femora_tag"] = w.i32(tag)
        children[f"{k:06d}"] = w.node(attrs=attrs, data={"elements": w.i32(els)})
    return {"uniform": w.node(groups=children)}


def write_model(m: dict, path: Path, femorax_vtkhdf: str = "femorax.vtkhdf", explicit_dt: float | None = None,
                recorder_dt: float | None = None, responses: str | None = None) -> None:
    nodes = sorted(m["nodes"])
    mats = {}
    for tag, p in sorted(m["materials"].items()):
        if p.get("type") == "ElasticIsotropic":
            mats[str(tag)] = w.node(attrs={"type": "ElasticIsotropic", "E": w.f64(p["E"]), "nu": w.f64(p["nu"]),
                                           "rho": w.f64(p["rho"])})
            continue
        kind = p.get("type", "PressureDependMultiYield")
        attrs = {"type": kind}
        for k in (PIMY_ORDER if kind == "PressureIndependMultiYield" else PDMY_ORDER):
            attrs[k] = w.i32(int(p[k])) if k in ("nd", "noYieldSurf") else w.f64(p[k])
        mats[str(tag)] = w.node(attrs=attrs)
    groups = {}
    keys = sorted({(mat, b) for (_e, _c, mat, b) in m["elements"]})
    for gi, (mat, b) in enumerate(keys):
        els = [(e, c) for (e, c, mm, bb) in m["elements"] if (mm, bb) == (mat, b)]
        groups[f"{gi:06d}"] = w.node(
            attrs={"type": "stdBrick", "material": w.i32(mat), "body_force": w.f64(list(b)),
                   "lumped": w.u8(explicit_dt is not None)},
            data={"tags": w.i32([e for e, _ in els]), "connectivity": w.i32([c for _, c in els])})
    fix_node, fix_dof = [], []
    fixed = set()
    for (axis, z, dofs, tol) in m["fixes"]:
        for (t, *xyz) in nodes:
            if abs(xyz[axis] - z) <= tol:
                for d, on in enumerate(dofs, start=1):
                    if on and (t, d) not in fixed:  # OpenSees: a second fix of a DOF is ignored
                        fixed.add((t, d))
                        fix_node.append(t)
                        fix_dof.append(d)
    for (t, flags) in m["node_fixes"]:
        for d, on in enumerate(flags, start=1):
            if on and (t, d) not in fixed:
                fixed.add((t, d))
                fix_node.append(t)
                fix_dof.append(d)
    ret, con, dof = [], [], []
    for (r, c, ds) in m["ties"]:
        for d in ds:
            ret.append(r)
            con.append(c)
            dof.append(d)
    # Beams (schema 1.1): one group per (points, section, transformation, mass, cMass); with
    # --explicit their consistent mass becomes lumped, as the bricks'.
    bkeys = sorted({k for (_e, _c, k) in m["beams"]}, key=str)
    for bi, key in enumerate(bkeys):
        nip, sec, tr, mpl, cmass, kind, iters, tol = key
        els = [(e, c) for (e, c, k) in m["beams"] if k == key]
        attrs = {"type": kind, "section": w.i32(sec), "transformation": w.i32(tr),
                 "integration_points": w.i32(nip), "consistent_mass": w.u8(cmass and explicit_dt is None)}
        if iters is not None:  # the force beam's element iteration (schema 1.3)
            attrs["max_iterations"] = w.i32(iters)
            attrs["tolerance"] = w.f64(tol)
        if mpl is not None:
            attrs["mass_per_length"] = w.f64(mpl)  # OpenSees' -mass; without it the beams' mass is nodal
        groups[f"beam{bi:06d}"] = w.node(attrs=attrs, data={"tags": w.i32([e for e, _ in els]),
                                                           "connectivity": w.i32([c for _, c in els])})
    sections = {str(t): w.node(attrs={"type": "Elastic", **{k: w.f64(v) for k, v in p.items()}})
                for t, p in sorted(m["sections"].items())}
    transfs = {str(t): w.node(attrs={k: (v if k == "type" else w.f64(v)) for k, v in p.items()})
               for t, p in sorted(m["transfs"].items())}
    if m["recorder"] is not None:
        _f, rdt, tcl_responses = m["recorder"]
        rdt = recorder_dt if recorder_dt is not None else rdt
        responses = responses if responses is not None else tcl_responses

    def typed(a: dict) -> dict:
        out = {}
        for k, v in a.items():
            if isinstance(v, str):
                out[k] = v
            elif k.startswith("modified_newton") or k == "initialize":
                out[k] = w.u8(bool(v))
            elif isinstance(v, int):
                out[k] = w.i32(v)
            elif isinstance(v, float):
                out[k] = w.f64(v)
            else:
                out[k] = v
        return out

    series_groups, patterns = {}, {}
    if m["series"] is not None:
        stag, times, vals, factor = m["series"]
        series_groups[str(stag)] = w.node(attrs={"type": "Path", "factor": w.f64(factor)},
                                          data={"time": w.f64(times), "values": w.f64(vals)})
    if m["pattern"] is not None:
        ptag, pdof, pseries = m["pattern"]
        patterns[str(ptag)] = w.node(attrs={"type": "UniformExcitation", "dof": w.i32(pdof), "accel": w.i32(pseries)})
    if m["drm"] is not None:
        d = m["drm"]
        patterns[str(d["tag"])] = w.node(
            attrs={"type": "H5DRM", "file": d["file"], "factor": w.f64(d["factor"]), "crd_scale": w.f64(d["crd_scale"]),
                   "distance_tolerance": w.f64(d["distance_tolerance"]),
                   "do_coordinate_transformation": w.i32(d["do_coordinate_transformation"])},
            data={"transform": w.f64(np.reshape(d["transform"], (3, 3)).tolist()), "origin": w.f64(d["origin"])}
            if d["do_coordinate_transformation"] else {})
    steps = {}
    for i, (kind, a, d) in enumerate(m["process"]):
        if kind == "analysis":
            if explicit_dt is not None and a["analysis_type"] == "Transient":
                a = {k: v for k, v in a.items() if k in ("name", "analysis_type", "final_time", "num_steps",
                                                          "constraint_handler", "numberer", "system")}
                a.update(integrator="ExplicitDifference", algorithm="Linear", dt=explicit_dt)
            a = typed(a)
        elif kind == "update_material_stage":
            a = {"stage": w.i32(a["stage"])}
            d = {"materials": w.i32(d["materials"])}
        steps[f"{i:06d}"] = w.node(attrs={"kind": kind, **a}, data=d)
    tree = w.node(attrs=w.root_attrs("femora_nonlinear_layered_soil_column",
                                     "femorax tests/system/femora_example/tcl_to_femorax.py"), groups={
        "nodes": w.node(data={"tags": w.i32([n[0] for n in nodes]), "coords": w.f64([list(n[1:]) for n in nodes]),
                              "ndf": w.i32([m["node_ndf"].get(n[0], 3) for n in nodes])}),
        "materials": w.node(groups={"nd": w.node(groups=mats)}),
        **({"sections": w.node(groups=sections)} if sections else {}),
        **({"transformations": w.node(groups=transfs)} if transfs else {}),
        "elements": w.node(groups=groups),
        **({"mass": w.node(data={"node": w.i32([x[0] for x in m["masses"]]), "dof": w.i32([x[1] for x in m["masses"]]),
                                 "value": w.f64([x[2] for x in m["masses"]])})} if m["masses"] else {}),
        "constraints": w.node(groups={
            "sp": w.node(groups={"fix": w.node(data={"node": w.i32(fix_node), "dof": w.i32(fix_dof)})}),
            "mp": w.node(groups={"equal_dof": w.node(data={"retained": w.i32(ret), "constrained": w.i32(con),
                                                           "dof": w.i32(dof)}),
                                 **mp_constraints(m)})}),
        "damping": w.node(groups={"rayleigh": w.node(groups={f"{k:06d}": w.node(
            attrs={"alphaM": w.f64(ray[0]), "betaK": w.f64(ray[1]), "betaKinit": w.f64(ray[2]),
                   "betaKcomm": w.f64(ray[3])}, data={"elements": w.i32(els)})
            for k, (els, ray) in enumerate(m["regions"])}), **uniform_group(m)}),
        "time_series": w.node(groups=series_groups),
        "patterns": w.node(groups=patterns),
        "recorders": w.node(groups={**({"vtkhdf": w.node(attrs={"type": "VTKHDF", "file": femorax_vtkhdf,
                                                               "dT": w.f64(rdt), "responses": responses})}
                                       if m["recorder"] is not None else {}),
                                    **{name: w.node(attrs={"type": "Node", "file": f, "response": resp,
                                                           "time": w.u8(t), "precision": w.i32(prec)},
                                                    data={"nodes": w.i32(nn), "dofs": w.i32(dd)})
                                       for name, (f, prec, t, nn, dd, resp) in m["node_recorders"].items()}}),
        "process": w.node(groups=steps),
    })
    w.write(path, tree)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tcl", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--explicit", type=float, default=None, metavar="DT")
    ap.add_argument("--recorder-dt", type=float, default=None, metavar="DT")
    ap.add_argument("--recorder", default="femorax.vtkhdf", metavar="FILE")
    ap.add_argument("--responses", default=None)
    o = ap.parse_args()
    try:
        write_model(convert(o.tcl), o.out, femorax_vtkhdf=o.recorder, explicit_dt=o.explicit,
                    recorder_dt=o.recorder_dt, responses=o.responses)
    except Unsupported as e:
        print(f"tcl_to_femorax: not in the recognised subset: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
