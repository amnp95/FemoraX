#!/usr/bin/env python3
"""Write the constraint comparison models (Tcl, run by OpenSees and converted for FemoraX).

    make_constraints.py <out_dir>

frame.tcl: a one-storey frame of four elastic dispBeamColumn columns (two elements each, bases
fixed), whose tops are the slaves of a rigid diaphragm (perp 3) with its master off the centre
(the floor's mass and rotary inertia on it; its uz, rx, ry fixed), and an eccentric mass hung on
a rigid link (beam) off the mid-height node of one column. A sin^2 base pulse in x, Newmark with
Newton, constraints Transformation. Records the master (x, y, rz), the eccentric mass (x, y, z)
and the linked column node (x, y).

embedded_<K>.tcl: an elastic steel pile through a 4 x 4 x 8 stdBrick soil column, off the soil
nodes (x = 2.3, y = 1.6, at the middle of each element layer), each pile node embedded with
ASDEmbeddedNodeElement -rot in four corners of its brick (the Kuhn tetrahedron that holds it),
with the penalty -K <K>; FemoraX ties the nodes exactly (the same corners, the barycentric
weights). The same base pulse and analysis. Records the pile's x displacement and y rotation.
"""
import math
import sys
from pathlib import Path

ANALYSIS = """puts "Starting analysis : shake"
constraints Transformation
numberer Plain
system BandGeneral
test NormDispIncr 1e-12 20
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
proc FemoraAnalyzeTransientStep {{dt level}} {{
	set Ok [analyze 1 $dt]
	return $Ok
}}
set FemoraAnalysisStep 0
while {{[getTime] < {T:.6f}}} {{
	set Ok [FemoraAnalyzeTransientStep {dt} 0]
	if {{$Ok != 0}} {{
		error "analysis failed"
	}}
	incr FemoraAnalysisStep
}}
wipeAnalysis"""

K_VALUES = ["1.0e8", "1.0e10", "1.0e12"]


def motion(out: Path, dt=0.001, n=401, T=0.05, amp=1.0):
    t = [i * dt for i in range(n)]
    a = [(amp * math.sin(math.pi * x / T) ** 2 if x < T else 0.0) for x in t]
    (out / "time.txt").write_text("\n".join(f"{x:.6f}" for x in t) + "\n")
    (out / "accel.txt").write_text("\n".join(f"{x:.12e}" for x in a) + "\n")


def frame() -> str:
    a, b, h = 2.0, 1.5, 3.0
    xy = [(-a, -b), (a, -b), (a, b), (-a, b)]
    L = ["wipe", "model BasicBuilder -ndm 3 -ndf 6"]
    for c, (x, y) in enumerate(xy):
        for k in range(3):
            L.append(f"node {10 * (c + 1) + k} {x} {y} {h * k / 2} -ndf 6")
    L.append("node 100 0.3 -0.2 3.0 -ndf 6")
    L.append(f"node 200 {a + 0.8} {b} {h / 2 + 0.5} -ndf 6")
    L.append("section Elastic 1 2.0e8 0.05 4.0e-4 3.0e-4 7.7e7 6.0e-4")
    L.append("geomTransf Linear 1 1.0 0.0 0.0")
    for c in range(4):
        for e in range(2):
            L.append(f"element dispBeamColumn {10 * (c + 1) + e} {10 * (c + 1) + e} {10 * (c + 1) + e + 1} 3 1 1")
        L.append(f"fix {10 * (c + 1)} 1 1 1 1 1 1")
    L.append("fix 100 0 0 1 1 1 0")
    L.append("rigidDiaphragm 3 100 12 22 32 42")
    L.append("rigidLink beam 31 200")
    jz = 40.0 * (4 * a * a + 4 * b * b) / 12.0
    L.append(f"mass 100 40.0 40.0 0.0 0.0 0.0 {jz}")
    L.append("mass 200 6.0 6.0 6.0 0.5 0.5 0.5")
    # the columns' mass (rho A over each node's tributary length) and a small rotary inertia as
    # nodal masses, the beams massless: the same model in both programs (FemoraX's -mass would add
    # the beam's own rotary inertia, docs/deviations.md)
    mc = 7.8 * 0.05 * h / 2
    for c in range(4):
        for k, f in ((1, 1.0), (2, 0.5)):
            L.append(f"mass {10 * (c + 1) + k} {mc * f} {mc * f} {mc * f} 0.01 0.01 0.01")
    L.append("timeSeries Path 1 -filePath accel.txt -fileTime time.txt -factor 3.0")
    L.append("pattern UniformExcitation 2 1 -accel 1")
    L.append("recorder Node -file opensees_master.txt -precision 16 -time -node 100 -dof 1 2 6 disp")
    L.append("recorder Node -file opensees_mass.txt -precision 16 -time -node 200 -dof 1 2 3 disp")
    L.append("recorder Node -file opensees_column.txt -precision 16 -time -node 31 -dof 1 2 disp")
    L += ANALYSIS.format(T=0.25, dt=0.001).splitlines()
    L.append("exit")
    return "\n".join(L) + "\n"


def embedded(K: str) -> str:
    nx = ny = 4
    nz = 8
    dz = 0.5
    NX, NY = nx + 1, ny + 1
    tag = lambda i, j, k: 1 + i + NX * (j + NY * k)
    L = ["wipe", "model BasicBuilder -ndm 3 -ndf 3"]
    for k in range(nz + 1):
        for j in range(NY):
            for i in range(NX):
                L.append(f"node {tag(i, j, k)} {float(i)} {float(j)} {-nz * dz + k * dz} -ndf 3")
    L.append("nDMaterial ElasticIsotropic 1 377000.0 0.3 2.0285")
    L.append("nDMaterial ElasticIsotropic 2 109200.0 0.3 2.0183")
    e = 1
    for k in range(nz):
        mat = 1 if k < nz // 2 else 2
        for j in range(ny):
            for i in range(nx):
                n = [tag(i, j, k), tag(i + 1, j, k), tag(i + 1, j + 1, k), tag(i, j + 1, k),
                     tag(i, j, k + 1), tag(i + 1, j, k + 1), tag(i + 1, j + 1, k + 1), tag(i, j + 1, k + 1)]
                L.append(f"element stdBrick {e} {' '.join(map(str, n))} {mat} 0.0 0.0 0.0")
                e += 1
    L.append("model BasicBuilder -ndm 3 -ndf 6")
    px, py = 2.3, 1.6
    levels = list(range(1, nz))  # pile nodes at the middle of element layers 1 .. nz - 1
    for k in levels:
        L.append(f"node {10000 + k} {px} {py} {-nz * dz + (k + 0.5) * dz} -ndf 6")
    L.append("section Elastic 1 2.0e8 0.05 4.0e-4 4.0e-4 7.7e7 8.0e-4")
    L.append("geomTransf Linear 1 1.0 0.0 0.0")
    for k in levels[:-1]:
        L.append(f"element dispBeamColumn {50000 + k} {10000 + k} {10001 + k} 3 1 1")
    m_node, i_node, ip_node = 7.8 * 0.05 * dz, 7.8 * 4.0e-4 * dz, 7.8 * 8.0e-4 * dz
    for k in levels:
        f = 0.5 if k in (levels[0], levels[-1]) else 1.0
        L.append(f"mass {10000 + k} {m_node * f} {m_node * f} {m_node * f} {i_node * f} {i_node * f} {ip_node * f}")
    # Each pile node in the Kuhn tetrahedron of its brick (cube [i, i+1] x [j, j+1] x [k, k+1]):
    # from the corner (0, 0, 0), unit steps along the axes in decreasing order of the local
    # coordinates.
    i0, j0 = int(px), int(py)
    for k in levels:
        z = (k + 0.5) * dz / dz  # in element layers
        k0 = int(z)
        loc = [px - i0, py - j0, z - k0]
        order = sorted(range(3), key=lambda d: -loc[d])
        p = [0, 0, 0]
        corners = [tag(i0, j0, k0)]
        for d in order:
            p[d] += 1
            corners.append(tag(i0 + p[0], j0 + p[1], k0 + p[2]))
        L.append(f"element ASDEmbeddedNodeElement {60000 + k} {10000 + k} {' '.join(map(str, corners))} -rot -K {K}")
    L.append(f"fixZ {-nz * dz} 1 1 1 -tol 1e-6")
    L.append("timeSeries Path 1 -filePath accel.txt -fileTime time.txt -factor 1.0")
    L.append("pattern UniformExcitation 2 1 -accel 1")
    pile = " ".join(str(10000 + k) for k in levels)
    L.append(f"recorder Node -file opensees_pile_ux.txt -precision 16 -time -node {pile} -dof 1 disp")
    L.append(f"recorder Node -file opensees_pile_ry.txt -precision 16 -time -node {pile} -dof 5 disp")
    L += ANALYSIS.format(T=0.2, dt=0.001).splitlines()
    L.append("exit")
    return "\n".join(L) + "\n"


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    motion(out)
    (out / "frame.tcl").write_text(frame())
    for K in K_VALUES:
        (out / f"embedded_{K}.tcl").write_text(embedded(K))
    return 0


if __name__ == "__main__":
    sys.exit(main())
