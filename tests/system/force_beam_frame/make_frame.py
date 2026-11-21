#!/usr/bin/env python3
"""Write the force-beam frame comparison model (Tcl, run by OpenSees and converted for FemoraX).

    make_frame.py <out_dir>

force_frame.tcl: a two-storey, one-bay 3D frame, one force-based element per member
(nonlinearBeamColumn: ForceBeamColumn3d with Lobatto points): eight columns (5 points, PDelta,
the upper ones 3 points) and eight girders (Linear; 7 points in x, 4 in y, the x girders of the
first floor with rigid joint offsets), elastic sections (a column and a girder section), bases
fixed. Nodal masses with rotary inertia, unequal over the plan (so the floors twist), the elements
massless: the same mass in both programs. A sin^2 base pulse in x, Newmark with Newton. Records
the displacements and rotations of one node per floor and the x displacement of every joint.
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


def motion(out: Path, dt=0.001, n=401, T=0.05, amp=1.0):
    t = [i * dt for i in range(n)]
    a = [(amp * math.sin(math.pi * x / T) ** 2 if x < T else 0.0) for x in t]
    (out / "time.txt").write_text("\n".join(f"{x:.6f}" for x in t) + "\n")
    (out / "accel.txt").write_text("\n".join(f"{x:.12e}" for x in a) + "\n")


def frame() -> str:
    a, b, h = 6.0, 4.0, 3.2
    xy = [(0.0, 0.0), (a, 0.0), (a, b), (0.0, b)]
    node = lambda c, k: 10 * (c + 1) + k  # column line c (0..3), level k (0 base, 1, 2)
    L = ["wipe", "model BasicBuilder -ndm 3 -ndf 6"]
    for c, (x, y) in enumerate(xy):
        for k in range(3):
            L.append(f"node {node(c, k)} {x} {y} {h * k} -ndf 6")
        L.append(f"fix {node(c, 0)} 1 1 1 1 1 1")
    L.append("section Elastic 1 2.0e8 0.02 5.0e-4 3.0e-4 7.7e7 6.0e-4")   # columns
    L.append("section Elastic 2 2.0e8 0.012 4.0e-4 6.0e-5 7.7e7 2.0e-5")  # girders
    L.append("geomTransf PDelta 1 1.0 0.0 0.0")                           # columns
    L.append("geomTransf Linear 2 0.0 0.0 1.0")                           # girders
    L.append("geomTransf Linear 3 0.0 0.0 1.0 -jntOffset 0.15 0.0 0.0 -0.15 0.0 0.0")  # x girders, floor 1
    e = 1
    for c in range(4):
        L.append(f"element nonlinearBeamColumn {e} {node(c, 0)} {node(c, 1)} 5 1 1 -iter 10 1e-12"); e += 1
        L.append(f"element nonlinearBeamColumn {e} {node(c, 1)} {node(c, 2)} 3 1 1 -iter 10 1e-12"); e += 1
    for k in (1, 2):
        for c in range(4):
            i, j = node(c, k), node((c + 1) % 4, k)
            along_x = c in (0, 2)
            if c == 2 or c == 3:
                i, j = j, i  # every girder runs towards +x or +y
            nip = 7 if along_x else 4
            tr = 3 if (along_x and k == 1) else 2
            L.append(f"element nonlinearBeamColumn {e} {i} {j} {nip} 2 {tr} -iter 10 1e-12"); e += 1
    # floor masses, unequal over the plan, with rotary inertia
    for k, m0 in ((1, 8.0), (2, 6.0)):
        for c, f in enumerate((1.0, 1.6, 0.7, 1.2)):
            m = m0 * f
            L.append(f"mass {node(c, k)} {m} {m} {m} {0.02 * m} {0.02 * m} {0.03 * m}")
    L.append("timeSeries Path 1 -filePath accel.txt -fileTime time.txt -factor 3.0")
    L.append("pattern UniformExcitation 2 1 -accel 1")
    L.append(f"recorder Node -file opensees_floor1.txt -precision 16 -time -node {node(1, 1)} -dof 1 2 3 4 5 6 disp")
    L.append(f"recorder Node -file opensees_floor2.txt -precision 16 -time -node {node(2, 2)} -dof 1 2 3 4 5 6 disp")
    joints = " ".join(str(node(c, k)) for k in (1, 2) for c in range(4))
    L.append(f"recorder Node -file opensees_joints.txt -precision 16 -time -node {joints} -dof 1 disp")
    L += ANALYSIS.format(T=0.3, dt=0.001).splitlines()
    L.append("exit")
    return "\n".join(L) + "\n"


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    motion(out)
    (out / "force_frame.tcl").write_text(frame())


if __name__ == "__main__":
    main()
