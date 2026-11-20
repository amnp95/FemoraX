#!/usr/bin/env python3
"""Write the pile-in-soil model (Tcl, run by OpenSees and converted for FemoraX) and its motion.

    make_pile.py <out_dir>

A 4 x 4 x 8 stdBrick soil column (two elastic layers, dz 0.5, base fixed, sides free, no body
force), an elastic steel pile (dispBeamColumn, 3 points, Linear) on its centre line from the first
node level above the base to the top, each pile node tied to its soil node in x, y, z (equalDOF);
the pile's mass and rotary inertia as nodal masses (the same model in OpenSees and FemoraX), a
sin^2 base pulse in x, Newmark with Newton. Femora's analysis block form, so the
converter (tests/system/femora_example/tcl_to_femorax.py) reads it.
"""
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
nx = ny = 4
nz = 8
dz = 0.5
NX, NY = nx + 1, ny + 1
tag = lambda i, j, k: 1 + i + NX * (j + NY * k)
L = []
L.append("wipe")
L.append("model BasicBuilder -ndm 3 -ndf 3")
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
# The pile: nodes 10001 + k at the centre line, k = 1 .. nz (OpenSees: frame elements and
# geomTransf need the builder at 6 DOFs, as Femora's exports switch it for beam parts).
L.append("model BasicBuilder -ndm 3 -ndf 6")
for k in range(1, nz + 1):
    L.append(f"node {10000 + k} 2.0 2.0 {-nz * dz + k * dz} -ndf 6")
L.append("section Elastic 1 2.0e8 0.05 4.0e-4 4.0e-4 7.7e7 8.0e-4")
L.append("geomTransf Linear 1 1.0 0.0 0.0")
for k in range(1, nz):
    L.append(f"element dispBeamColumn {50000 + k} {10000 + k} {10001 + k} 3 1 1")
# rho A and the rotary inertia rho I (bending), rho Ip (torsion) per length, times the tributary
# length; the rotations need it: the ties hold only the translations, so the pile's twist about
# its axis has no stiffness (a massless rigid mode makes the implicit system singular).
m_node, i_node, ip_node = 7.8 * 0.05 * dz, 7.8 * 4.0e-4 * dz, 7.8 * 8.0e-4 * dz
for k in range(1, nz + 1):
    f = 0.5 if k == nz else 1.0
    m, ib, ip = m_node * f, i_node * f, ip_node * f
    L.append(f"mass {10000 + k} {m} {m} {m} {ib} {ib} {ip}")
for k in range(1, nz + 1):
    L.append(f"equalDOF {tag(2, 2, k)} {10000 + k} 1 2 3")
L.append(f"fixZ {-nz * dz} 1 1 1 -tol 1e-6")
L.append("timeSeries Path 1 -filePath accel.txt -fileTime time.txt -factor 1.0")
L.append("pattern UniformExcitation 2 1 -accel 1")
pile = " ".join(str(10000 + k) for k in range(1, nz + 1))
L.append(f"recorder Node -file opensees_pile.txt -precision 16 -time -node {pile} -dof 1 disp")
L += """puts "Starting analysis : shake"
constraints Transformation
numberer Plain
system BandGeneral
test NormDispIncr 1e-12 20
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
proc FemoraAnalyzeTransientStep {dt level} {
	set Ok [analyze 1 $dt]
	return $Ok
}
set FemoraAnalysisStep 0
while {[getTime] < 0.200000} {
	set Ok [FemoraAnalyzeTransientStep 0.001 0]
	if {$Ok != 0} {
		error "analysis failed"
	}
	incr FemoraAnalysisStep
}
wipeAnalysis""".splitlines()
L.append("exit")
(out / "pile.tcl").write_text("\n".join(L) + "\n")
t = [i * 0.001 for i in range(0, 301)]
a = [(math.sin(math.pi * x / 0.05) ** 2 if x < 0.05 else 0.0) for x in t]
(out / "time.txt").write_text("\n".join(f"{x:.6f}" for x in t) + "\n")
(out / "accel.txt").write_text("\n".join(f"{x:.12e}" for x in a) + "\n")
