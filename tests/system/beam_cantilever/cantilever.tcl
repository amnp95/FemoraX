# A Femora-style Tcl cantilever for the converter's beam commands (schema 1.1): six
# dispBeamColumn elements along z (6-DOF nodes), section Elastic, geomTransf PDelta with a
# joint offset, OpenSees' -mass on the elements plus a nodal mass on the top, base shaking in x,
# Newmark with Newton. Converted by tcl_to_femorax.py and run by FemoraX in beam_check.sh.
wipe
model BasicBuilder -ndm 3 -ndf 6
node 1 1.0 2.0 -6.0 -ndf 6
node 2 1.0 2.0 -5.0 -ndf 6
node 3 1.0 2.0 -4.0 -ndf 6
node 4 1.0 2.0 -3.0 -ndf 6
node 5 1.0 2.0 -2.0 -ndf 6
node 6 1.0 2.0 -1.0 -ndf 6
node 7 1.0 2.0 0.0 -ndf 6
section Elastic 1 2.5e7 0.5 0.02 0.015 1.0e7 0.03
geomTransf PDelta 1 1.0 0.0 0.0 -jntOffset 0.0 0.0 0.0 0.0 0.0 0.05
element dispBeamColumn 1 1 2 5 1 1 -mass 1.2
element dispBeamColumn 2 2 3 5 1 1 -mass 1.2
element dispBeamColumn 3 3 4 5 1 1 -mass 1.2
element dispBeamColumn 4 4 5 5 1 1 -mass 1.2
element dispBeamColumn 5 5 6 5 1 1 -mass 1.2
element dispBeamColumn 6 6 7 5 1 1 -mass 1.2
mass 7 0.5 0.5 0.5 0.0 0.0 0.0
fixZ -6.0 1 1 1 1 1 1 -tol 1e-6
timeSeries Path 1 -filePath accel.txt -fileTime time.txt -factor 1.0
pattern UniformExcitation 2 1 -accel 1
recorder Node -file opensees_tip.txt -precision 12 -time -node 7 -dof 1 2 3 4 5 6 disp
puts "Starting analysis : shake"
constraints Transformation
numberer Plain
system BandGeneral
test NormDispIncr 1e-12 10
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
set FemoraAnalysisTotal 200
set ok [FemoraAnalyzeTransientStep 0.001 0]
wipeAnalysis
