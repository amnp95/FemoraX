# Reference data for tests/unit/test_uniform_damping.cpp (written to tests/reference/data/damping
# by tests/reference/damping/make_data.sh).
# Single-mode free vibration: a mass of 1 on a zeroLength spring (stiffness k = (2 pi f)^2),
# released from u0 = 1. DAMP = rayleigh | uniform. Uniform: ratio XI over [F1, F2].
# Records the displacement for NCYC cycles at 400 steps per period (Newmark, exact solves).
lassign $argv DAMP f XI F1 F2 NCYC
wipe
model basic -ndm 1 -ndf 1
node 1 0.0; node 2 0.0 -mass 1.0
fix 1 1
set w [expr 2.0*3.14159265358979323846*$f]
uniaxialMaterial Elastic 1 [expr $w*$w]
if {$DAMP == "uniform"} {
  damping Uniform 1 $XI $F1 $F2
  element zeroLength 1 1 2 -mat 1 -dir 1 -damp 1
} else {
  element zeroLength 1 1 2 -mat 1 -dir 1
  region 1 -ele 1 -rayleigh 0.0 [expr 2.0*$XI/$w] 0.0 0.0
}
pattern Plain 1 Linear { load 2 [expr $w*$w] }
constraints Plain; numberer Plain; system FullGeneral; test NormDispIncr 1e-14 20; algorithm Newton
integrator LoadControl 1.0; analysis Static; analyze 1
loadConst -time 0.0; remove loadPattern 1; wipeAnalysis
set dt [expr 1.0/$f/400.0]
recorder Node -file sdof_${DAMP}_${f}.txt -precision 16 -time -node 2 -dof 1 disp
constraints Plain; numberer Plain; system FullGeneral; test NormDispIncr 1e-14 20; algorithm Newton
integrator Newmark 0.5 0.25; analysis Transient
analyze [expr int($NCYC*400)] $dt
wipe
