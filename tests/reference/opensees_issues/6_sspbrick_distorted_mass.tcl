# SSPbrick: the lumped mass and the body force of a distorted element do not add up to
# rho*V and b*V.
#
# SSPbrick::getMass and SSPbrick::getResistingForce (applyLoad == 0) give node i
#   massTerm = density * J[0] * (1 + (... + J[7] + J[8] + J[9])/3 + (...)/9 + (...)/27)
#   polyJac  = J[0] * (same bracket)
# The bracket adds 1 to terms that have units of volume (J[k] are triple products of the
# nodal coordinate moments), so the formula is exact only when J[4..19] vanish: for
# parallelepipeds (every Femora grid). For other shapes the element's total mass and total
# body force differ from rho*V and b*V (V: SSPbrick's own mVol, equal to stdBrick's volume).
#
# One brick, its nodes moved by up to `amp` of the edge length, a body force b = (0 0 -10)
# and no displacement: eleForce gives -b * polyJac (SSPbrick) and -b * integral(N) (stdBrick)
# per node; their sums are compared with b * V.
# Found while porting SSPbrick to FemoraX; OpenSees commit 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea.

proc total_z {type amp} {
  wipe
  model basic -ndm 3 -ndf 3
  nDMaterial ElasticIsotropic 1 1000.0 0.25 1.0
  # unit cube, then a fixed pseudo-random perturbation of every node
  set d {0.11 -0.07 0.05  -0.09 0.13 -0.04  0.06 0.08 -0.12  -0.1 -0.05 0.09
         0.07 -0.11 -0.06  -0.08 0.04 0.1  0.12 -0.09 0.07  -0.05 0.1 -0.08}
  set k 1
  set c 0
  foreach z {0.0 1.0} {
    foreach {x y} {0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0} {
      node $k [expr {$x + $amp * [lindex $d $c]}] [expr {$y + $amp * [lindex $d [expr {$c+1}]]}] [expr {$z + $amp * [lindex $d [expr {$c+2}]]}]
      incr k
      incr c 3
    }
  }
  element $type 1 1 2 3 4 5 6 7 8 1 0.0 0.0 -10.0
  set f [eleForce 1]
  set s 0.0
  for {set i 2} {$i < 24} {incr i 3} { set s [expr {$s + [lindex $f $i]}] }
  return $s
}

set ok 1
foreach amp {0.0 1.0 2.0} {
  set fs [total_z SSPbrick $amp]
  set fb [total_z stdBrick $amp]
  # stdBrick integrates N exactly (2x2x2 Gauss), so its total is b * V.
  set rel [expr {($fs - $fb) / $fb}]
  puts [format "amp %.1f  total body force z: SSPbrick %.12f  stdBrick (= b*V) %.12f  relative %.3e" $amp $fs $fb $rel]
  if {abs($rel) > 1.0e-12} {
    puts [format "WRONG: amp %.1f: SSPbrick's body force (and mass) total is %.3e off b*V" $amp $rel]
    set ok 0
  }
}
if {$ok} { puts "correct" }
