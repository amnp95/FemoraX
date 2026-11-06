# Newmark with stdBrick -lumped: the lumped mass is used in the tangent only.
#
# Brick::formInertiaTerms computes the inertia force M*a of the residual with the consistent
# mass whatever massType says (getResistingForceIncInertia); only getMass(), which the
# Newmark tangent uses, honours -lumped. So:
#   - with Newton, -lumped silently converges to the consistent-mass answer;
#   - with Linear (one solve per step), the step solves
#       (c1*K + c3*M_L) dU = R(M_C)
#     which is neither model: the error acts like an extra stiffness ~ (M_L - M_C)/dt^2 and
#     grows as the step gets smaller.
#
# One 1x1x1 brick, base fixed, the four top nodes tied in z, undamped, a suddenly applied
# body force. The peak top displacement is compared with:
#   consistent mass (Newton), and a true lumped mass (massless brick + nodal masses rho*V/8).
# Found while building FemoraX's Newmark; OpenSees commit 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea.

proc run {mass algo dt} {
  wipe
  model basic -ndm 3 -ndf 3
  set rho 1.0
  if {$mass == "nodal"} { set rhoe 0.0 } else { set rhoe $rho }
  nDMaterial ElasticIsotropic 1 1000.0 0.25 $rhoe
  set k 1
  foreach z {0.0 1.0} {
    foreach {x y} {0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0} {
      node $k $x $y $z
      incr k
    }
  }
  set flag [expr {$mass == "lumped" ? "-lumped" : ""}]
  eval element stdBrick 1 1 2 3 4 5 6 7 8 1 0.0 0.0 -10.0 $flag
  foreach n {1 2 3 4} { fix $n 1 1 1 }
  foreach n {5 6 7 8} { fix $n 1 1 0 }
  foreach n {6 7 8} { equalDOF 5 $n 3 }
  if {$mass == "nodal"} {
    foreach n {5 6 7 8} { mass $n 0.125 0.125 0.125 }
  }
  constraints Transformation
  numberer Plain
  system BandGeneral
  test NormDispIncr 1.0e-14 50 0
  algorithm $algo
  integrator Newmark 0.5 0.25
  analysis Transient
  set peak 0.0
  set n [expr {int(round(0.5 / $dt))}]
  for {set i 0} {$i < $n} {incr i} {
    analyze 1 $dt
    set u [nodeDisp 5 3]
    if {abs($u) > abs($peak)} { set peak $u }
  }
  return $peak
}

set ok 1
foreach dt {0.004 0.001 0.00025} {
  set c [run consistent Newton $dt]
  set t [run nodal Newton $dt]
  set ln [run lumped Newton $dt]
  set ll [run lumped Linear $dt]
  puts [format "dt %-8g peak top uz: consistent %.6e  true lumped %.6e  -lumped Newton %.6e  -lumped Linear %.6e" \
          $dt $c $t $ln $ll]
  if {abs($ln - $c) < 1.0e-9 * abs($c) && abs($ln - $t) > 1.0e-3 * abs($t)} {
    puts [format "WRONG: dt %g: -lumped with Newton gives the consistent-mass answer %.6e, not the lumped-mass one %.6e" $dt $ln $t]
    set ok 0
  }
  if {abs($ll - $t) > 0.01 * abs($t)} {
    puts [format "WRONG: dt %g: -lumped with Linear gives %.6e, %.1f%% from the lumped-mass answer %.6e" \
            $dt $ll [expr {100.0 * abs($ll - $t) / abs($t)}] $t]
    set ok 0
  }
}
if {$ok} { puts "correct" }
wipe
