# PressureDependMultiYield: one global pAtm for every material.
#
# PressureDependMultiYield keeps the atmospheric pressure (the last constructor argument
# before the cohesion, Femora's `pa`) in one static member, pAtm, which every constructor
# overwrites. The yield surfaces are set up with each material's own value, but at run time
# isCriticalState (PressureDependMultiYield.cpp, lines 2047-2051) reads the static, so every
# PressureDependMultiYield material uses the pa of the last one defined. Defining another,
# unused material therefore changes the response of the first.
#
# One SSPbrick of a dense sand near its critical void ratio (e = 0.86), its strain driven through
# the nodes: consolidation in the elastic stage, the switch to the plastic stage, cyclic
# shear with a vertical strain cycle. Run twice: material 1 alone, and material 1 followed by
# an unused material 2 with pa = 0.01. The stresses must not change.
# Found while porting PressureDependMultiYield to FemoraX; OpenSees commit
# 955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea.

proc run {second} {
  wipe
  model basic -ndm 3 -ndf 3
  nDMaterial PressureDependMultiYield 1 3 2.03 145000.0 314166.7 40.0 0.1 80.0 0.5 27.0 0.03 0.8 5.0 0.0 0.0 0.0 20 0.86 0.9 0.02 0.7 101.0 0.3
  if {$second} {
    nDMaterial PressureDependMultiYield 2 3 2.03 145000.0 314166.7 40.0 0.1 80.0 0.5 27.0 0.03 0.8 5.0 0.0 0.0 0.0 20 0.86 0.9 0.02 0.7 0.01 0.3
  }
  set k 1
  foreach z {0.0 1.0} {
    foreach {x y} {0.0 0.0 1.0 0.0 1.0 1.0 0.0 1.0} {
      node $k $x $y $z
      incr k
    }
  }
  element SSPbrick 1 1 2 3 4 5 6 7 8 1 0.0 0.0 0.0
  foreach n {1 2 3 4} { fix $n 1 1 1 }
  # shear gamma_xz(t) on the top x DOFs, vertical strain e_zz(t) on the top z DOFs
  set g {}
  set ez {}
  for {set i 0} {$i <= 130} {incr i} {
    if {$i <= 10} {
      lappend g 0.0
      lappend ez [expr {-1.0e-3 * $i / 10.0}]
    } else {
      set t [expr {($i - 10) / 120.0}]
      lappend g [expr {2.0e-3 * sin(2 * 3.141592653589793 * 3 * $t)}]
      lappend ez [expr {-1.0e-3 - 1.5e-3 * sin(3.141592653589793 * $t)}]
    }
  }
  timeSeries Path 1 -dt 1.0 -values $g
  timeSeries Path 2 -dt 1.0 -values $ez
  pattern Plain 1 1 { foreach n {5 6 7 8} { sp $n 1 1.0 } }
  pattern Plain 2 2 { foreach n {5 6 7 8} { sp $n 3 1.0 } }
  constraints Transformation
  numberer Plain
  system FullGeneral
  test NormDispIncr 1.0e-12 30 0
  algorithm Newton
  integrator LoadControl 1.0
  analysis Static
  analyze 10
  updateMaterialStage -material 1 -stage 1
  analyze 120
  return [eleResponse 1 stress]
}

set a [run 0]
set b [run 1]
puts "material 1 alone:          stress $a"
puts "material 1, then unused 2: stress $b"
set ok 1
set scale 0.0
foreach x $a { if {abs($x) > $scale} { set scale [expr {abs($x)}] } }
foreach x $a y $b {
  if {abs($x - $y) > 1.0e-9 * $scale} { set ok 0 }
}
if {$ok} {
  puts "correct"
} else {
  puts "WRONG: defining an unused PressureDependMultiYield material with another pa changed the stress of the first"
}
