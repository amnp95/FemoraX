# Shared setup for the issue examples: 2D, 2 DOFs per node. Every free node gets a unit
# spring to its own fixed ground node in x and y (zeroLength), so the system is always
# well posed and each answer can be checked by hand.
wipe
model basic -ndm 2 -ndf 2
uniaxialMaterial Elastic 1 1.0

proc free_node {tag x y} {
  node $tag $x $y
  set g [expr {1000 + $tag}]
  node $g $x $y
  fix $g 1 1
  element zeroLength $tag $g $tag -mat 1 1 -dir 1 2
}

proc solve {} {
  constraints Transformation
  numberer Plain
  system BandGeneral
  test NormDispIncr 1.0e-12 10
  algorithm Linear
  integrator LoadControl 1.0
  analysis Static
  analyze 1
}

proc report {tags expected} {
  # expected: list of {tag dof value} with the correct displacement
  foreach t $tags {
    puts [format "node %d  disp %-28s DOF equations %s" $t [nodeDisp $t] [nodeDOFs $t]]
  }
  set ok 1
  foreach e $expected {
    lassign $e t d want
    set got [nodeDisp $t $d]
    if {abs($got - $want) > 1.0e-9} {
      puts [format "WRONG: node %d DOF %d is %.6g, the constraints give %.6g" $t $d $got $want]
      set ok 0
    }
  }
  if {$ok} { puts "correct" }
}
