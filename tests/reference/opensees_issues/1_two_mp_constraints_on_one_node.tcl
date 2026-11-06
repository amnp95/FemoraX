# Issue 1: a node constrained by two equalDOF commands.
# Node 3 DOF 1 is tied to node 1 and node 3 DOF 2 to node 2. The Transformation handler
# keeps only the first MP_Constraint that names a node (TransformationConstraintHandler::
# handle, constrainedNodesMP.getLocation) and silently ignores the second.
#
# Correct answer for a unit load on node 3, DOF 2: nodes 2 and 3 share DOF 2, two unit
# springs, so u2y = u3y = 0.5. OpenSees gives u3y = 1, u2y = 0.
source common.tcl
free_node 1 0 0
free_node 2 1 0
free_node 3 2 0
equalDOF 1 3 1
equalDOF 2 3 2
pattern Plain 1 Linear { load 3 0.0 1.0 }
solve
report {1 2 3} {{2 2 0.5} {3 2 0.5}}
