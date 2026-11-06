# Issue 4: a fix on a DOF that an equalDOF constrains.
# Node 2 is tied to node 1 (both DOFs), and node 2 DOF 1 is fixed. In
# TransformationDOF_Group::addSP_Constraint the -1 goes to position loc = (number of
# unconstrained DOFs below the fixed DOF) = 0, which is a retained-node entry that doneID
# later overwrites, so the fix is lost.
#
# Correct answer for a unit load on node 1, DOF 1: the fix holds node 2, the tie holds
# node 1, so u1x = u2x = 0. OpenSees gives 0.5 (two springs, no fix).
source common.tcl
free_node 1 0 0
free_node 2 1 0
equalDOF 1 2 1 2
fix 2 1 0
pattern Plain 1 Linear { load 1 1.0 0.0 }
solve
report {1 2} {{1 1 0.0} {2 1 0.0}}
