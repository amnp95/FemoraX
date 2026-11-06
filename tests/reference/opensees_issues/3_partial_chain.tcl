# Issue 3: a partial chain.
# Node 2 DOF 1 is tied to node 1; node 3 (both DOFs) is tied to node 2. Node 3's
# constrained DOFs are mapped through node 2's transformed ID (TransformationDOF_Group::
# doneID: otherID(dof)), whose layout is [node 2's free DOF 2, node 1's DOF 1], not node 2's
# nodal DOFs. Node 3's x and y end up crossed.
#
# Correct answer for a unit load on node 3, DOF 1: nodes 1, 2, 3 share DOF 1 (three
# springs, u = 1/3); DOF 2 of nodes 2 and 3 is unloaded (0). OpenSees ties node 3 DOF 1 to
# node 2 DOF 2.
source common.tcl
free_node 1 0 0
free_node 2 1 0
free_node 3 2 0
equalDOF 1 2 1
equalDOF 2 3 1 2
pattern Plain 1 Linear { load 3 1.0 0.0 }
solve
report {1 2 3} {{1 1 0.333333333333} {2 1 0.333333333333} {3 1 0.333333333333} {2 2 0.0} {3 2 0.0}}
