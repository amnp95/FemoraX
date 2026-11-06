# Issue 2: a chain whose intermediate node has a larger tag.
# Node 2 is tied to node 3, and node 3 is tied to node 1 (all DOFs). DOF groups are
# completed in ascending tag order (TransformationDOF_Group::doneID), so node 2 reads node
# 3's equation numbers before they are set (still -1) and becomes fixed.
#
# Correct answer for a unit load on node 2, DOF 1: all three nodes move together, three
# unit springs, so u1x = u2x = u3x = 1/3. OpenSees fixes node 2.
source common.tcl
free_node 1 0 0
free_node 2 1 0
free_node 3 2 0
equalDOF 3 2 1 2
equalDOF 1 3 1 2
pattern Plain 1 Linear { load 2 1.0 0.0 }
solve
report {1 2 3} {{1 1 0.333333333333} {2 1 0.333333333333} {3 1 0.333333333333}}
