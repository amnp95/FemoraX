# Control: patterns the Transformation handler resolves correctly (no issue).
# A full chain in ascending tag order (2 -> 1, 3 -> 2, all DOFs), and a fixed retained node
# with a tied node.
#
# Unit load on node 3, DOF 1: nodes 1-3 move together, u = 1/3. Node 5 is tied to node 4,
# and node 4 DOF 1 is fixed: node 5 DOF 1 stays 0 under its load.
source common.tcl
free_node 1 0 0
free_node 2 1 0
free_node 3 2 0
equalDOF 1 2 1 2
equalDOF 2 3 1 2
free_node 4 0 1
free_node 5 1 1
equalDOF 4 5 1 2
fix 4 1 0
pattern Plain 1 Linear {
  load 3 1.0 0.0
  load 5 1.0 0.0
}
solve
report {1 2 3 4 5} {{1 1 0.333333333333} {3 1 0.333333333333} {5 1 0.0}}
