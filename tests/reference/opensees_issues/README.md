# OpenSees TransformationConstraintHandler: four silently wrong constraint patterns

Standalone examples for the OpenSees developers. Each is a few nodes in 2D, every free
node held by unit springs (`zeroLength`) to its own fixed ground node, one unit load, and
a check against the displacement the constraints imply (worked out by hand in each
script). Found while building FemoraX's DOF numbering and confirmed with OpenSees at commit
`955ceb0a55f7b5764856ea82fc9a9cc7f07a1eea` (2026-10-31), built from source.

Run one with `OpenSees 1_two_mp_constraints_on_one_node.tcl`, or all with `./run.sh
[path/to/OpenSees]`. Each script prints every node's displacement and DOF equation numbers
(`nodeDOFs`), then either `correct` or a `WRONG:` line per wrong value. None of the four
produces a warning.

| # | Pattern | What OpenSees does | Where |
|---|---|---|---|
| 1 | a node constrained by two `equalDOF` commands | keeps only the first MP_Constraint that names the node; the second tie is ignored | `TransformationConstraintHandler::handle` (`constrainedNodesMP.getLocation` returns the first) |
| 2 | a chain whose intermediate node has a larger tag than the node tied to it | the node tied to it becomes fixed | `TransformationDOF_Group::doneID` runs in ascending tag order and copies the intermediate node's IDs before they are set (-1) |
| 3 | a partial chain (the inner tie covers fewer DOFs, or another order) | DOFs get connected crosswise | `doneID` indexes the retained node's transformed ID (`otherID(dof)`), whose layout is not the nodal DOF order |
| 4 | a `fix` on a DOF that an `equalDOF` constrains, retained DOF free | the fix is lost (the node moves with the retained node); the node then also reports zero displacement | `TransformationDOF_Group::addSP_Constraint` puts the -1 at `loc` = number of unconstrained DOFs below the fixed DOF, a retained-node entry that `doneID` overwrites |

`0_control_correct_patterns.tcl` shows the related patterns the handler resolves correctly
(a full chain in ascending tag order; a fixed retained node with a tied node).

Suggested fixes: warn or error on patterns 1-4; for 2-3, resolve chains to the final
retained node before building the DOF groups; for 4, either apply the fix to the retained
node's group or reject it.

## Output at commit 955ceb0

### 1_two_mp_constraints_on_one_node.tcl
```
node 1 disp 0.00000000000000000000 0.00000000000000000000 DOF equations 0 1 
node 2 disp 0.00000000000000000000 0.00000000000000000000 DOF equations 2 3 
node 3 disp 0.00000000000000000000 1.00000000000000000000 DOF equations 4 0 
WRONG: node 2 DOF 2 is 0, the constraints give 0.5
WRONG: node 3 DOF 2 is 1, the constraints give 0.5
```
### 2_chain_with_larger_intermediate_tag.tcl
```
node 1 disp 0.00000000000000000000 0.00000000000000000000 DOF equations 0 1 
node 2 disp 0.00000000000000000000 0.00000000000000000000 DOF equations -1 -1 
node 3 disp 0.00000000000000000000 0.00000000000000000000 DOF equations 0 1 
WRONG: node 1 DOF 1 is 0, the constraints give 0.333333
WRONG: node 2 DOF 1 is 0, the constraints give 0.333333
WRONG: node 3 DOF 1 is 0, the constraints give 0.333333
```
### 3_partial_chain.tcl
```
node 1 disp 0.00000000000000000000 0.00000000000000000000 DOF equations 0 1 
node 2 disp 0.00000000000000000000 0.50000000000000000000 DOF equations 2 0 
node 3 disp 0.50000000000000000000 0.00000000000000000000 DOF equations 2 0 
WRONG: node 1 DOF 1 is 0, the constraints give 0.333333
WRONG: node 2 DOF 1 is 0, the constraints give 0.333333
WRONG: node 3 DOF 1 is 0.5, the constraints give 0.333333
WRONG: node 2 DOF 2 is 0.5, the constraints give 0
```
### 4_fix_on_constrained_dof.tcl
```
node 1 disp 0.50000000000000000000 0.00000000000000000000 DOF equations 0 1 
node 2 disp 0.00000000000000000000 0.00000000000000000000 DOF equations 0 1 
WRONG: node 1 DOF 1 is 0.5, the constraints give 0
```
### 0_control_correct_patterns.tcl
```
node 1 disp 0.33333333333333331483 0.00000000000000000000 DOF equations 0 1 
node 2 disp 0.33333333333333331483 0.00000000000000000000 DOF equations 0 1 
node 3 disp 0.33333333333333331483 0.00000000000000000000 DOF equations 0 1 
node 4 disp 0.00000000000000000000 0.00000000000000000000 DOF equations -1 2 
node 5 disp 0.00000000000000000000 0.00000000000000000000 DOF equations -1 2 
correct
```

## FemoraX

FemoraX refuses all four patterns with a message saying what is wrong and how to fix the
model (`src/compile/dof_numbering.cpp`, `docs/deviations.md` "Constraint handling"), and
accepts the control patterns. Its emulation of the handler reproduces OpenSees on these
patterns (`tests/unit/test_dof_numbering.cpp`).
