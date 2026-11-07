# OpenSees issues found while building FemoraX

Issues 1-4: OpenSees TransformationConstraintHandler, four silently wrong constraint
patterns. Issue 5: Newmark with stdBrick `-lumped`. Issue 6: SSPbrick's mass and body force
on distorted elements (both below the table).

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

### 5. Newmark with stdBrick `-lumped`

`5_newmark_lumped_brick.tcl` (3D, one brick, no constraint involved). `Brick::formInertiaTerms`
computes the residual's inertia force `M*a` with the consistent mass whatever `massType`
says; only `getMass()`, which the Newmark tangent uses, honours `-lumped`. With Newton the
analysis silently converges to the consistent-mass answer; with Linear (one solve per step)
the step solves `(c1*K + c3*M_L) dU = R(M_C)`, which is neither model, and the error grows
as `dt` shrinks (51%, 70%, 90% of the peak at dt = 0.004, 0.001, 0.00025). The same mix
appears in CentralDifference. Suggested fix: use the mass of `massType` in
`formInertiaTerms`' residual as well (`resid += mass * accel` with the assembled `mass`).

### 6. SSPbrick mass and body force on distorted elements

`6_sspbrick_distorted_mass.tcl` (3D, one element, no analysis: `eleForce` at zero
displacement). `SSPbrick::getMass` and the body force of `getResistingForce` use, at node i,
`J[0] * (1 + (J[1] xi_i + ... + J[7] + J[8] + J[9])/3 + (...)/9 + (...)/27)`: the bracket
adds 1 to terms that have units of volume (the `J[k]` are triple products of the nodal
coordinate moments). It is exact when `J[4..19]` vanish, i.e. for parallelepipeds (all
Femora grids); for other shapes the element's total mass and body force are not `rho*V`
and `b*V` (`V` = SSPbrick's own `mVol`, equal to stdBrick's volume): 0.46% off with nodes
moved by up to 13% of the edge, 2.1% at 26%. Suggested fix: integrate `N_i det J` (the
row sums of the consistent mass), or use `J[0] + (...)/3 + ...` with the bracket's terms
scaled to the same units.

`0_control_correct_patterns.tcl` shows the related patterns the handler resolves correctly
(a full chain in ascending tag order; a fixed retained node with a tied node).

Suggested fixes: warn or error on patterns 1-4; for 2-3, resolve chains to the final
retained node before building the DOF groups; for 4, either apply the fix to the retained
node's group or reject it.

## Output at commit 955ceb0

### 6_sspbrick_distorted_mass.tcl
```
amp 0.0  total body force z: SSPbrick 10.000000000000  stdBrick (= b*V) 10.000000000000  relative 1.776e-16
amp 1.0  total body force z: SSPbrick 10.180534584538  stdBrick (= b*V) 10.133474166667  relative 4.644e-03
WRONG: amp 1.0: SSPbrick's body force (and mass) total is 4.644e-03 off b*V
amp 2.0  total body force z: SSPbrick 10.317029576158  stdBrick (= b*V) 10.107126666667  relative 2.077e-02
WRONG: amp 2.0: SSPbrick's body force (and mass) total is 2.077e-02 off b*V
```

### 5_newmark_lumped_brick.tcl
```
dt 0.004    peak top uz: consistent -8.303442e-03  true lumped -8.312953e-03  -lumped Newton -8.303442e-03  -lumped Linear -4.051145e-03
WRONG: dt 0.004: -lumped with Newton gives the consistent-mass answer -8.303442e-03, not the lumped-mass one -8.312953e-03
WRONG: dt 0.004: -lumped with Linear gives -4.051145e-03, 51.3% from the lumped-mass answer -8.312953e-03
dt 0.001    peak top uz: consistent -8.331352e-03  true lumped -8.332053e-03  -lumped Newton -8.331352e-03  -lumped Linear -2.471102e-03
WRONG: dt 0.001: -lumped with Linear gives -2.471102e-03, 70.3% from the lumped-mass answer -8.332053e-03
dt 0.00025  peak top uz: consistent -8.333215e-03  true lumped -8.333255e-03  -lumped Newton -8.333215e-03  -lumped Linear -8.393286e-04
WRONG: dt 0.00025: -lumped with Linear gives -8.393286e-04, 89.9% from the lumped-mass answer -8.333255e-03
```

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
