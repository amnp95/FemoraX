# Deviations from OpenSees

FemoraX matches OpenSees where OpenSees is correct and deviates deliberately where we know
better (CLAUDE.md rule 1). Every difference from the OpenSees class a component comes
from, with the reason and the evidence, and every tolerance looser than the default, is
listed here. A component is not finished until its deviations are.

Default tolerances (until a component states otherwise):

- Material strain-path tests: relative 1e-14 on stress and tangent.
- Element tests: relative 1e-12 on force, stiffness and mass.
- System tests: to be set with the first system test, with the reason.

## Entry format

```
### <Component> — <one-line summary>
- OpenSees: <file>, commit <hash>
- Difference: <what FemoraX does differently>
- Why: <reason>
- Effect on results: <none / roundoff / bounded by ...>
- Test: <test that covers it>
```

## Entries

### stdBrick — Rayleigh damping forces summed point by point
- OpenSees: `SRC/element/Element.cpp` (`getRayleighDampingForces`) with
  `SRC/element/brick/Brick.cpp`, commit 955ceb0
- Difference: OpenSees forms the element damping matrix
  `C = alphaM*M + betaK*K + betaK0*K0 + betaKc*Kc` and multiplies it by the nodal
  velocities. FemoraX never forms `C`: at each Gauss point it pushes the damping stress
  `(betaK*Dt + betaK0*D0 + betaKc*Dc) * (B v)` through `B^T`, and adds
  `alphaM * rho * N_j * dvol * v` per point (consistent mass: `v` interpolated at the point;
  lumped mass: the node's `v`).
- Why: the explicit path must not form matrices (docs/design.md, "Explicit analysis";
  CLAUDE.md, Performance), and the matrix-free form is what the GPU kernels run.
- Effect on results: rounding only. Against OpenSees on 4 geometries x 2 materials x 4
  velocity fields, the largest difference is 2.0e-15 of the largest entry; 271 of 1,536
  values are bit-identical. Internal force, tangent, initial tangent and both mass
  matrices are bit-identical (47,119 of 48,384 values in total).
- Test: `tests/unit/test_std_brick_reference.cpp` (gate: relative 1e-12, the element
  default).

### stdBrick — features not ported
- OpenSees: `SRC/element/brick/Brick.cpp`, commit 955ceb0
- Difference: no `eleLoad` body-force path (`applyLoad`, `appliedB`: schema v0 has no
  `eleLoad`; the constant body force `b` is applied every step, as OpenSees does without
  one), and no element `Damping` objects (`-damp`, `theDamping`).
- Why: Femora's milestone-1 models use neither. Schema v0 has no field for either, so
  strict reading rejects any file that tries to carry them.
- Effect on results: none for supported models.
- Test: none specific (strict reading, `tests/io`, covers unknown items in general).

### Constraint handling — refuse what OpenSees' Transformation handler silently gets wrong
- OpenSees: `SRC/analysis/handler/TransformationConstraintHandler.cpp`,
  `SRC/analysis/dof_grp/TransformationDOF_Group.cpp`, commit 955ceb0
- Difference: FemoraX numbers DOFs from the constraints as written (union-find; a group
  with a fixed DOF is fixed). It also emulates the handler (first MP per constrained node,
  `addSP_Constraint` position arithmetic, `doneID` in ascending tag order). Where the two
  agree, the model runs, including chains in ascending tag order over all DOFs and fixed
  DOFs inside a fixed group (laminar bounds that include a fixed base). Where they
  disagree, OpenSees is wrong, and FemoraX refuses the model with a message that says what
  is wrong and how to fix it:
  1. a node constrained by more than one `equalDOF` command (OpenSees applies only the
     first; command boundaries come from schema 0.2's `command` field, or from distinct
     node pairs without it);
  2. a chain whose intermediate node has a larger tag (OpenSees fixes the tied node);
  3. a partial chain (OpenSees connects DOFs crosswise);
  4. a fix on a tied DOF whose retained DOF is free (OpenSees loses the fix).
- Why: the four patterns are OpenSees bugs: its results contradict the constraints, without
  a warning. FemoraX does not copy a wrong answer (rule 1) and does not silently give a
  different one either; it asks for a model both programs read the same way. Femora's
  laminar boundary (a star at each elevation, base excluded) never produces them.
- Evidence: `tests/reference/opensees_issues/` holds a standalone OpenSees script per
  pattern with hand-computed correct displacements; all four reproduce at commit 955ceb0
  (for example, issue 2 leaves a loaded, free node at zero displacement), and the control
  script shows the related correct patterns. Prepared for reporting to the OpenSees
  developers.
- Effect on results: none for accepted models: FemoraX's DOF classes equal OpenSees' on all
  accepted cases, also after renumbering.
- Test: `tests/unit/test_dof_numbering.cpp` against the `dof_numbering` driver on 11 cases
  (1x1, 2x2, 3x2 soil columns with corner, edge and face nodes; chains; a fixed base in the
  laminar bounds; the refused patterns, each checked for its specific message). The
  emulation reproduces OpenSees in all 9 cases it applies to, including the 3 where
  OpenSees is wrong.
