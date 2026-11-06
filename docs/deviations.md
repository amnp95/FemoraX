# Deviations from OpenSees

Every deliberate difference between FemoraX and the OpenSees class it ports, and every
tolerance that is looser than the default. A port is not finished until its deviations are
listed here.

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
  velocity fields, the largest difference is 2.3e-15 of the largest entry; 246 of 1,536
  values are bit-identical. Internal force, tangent, initial tangent and both mass
  matrices are bit-identical (47,094 of 48,384 values in total).
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

### Constraint handling — refuse what OpenSees' Transformation handler silently mishandles
- OpenSees: `SRC/analysis/handler/TransformationConstraintHandler.cpp`,
  `SRC/analysis/dof_grp/TransformationDOF_Group.cpp`, commit 955ceb0
- Difference: FemoraX numbers DOFs from the constraints as written (union-find; a group
  with a fixed DOF is fixed). It also emulates the handler (first MP per constrained node,
  `addSP_Constraint` position arithmetic, `doneID` in ascending tag order) and, where the
  two disagree, refuses the model with a message saying what OpenSees would do. Cases
  where they agree run unchanged, including chains in ascending tag order over all DOFs
  and fixed DOFs inside a fixed group (laminar bounds that include a fixed base).
- Refused (OpenSees gives a different, physically wrong answer without a warning):
  - a node tied to two retained nodes (OpenSees applies only the first MP);
  - a chain whose intermediate node has a larger tag than the node tied to it (OpenSees
    reads its IDs before they are set and fixes the DOFs);
  - a partial chain (the retained node's transformed ID layout no longer matches its
    nodal DOFs; OpenSees crosses DOFs);
  - a fix on a constrained DOF whose retained DOF is free (OpenSees loses the fix).
- Why: accepting them would give answers different from OpenSees (rule 1); copying
  OpenSees would give answers different from the model. Neither occurs in Femora's
  laminar boundary (a star at each elevation, base excluded).
- Effect on results: none for accepted models: FemoraX's DOF classes equal OpenSees' on all
  accepted cases, also after renumbering.
- Test: `tests/unit/test_dof_numbering.cpp` against the `dof_numbering` driver on 11 cases
  (1x1, 2x2, 3x2 soil columns with corner, edge and face nodes; chains; fixed base in
  the laminar bounds; the refused patterns). The emulation itself reproduces OpenSees in
  all 9 cases it applies to, including the 3 where OpenSees is wrong.
- Known schema gap: separate one-DOF `equalDOF` commands on the same node pair are one MP
  per command in OpenSees (only the first applies), but schema v0 rows cannot show command
  boundaries, so FemoraX ties all the DOFs. The `split_equal_dof_commands` case records it.
  Femora writes one command per pair with all DOFs, so its models are not affected.
