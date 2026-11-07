# Deviations from OpenSees

FemoraX matches OpenSees where OpenSees is correct and deviates deliberately where we know
better (CLAUDE.md rule 1). Every difference from the OpenSees class a component comes
from, with the reason and the evidence, and every tolerance looser than the default, is
listed here. A component is not finished until its deviations are.

Default tolerances (until a component states otherwise):

- Material strain-path tests: relative 1e-14 on stress and tangent.
- Element tests: relative 1e-12 on force, stiffness and mass.
- System tests (Femora-style models run through both programs, every recorded value):
  relative 1e-12 of the largest magnitude of each response (displacement, velocity,
  acceleration), for every renumbering, layout and backend. Bit-identity is not expected
  even without renumbering: FemoraX sums element contributions in its own order (the
  gather scatter) and sums damping forces point by point (stdBrick entry below), and the
  differences then grow through the time steps. The first system test
  (`tests/system/soil_column`, 10x10x40 column, 4000 steps) shows 4.8e-13 at most
  (acceleration; 1.8e-15 on displacement), with renumbering off or on, on the CPU and the
  GPU. The same column with SSPbrick: 5.8e-13 (acceleration; 1.8e-14 displacement), CPU
  with renumbering off and on, GPU.
- Implicit system tests (Newmark, LoadControl): relative 1e-8, for the same reason plus the
  iterative solves (FemoraX's Krylov solves stop at a relative residual of 1e-12, OpenSees
  solves directly; the acceleration, c3 dU with c3 = 1/(beta dt^2), amplifies the
  difference most). Observed: 2x2x10 column, all four gravity/algorithm variants, 6.5e-10;
  10x10x40 column (9840 equations, 500 steps): implicit-dynamic gravity and Linear,
  4.2e-9 (acceleration; 8e-12 displacement), CPU and GPU, renumbering off and on; static
  gravity and Newton, 5.6e-9 (acceleration; 8.8e-12 displacement), CPU and GPU. With
  SSPbrick (lumped mass): 2x2x10, all four variants, 1.4e-10; 10x10x40, dynamic gravity
  and Linear 2.5e-10, static gravity and Newton 2.9e-10 (acceleration; 4.4e-12
  displacement), CPU (renumbering off and on) and GPU.

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

### CentralDifference with lumped stdBrick — lumped mass in the residual too
- OpenSees: `SRC/analysis/integrator/CentralDifference.cpp`,
  `SRC/analysis/integrator/TransientIntegrator.cpp` (`formEleResidual`),
  `SRC/element/brick/Brick.cpp` (`formInertiaTerms`), commit 955ceb0
- Difference: OpenSees forms the CentralDifference residual with
  `getResistingForceIncInertia`, and `Brick::formInertiaTerms` computes the inertia force
  `M*a` with the consistent mass whatever `massType` says; the system matrix
  `c3*M + c2*C` uses the lumped mass (`getMass`). The step solved is
  `M_L*U/dt^2 = P - K*u + M_C*(2u - u_prev)/dt^2 - ...`, which carries the spurious term
  `(M_L - M_C)*(2u - u_prev)/dt^2`: an artificial stiffness that grows as `1/dt^2`.
  FemoraX uses the lumped mass in both places (textbook central difference with a lumped
  mass, the same mass ExplicitDifference uses).
- Why: OpenSees' result is wrong and depends on the step size. On a 2x2x10 column under
  gravity and a base motion, alphaM damping only, the peak settlement of the top is
  1.620e-3 for ExplicitDifference (FemoraX and OpenSees agree to 7e-15) and for FemoraX
  CentralDifference (1.620e-3 at dt = 2.5e-4, 1.620e-3 at 1.25e-4), but 2.73e-4 for
  OpenSees CentralDifference at dt = 2.5e-4 and 8.19e-5 at 1.25e-4. The closed form of
  the first steps confirms the cause: after step 1 (identical in both), the step-2
  displacement next to the fixed base differs by exactly `u1/3`, the consistent-mass
  coupling to the base nodes.
- Effect on results: FemoraX does not match OpenSees for CentralDifference with lumped
  stdBrick elements; it matches ExplicitDifference and the physics. Consistent-mass models
  are refused in explicit stages anyway.
- Also in Newmark: see the next entry.
- Test: `tests/system/soil_column/make_models.py <dir> 2 2 10 1.0 2.5e-4 --integrator
  CentralDifference` (and the same with `ExplicitDifference --mass-damping-only`), compared
  with `compare.py`.

### Newmark with lumped stdBrick — a true lumped mass (decided 2026-11-06, milestone 1b)
- OpenSees: `SRC/analysis/integrator/Newmark.cpp` (`formEleTangent`: `addMtoTang`, which
  uses `getMass()`), `TransientIntegrator::formEleResidual` (`getResistingForceIncInertia`),
  `SRC/element/brick/Brick.cpp` (`formInertiaTerms`), commit 955ceb0
- Difference: OpenSees honours `-lumped` only in the tangent; the residual's inertia force
  uses the consistent mass. With Newton the analysis silently converges to the
  consistent-mass answer; with Linear each step solves `(c1*K + c3*M_L) dU = R(M_C)`,
  which is neither model. FemoraX uses a true lumped mass (in residual and tangent alike)
  when the model asks for lumped mass, as its explicit integrators do; consistent mass
  is the consistent mass in both places, as in OpenSees.
- Why: the requested model is the lumped one, and OpenSees' Linear result is wrong and
  depends on the step size.
- Evidence: 2x2x10 column, gravity plus the x sweep, Newmark 0.5/0.25, no damping; largest
  top x displacement: consistent with Newton or Linear 1.835e-3 (dt 2.5e-4); `-lumped`
  with Newton the same to 1e-9; `-lumped` with Linear 2.69e-4, 7.13e-4, 1.42e-3 at dt =
  2.5e-4, 1e-3, 5e-3 (97%, 77%, 28% off); a true lumped mass (massless bricks + nodal
  masses) 1.856e-3, 2% from consistent, a genuine modelling difference. Standalone
  reproduction: `tests/reference/opensees_issues/5_newmark_lumped_brick.tcl`.
- Effect on results: models with lumped stdBrick in implicit stages differ from OpenSees
  (about 2% here against `-lumped` with Newton). Femora's default is consistent mass
  (`lumped=False`), and the system tests against OpenSees use consistent mass.
- Test: to come with the Newmark step (lumped against OpenSees with nodal masses and
  massless elements, which is the same model).

### Stage end time — the Tcl loop's rounding of getTime
- OpenSees: `SRC/tcl/commands.cpp` (`getTime` prints the domain time with `"%f"`);
  Femora: `src/femora/components/analysis/analysis.py` (`while {[getTime] < %f}`)
- Difference: none intended; recorded because it is easy to get wrong. A stage with
  `final_time` runs while `time < final_time` with both rounded to six decimals, as the
  Tcl loop compares them. With the accumulated time `0.9999999999999176` after 4000 steps
  of 2.5e-4, the loop prints `1.000000` and stops, so a 1 s stage has 4000 steps, not 4001.
- Test: `tests/system/soil_column` (both programs record the same 4000 steps).

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

### SSPbrick — Kstab applied matrix-free, damping summed term by term
- OpenSees: `SRC/element/UWelements/SSPbrick.cpp` with `SRC/element/Element.cpp`
  (`getRayleighDampingForces`), commit 955ceb0
- Difference: OpenSees forms the 24 x 24 stabilization `Kstab = Mben^T interior Mben` once
  and multiplies it by the displacements; FemoraX stores the hourglass vectors and the
  upper triangle of the 12 x 12 `interior` (element constants) and applies
  `Mben^T (interior (Mben v))`, so `Kstab` is exactly symmetric (OpenSees' product
  `KuT FeCFe^-1 FeCFhg` is symmetric to rounding only). The 3 x 3 and 9 x 9 inverses are
  an LU with partial pivoting (OpenSees: LAPACK). Damping forces are summed term by term,
  as for stdBrick: `(betaK + betaK0) Dt + betaKc Dc` through the point (OpenSees'
  `getInitialStiff` returns the tangent, so `betaK0` uses `Dt`, kept), `(betaK + betaK0 +
  betaKc) Kstab v`, and `alphaM` times the lumped mass. As for stdBrick, there is no
  `eleLoad` body-force path (`applyLoad`, `appliedB`; schema v0 has no `eleLoad`).
- Why: performance rule 3a (no full element matrix in one thread; the explicit path forms
  no matrices) and the matrix-free implicit operator.
- Effect on results: rounding only. Against OpenSees on 4 geometries x 2 materials x 4
  fields: mass bit-identical; tangent and initial tangent within 9.0e-15 of the largest
  entry, force within 1.2e-14, damping within 3.2e-14; 49,060 of 81,408 values
  bit-identical.
- Test: `tests/unit/test_ssp_brick_reference.cpp` (gate: relative 1e-12, the element
  default); the system tests above.

### SSPbrick — OpenSees' mass and body force of distorted elements (ported as is)
- OpenSees: `SRC/element/UWelements/SSPbrick.cpp` (`getMass`, `getResistingForce`),
  commit 955ceb0
- Difference: none; FemoraX reproduces it (`ssp_detail::poly_bracket`). Recorded because
  it is wrong for some meshes: node i's mass and body-force weight are `J[0] * (1 +
  (... + J[7] + J[8] + J[9])/3 + (...)/9 + (...)/27)`, which adds 1 to terms with units of
  volume. It is exact when `J[4..19]` vanish, i.e. for parallelepipeds (every Femora grid),
  and wrong otherwise: the element's total mass and body force differ from `rho V` and
  `b V` (`V`, SSPbrick's own volume, equals stdBrick's): 3.6e-3 relative with nodes moved
  by up to 15% of the edge, 0.46% / 2.1% in the OpenSees example at 13% / 26%; the nodal
  split differs from the consistent-mass row sums by up to 20%. The stable step uses the
  same masses.
- Why kept: matching OpenSees until it is decided; it changes nothing for Femora's
  structured grids. A fix (integrating `N_i det J`, as stdBrick's lumped mass does) is a
  one-function change in the element.
- Effect on results: none for parallelepiped elements; for distorted ones, the mass and
  gravity load errors above, as in OpenSees.
- Test: `SSPbrick.MassIsDiagonalAndExactForParallelepipeds` (exact for boxes and sheared
  parallelepipeds; reports the distorted case); the standalone OpenSees example
  `tests/reference/opensees_issues/6_sspbrick_distorted_mass.tcl`.

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
