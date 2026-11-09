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

- Nonlinear system tests (PressureDependMultiYield; stored OpenSees output,
  `tests/system/soil_column/nonlinear_check.sh`): explicit stages relative 1e-12 (observed
  1.4e-13 and 2.4e-13 on one SSPbrick of dense and of loose sand, 1000 steps with plastic
  flow); implicit stages per case from the first runs: one stdBrick 1e-6 (observed
  2.8e-7), one SSPbrick 5e-5 (observed 1.9e-5, acceleration; displacement 4.7e-7). The
  implicit cases agree to 1e-16 for the first 45 shaking steps, then a rounding-level
  difference changes a plastic decision and the Newton iterates converge, within the
  NormDispIncr 1e-10 test, to slightly different answers. It is not FemoraX-specific: a
  host replica of OpenSees' algorithm with FemoraX's physics and exact 3 x 3 solves
  differs from itself by 4.7e-7 (SSPbrick) and 1.8e-7 (stdBrick) in displacement when only
  the pivoting of the linear solve changes, and from OpenSees by 1.8e-7 and 2.5e-7. The
  Newton iteration counts agree step by step for the stdBrick case (482 in 100 steps) and
  differ in one step of the SSPbrick case (95: 6 against 5). The explicit cases, with no
  linear solve, stay at rounding: the material, state, commit and stage logic are those of
  OpenSees. On the GPU (fused multiply-adds, CUDA's pow, exp and log) the same rounding
  sensitivity changes plastic decisions in every case: explicit 7.1e-7 and 1.2e-6,
  implicit 1.9e-4 (stdBrick) and 7.3e-5 (SSPbrick); tolerances 5e-6 (explicit), 5e-4 and
  2e-4 (implicit). The Gauss-point paths (test_pdmy_reference, DeviceBackend) separate
  rounding from branch flips: 31 of 32 within 2.3e-11, one flipped.

- Femora's nonlinear soil column (its example `nonlinear_layered_soil_column.py`, exported
  by Femora unchanged, on the first 2 s of shaking, 1.9-3.9 s of FrequencySweep; stored
  OpenSees output, `tests/system/femora_example/femora_example_check.sh`): retry events
  equal and in order (OpenSees and FemoraX both retry re-equilibration steps 4 and 5 once);
  displacement, stress3D6 and strain3D6 relative to their largest values, and the surface
  PGA and 5% response spectrum (100 periods, 0.02-5 s) relative to their peaks: 1e-9 on
  host backends (observed 4.2e-13, 9.2e-13, 1.1e-11; 6.9e-13 and 1.8e-12), 5e-4 on GPUs
  (observed 3.0e-6, 8.7e-6, 5.6e-5; 2.1e-6 and 2.8e-5: fused multiply-adds change plastic
  decisions, which the implicit steps carry forward). Acceleration and velocity histories
  are reported, not compared step by step: the shaking stage converges to NormDispIncr
  1e-5 m, which allows up to 40 m/s^2 of acceleration at dt = 0.001, so step values carry
  that slack (on the CPU they still agree to 2.6e-11 and 1.3e-12; on the GPU 7.0e-3 and
  9.9e-5); the PGA and the spectrum are what site response uses.

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

### PressureDependMultiYield — per-material pAtm, and the port's structure
- OpenSees: `SRC/material/nD/soil/PressureDependMultiYield.cpp` (with `T2Vector.cpp`,
  `MultiYieldSurface.cpp`), commit 955ceb0
- Difference:
  - `pAtm` (Femora's `pa`): OpenSees keeps one static `pAtm`, set by each constructor, so
    at run time every PressureDependMultiYield material uses the last constructed one's
    (`isCriticalState`, lines 2047-2051; the surfaces are set up with each material's own).
    FemoraX uses each material's own `pa` everywhere. Femora's example gives every soil
    the same `pa`, so it makes no difference there.
  - The per-material static arrays are Params and the static workspaces are locals; every
    workspace was checked to be written before it is read within one call, so none carries
    a value between calls or between points (OpenSees' would carry it between points if
    any function read one first; none does).
  - `elast2Plast` (the switch to the plastic stage) changes the committed state on the
    first `getStress`/`getTangent` after the switch, and a revert does not undo it. FemoraX
    runs it inside the update on a copy of the committed state, and the result becomes
    committed at the commit. Being a pure function of the committed state, it gives the
    same values, and a reverted step that is redone sees the same input.
  - Recursion (`stressCorrection`, `deviatorScaling`) is a loop; OpenSees' `exit()` on
    impossible states (`secondOrderEqn`'s complex or negative roots, "error in surface
    motion") makes the stress NaN instead, so the step fails and the analysis can retry.
    A conversion of an out-of-range double to int in `setSubStrainRate` (undefined in C++;
    only at zero confinement) is clamped. `strainPTOcta` starts at 0 (OpenSees leaves it
    unset if no pair of surfaces brackets the phase-transformation ratio).
  - Load stage 2 (elastic with the initial pressure, for u-p elements) is not ported:
    schema 0.4 has stages 0 and 1.
- Why: correctness (`pa`), GPU kernels without static or global state (CLAUDE.md rule 2),
  and a failed step instead of a stopped program.
- Effect on results: none on the verification data: 208,152 of 208,152 stress and tangent
  values bit-identical to OpenSees along 32 strain paths (Femora's dense Ottawa, loose
  Ottawa and dense Monterey, and a user G/Gmax backbone; consolidation in the elastic
  stage, the switch, monotonic and cyclic shear from 1e-5 to 1e-2, reversals, confinement
  changes, extension, switching back and forth, discarded trials). Different `pa` values
  across materials change the critical-state terms. On the GPU (nvcc fuses multiply-adds;
  CUDA's pow, exp and log are not glibc's) the same paths agree with the host to 2.3e-11
  or better in 31 of 32 files; in one (loose Ottawa, random reversals near liquefaction) a
  different plastic branch is taken and the result differs by 3.2e-6 of the largest value
  (`PressureDependMultiYieldReference.DeviceBackend` reports both).
- Test: `tests/unit/test_pdmy_reference.cpp` (gate: relative 1e-14, the material default;
  observed: bit-identical), `tests/unit/test_pdmy.cpp`; the standalone OpenSees example
  `tests/reference/opensees_issues/7_pdmy_global_patm.tcl` (an unused second material with
  another `pa` changes the first one's stress by 4%).

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
  fields (with OpenSees' mass formula, entry below): mass within 2.0e-16, tangent and
  initial tangent within 9.0e-15 of the largest entry, force within 1.2e-14, damping
  within 3.2e-14; 49,007 of 81,408 values bit-identical.
- Test: `tests/unit/test_ssp_brick_reference.cpp` (gate: relative 1e-12, the element
  default); the system tests above.

### SSPbrick — exact lumped mass and body force (decided 2026-11-07, milestone 2)
- OpenSees: `SRC/element/UWelements/SSPbrick.cpp` (`getMass`, `getResistingForce`),
  commit 955ceb0
- Difference: OpenSees gives node i the mass `density * J[0] * (1 + (... + J[7] + J[8] +
  J[9])/3 + (...)/9 + (...)/27)` and the body force `b` times the same weight (`polyJac`).
  The bracket adds 1 to terms with units of volume, so it is exact only when `J[4..19]`
  vanish, i.e. for parallelepipeds. FemoraX weights node i by the integral of `N_i` over
  the element (2 x 2 x 2 Gauss on the trilinear map, exact; the same points as stdBrick's
  lumped mass): mass `rho * w_i`, body force `b * w_i`, so the totals are `rho V` and `b V`
  for every shape. OpenSees' formula stays as a test-only variant
  (`SSPbrick<Mat, SSPbrickMass::OpenSees>`).
- Why: OpenSees' mass and gravity load are wrong for distorted elements: the totals are
  off by 3.6e-4, 3.6e-3 and 1.7e-2 relative with nodes moved by up to 5%, 15% and 30% of
  the edge (0.46% and 2.1% in the OpenSees example at 13% and 26%), and the nodal split
  differs from the consistent-mass row sums by up to 20%. The stable step uses the same
  masses.
- Effect on results: rounding only for parallelepiped elements (every Femora grid):
  against OpenSees on the cube and soil-column geometries the production element's mass
  is within 6.6e-16, force within 5.4e-16, damping within 4.0e-15, tangent bit-identical.
  On distorted elements FemoraX's mass and gravity load are the exact ones, OpenSees' are
  not (above).
- Test: `SSPbrick.TotalMassAndBodyForceAreRhoVAndBVOnDistortedElements` (totals rho V and
  b V to 1e-13, nodal masses equal stdBrick's lumped mass), `SSPbrickReference.*` (every
  geometry through the OpenSees-formula variant; the production element on the
  parallelepipeds); the standalone OpenSees example
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
