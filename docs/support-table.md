# Supported components

A component is **supported** only when its port and its verification test against OpenSees
pass (CLAUDE.md rule 1). Anything not marked supported makes the model reader stop with a
message that names it and suggests the OpenSees backend.

Status: **supported**, **in progress** (milestone named), or **planned**.

## Elements

| Femora / OpenSees | Status | Notes |
|---|---|---|
| stdBrick | **supported** | consistent and lumped mass; force, tangent and masses bit-identical to OpenSees `955ceb0`, damping forces within 2.3e-15 (matrix-free, docs/deviations.md). Explicit force kernels (flat and team layouts; gather, atomic and colored scatter) verified against the physics. Lumped mass runs end to end in explicit stages; system test against OpenSees within 4.8e-13 (`tests/system/soil_column`); implicit stages (1b) within 5.6e-9 |
| SSPbrick | **supported** | port of OpenSees `SSPbrick.cpp` at `955ceb0` through the element registry plus the generic element-term extension (docs/design.md). Against OpenSees: force, tangent, initial tangent within 1.2e-14, damping within 3.2e-14, mass within 2e-16 with OpenSees' mass formula (`test_ssp_brick_reference`, 4 geometries x 2 materials x 4 fields). Every layout, scatter and geometry mode of the force kernels and the matrix-free operator verified against the host physics. System tests against OpenSees on the 10x10x40 column: explicit 5.8e-13, implicit (dynamic gravity + Linear, static gravity + Newton) 2.9e-10, CPU and GPU. FemoraX uses the exact lumped mass and body force (OpenSees' formula is exact for parallelepipeds only; docs/deviations.md) |
| PML3D, SSPquad, ElasticBeamColumn, DispBeamColumn, ForceBeamColumn, Truss, ASDEmbeddedNodeElement3D, embedded node contact, ZeroLengthContactASDimplex | planned | |

## nD materials

| Femora / OpenSees | Status | Notes |
|---|---|---|
| ElasticIsotropic | **supported** | `ThreeDimensional` form; bit-identical to OpenSees `955ceb0` on 76,650 stress/tangent values (`test_elastic_isotropic_reference`) |
| PressureDependMultiYield | in progress (milestone 2): material verified | 3D port of OpenSees `955ceb0` (`physics/materials/pressure_depend_multi_yield.hpp`); bit-identical to OpenSees on 208,152 stress and tangent values along 32 strain paths (Femora's three soils and a user backbone; stage switches, discarded trials; `test_pdmy_reference`). In the kernels (committed and trial state, commit as a swap, the stage action, element output) and run end to end against stored OpenSees output: single elements and SSPbrick columns of 4 and 25 elements through Femora's staging (explicit shaking within 6.4e-13, implicit within 2.8e-7 for stdBrick and 1.9e-5 for SSPbrick: rounding-amplified plastic decisions, docs/deviations.md), and Femora's own example column with its retry recovery (fields within 1.1e-11 on the CPU, 5.6e-5 on the GPU). Next: the compacted corrector kernel. `pa` per material (docs/deviations.md) |
| PressureIndependMultiYield | **supported** (milestone 2) | 3D port of OpenSees `955ceb0` (`physics/materials/pressure_independ_multi_yield.hpp`); bit-identical to OpenSees on 201,516 stress and tangent values along 34 strain paths (Femora's basin clay, a frictional material, user backbones; `test_pimy_reference`); in both elements, both force paths (fused and compacted corrector), stored OpenSees comparisons of single elements and 4-element columns (explicit at rounding; with ModifiedNewton re-equilibration a rounding-decided reversal at the switch, docs/deviations.md). Up to 40 surfaces; `betaKinit` damping and ModifiedNewton `-initial` refused with it (docs/deviations.md) |
| DruckerPrager, J2CyclicBoundingSurface, LinearElasticGGmax | planned | |

## Uniaxial materials and sections

| Femora / OpenSees | Status | Notes |
|---|---|---|
| all | planned | milestone 3 onward |

## Constraints

| Femora / OpenSees | Status | Notes |
|---|---|---|
| fix | in progress (1a) | |
| equalDOF | in progress (1a) | |
| laminar boundary | in progress (1a) | Femora exports it as equalDOF pairs |
| rigidDiaphragm, rigidLink | planned | |

## Damping

| Femora / OpenSees | Status | Notes |
|---|---|---|
| Rayleigh, frequency Rayleigh (per region) | in progress (1a) | |
| Uniform (`damping Uniform`, per region with `-damp`) | supported (schema 0.7) | stdBrick and SSPbrick with ElasticIsotropic, PressureIndependMultiYield and PressureDependMultiYield (20 surfaces); explicit and implicit stages; `-activateTime`, `-deactivateTime`; FemoraX's fit by default, OpenSees' with `fit = "opensees"` (docs/deviations.md); not yet: `-fact`, multi-yield materials with more than 20 surfaces, damped DRM-layer elements |
| modal, secant stiffness proportional | not planned | Uniform damping covers them (docs/design.md) |

## Patterns and time series

| Femora / OpenSees | Status | Notes |
|---|---|---|
| UniformExcitation | in progress (1a) | runs on the device; `-accel`, `-fact`, `-vel0`; matches OpenSees in the system test |
| Path time series | in progress (1a) | uniform `-dt` and `-time` forms, evaluated on the device exactly as PathSeries / PathTimeSeries (`test_ground_motion`); `-time` with `use_last` or `prepend_zero` refused (OpenSees ignores them) |
| H5DRM | in progress (milestone 4) | schema 0.6; runs in explicit stages, and implicit stages apply it too (not yet verified there) (OpenSees H5DRMLoadPattern: nearest-station matching with the coordinate transformation, linear interpolation, the file's downward z axis, forces from the DRM elements' own K and M). The DRM layer must be linear elastic; factor other than 1 refused (OpenSees never applies it). Histories held whole on the device for now (streaming: next step). Verified: a box inside a DRM layer reproduces a given free field to 2e-14 |
| Plain, MultipleSupport | planned | |

## Integrators

| Femora / OpenSees | Status | Notes |
|---|---|---|
| CentralDifference | in progress (1a) | lumped mass, alphaM damping only (others need the 1b solver); deliberately differs from OpenSees for lumped stdBrick (docs/deviations.md) |
| ExplicitDifference | in progress (1a) | all four Rayleigh terms; stable-step check per stage; matches OpenSees within 4.8e-13 on every recorded value (`tests/system/soil_column`) |
| Newmark | in progress (1b) | form D, matrix-free with CG or GMRES (chosen from the tangent symmetry) and Jacobi; Linear, Newton, ModifiedNewton; NormUnbalance, NormDispIncr, EnergyIncr; equals the OpenSees recurrence with exact solves to 1e-8 (`test_implicit`); against OpenSees within 5.6e-9 on the 10x10x40 column, CPU and GPU (`system.implicit_vs_opensees`, `make_implicit.py`) |
| LoadControl (Static) | in progress (1b) | adaptive increment as OpenSees; static gravity equals the exact solution (`test_implicit`) |

## Recorders

| Femora / OpenSees | Status | Notes |
|---|---|---|
| Node | in progress (1a) | disp, vel, accel; OpenSees recording rule and text format; device gather, asynchronous copies, writer thread; `-timeSeries` refused for now |
| VTKHDF | in progress (1a) | `disp vel accel stress3D6 strain3D6` (element output since milestone 2: 4-element PressureDependMultiYield column within 3.4e-14 of OpenSees' file, explicit), whole model or an element list; OpenSees' layout dataset for dataset (types, chunks, offsets, tags), `femora.results` returns the same arrays as for OpenSees' file (`system.vtkhdf_vs_opensees`, `compare_vtkhdf.py`); FemoraX writes with the system HDF5 (1.10 file format objects), OpenSees with its own; optional single-precision responses (`femorax run --vtkhdf-precision single`, a FemoraX option: float32 response datasets, the rest unchanged; `system.vtkhdf_single_precision`) |
| MPCO | planned | |
