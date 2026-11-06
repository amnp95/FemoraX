# Supported components

A component is **supported** only when its port and its verification test against OpenSees
pass (CLAUDE.md rule 1). Anything not marked supported makes the model reader stop with a
message that names it and suggests the OpenSees backend.

Status: **supported**, **in progress** (milestone named), or **planned**.

## Elements

| Femora / OpenSees | Status | Notes |
|---|---|---|
| stdBrick | in progress (1a): physics verified | consistent and lumped mass; force, tangent and masses bit-identical to OpenSees `955ceb0`, damping forces within 2.3e-15 (matrix-free, docs/deviations.md). Explicit force kernels (flat and team layouts; gather, atomic and colored scatter) verified against the physics. Lumped mass runs end to end in explicit stages; system test against OpenSees within 4.8e-13 (`tests/system/soil_column`) |
| SSPbrick | in progress (1a, after stdBrick) | |
| PML3D, SSPquad, ElasticBeamColumn, DispBeamColumn, ForceBeamColumn, Truss, ASDEmbeddedNodeElement3D, embedded node contact, ZeroLengthContactASDimplex | planned | |

## nD materials

| Femora / OpenSees | Status | Notes |
|---|---|---|
| ElasticIsotropic | **supported** | `ThreeDimensional` form; bit-identical to OpenSees `955ceb0` on 76,650 stress/tangent values (`test_elastic_isotropic_reference`) |
| DruckerPrager, J2CyclicBoundingSurface, LinearElasticGGmax, PressureDependMultiYield, PressureIndependMultiYield | planned | |

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
| modal, secant stiffness proportional, uniform | planned | |

## Patterns and time series

| Femora / OpenSees | Status | Notes |
|---|---|---|
| UniformExcitation | in progress (1a) | runs on the device; `-accel`, `-fact`, `-vel0`; matches OpenSees in the system test |
| Path time series | in progress (1a) | uniform `-dt` and `-time` forms, evaluated on the device exactly as PathSeries / PathTimeSeries (`test_ground_motion`); `-time` with `use_last` or `prepend_zero` refused (OpenSees ignores them) |
| Plain, MultipleSupport, H5DRM | planned | |

## Integrators

| Femora / OpenSees | Status | Notes |
|---|---|---|
| CentralDifference | in progress (1a) | lumped mass, alphaM damping only (others need the 1b solver); deliberately differs from OpenSees for lumped stdBrick (docs/deviations.md) |
| ExplicitDifference | in progress (1a) | all four Rayleigh terms; stable-step check per stage; matches OpenSees within 4.8e-13 on every recorded value (`tests/system/soil_column`) |
| Newmark | in progress (1b) | with Newton and the Eigen direct solver |

## Recorders

| Femora / OpenSees | Status | Notes |
|---|---|---|
| Node | in progress (1a) | disp, vel, accel; OpenSees recording rule and text format; device gather, asynchronous copies, writer thread; `-timeSeries` refused for now |
| VTKHDF | in progress (1a) | `disp vel accel`, whole model or an element list; OpenSees' layout dataset for dataset (types, chunks, offsets, tags), `femora.results` returns the same arrays as for OpenSees' file (`system.vtkhdf_vs_opensees`, `compare_vtkhdf.py`); FemoraX writes with the system HDF5 (1.10 file format objects), OpenSees with its own |
| MPCO | planned | |
