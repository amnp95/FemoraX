# Supported components

A component is **supported** only when its port and its verification test against OpenSees
pass (CLAUDE.md rule 1). Anything not marked supported makes the model reader stop with a
message that names it and suggests the OpenSees backend.

Status: **supported**, **in progress** (milestone named), or **planned**.

## Elements

| Femora / OpenSees | Status | Notes |
|---|---|---|
| stdBrick | in progress (1a): physics verified | consistent and lumped mass; force, tangent and masses bit-identical to OpenSees `955ceb0`, damping forces within 2.3e-15 (matrix-free, docs/deviations.md). Explicit force kernels (flat and team layouts, atomic and colored scatter) verified against the physics; runnable once the explicit time loop lands |
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
| UniformExcitation | in progress (1a) | |
| Path time series | in progress (1a) | |
| Plain, MultipleSupport, H5DRM | planned | |

## Integrators

| Femora / OpenSees | Status | Notes |
|---|---|---|
| CentralDifference | in progress (1a) | see the plan's open question on damping |
| ExplicitDifference | in progress (1a) | see the plan's open question on damping |
| Newmark | in progress (1b) | with Newton and the Eigen direct solver |

## Recorders

| Femora / OpenSees | Status | Notes |
|---|---|---|
| Node | in progress (1a) | |
| VTKHDF | in progress (1a) | |
| MPCO | planned | |
