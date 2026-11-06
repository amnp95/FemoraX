# FemoraX design

## Purpose and scope

- FemoraX reads a model exported by Femora and runs the whole analysis on the GPU,
  or on the CPU through the same Kokkos code.
- OpenSees is the reference implementation. For the same model, FemoraX must reproduce
  OpenSees results.
- Scope is what Femora can emit, not all of OpenSees. The list is at the end of this file.
- Arbitrary Tcl (Femora's `model.actions.tcl`) is not supported. The reader rejects it.

## Architecture

Host side, once per analysis stage:

1. Read the model file.
2. Number DOFs, folding `fix`, `equalDOF` and the laminar boundary into the numbering
   (slave DOFs map to their master's equation; fixed DOFs map to -1).
3. Group elements into batches by (element type, material type).
4. Renumber nodes and elements for memory locality (RCM or a space-filling curve).
5. Build the sparsity pattern and the per-element assembly maps.
6. Allocate device memory for all state.

Device side, every Newton iteration:

1. State determination: one kernel per batch computes stress, tangent, internal force.
2. Assemble the effective stiffness and the residual.
3. Linear solve.
4. Update displacements; compute the convergence norm (one scalar returns to the host).
5. On convergence: commit state (swap buffers), copy recorder data asynchronously.

## Three code layers

1. **Physics** (`src/physics/`): element and material math as framework-free functions.
   This is most of the code and must outlive any framework choice.
2. **Launch** (`src/exec/`): Kokkos `parallel_for`, Views and Graphs that apply the physics
   functions over batches. Small and replaceable.
3. **Native** (`src/native/`): CUDA or HIP versions of a kernel, only when profiling shows
   the portable kernel losing real time.

## Data layout

- Structure of arrays for everything indexed by element, Gauss point or node.
- Material state is double-buffered: committed and trial.
  `commitState` = swap the two buffers, `revertToLastCommit` = copy committed into trial.
  This keeps OpenSees semantics, so retry-with-smaller-step algorithms work unchanged.
- State size is fixed per batch (for example, the maximum number of yield surfaces in a
  PressureDependMultiYield batch).

## Element and material interface (sketch)

```cpp
// Material: pure function, same math as the OpenSees class.
struct ElasticIsotropic {
  struct Params { double E, nu, rho; };
  static constexpr int NSTATE = 0;
  FEMORAX_HD static void update(const Params& p,
                                const double* committed, double* trial,
                                const double eps[6],      // total trial strain, as in setTrialStrain
                                double sig[6], double D[6][6]);
};

// Element: computes internal force, and the tangent only when asked,
// calling the material at each integration point.
template <class Mat>
struct StdBrick {
  static constexpr int NNODE = 8, NDOF = 24, NGP = 8;

  // WantK = false is the explicit path: Ke is never formed (pass nullptr).
  template <bool WantK>
  FEMORAX_HD static void compute(const typename Mat::Params& mp,
                                 const double xe[NNODE][3], const double ue[NDOF],
                                 const double ve[NDOF], double betaK,   // stiffness-proportional damping
                                 const double* committed, double* trial,
                                 double fe[NDOF], double (*Ke)[NDOF]);

  // Largest stable explicit time step for this element.
  FEMORAX_HD static double critical_dt(const typename Mat::Params& mp,
                                       const double xe[NNODE][3], const double* state);
};
```

The launch layer gathers nodal data, calls `compute`, and scatters `fe` (and `Ke` when
formed) through the assembly maps. Large elements may later compute and scatter `Ke` in
pieces instead of holding the full matrix.

## Assembly

- The sparsity pattern is fixed within an analysis stage, so maps from each element's local
  entries to positions in the global value array are built once.
- Maps are stored per node block (block CSR) to cut index memory.
- Two scatter paths: atomic add (fast, non-deterministic summation order) and graph coloring
  (deterministic). Validation tests always use coloring.
- Constraints use DOF elimination (transformation), not penalty, because penalty ruins
  iterative-solver convergence.
- Newmark and Rayleigh coefficients are folded into one scalar so the effective stiffness
  is assembled in a single pass.

## Time integration and algorithms

- First: explicit central difference (needs no linear solver; see the next section).
- Then: implicit Newmark with full Newton, matching the Femora examples.
- Later: ModifiedNewton, KrylovNewton, ExplicitDifference.
- The Newton loop stays on the device; only the norm comes back to the host.
  Kokkos::Graph can capture the iteration later to cut launch overhead.
- Staged analysis: the material stage (elastic/plastic) is a flag in `Params`; element
  activation uses masks. Rebuild the pattern only when it really changes.

## Explicit analysis

Explicit dynamics uses the same physics functions, data layout and batches. It is where the
GPU gains most, because almost all the work is element state determination.

- **No global matrix and no solver.** Each step: element kernels compute internal and damping
  forces → acceleration = (external − internal − damping force) / lumped mass → update
  velocity and displacement → commit. An explicit-only stage builds no sparsity pattern.
- **Force-only element path.** `compute<false>` never forms `Ke`, which removes most of the
  arithmetic and register pressure.
- **Lumped (diagonal) mass after DOF elimination.** A slave DOF's mass is added to its
  master's equation; fixed DOFs drop out.
- **Stable time step.** Every element provides `critical_dt` (characteristic length over the
  dilatational wave speed). A min-reduction over all elements gives the critical step, which
  is checked against the user's step at the start of each stage. Pressure-dependent soil
  moduli grow with confinement, so nonlinear runs re-check it periodically.
- **Damping without a matrix.** Mass-proportional damping is a diagonal term.
  Stiffness-proportional damping is computed inside the same element kernel from the
  velocity (strain rate times the material tangent), so no stiffness matrix is formed.
  It also lowers the stable step; `critical_dt` accounts for it.
- **Many small steps.** Run many steps per launch sequence (Kokkos::Graph) without returning
  to the host. Check stability and NaNs every N steps. Recorders sample every k-th step and
  copy asynchronously.
- **Match OpenSees.** Implement OpenSees `CentralDifference` and `ExplicitDifference` exactly
  (Femora exposes both), so results can be compared step by step.
- **Limits.** Stiff components shrink the stable step: penalty-type interfaces and contact
  (ASDEmbeddedNode, contact elements), nearly incompressible or saturated materials, and very
  small elements. PML3D carries its own integration parameters; check its OpenSees
  implementation before supporting it in explicit runs.

## Linear solvers

- Interface: `analyze(pattern)` once per stage, `setup(values)` per factorization,
  `solve(rhs, x)` per iteration.
- Backends to evaluate: a CPU direct solver for milestone 1; Trilinos (Kokkos-native) or
  PETSc for portable GPU iterative solvers with AMG; cuDSS on NVIDIA for direct solves.
- In dynamics the effective stiffness is well conditioned, and the previous step's solution
  is a good initial guess, so iterative solvers usually converge quickly. The gravity stage
  is harder and may need a direct solver.

## Precision and hardware

- FP64 throughout. Laptop and consumer GPUs have weak FP64; they are for development and
  medium models. Large runs go to data-center GPUs (for example through Femora's TACC jobs).
- Apple GPUs have no FP64; on Macs FemoraX runs on the CPU backend.
- Kokkos enables one device backend per build, so FemoraX ships separate builds
  (CPU, CUDA, ROCm).

## Hard parts (later milestones)

- PressureDependMultiYield / PressureIndependMultiYield: branchy return mapping. Use an
  elastic predictor on all points, then compact the yielding points and run the plastic
  corrector only on those. OpenSees uses static workspaces and dynamic yield surfaces here;
  rewrite both as fixed-size local data.
- ForceBeamColumn with fiber or RC sections: iterates inside the element. Use one team
  (thread block) per element, threads over fibers.
- PML3D: extra DOFs and its own integration parameters; needs its own kernel.
- H5DRM: large HDF5 input every step; prefetch to the device asynchronously.
- Contact (ZeroLengthContactASDimplex, embedded node contact): branchy state updates.
- Modal damping: needs an eigenvalue solve; do it last.

## Model file (the contract with Femora)

- HDF5 with a top-level `schema_version`.
- Planned groups: `/nodes` (coordinates, ndf), `/elements/<batch>` (type, connectivity,
  material reference, parameters), `/materials`, `/sections`, `/constraints` (sp, mp),
  `/mass`, `/damping`, `/time_series`, `/patterns`, `/recorders`, `/process` (ordered steps).
- Femora owns the exporter. FemoraX owns the reader and the specification in `schema/`.
- An unsupported component makes the reader stop with a message that names it and suggests
  the OpenSees backend.

## Verification

1. **Materials:** a small driver built against the OpenSees source calls `setTrialStrain`,
   `getStress`, `getTangent` and `commitState` along strain paths (monotonic, cyclic, with
   reversals). Its output is stored in `tests/reference/`; FemoraX must match it.
2. **Elements:** patch tests, plus single-element comparisons with OpenSees.
3. **Systems:** Femora examples run through both OpenSees and FemoraX; compare time
   histories with documented tolerances.

## Supported-component target (from Femora's API)

- Elements: SSPbrick, stdBrick, PML3D, SSPquad, ElasticBeamColumn, DispBeamColumn,
  ForceBeamColumn, Truss, ASDEmbeddedNodeElement3D, embedded node contact,
  ZeroLengthContactASDimplex.
- nD materials: ElasticIsotropic, DruckerPrager, J2CyclicBoundingSurface,
  LinearElasticGGmax, PressureDependMultiYield, PressureIndependMultiYield.
- Uniaxial materials: Elastic, Steel01.
- Sections: Elastic, Uniaxial, WFSection2d, Fiber, RC, Aggregator, Parallel,
  ElasticMembranePlate, PlateFiber, Bidirectional, Isolator2Spring.
- Constraints: fix, equalDOF, laminar boundary, rigidDiaphragm, rigidLink.
- Damping: Rayleigh, frequency Rayleigh, modal, secant stiffness proportional, uniform.
- Patterns: Plain, UniformExcitation, MultipleSupport, H5DRM.
- Recorders: node, VTKHDF (so `femora.results` reads FemoraX output unchanged), MPCO later.

## Roadmap

1. **Layered elastic soil column** (CPU backend), in two steps:
   - **1a, explicit:** stdBrick and SSPbrick, ElasticIsotropic, fix, laminar boundary,
     Rayleigh damping, UniformExcitation, CentralDifference, node and VTKHDF recorders.
     Needs no solver, so the whole pipeline runs end to end early.
   - **1b, implicit:** the same model with Newmark + Newton and a CPU direct solver behind
     `LinearSolver`.
2. **Nonlinear layered soil column**: CUDA build, GPU solvers, PressureDependMultiYield,
   material stages.
3. **Laterally loaded pile**: beams, sections, embedded interfaces.
4. **DRM basin and dynamic pile-soil interaction**: H5DRM, PML, absorbing boundaries.
5. **Rotated two-pile bent**: fiber sections, ForceBeamColumn, multi-GPU using Femora's
   METIS partitions.

## Open decisions

- Test framework (GoogleTest or Catch2).
- Python bindings (nanobind or pybind11).
- Solver library for GPU (Trilinos or PETSc) and the milestone-1 CPU direct solver.
- Details of the HDF5 schema v0.
- FemoraX license. Code ported from OpenSees is a derivative work, and the OpenSees license
  covers educational, research and non-profit use; settle this before the first public release.
