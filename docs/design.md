# FemoraX design

## Purpose and scope

- FemoraX reads a model exported by Femora and runs the whole analysis on the GPU,
  or on the CPU through the same Kokkos code.
- Correct physics first, verified against analytical solutions, manufactured solutions
  and published benchmarks. OpenSees is where the first formulations come from and the
  main cross-check: FemoraX matches it where OpenSees is correct, and deviates deliberately,
  with the reason and the evidence in `docs/deviations.md`, where we know better (bug
  fixes, better algorithms, new elements and materials). FemoraX is its own solver.
- Scope is what Femora can emit, not all of OpenSees. The list is at the end of this file.
- Arbitrary Tcl (Femora's `model.actions.tcl`) is not supported. The reader rejects it.

## Architecture

Host side, once per analysis stage:

1. Read the model file.
2. Number DOFs, folding `fix`, `equalDOF` and the laminar boundary into the numbering
   (slave DOFs map to their master's equation; fixed DOFs map to -1). Any number of DOFs
   per node, and element-owned DOFs (see "DOF numbering").
3. Group elements into batches by (element type, material type).
4. Renumber nodes and elements for memory locality (RCM or a space-filling curve).
5. Build the per-element equation maps (no global sparsity pattern: matrix-free).
6. Allocate device memory for all state.

Device side, every Newton iteration:

1. State determination: one kernel per batch computes stress, internal force and the
   residual (and stores the Gauss-point tangent when the operator uses stored tangents).
2. Linear solve with the matrix-free operator: each CG iteration applies `K_eff v`
   element by element (no global matrix).
4. Update displacements; compute the convergence norm (one scalar returns to the host).
5. On convergence: commit state (swap buffers), copy recorder data asynchronously.

## Three code layers

1. **Physics** (`src/physics/`): element and material math as framework-free functions.
   This is most of the code and must outlive any framework choice.
2. **Launch** (`src/exec/`): Kokkos `parallel_for`, Views and Graphs that apply the physics
   functions over batches. Small and replaceable.
3. **Native** (`src/native/`): CUDA or HIP versions of a kernel, only when profiling shows
   the portable kernel losing real time.

The solvers (`src/solvers/`) are launch-layer code too: Kokkos is allowed there, and only
there besides `src/exec/` (CLAUDE.md rule 3).

## Data layout

- Structure of arrays for everything indexed by element, Gauss point or node.
- Material state is double-buffered: committed and trial.
  `commitState` = swap the two buffers, `revertToLastCommit` = copy committed into trial.
  This keeps OpenSees semantics, so retry-with-smaller-step algorithms work unchanged.
- State size is fixed per batch (for example, the maximum number of yield surfaces in a
  PressureDependMultiYield batch).

## Element and material contract

Adding an element is one header in `src/physics/elements/` plus one line in
`src/exec/element_registry.hpp`. Adding a material is the same with
`src/physics/materials/` and `src/exec/material_registry.hpp`. The concepts in
`src/physics/contracts.hpp` check every entry at compile time, so a missing or inconsistent
trait is a compile error, not a runtime surprise.

### Element traits

Every size is a compile-time constant, so the launch layer and the compiler see fixed-size
local arrays.

| Trait | Meaning |
|---|---|
| `NDM` | spatial dimension of the node coordinates the element reads |
| `NNODE` | number of nodes |
| `NODE_NDF` | `std::array<int, NNODE>`: DOFs per node, node by node (3 for a brick, 6 for a 3D beam, 4 at the corner nodes of a u-p element, 9 or more for PML) |
| `NEXTRA` | DOFs the element owns that belong to no node (0 for most elements) |
| `NDOF` | `sum(NODE_NDF) + NEXTRA`; checked by the concept |
| `NGP` | integration points |
| `NSTATE` | doubles of history state per element (material state at every point plus element state) |
| `MASS` | `Consistent`, `Lumped` or `Selectable` (chosen per element group, as stdBrick's `-lumped`) |
| `EXPLICIT` | whether the element may appear in an explicit stage (needs a lumped mass and `critical_dt`) |
| `GEOMETRY_CACHEABLE` | whether its geometry (shape-function derivatives, det J) may be computed once and reused: true only for small-strain elements. Large-deformation or corotational elements must recompute or update it |
| `MATERIAL_KIND` | which material contract it calls (`NDMaterial3D`, `NDMaterialPlaneStrain`, `Uniaxial`, `Section...`) |
| `Params` | plain struct: material `Params` plus element data (body force, mass type, ...) |

Element DOF order is the OpenSees order: node 0's DOFs, node 1's DOFs, ..., then the extra
DOFs.

```cpp
template <class Mat>
struct StdBrick {
  static constexpr int NDM = 3, NNODE = 8, NEXTRA = 0, NGP = 8;
  static constexpr std::array<int, NNODE> NODE_NDF = {3, 3, 3, 3, 3, 3, 3, 3};
  static constexpr int NDOF = 24;
  static constexpr int NSTATE = NGP * Mat::NSTATE;
  static constexpr MassType MASS = MassType::Selectable;
  static constexpr bool EXPLICIT = true;
  static constexpr MaterialKind MATERIAL_KIND = MaterialKind::NDMaterial3D;
  struct Params { typename Mat::Params mat; double b[3]; MassType mass; };

  // Internal force plus all four Rayleigh damping forces of this element
  //   fe = f_int(ue) + alphaM*Me*ve + (betaK*Kt + betaKinit*K0 + betaKcomm*Kc)*ve,
  // the same as OpenSees getResistingForce + getRayleighDampingForces.
  // WantK = false is the explicit path: Ke is never formed (pass nullptr).
  template <bool WantK>
  FEMORAX_HD static void compute(const Params& p, const Rayleigh& damp,
                                 const double xe[NNODE][NDM], const double ue[NDOF],
                                 const double ve[NDOF],
                                 const double* committed, double* trial,
                                 double fe[NDOF], double (*Ke)[NDOF]);

  // Me as a diagonal (lumped) or a full matrix (consistent), per p.mass.
  FEMORAX_HD static void mass(const Params& p, const double xe[NNODE][NDM], double* Me);

  // Upper bound of the element's largest eigenvalue omega^2 of K*phi = omega^2*M*phi with the
  // lumped mass. The element eigenvalue theorem makes the largest element value an upper
  // bound for the assembled model.
  FEMORAX_HD static double max_omega_sq(const Params& p, const double xe[NNODE][NDM],
                                        const double* committed);

  // Stable step for this element under the given scheme and damping (see "Stable time step").
  FEMORAX_HD static double critical_dt(const Params& p, const Rayleigh& damp,
                                       ExplicitScheme scheme,
                                       const double xe[NNODE][NDM], const double* committed);
};
```

Most elements implement `critical_dt` as one call to the shared
`physics/explicit_stability.hpp` helper with their `max_omega_sq`.

### Material traits

```cpp
struct ElasticIsotropic {
  static constexpr MaterialKind KIND = MaterialKind::NDMaterial3D;
  static constexpr int NSTRAIN = 6;   // OpenSees order: 11 22 33 12 23 31, engineering shear
  static constexpr int NSTATE = 0;    // doubles of history per integration point
  struct Params { double E, nu, rho; };
  // Same math as setTrialStrain + getStress + getTangent.
  FEMORAX_HD static void update(const Params& p, const double* committed, double* trial,
                                const double eps[NSTRAIN],
                                double sig[NSTRAIN], double D[NSTRAIN][NSTRAIN]);
  // getInitialTangent, for betaKinit damping.
  FEMORAX_HD static void initial_tangent(const Params& p, double D[NSTRAIN][NSTRAIN]);
  FEMORAX_HD static double density(const Params& p);
};
```

The committed tangent that `betaKcomm` damping needs comes from `update` at the committed
state, or from a stored copy when the material is expensive; the material decides.

### Registration

`src/compile/registry.hpp` is the one place that maps model types onto physics types:

```cpp
struct StdBrickEntry {                     // the element's adapter
  template <class Mat> using Element = StdBrick<Mat>;
  static constexpr io::ElementType type = io::ElementType::StdBrick;
  template <class Mat>
  static typename Element<Mat>::Params params(const io::ElementGroup&, const typename Mat::Params&);
};
using Elements  = TypeList<StdBrickEntry>;          // one line per element
using Materials = TypeList<ElasticIsotropicEntry>;  // one line per material
```

Adding an element is its physics header, its adapter (model data to `Params`) and one line
in the list. Batches are built for every registered pair whose material kinds match; the
compile step (batches, lumped mass, parameter sets stored as the element's own `Params`)
and the kernels (`src/exec/element_kernels.hpp`: every layout x scatter x geometry, generic
over the element) are generic over the entries. A pair that is not registered is an
unsupported component.

## DOF numbering

Nothing assumes a fixed number of DOFs per node.

1. **DOF space.** Each node has `ndf[node]` DOFs, read from `/nodes`. Each element's
   `NODE_NDF[i]` must equal the `ndf` of its node `i` (as in OpenSees), checked when the
   model is read. Node DOFs are numbered by a prefix sum, `dof_offset[node]`, so node `n`'s
   DOF `d` is `dof_offset[n] + d`. Elements with `NEXTRA > 0` get their DOFs after all node
   DOFs, through `extra_offset[element]`.
2. **Constraints.** `equalDOF` (and the laminar boundary, which Femora writes as equalDOF
   pairs) joins DOFs in a union-find over DOF ids. Each DOF named in a constraint must exist
   on both nodes. `fix` marks DOFs; if any DOF of a joined group is fixed, the whole group is
   fixed. The result is checked against an emulation of OpenSees' Transformation handler;
   patterns that OpenSees silently mishandles are refused (docs/deviations.md,
   "Constraint handling"). `src/compile/dof_numbering.cpp`.
3. **Equations.** Each surviving group gets one equation, numbered deterministically (in
   order of the group's lowest DOF id after renumbering). The result is `eq[dof]`, with -1
   for fixed DOFs.
4. **Element maps.** Every batch stores, per element, its `NDOF` equation numbers
   (`int32`, structure of arrays `[k * nelem + e]`), built from `NODE_NDF` and the
   offsets. The launch layer never computes DOF positions on the fly.
5. **Lumped mass and loads** sum through the same maps, so a slave's mass lands on its
   master's equation and fixed DOFs drop out.

Indices are 32-bit (nodes, elements, DOFs, equations, map entries). A count that can pass
2^31 (the nonzeros of a large stiffness matrix) uses 64-bit row offsets; the compile step
checks this.

## Stable time step

Each explicit scheme has its own stability limit for a mode with frequency `omega` and
Rayleigh damping. With `alpha = alphaM` and
`beta = betaK + betaKinit + betaKcomm` (conservative: all stiffness terms at the largest
`omega`):

- **ExplicitDifference** (OpenSees; damping uses the half-step velocity, so it is lagged).
  For one mode the update has trace `2 - W^2 - 2*xi*W` and determinant `1 - 2*xi*W`, with
  `W = omega*dt`. Stability requires `W^2 + 4*xi*W <= 4`. With
  `xi = alpha/(2*omega) + beta*omega/2` this is
  `omega^2*dt^2 + 2*(alpha + beta*omega^2)*dt - 4 <= 0`, so

  `dt_crit = (sqrt(c^2 + 4*omega^2) - c) / omega^2`, with `c = alpha + beta*omega^2`.

  Without damping this is `2/omega`. Stiffness-proportional damping lowers it, which
  matters for soil columns with `betaK` damping on fine meshes.
- **CentralDifference** (OpenSees; damping is centred and implicit). The limit is
  `omega*dt <= 2`, unaffected by damping. In milestone 1a only the diagonal case
  (lumped mass, `alphaM` only) is accepted; the general case needs the solver (1b).

`omega^2` is the largest `max_omega_sq` over all elements, a min-reduction of
`critical_dt` gives the stage's stable step, and both are checked at the start of each
explicit stage. A step above `dt_crit` stops the run with the two values and the element
that sets the limit; `allow_unstable_dt` turns the stop into a warning. Nonlinear runs
re-check every N steps. Tests run a single mode and a column just below and just above
`dt_crit` for both schemes, with and without each damping term.

## Element kernels

Applies from the stdBrick kernel on. The goal is that every element runs in the thread
layout that is fastest for it on each backend, with the physics written once.

### Pointwise physics functions

Each element exposes its work at one integration point as separate physics functions, in
the style of libCEED's pointwise QFunctions. The whole-element `compute` is built from
them, so the one-thread-per-element path and the team paths run the same arithmetic.

```cpp
template <class Mat>
struct StdBrick {
  struct PointShape { double shp[4][8]; double dvol; };  // dN/dx, dN/dy, dN/dz, N; w*det(J)
  // Shape functions of point gp (OpenSees order gp = 4i + 2j + k) from coordinates xl[3][8].
  FEMORAX_HD static void point_geometry(const double xl[3][8], int gp, PointShape& s);
  // B u: strain from nodal displacements, or strain rate from nodal velocities.
  FEMORAX_HD static void point_strain(const PointShape& s, const double u[NDOF],
                                      double strain[6]);
  // This point's share of node j's force: B_j^T (stress * dvol).
  FEMORAX_HD static void point_node_force(const PointShape& s, const double stress[6],
                                          int j, double fj[3]);
  // This point's body force at node j, component q: dvol * b[q] * N_j.
  FEMORAX_HD static double point_body_force(const Params& p, const PointShape& s, int j, int q);
  // This point's share of the 3x3 block of node pair (J, K): B_J^T (D * dvol) B_K.
  FEMORAX_HD static void point_block_tangent(const PointShape& s, const double (*D)[6],
                                             int J, int K, double KJK[3][3]);
};
```

The material update sits between `point_strain` and `point_node_force` (in `compute`, or
in the launch layer's point threads). Stiffness-proportional damping is a damping stress
`(betaK*Dt + betaKinit*D0 + betaKcomm*Dc) * strain_rate` pushed through
`point_node_force`; mass-proportional damping is `alphaM * N_j * dvol * rho * v` per point
(consistent: v interpolated at the point; lumped: the node's own v). The `PointwiseElement`
concept in `contracts.hpp` checks these functions.

### Thread layouts

The launch layer offers several layouts per element; each is one fused
gather-compute-scatter kernel.

- **Explicit stdBrick, team layout:** a team of 8 threads per element, one per Gauss point,
  and several elements per team (for example 4 elements = 32 threads). Each thread does
  `point_geometry` and `point_update`; the element's force is reduced over its 8 points in
  team scratch (shared) memory in a fixed order; then one scatter per node.
- **Explicit stdBrick, flat layout:** one thread per element runs the whole element.
- **SSPbrick** (one integration point): one thread per element.
- **Implicit assembly:** never hold the full 24x24 tangent in one thread. A team per element
  first runs `point_update` for every point (stress and `D` into scratch memory), then its
  threads compute the 64 node-pair 3x3 blocks in parallel (`point_block_tangent` summed
  over the points) and scatter each block directly through the assembly map. The force is
  reduced as in the explicit team layout.

Reductions within a team use a fixed order, and the team layout reduces in the same order
as the flat one, so all layouts give the same bits on a given backend (tested on OpenMP and
CUDA). Across backends results agree to rounding only: nvcc fuses multiply-adds
(`--fmad=true`), the host compiler does not (GCC with `-std=c++20` does not contract), so GPU
results are verified against host references and OpenSees at a documented tolerance.

### Settings, defaults and auto-tune

Everything above the kernels (the time loop, future elements and materials) asks for forces
through one interface, `ExplicitForce::compute(ForceSettings)`, with layout, scatter,
elements per team and geometry mode as settings. No code assumes a variant; every variant
stays in the code, the tests and the benchmarks.

- **Defaults** are per device, chosen by benchmark (`src/exec/force_settings.cpp`), for two
  purposes: production (fastest) and deterministic (validation, bit-identical run to run).
  An entry changes only with benchmark numbers in the commit message.
- **Auto-tune** (`exec::autotune`) times every valid variant on the actual model and
  hardware at startup and returns the fastest, for when the table has no entry for the
  device or a run wants to check it.
- **Scatter modes:** atomic (production); colored (one launch per color); gather (each
  element writes its 24 values to its own slots in locality order, then each equation adds
  its slots in a fixed precomputed order). Colored and gather are deterministic. Elements
  are never sorted globally by color: production needs the locality order. If coloring
  stays needed, the compile step uses color order only for deterministic runs.

### Geometry cache

- The recompute path is permanent. Cached geometry (25 doubles per Gauss point, 1.6 KB per
  brick) is used only for elements with `GEOMETRY_CACHEABLE`, and only if it fits: with
  `GeometryCache::Auto` the cache is built when the model, the cache and the work vectors
  fit in 80% of the device's free memory (`exec::device_free_memory_bytes`). Otherwise
  `Geometry::Cached` falls back to recomputing, with the same bits.
- Later: share the geometry between elements of identical shape (regular grids, the common
  case in soil columns and boxes): store each distinct shape once and give elements a shape
  index, so the cache costs almost no memory. Elements are identical when their nodal
  coordinates differ by a translation only.

### Later

- **Matrix-free operator:** now the implicit path itself ("Linear solvers").
- **libCEED comparison:** implement elastic stdBrick in libCEED as a speed benchmark for
  our kernels (benchmark code only, never a dependency of FemoraX).

## Machine-learned constitutive models (design note, no code yet)

- **Small networks** run inline at each Gauss point like any material. Their weights are
  material parameters, held once in a shared device array that every point reads.
- **Larger networks** use a **batched material path**: the element kernel writes the
  Gauss-point inputs to a buffer, one batched GEMM (or a short sequence of them) evaluates
  the network for all points, and a second element kernel finishes (force, tangent,
  scatter). The batched path is an official material option in the material contract, not
  an ML special case; the elastic-predictor / plastic-corrector split of the multi-yield
  materials uses the same mechanism (predict everywhere, then correct the compacted set).
- **Tangent:** the analytic derivative of the network, a network that outputs the tangent
  as well, or finite differences, chosen per material. Explicit analysis needs stress only.
- **Precision:** an exception to the FP64 rule. An ML material may compute internally in
  FP32 or lower; its inputs and outputs are converted to and from FP64 at the material
  boundary, so element, state and assembly code stay FP64.
- **History:** hidden states of recurrent networks are ordinary material state, in the
  committed/trial buffers, so commit and revert work unchanged.
- **Training** happens outside FemoraX. FemoraX loads exported weights from a file named in
  the model (a schema addition when the first ML material arrives).

## Memory and parallelism contract

Rules every kernel follows, so the same physics code is fast on every backend. Tested in
`tests/unit/test_memory_contract.cpp`.

1. **Strided state access.** Physics code never receives the state of a point as a plain
   contiguous pointer. It gets `Strided<T>` (`src/physics/common/strided.hpp`): value `k`
   of the point is `ref[k] = base[k * stride]`. An element gets `PointArray<T>`, whose
   `point(gp)` is that reference for point `gp`. With `stride = nPoints` the storage is
   structure of arrays and coalesced; the launch layer may pick another stride per backend
   (contiguous per point for some CPU kernels, a padded stride) without touching physics
   code. Offsets are computed in 64 bits; indices stay 32-bit. The same rule applies to
   any per-element or per-point array physics code reads from global memory.
2. **Storage order follows the thread mapping.** Per-element data: `value[k * nElements + e]`.
   Per-Gauss-point data: `value[k * nPoints + (e * NGP + gp)]`. Consecutive threads read
   consecutive addresses in the flat layout (thread = element) and in the team layout
   (thread = Gauss point).
3. **Inner loops through our own helpers.** Work inside an element or point that can use
   several threads (node-pair tangent blocks, yield surfaces, fibers, network neurons) is
   written with `inner_for(ctx, n, f)` and `inner_reduce(ctx, n, f)`
   (`src/physics/common/inner.hpp`). `SerialInner` runs plain loops; the launch layer's
   `exec::VectorLanes` (`src/exec/inner_kokkos.hpp`) maps them to
   `Kokkos::ThreadVectorRange`. Three levels: team = group of elements, team thread =
   element or Gauss point, vector lane = work inside a point; the mapping is chosen per
   element-material pair. To give the same bits under every mapping:
   - outputs of an `inner_for` go to memory the caller provides (team scratch in
     lane-parallel layouts), because locals are private per lane on a GPU;
   - a lane reads only what the same index wrote (combine with an `inner_for` over the
     same index space); there is no lane barrier;
   - `inner_reduce` adds terms in index order, whatever the split;
   - code outside inner loops runs redundantly on every lane; its state writes are pure
     functions of the inputs.
4. **Team sizes come from the backend at run time** (`exec::simd_width()`: 32 on NVIDIA,
   64 on AMD data-center GPUs, 1 on CPU backends); nothing assumes 32. The team layout fills
   a warp or wavefront with several elements (`elements_per_team = simd_width / NGP`).
5. **Nodal gathers.** Nodes and elements are renumbered for cache locality in the compile
   step (it can be turned off). In the team layout, the element's threads load its nodal
   coordinates and values cooperatively into team scratch once, and all Gauss points read
   from there.
6. **Memory management.**
   - Everything is allocated once in the compile step; nothing is allocated in the time
     loop.
   - All data is device-resident; host copies exist only for I/O (recorders, reader).
   - Per-element and per-point work arrays live in Kokkos team scratch, not global memory.
   - Batches are padded to a multiple of the SIMD width; padding elements are inert
     (zero mass and stiffness, scattered nowhere).
   - Before a run, a memory estimate (state, maps, vectors, recorder buffers, per backend)
     is compared with the device's free memory; the run stops with a clear message naming
     the estimate and the available memory if the model will not fit.

## Multi-GPU (plan; one subdomain today)

Femora already partitions models with METIS. FemoraX will run one subdomain per GPU:

- **Compile step:** each subdomain gets its owned elements, its owned nodes and its ghost
  nodes (nodes of owned elements that another subdomain owns). Today the model is one
  subdomain: every node owned, no ghosts (`compile::Subdomain`). Everything after the
  compile step works in the subdomain's local numbering (local nodes, local equations);
  global tags live only in the compile step's maps, for input and output.
- **Explicit step:** each GPU computes its element forces into its local vector, then
  exchanges border-node forces with its neighbours (halo exchange: send the force
  contributions of ghost nodes to their owners, receive the owners' summed values back),
  overlapping the exchange with interior work. Lumped masses are exchanged the same way
  once per stage.
- **Constraints across subdomains:** a tie whose nodes sit in different subdomains is
  resolved by the owner of the retained node; the constrained node becomes a ghost of
  that subdomain, so the shared equation has one owner.
- **Global reductions:** the stable step (minimum), the NaN/instability check (maximum) and
  convergence norms in 1b are MPI reductions of per-subdomain values; one scalar each.
- **Output:** each subdomain writes its owned nodes and elements (a VTKHDF part per
  subdomain, as OpenSeesMP with Femora's `$pid` file names); `femora.results` already reads
  partitioned output.
- **Determinism:** the gather scatter stays deterministic within a subdomain; across
  subdomains the halo sums are added in a fixed (rank) order.

## Assembly

There is no production assembly of global matrices: the implicit path is matrix-free
("Linear solvers"). Global vectors (internal force, the operator's product, its diagonal)
are added into equations through the force kernels' scatter (gather by default,
deterministic; atomic; colored), which handles any merged equations from the constraints.
Constraints use DOF elimination (transformation), not penalty, because penalty ruins
iterative-solver convergence. Assembled matrices are built only in tests, on the host from
element matrices, for small reference models.

## Time integration and algorithms

- Explicit: ExplicitDifference and CentralDifference (lumped mass), no solver.
- Implicit (`src/exec/implicit_stage.cpp`): Newmark (form D) for Transient stages and
  LoadControl for Static ones, each with the Linear, Newton or ModifiedNewton algorithm
  and the NormUnbalance, NormDispIncr or EnergyIncr test, as OpenSees runs them. Every
  linear solve is matrix-free (CG or GMRES from the declared tangent symmetry, Jacobi).
  Solver settings are automatic unless the model sets them (schema 0.3): relative
  tolerance 1e-12 (Newton-type algorithms accept a solve that hit the iteration limit
  after reducing the residual by six orders), at most min(n, 10000) iterations (at least
  200), the Jacobi preconditioner; each solve starts from the previous increment.
- Later: KrylovNewton, Femora's retry and substep recovery (a failed step stops the run
  today), Newmark forms V and A.
- The Newton loop stays on the device; only norms come back to the host.
  Kokkos::Graph can capture the iteration later to cut launch overhead.
- Staged analysis: the material stage (elastic/plastic) is a flag in `Params`; element
  activation uses masks.

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

Decided for milestone 1b (2026-11-06): **the implicit path is matrix-free.** Goals: every
CPU and GPU (NVIDIA, AMD, Intel), very large models (billions of elements), few
dependencies, and any element, material (elastic or plastic) and constraint (laminar
boundary, rigid diaphragm, equalDOF, embedded nodes, contact), with the tangent changing
every Newton iteration. A global matrix would be rebuilt every iteration and would not fit
at the target scale; a direct solver (memory ~ n^(4/3), work ~ n^2 on 3D meshes) is out.
No Trilinos (it would pin our Kokkos version and is too heavy).

### The interface

Everything outside `src/solvers/` sees only `femorax::LinearSolver` and the operator and
preconditioner types beside it (`src/solvers/linear_solver.hpp`); no solver-library type
appears anywhere else (CLAUDE.md rule 6). Vectors are device-resident FP64 views in the
compile step's equation numbering.

- `LinearOperator`: `apply(x, y)` (y = A x) and `diagonal(d)`, plus a description:
  symmetry (`SymmetricPositiveDefinite`, `Symmetric`, `General`) and fields (equation
  ranges, for coupled problems such as u-p). Matrix-free in production; an assembled
  operator (`CsrOperator`) exists for tests and exposes its CSR arrays.
- `Preconditioner`: `setup(operator)`, `apply(r, z)`.
- `LinearSolver`: `setup(operator, preconditioner)` when the operator changes,
  `solve(b, x)` with `x` holding the initial guess; returns converged, iterations and the
  residual norms. Options: relative and absolute tolerance, maximum iterations.

Non-symmetric operators and block structure need no change for callers.

### The matrix-free operator

`K_eff v = c1 K_t v + c2 C v + c3 M v` (the Newmark and Rayleigh factors of the step) is
computed element by element, in the same kernel shape as the internal force:

- at each Gauss point, the tangent times the strain of `v`: `B^T D_t (B v) dvol` (the
  stiffness-proportional damping terms use `D_t`, `D_0` and the committed `D_c` the same
  way, as the explicit damping force already does); the mass term is
  `rho N_j N_k dvol v_k` per point (consistent) or the node's lumped mass times `v`;
- each element's result is added into the equations with the existing deterministic
  force gather (or atomic or colored scatter), so merged equations from any constraint
  are handled the way the force already is.

The material tangent `D_t` comes from the state determination of the current Newton
iterate. Two modes, benchmarked against each other per element-material pair:
**stored** (each state determination writes `D_t` per Gauss point, 21 values when the
tangent is symmetric, 36 otherwise; elastic materials need one copy per material) and
**recomputed** (`apply` calls the material's tangent again from the trial state). The
pieces are composable kernel parts (`point_strain`, the material tangent product,
`point_node_force`, the mass and damping terms), so every element and material that
provides the pointwise functions gets the operator, its diagonal and the preconditioners
without new kernel code.

### Symmetry of the tangent, and the solver choice

Plastic soil models with non-associative flow, frictional contact and u-p elements give
non-symmetric or indefinite tangents; CG needs a symmetric positive definite operator.
So symmetry is declared, not assumed:

- Every material declares its tangent: `TANGENT = SymmetricPositiveDefinite`,
  `Symmetric` (possibly indefinite) or `General`; every element declares what its own
  terms add (u-p coupling, contact). The concepts in `contracts.hpp` require it. Elastic
  ElasticIsotropic is `SymmetricPositiveDefinite`.
- The stage's operator takes the weakest declaration over all element-material pairs in
  it (and over the constraints, which never break symmetry under DOF elimination).
- The solver follows automatically; users never choose:
  `SymmetricPositiveDefinite` → CG; `Symmetric` or `General` → GMRES. If CG detects a
  non-positive curvature (`p^T A p <= 0`), it stops and the solve is repeated with GMRES,
  with a warning naming the stage (a declaration was too optimistic).
- Tangent storage follows the declaration too: the packed 21-value form only when the
  material declares a symmetric tangent, the full 36 values otherwise.

### Preconditioners (all matrix-free)

1. **Jacobi** (first): the diagonal of `K_eff`, gathered element by element like a force
   vector (each element's diagonal entries, through the same scatter).
2. **Chebyshev with Jacobi** (next, after the CG iteration study): a fixed polynomial in
   the Jacobi-scaled operator; needs only `apply` and an eigenvalue estimate.
3. **Matrix-free geometric multigrid** (a later milestone; plan only): for static gravity
   and big models. Smoothing by Chebyshev-Jacobi on every level, transfers between
   levels, and only the small coarsest level stored and solved directly.

### Solvers

- **CG** (our own, in Kokkos, all backends): for symmetric positive definite stages. Each solve
  starts from the previous solution. Dot products use a fixed-chunk two-stage reduction,
  so results are bit-identical for any thread count on a backend
  (`src/solvers/vector_ops.hpp`). Implicit dynamics suits it: at practical steps `K_eff`
  is dominated by `M/dt^2`, so its condition number stays bounded as the mesh grows.
- **GMRES(m)** (our own, in Kokkos, all backends): for symmetric indefinite and
  non-symmetric stages. Restarted GMRES with right preconditioning in its flexible form
  (FGMRES), so preconditioners that change between iterations (Chebyshev, later
  multigrid) are allowed. Orthogonalization by classical Gram-Schmidt with one
  re-orthogonalization (CGS2): all dot products of one iteration are fused into one
  deterministic multi-dot reduction, so an iteration needs two global reductions instead
  of k, which matters at scale (and across GPUs). Restart length m = 30 by default
  (memory: about 2m vectors); the residual norm never increases, and it stays robust for
  indefinite and strongly non-normal tangents. Chosen over BiCGStab, whose memory is
  fixed (about 7 vectors) and which needs no restart, but whose residual is irregular and
  which can break down or stagnate on indefinite and non-normal systems (plasticity,
  contact), exactly the cases it would be there for; BiCGStab can be added behind the same
  interface if memory ever forces it.
- **Eigen `SimplicialLDLT`**, and `SparseLU` for non-symmetric cases: in tests only, as the
  reference solution on small models,
  with the global matrix assembled on the host from element matrices
  (`compile/sparsity.hpp` gives the pattern).

### Optional for the future: hypre BoomerAMG

Not on the main path. hypre's BoomerAMG would be an optional preconditioner for an
assembled matrix, should a stage ever need one. Notes from the hypre documentation and
changelog (checked 2026-11-06):

- Current release 3.2.1 (2026-11-02); GPU backends CUDA, HIP and SYCL, one per build;
  ROCm 7.0 from 3.1.0. CUDA 13: no changelog entry names it; 2.x tags do not build with
  it, and PETSc builds hypre 3.x with CUDA 13.x.
- GPU-supported BoomerAMG options: PMIS coarsening; interpolation direct, BAMG-direct,
  extended, extended+i, extended+e; aggressive coarsening; smoothers Jacobi, l1-Jacobi,
  hybrid Gauss-Seidel/SSOR, two-stage Gauss-Seidel, Chebyshev. Systems AMG, nodal
  coarsening and rigid-body interpolation vectors are not listed as GPU-supported.
- The IJ interface copies values into hypre's storage; zero-copy needs hypre's internal
  ParCSR structures. Full 64-bit integers are not supported on the GPU; mixed 32/64-bit
  carries no documented GPU restriction.

## Precision and hardware

- FP64 throughout (one exception: the internals of machine-learned materials, see
  "Machine-learned constitutive models"). Laptop and consumer GPUs have weak FP64; they are for development and
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

Correctness is checked against physics first and OpenSees second.

1. **Analytical and manufactured solutions:** closed-form material responses, patch tests
   (linear fields reproduced exactly), rigid-body modes, energy conservation, wave speeds
   and natural frequencies of soil columns, manufactured solutions for the time
   integrators.
2. **Published benchmarks:** standard site-response and soil-structure problems with
   published results, as the elements and materials that need them arrive.
3. **OpenSees comparisons** (the main bug-finding tool): small drivers built against the
   OpenSees source call each ported class directly (materials along strain paths with
   commit and revert; elements in a real Domain; the constraint handler), and Femora
   examples run through both programs. FemoraX matches OpenSees where OpenSees is
   correct; every deliberate difference is in `docs/deviations.md`. Reference data lives
   in `tests/reference/`.

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
   - **1b, implicit:** the same model with Newmark + Newton, matrix-free operator and our
     CG with Jacobi behind `LinearSolver` (Eigen only as the test reference); gravity
     stages.
   Milestone 1b plan (approved 2026-11-06; one commit per step):
   1. `LinearSolver` interface, deterministic vector kernels (done).
   2. Sparsity pattern and CSR operator, for tests and as the reference (done).
   3. Reference assembly for tests: global K, M, C on the host from element matrices;
      Eigen `SimplicialLDLT` / `SparseLU` (test-only dependency).
   4. Matrix-free operator `K_eff v` with its Jacobi diagonal: state determination that
      can store the Gauss-point tangent (packed only when declared symmetric), stored and
      recomputed modes, gather / atomic / colored scatter; tangent-symmetry traits on
      materials and elements. Tests against step 3, bit-identity across variants;
      benchmarks.
   5. CG with Jacobi; tests against Eigen on the soil column (static and dynamic).
   6. GMRES(m) (FGMRES, CGS2) with Jacobi and the automatic choice from the operator's
      symmetry (with CG's fall-back on non-positive curvature); tests against Eigen
      `SparseLU` on an artificial non-symmetric operator, and the same answer as CG on the
      symmetric soil column.
   7. Schema 0.3: Static analysis with LoadControl, ModifiedNewton, optional solver
      attributes (tolerance, maximum iterations, preconditioner).
   8. Newmark (OpenSees form D), Newton, ModifiedNewton, Linear, the convergence tests,
      Static with LoadControl; true lumped mass when lumped is requested.
   9. System tests against OpenSees: implicit-dynamic gravity, static gravity, shaking;
      consistent mass; CPU and GPU.
   10. Krylov iteration study (static, and dynamic at 0.5-16x the explicit stable step, on
       three column sizes).
   11. Chebyshev with Jacobi.
   Later milestone: matrix-free geometric multigrid (plan first).
2. **Nonlinear layered soil column**: CUDA build, GPU solvers, PressureDependMultiYield,
   material stages.
3. **Laterally loaded pile**: beams, sections, embedded interfaces.
4. **DRM basin and dynamic pile-soil interaction**: H5DRM, PML, absorbing boundaries.
5. **Rotated two-pile bent**: fiber sections, ForceBeamColumn, multi-GPU using Femora's
   METIS partitions.

## Open decisions

- Test framework (GoogleTest or Catch2).
- Python bindings (nanobind or pybind11).
- Solver: decided ("Linear solvers"): matrix-free operator, own CG (symmetric positive
  definite) and GMRES (otherwise), chosen from the declared tangent symmetry; Jacobi, then
  Chebyshev and matrix-free multigrid; Eigen in tests only; hypre optional; no Trilinos.
- Details of the HDF5 schema v0.
- FemoraX license. Code ported from OpenSees is a derivative work, and the OpenSees license
  covers educational, research and non-profit use; settle this before the first public release.
