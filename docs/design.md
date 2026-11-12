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
state, or from a stored copy when the material is expensive; the material decides. An
optional `stability_tangent(p, committed, D)` gives the stable step its tangent ("Stable
time step"); without it the committed tangent serves.

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

### Element-level terms and element constants

Most elements are sums over integration points of `B^T D B` and point masses. Some also have
a stiffness of their own: SSPbrick's hourglass stabilization `Kstab`, later beams' element
matrices and interface elements. The contract has an optional part for them
(`ElementTermElement` in `contracts.hpp`), used by every kernel generically:

- **Element constants:** `NCONST` doubles per element, built once from the coordinates and
  the parameters by `element_constants(p, xl, c)`. They live in the element-constants array
  (per-element layout, `value[k * nElements + e]`), built together with the geometry cache;
  without the cache every kernel recomputes them with the same function, so both paths give
  the same bits. The element's geometry sources carry them (`geometry.constants()`), made
  by the element's own factories `recomputed_geometry(xl, p)` and `cached_geometry(g, c)`.
- **The term `K_el v` without forming `K_el`** (performance rule 3a): an element-wide
  reduction `element_term_reduce(p, c, v, z)` (`NTERM` values; for SSPbrick `z = interior *
  (Mben v)`, 12 values) and a per-node expansion `element_term_node(p, c, z, j, out)`, so the
  reduction runs once per element and the nodes split across lanes. `element_term_entry(p, c,
  a, b)` gives single entries for diagonals (Jacobi) and test references.
- **Semantics:** `K_el` is constant and part of the current, initial and committed
  stiffness alike, so every stiffness-proportional damping term includes it
  (`physics/common/element_term.hpp`). An element whose initial stiffness is its tangent
  (`INITIAL_STIFFNESS_IS_TANGENT`, as SSPbrick's `getInitialStiff`) uses `D_t` for
  `betaKinit`.
- **Where it enters:** the force (the element's `compute` starts from `K_el u` and the
  damping from `(betaK + betaKinit + betaKcomm) K_el v`, then adds the point terms; the team
  layout writes the same initial values in an extra stage), the matrix-free operator and its
  diagonal (`cK + cC (betaK + betaKinit + betaKcomm)` times `K_el`), the stable step, and the
  host reference assembly of the tests.

Adding SSPbrick took its physics header, its registry adapter and list entry, and this
generic extension (written once for every later element that needs it).

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
that sets the limit; `allow_unstable_dt` turns the stop into a warning. When a material
has history, the step is re-checked every `stable_step_check_interval` steps (100 by
default, `exec::RunOptions`): a limit below `dt` stops the run with the step, the time, the
new limit and the one at the start of the stage (with `allow_unstable_dt`, one warning per
stage), and the stage summary reports the smallest limit seen. The element's `omega^2`
reads the material's stability tangent (`stability_tangent` in `contracts.hpp`, optional;
the committed tangent when absent): the stiffest response the next step can show. An
elastoplastic tangent is softer than the elastic unloading stiffness and would overstate
the step, so PressureDependMultiYield gives its elastic moduli at the committed
confinement (reference moduli times `getModulusFactor` of the committed stress). Tests run
a single mode and a column just below and just above `dt_crit` for both schemes, with and
without each damping term.

**The periodic re-check, made cheap (2026-11-12).** The exact check builds each element's
24 x 24 stiffness and solves its eigenproblem (about 110 ms for the 46,080-element basin on a
112-core spr node, against 1.3 ms for a force evaluation), so re-checking every 100 steps
doubled the step time of plastic models. Two rules make it cheap without ever overstating
the step:
- Only batches whose stable step can change are re-checked (`stability_may_change` in
  `contracts.hpp`: a material without state cannot, and a material declares
  `STABILITY_CONSTANT` when its stable tangent is fixed despite state; Uniform damping on
  ElasticIsotropic is). The others keep the value from the start of the stage.
- A material whose stability tangent is a fixed reference times one factor per point
  (`ScaledStability`: PressureDependMultiYield, PressureIndependMultiYield and the damped
  wrapper of either) lets elements bound their omega^2 from the factors alone: at the start
  of each explicit stage every element stores reference eigenvalues
  (`max_omega_sq_reference`), and each re-check only reads its points' factors
  (`max_omega_sq_bound`). stdBrick: the largest factor times the reference (exact when every
  point has the same factor). SSPbrick (with its constant Kstab): the eigenvalue at 13
  factors from 0.25 to 16, interpolated linearly, an upper bound because the eigenvalue is
  convex in the factor. The bound is never below the true value; when it puts the step below
  `dt`, the exact check decides, so a run stops only when the stable step itself is below
  `dt`. The stage summary's smallest limit is the bound's (within about 1 % of the exact one
  on the test columns).

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
  stdBrick; SSPbrick: 33 doubles at its one point plus 110 element constants, 1.1 KB) is
  used only for elements with `GEOMETRY_CACHEABLE`, and only if it fits: with
  `GeometryCache::Auto` the cache is built when the model, the cache and the work vectors
  fit in 80% of the device's free memory (`exec::device_free_memory_bytes`). Otherwise
  `Geometry::Cached` falls back to recomputing, with the same bits on host backends; on
  GPUs the two are different kernel instantiations, which nvcc may contract differently
  (seen with PressureIndependMultiYield on SSPbrick: last bits), so there the auto-tune
  keeps to the default's geometry mode.
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

## Frequency-independent ("Uniform") damping (plan approved 2026-11-11, with schema 0.7; before H5DRM streaming)

Rayleigh damping (supported) is accurate at two frequencies only, and its stiffness part
(`betaK`) cuts the explicit stable step (7x in Femora's basin, through the absorber). Uniform
damping gives one damping ratio over a frequency band to every element, and leaves the step
almost unchanged at normal damping levels. It is what Femora's `model.damping.uniform`
already exports for OpenSees (`damping Uniform <tag> <ratio> <f1> <f2> [-activateTime Ta]
[-deactivateTime Td] [-fact tsTag]`, attached to element regions with `region ... -damp tag`;
Femora attaches every damping through regions).

### Decisions taken with the user

- Damping for FemoraX: Rayleigh (supported), Uniform (this plan), PML for absorbing
  boundaries (milestone 4). Modal damping is not planned: it needs an eigen solve, couples
  the whole model, damps a building's rigid motion with the soil unless that is removed, and
  Uniform gives nearly the same per-mode damping without these problems. Secant-stiffness
  damping is not planned either (Uniform already follows yielding).
- Compatibility levels: (1) the same Femora input means the same thing in both programs and
  FemoraX matches OpenSees where OpenSees is correct; (2) where OpenSees is incomplete or wrong,
  FemoraX does the right thing and records it in docs/deviations.md (and Femora's OpenSees
  export should warn); (3) FemoraX-only options are optional, off by default, and never an
  error: Femora's OpenSees export writes the closest OpenSees equivalent with a warning.
- The fit: each Uniform damping definition chooses how the filter weights are computed,
  `fit = "femorax"` (the default: the measured damping ratio equals the target across the
  band) or `fit = "opensees"` (OpenSees' weights, which reproduce an OpenSees run to rounding;
  for a 5 % target over 0.1-10 Hz they give 4.7 % at 0.1 Hz down to 2.8 % at 10 Hz). The run
  summary names the fit and how to switch. Decided with the user, 2026-11-11.
- Not for absorbing layers: frequency-independent damping must stiffen the response at high
  frequency to stay causal; at high ratios over a wide band the element stiffens several
  times and explicit runs go unstable (the user saw it at 95 %). Normal use: up to about
  10-20 %. Absorbers are PML's job.

### Findings that shape the design (OpenSees `955ceb0`, Stampede3, 2026-11-11)

Scripts and records: `/work2/08189/amnp95/femorax_plots/damping_test/`.

1. Element support: OpenSees filters stdBrick (6 stresses per Gauss point), `quad` (3),
   the beam-column elements (basic forces: 3 in 2D, 6 in 3D), the shells MITC4, DKGQ/DKGT
   and their NL versions, ASDShellQ4/T3 (8 section resultants per point) and zeroLength
   (one force per spring). Every other element, SSPbrick included, prints "Damping is yet to
   be implemented" and runs **undamped** (a 1 x 1 x 10 column: 0.00 % on SSPbrick). This is
   the "no effect" the user saw.
2. Level: on single-mode systems (a mass on a zeroLength spring, 5 % over 0.1-10 Hz) the
   measured damping falls from 4.69 % at 0.1 Hz to 4.45 % at 1 Hz, 3.71 % at 3 Hz and 2.79 % at
   10 Hz, the same at 50, 400 and 3200 steps per period. OpenSees fits its filters so that
   the damping term is flat within 5 %, but the filters also add stiffness that grows with
   frequency, and the damping ratio is the damping relative to the total stiffness, so the
   measured ratio drops at high frequency. The port reproduces these histories to 1.3e-14
   (step 1, `test_uniform_damping`), so the drop is the formulation's, not a bug in a
   particular element. Note: OpenSees' command passes `zeta * 2.0` to the class, which works
   with the loss factor `eta = 2 zeta` (`TclDampingCommand.cpp`).
   Correction (step 2a): the peak-to-peak decay above is biased at high frequency by the
   filters' slow relaxation; with that part removed the oscillation's damping is 4.75 % at
   0.1 Hz, 4.21 % at 1 Hz and 3.42 % at 10 Hz, and the continuous-limit formula
   (`uniform_damping_mode`) gives 4.69 / 4.18 / 3.37 %. The same formula shows the
   stiffening: the natural frequency rises by 4 % at 0.1 Hz and 19 % at 10 Hz (OpenSees fit,
   5 % over 0.1-10 Hz).
4. Stiffening is the price of causal frequency-independent damping and grows with the ratio
   and the band (FemoraX fit: up to x1.08 for 2 %, x1.22 for 5 %, x1.49 for 10 % over 0.1-10 Hz,
   at the top of the band). For buildings this shortens the periods. Default (proposed to the
   user in plain terms, 2026-11-11, who left the technical choice to us): with
   `fit = "femorax"` FemoraX scales each damped element's stiffness so the damped response has
   the specified stiffness at a reference frequency, the band's geometric mean
   `sqrt(f1 f2)` (as SW4 does for its moduli), and the run summary always reports how much the
   damping stiffens the response at f1 and f2; `fit = "opensees"` keeps OpenSees' behaviour
   (no correction). Designed and verified in step 2b.
3. Stability: the element stiffness is multiplied by `1 + sum_i 4 alpha_i xi / (2 + w_i dt)`
   (`UniformDamping::getStiffnessMultiplier`), which grows with the ratio and the band width.

### Formulation

- Mechanism (Huang, Sturt & Willford 2019; OpenSees `SRC/damping/UniformDamping.cpp`): a
  set of first-order filters at frequencies `w_i` spread over the band with weights
  `alpha_i`; each element filters its own damping quantities `q` (below) and adds the
  filtered difference as a damping force. Static stages track `q` without damping; damping
  is applied for `Ta < t < Td`, times the optional time-series factor. Rigid motion gives
  `q = 0`, hence no damping; a yielding member's `q` stops growing, and so does its damping.
- **Deliberate difference: the fit target.** FemoraX fits `alpha_i` so that the damping
  ratio as measured (the damping relative to the stiffened response) equals the target over
  the band, within the same 5 %, instead of the damping term alone (step 2). The OpenSees fit
  stays available (the port of step 1) for comparisons with OpenSees; the difference is
  recorded in docs/deviations.md with this evidence.
- **Damping quantities per element type** (element contract, `contracts.hpp`: an optional
  `DampedElement` part, like `ElementTermElement`): the element declares `NDAMPQ` values
  per point (or per element), how to compute them from its state (stresses for solids,
  basic forces for beams, section resultants for shells, spring forces for zero-length
  elements), how to turn a damping quantity back into nodal forces (the element's `B^T`
  for solids, its basic-to-global transformation for beams), and a **group label** per
  value (`volume` / `shear` for solids, `axial` / `bending` / `torsion` for beams,
  `membrane` / `bending` / `transverse_shear` for shells). First version: one ratio for
  every group; per-group ratios are a later, optional extension that needs no redesign.
- State: per point (or per element) `NDAMPQ * (nFilter + 2)` doubles, in the committed and
  trial buffers like material state (commit swaps; retries and substeps work unchanged);
  about 5.4 KB per stdBrick with 5 filters, 0.7 KB per SSPbrick or 3D beam. `nFilter` is
  fixed per damping definition, so it is a compile-time bucket like the yield surfaces.
- **Zero cost when unused:** damped and undamped elements are different batches, so an
  undamped element keeps exactly today's code, state and memory (tested: the same bits and
  memory estimate as before). A ratio of 0 is no damping.
- Explicit stages: the damping force joins the element force in the same fused kernel. The
  stable step uses the stiffened response (`1 + km`, exact per element): a step above it
  stops the run with a message, as for Rayleigh damping.
- Implicit stages: the operator gets the stiffened tangent (`(1 + km) K_t` plus the damping
  term), so Newton stays consistent; the matrix-free operator, its Jacobi diagonal and the
  stored-tangent path include it. Symmetry: unchanged (a scalar multiple per element).

### The step context and the damped-material wrapper (plan for step 2b-2, 2026-11-11; approved by the user, who left the implementation choices to us)

**The blocker.** The filters need the step size `dT`, the trial time `t` and whether the stage
is dynamic, at every step and every substep. Nothing passes them today: a material's
`update(p, committed, trial, eps, sig, D)` and an element's `compute` / `compute_with` /
`point_update_shape` see parameters, state, strains and velocities only, and the parameter
sets on the device (`pr.p`, `pr.damp`) are fixed for a stage.

**Options.**
1. *Recommended:* a plain struct `StepContext { double t, dT; int dynamic; }`
   (`physics/step_context.hpp`), passed by value from the time loop to every kernel launch
   (a launch argument: no device copy, nothing allocated, the same value for every thread,
   so determinism is unchanged), and on to the elements and materials. Materials that ignore
   it keep their current `update`; the material concept accepts either form, and a small
   adapter calls the old one, so ElasticIsotropic, PressureDependMultiYield and
   PressureIndependMultiYield do not change. Elements forward it from `compute`,
   `compute_with` and `point_update_shape` (stdBrick, SSPbrick, and the compacted path's
   `Replay`). Substeps and retries pass their own `dT` and `t`; static stages pass
   `dynamic = 0`.
2. Store `dT` and `t` in the device parameter sets and update them every step: a host-to-
   device copy per step and per substep, against "nothing in the time loop"; rejected.
3. A device-wide "current step" value read by the physics: global mutable state, forbidden
   in the physics layer (CLAUDE.md rule 2); rejected.

**The damping as a material wrapper.** With the step context, Uniform damping becomes a
material wrapper `UniformDamped<Mat, NF>` (NF = a compile-time bucket of filters, for
example 8, 16 and 32): its state is `Mat`'s state followed by the filter state of the 6
stresses; its `update` calls `Mat::update` (stress `sig`, tangent `D`), filters `sig`
(`uniform_damping_update`), returns `sig + qd` and `D * (1 + km)` (OpenSees' Brick:
`dd *= getStiffnessMultiplier()`). The element force `B^T sig dvol` then includes the damping
force exactly as `uniform_damping_element_force` computes it (step 2b-1), and:
- every kernel variant (flat, team, compacted; cached and recomputed geometry; the implicit
  operator and its stored tangents) includes the damping with no kernel change beyond
  forwarding the context, so bit-identity across variants is kept by construction
  (`test_memory_contract` extended to a damped material);
- damped and undamped elements are different batches automatically (a different material
  type), so undamped models keep today's code, state and bits;
- recorders: `stress3D6` reports the material's own stress, without the damping stress (as
  OpenSees' `getStress`); the wrapper keeps the material stress in its state for this;
- the stable step: the wrapper's stability tangent is `Mat`'s times the largest multiplier,
  `1 + 2 eta sum_i alpha_i` (its limit as `dT -> 0`), a conservative bound with no `dT`;
- the stiffness correction (default with `fit = "femorax"`) lives in the wrapper too: it
  stores the stress at activation and subtracts `R_ref (sig - sig_activation)` from the
  damping stress, so gravity stresses are untouched.

**Cost to watch.** Each (element, material, NF bucket) triple is one more instantiation of
every kernel: 2 elements x 5 material entries x 3 buckets = 30 more, which lengthens the
CUDA and SYCL builds. Mitigation: the buckets 8 and 32 only at first; registration of the
damped wrapper for ElasticIsotropic and PressureIndependMultiYield first (the basin's
materials), the others when a model needs them.

**Steps of 2b-2 (one commit each):** (i) the step context through the time loops, kernels,
elements and the material concept, no behaviour change (all tests unchanged, bit for bit).
Done (2026-11-11), with less plumbing than planned: `StepContext` (`physics/contracts.hpp`)
travels inside the `Rayleigh` struct every element already receives where it updates its
material, and inside each batch's `DevBatch::step`, which every kernel already captures;
`InternalForce::step` is set by the time loops (ExplicitDifference and CentralDifference at
the force evaluation, every implicit attempt and substep) and handed to the batches before
each launch. Elements and kernel helpers call materials through `material_update<Mat>`,
which passes the step to a material whose `update` takes one and calls the others as
before; no element, material or kernel signature changed. All tests unchanged, and the
explicit force kernels within run-to-run noise (40^3 stdBrick, 96 threads: flat_gather
1261 / 1116 us, flat_atomic 1335 / 1347 us before / after);
(ii)-a the wrapper as physics. Done (2026-11-11): `physics/damping/uniform_damped.hpp`,
`UniformDamped<Mat, NF>` (state: the material's, then the filters'; the update with the step:
the material's stress plus the damping stress and its tangent times the multiplier; static
without a step; the stability tangent with the bound `1 + 2 eta sum alpha_i`; the recorded
stress is the material's own). Tests through the element's own `compute<true>` with the step
in the `Rayleigh` struct: static steps equal the undamped stdBrick bit for bit (force and
tangent), a damped stdBrick mode decays like the single-mode spring within 9e-15 (both fits),
and recorders read the material's stress.
(ii)-b damped batches. Done (2026-11-11): `io::UniformDampingRegion` and `Model::uniform`
(built by tests until the schema 0.7 reader, step 4); the registry's damped entries
`UniformDampedEntry<Base, NF>` (ElasticIsotropic in the buckets 8 and 32 so far),
`registry::pair_index(type, material, filters)` picking the smallest bucket that holds a
region's fit (undamped entries for elements outside any region), and
`registry::uniform_damping_filters` (the region's fit, once per region, `fit = "femorax"` by
default); batches key their parameter sets by (group, Rayleigh region, Uniform region). Test
through the real force kernels (stdBrick and SSPbrick columns): a static step of the damped
column equals the undamped column bit for bit, a dynamic step at a changed displacement gives
the damping force, and every tunable kernel variant gives the same bits.
Open for the next steps: the DRM layer refuses materials with state, so a damped elastic DRM
layer is refused today (decide what DRM forces do with Uniform damping); the H5DRM kernel and
the constant-tangent operator need CONSTANT_TANGENT; a free-vibration run through
`run_model` once the schema reader exists;
(ii) the wrapper for ElasticIsotropic with the free-vibration tests through the real
kernels; (iii) the stable step and the stiffness correction; (iv) PressureIndependMultiYield
and PressureDependMultiYield.
(iii)-a the stable step. Done (2026-11-11): through the real kernels the damped column's
stable explicit step is the undamped one over `sqrt(1 + 2 eta sum alpha_i)`, to 1e-12, for
stdBrick and SSPbrick (5 % over 0.1-10 Hz, FemoraX's fit: 0.7932 of the undamped step, a 21 %
smaller step: the stiffening at the mesh's highest frequencies; Rayleigh `betaK` cost 7x in
the basin). Next: the stiffness correction (iii)-b.
(iii)-b the stiffness correction. Done (2026-11-11): `UniformDampingFilters::kcorr`, the
filters' stiffening at `w_r = 2 pi sqrt(f1 f2)`, `sum_i 2 eta alpha_i w_r^2 / (w_r^2 +
omegac_i^2)` (`uniform_damping_correction`), set by FemoraX's fit, which now refits the
weights with the correction on (0 for OpenSees' fit). The wrapper stores the material
stress when the damping starts to act (6 more values per point; it follows the stress in
static steps and before `ta`) and, while the damping acts, subtracts `kcorr (sig - sig_act)`
from the stress and `kcorr D` from the tangent; the stability tangent uses
`1 + 2 eta sum alpha_i - kcorr`. 5 % over 0.1-10 Hz: kcorr 0.206 with 7 filters, damping
0.962-1.015 of the target, frequency x1.002 at the reference and x0.93-x1.09 over the band
(without the correction up to x1.19), the stable step 0.890 of the undamped one (0.793
before). A gravity stress at activation gives the material's stress bit for bit.
(iv) the multi-yield materials wrapped. Done (2026-11-11): registry entries
`UniformDampedEntry<PressureDependMultiYieldEntry<20>, NF>` and the same for
PressureIndependMultiYield, NF 8 and 32 (the 40-surface buckets are not wrapped yet: a damped
material with more than 20 surfaces is an unsupported pair; one line each when a model needs
it). The material stage reaches the wrapped Params (`set_material_stage` looks for
`mat.loadStage` too); the wrapper forwards `recorded_strain`. A wrapped material runs the
fused update (the compacted predictor / corrector path needs a split wrapper, later). Tests: a
wrapped PIMY point equals the plain material bit for bit in static steps through the stage
switch and into yielding; under shear at a constant rate onto the perfectly plastic surface
the damping stress dies out (1.7e-8 after 24 s on the plateau, against 159 for the elastic
material at the same rate; its largest value 2.0) (verification item 5); damped PIMY and PDMY
columns through the real kernels take the stage, equal the undamped column in static
evaluations bit for bit, and give the same bits in every kernel variant.

### Verification

1. Single-mode free vibration (a mass on a spring, then a brick, then a beam) at 8-10
   frequencies across the band: the measured ratio equals the target within 5 % (the
   design accuracy); outside the band it falls off as the fit predicts.
2. The same systems against OpenSees on the elements OpenSees supports (stdBrick first,
   beams with milestone 3): the force history at the OpenSees fit (a test-only option) to
   rounding, so the mechanism itself is verified; the default fit then differs only by the
   documented target change.
3. SSPbrick damped as specified (OpenSees: none; a documented deviation).
4. Rigid motion: a block translating and rotating gets no damping force.
5. Yielding: a PressureIndependMultiYield column, damping force bounded by the stress.
6. Stable step: the check refuses a ratio and band whose stiffening breaks the explicit step,
   and a run just below the limit is stable.
7. Plane-wave attenuation in an elastic column against the analytical constant-Q solution
   (frequency-independent attenuation over the band), and the Hayward basin with Uniform in
   the soil against Rayleigh. Plane wave done (2026-11-12,
   `UniformDampingWave.PlaneWaveAttenuationIsTheConstantQOne`): a 2 Hz Ricker S wave in a
   600 m column (Vs 200 m/s, 5 % over 0.1-10 Hz): the spectral ratio between 100 and 300 m
   matches the damped material's complex modulus within 0.2 % over 0.7-4 Hz, and the measured
   damping is 0.963-0.976 of the target (the fit's range: 0.962-1.015). The basin: next.
8. Undamped models: bit-identical and the same memory as before.

### Schema 0.7 (proposal; asks the user before any change)

`/damping/uniform/<index>` (optional), one per Femora damping region, like
`/damping/rayleigh/<index>`: attrs `ratio` (float64, > 0), `freq1`, `freq2` (float64,
0 < f1 < f2), optional `activate_time`, `deactivate_time` (float64), optional `factor_series`
(int32, a time-series tag), optional `fit` (string, `"femorax"` by default or `"opensees"`);
dataset `elements` (int32 tags). An element in both a Rayleigh
and a Uniform region gets both (as OpenSees allows). Later, optional: per-group ratios
(`ratio_axial`, ... ) as more attributes.

### Steps (one commit each, after approval)

1. The filter fit and the single-mode tests (host, physics only). Done (2026-11-11):
   `physics/damping/uniform_damping.hpp` (OpenSees' update and stiffness multiplier, state
   through `Strided` views) and `compile/uniform_damping_fit.hpp` (the fits, host code), `test_uniform_damping` against stored OpenSees records
   (`tests/reference/damping/`, `tests/reference/data/damping/`): within 1.3e-14.
2. (a) FemoraX's fit. Done (2026-11-11): `uniform_damping_mode` (the oscillation's damping
   and frequency, the continuous limit of the filter, confirmed against step-by-step
   histories within 1.5 %) and `uniform_damping_fit_femorax` (Gauss-Newton on OpenSees'
   filters, from OpenSees' weights, one more filter when needed): within 0.95-1.05 of the
   target over the band for 2, 5 and 10 % over 0.1-10, 0.5-20 and 0.2-2 Hz (2-12 filters);
   step by step, 5 % over 0.1-10 Hz gives 4.98 / 5.03 / 4.87 % at 0.1 / 1 / 10 Hz.
   (b-1) Inside solid elements, physics only. Done (2026-11-11):
   `physics/damping/element_damping.hpp` (the `UniformDampedSolid` concept: the point
   stresses are the damping quantities; the damping force `B^T qd dvol` through the element's
   own `point_node_force`, generic over stdBrick and SSPbrick); a stdBrick started in one of
   its modes decays exactly like the single-mode spring of that frequency (within 5e-15,
   both fits) and rigid motion gives no damping force (stdBrick and SSPbrick).
   (b-2) The kernels: damping state per batch (committed and trial, the number of filters a
   compile-time bucket), damped and undamped batches, the damping force in every force-kernel
   variant (bit-identical across layouts), the stable step with the stiffened response, and
   the stiffness correction at the band's geometric mean. The correction must act only on the
   change of the damping quantities since the damping was activated: applied to the full
   stress it would push on the gravity stresses when shaking starts.
3. Implicit stages (operator, diagonal), tests against the reference assembly and OpenSees.
   Done (2026-11-11), with no code change: the wrapper's tangent `D (km - kcorr)` is the exact
   derivative of its stress, and the stored-tangent operator, its Jacobi diagonal and every
   implicit attempt and substep already carry it and the step context. Test
   (`Implicit.UniformDampingNewmarkMatchesTheHostRecurrence`): Newmark with Newton on the
   damped column (Rayleigh damping and a base motion too) through `run_model` against the
   host recurrence whose residual comes from the production force kernel: within 9e-14
   (stdBrick) and 2.1e-13 (SSPbrick) of the peak, 1.6 Newton iterations per step (the
   consistent tangent; 5-7 with the host's approximate matrix). The OpenSees comparison of an
   implicit damped stdBrick is the single-mode test of step 2b (Newmark with Newton through
   the element's own `compute<true>`, against the spring that matches OpenSees' records).
4. Schema 0.7, reader and fixtures; the converter; support table.
   Done (2026-11-12): `/damping/uniform/<index>` as proposed (schema/v0/README.md 3.7.1),
   reader 0.7; `factor_series` is read and refused as unsupported until `-fact` is
   implemented; a damped multi-yield material with more than 20 surfaces is refused by
   `check_supported` with a message that says so; 14 fixtures. The converter reads
   `damping Uniform` and `region ... -damp`. The run summary names each region's fit, the
   damping range over the band, the frequency change at f1 and f2 and how to switch fits.
5. Tests 5-7; benchmarks of the damped kernels.
Beams, shells and springs get it with their own milestones by declaring their damping
quantities.

## Damped DRM layer (plan approved by the user, 2026-11-12)

**Decision with the user (2026-11-12):** the DRM layer is soil. Damping assigned to the soil
applies to the DRM-layer elements too, and the DRM forces include the layer's damping, so
the free field is reproduced by a damped layer exactly as by an undamped one. A deliberate
difference from OpenSees, whose H5DRMLoadPattern forces use the layer's mass and stiffness
only (with a damped layer there the forces are inconsistent by about the damping ratio times
the stiffness forces). Today FemoraX refuses a damped DRM-layer element (unsupported).

**Formulation.** For a DRM element with boundary nodes b and exterior nodes e, with the free
field u0 (and its velocity v0, acceleration a0) at its nodes, the effective forces are the
element's own response to the free field restricted to one side,
`Peff_b = -[M a0_e + F(u0_e, v0_e)]_b` and `Peff_e = [M a0_b + F(u0_b, v0_b)]_e`, where `F` is
the element's force for a linear material: today `K u`; with damping
`K u + C v` (Rayleigh: `alphaM M v + (betaK + betaKinit + betaKcomm) K v`, the layer's
coefficients) plus the Uniform damping stress of the restricted field. The Uniform filters
are linear but have memory, so each of the two restricted fields of a DRM element has its
own filter state per Gauss point, updated at every load evaluation with the step context,
committed and reverted with the model's state (the stiffness correction acts on the change
since activation, as in the wrapper). The layer's material must still be linear (elastic,
with or without damping); a material that can yield stays refused.

**Inputs.** The velocity of the free field: the H5DRM file's velocity dataset when it has
one (to be checked in OpenSees' format and SW4's files), otherwise from the displacement
samples (central difference), verified against a file that carries both.

**Cost** (estimate, to be measured with a benchmark): the DRM layer is a one-element shell,
a few percent of the elements; per DRM element the force kernel about doubles (one more
strain-to-stress pass and the filter update). Memory: two filter states per Gauss point,
about 15 KB per damped stdBrick DRM element with 8 filters (both buffers); about 50 MB for a
basin-sized layer. Reading the histories stays the expensive part of DRM (streaming, step 4
of milestone 4).

**Verification.** The free-field test (`test_drm_free_field`) with damping: the free field of
a damped laminar column, computed by FemoraX, drives a damped box inside a damped DRM layer;
the box must move as the free field to rounding (2.2e-14 today without damping), for
Rayleigh and for Uniform damping, explicit and implicit, and the scattered field outside must
vanish. The same run with the old forces measures the inconsistency they leave.

**Steps (one commit each, after approval):** (1) the free-field velocity input and the
Rayleigh terms; (2) the Uniform damping state and terms; (3) the deviation entry, the
support table and the basin with Uniform damping in all its soil. To be done together with
the H5DRM streaming rework of the same kernel.

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

### Krylov iteration study (milestone 1b step 10, 2026-11-07)

CG with Jacobi, relative tolerance 1e-12, consistent mass, n x n x 40 soil columns
(`tests/system/soil_column/krylov_study.py`; GPU, average iterations per solve). Implicit
dynamic steps at multiples of the explicit stable step (5.28e-4 s for this mesh, with
damping):

| column | equations | static gravity | 0.5 dt_crit | 1 dt_crit | 4 dt_crit | 16 dt_crit |
|---|---|---|---|---|---|---|
| 10x10x40, Rayleigh damping | ~14.5k | 145 | 42 | 39 | 41 | 77 |
| 20x20x40, Rayleigh damping | ~53k | 214 | 43 | 39 | 41 | 90 |
| 40x40x40, Rayleigh damping | ~202k | 319 | 43 | 38 | 41 | 95 |
| 10x10x40, no damping | ~14.5k | 145 | 47 | 43 | 38 | 74 |
| 20x20x40, no damping | ~53k | 214 | 47 | 43 | 38 | 86 |
| 40x40x40, no damping | ~202k | 319 | 47 | 42 | 38 | 90 |

What it shows:
- **Dynamic steps up to a few times the explicit step:** about 40 iterations, flat in the
  model size, as the theory says (`M/dt^2` dominates `K_eff`). The count is set by the
  tolerance and the consistent mass's own conditioning, not by the mesh.
- **Larger steps** (16 dt_crit): 75-95, growing slowly with the model.
- **Static gravity:** 145 -> 214 -> 319 as the column widens (about n^0.55 here); at the
  target scale this becomes thousands. Static and gravity stages need multigrid.
- **Explicit versus implicit on this mesh:** one implicit step costs about as many
  operator applications as its CG iterations, each about one explicit force evaluation.
  At 16 dt_crit an implicit step (~90 applies) costs more than the 16 explicit steps it
  replaces. With Jacobi, implicit dynamics pays only where the explicit step is very
  small for other reasons (stiff pore water, contact, tiny elements), or with a better
  preconditioner.
- **Next:** Chebyshev-Jacobi (step 11) to cut the dynamic counts; matrix-free multigrid
  (later milestone) for static gravity and large steps.

Chebyshev-Jacobi (step 11, `src/solvers/chebyshev.hpp`; 40x40x40, damped; CG iterations,
and operator applications = iterations x degree):

| preconditioner | static gravity | 1 dt_crit | 16 dt_crit | GPU wall time, 10 steps at 1 / 16 dt_crit |
|---|---|---|---|---|
| Jacobi | 319 (319) | 38 (38) | 95 (95) | 4.4 s / 7.1 s |
| Chebyshev degree 2 | 167 (334) | 21 (42) | 50 (101) | 4.7 s / 7.4 s |
| Chebyshev degree 4 | 89 (356) | 19 (76) | 26 (105) | 6.3 s / 7.4 s |
| Chebyshev degree 8 | 52 (416) | 10 (78) | 15 (116) | 6.5 s / 8.0 s |

It cuts the CG iterations (and with them the global reductions: two dot products per
iteration) by the degree, but not the operator applications, so on one GPU it is no
faster. Jacobi stays the automatic choice. Chebyshev is for later: across many GPUs, where
each global reduction is a synchronization of every device, and as the smoother inside
matrix-free multigrid. It stays available behind the developer override
`FEMORAX_PRECONDITIONER=chebyshev[:degree]` for such studies.

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
  rewrite both as fixed-size local data. Done for PressureDependMultiYield (2026-11-09):
  `MaterialPath::Compacted` (`src/exec/compacted_kernels.hpp`), for any `SplitMaterial`
  in an element with `WithMaterial`: predictor one thread per point, a scan of the plastic
  points, the corrector one thread per plastic point, then any force-kernel variant with
  the element instantiated on `Replay<Mat>` (reads the stored stress and tangent).
  Bit-identical to the fused path on host backends (in the auto-tune there), rounding on
  GPUs; slower than the fused team layout on the laptop GPU (benchmarks/README.md), kept
  for the TACC benchmarks. Not yet in the stored-tangent state determination of implicit
  stages, which runs the fused update.
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
2. **Nonlinear layered soil column**. Plan approved 2026-11-07 (one commit per step, a
   report after each):
   1. SSPbrick through the element registry, verified against OpenSees like stdBrick, with
      the generic element-term extension ("Element-level terms and element constants")
      (done).
   2. Schema 0.4 (approved): PressureDependMultiYield and PressureIndependMultiYield with
      Femora's parameters and optional `pairs` (nd = 3 only), the `updateMaterialStage`
      action (stage 0 or 1), `stress3D6` and `strain3D6` in VTKHDF. Then PDMY, then PIMY:
      fixed-size state with a compile-time maximum number of yield surfaces per batch
      (buckets 20 and 40, a new bucket is one line), automatic surfaces (positive N) and
      user backbones (negative N with pairs); no heap and no static workspaces; state
      through the strided views; surface loops through `inner_for` / `inner_reduce`; the
      material stage as OpenSees' `updateMaterialStage`; the tangent symmetry declared from
      the OpenSees source; a fused kernel and a compacted plastic-corrector variant, both
      kept for TACC benchmarks; the state size per point reported. `pa` per material (a
      deviation: OpenSees keeps one global `pAtm`, the last constructed material's), with
      a standalone OpenSees issue example.
      Done so far: schema 0.4 (reader, fixtures; the build refuses the new items until each
      runs); the PressureDependMultiYield material (`physics/materials/pressure_depend_multi_yield.hpp`,
      bit-identical to OpenSees along 32 strain paths; state 89 + 6 NS doubles per buffer,
      3.3 KiB per Gauss point for NS = 20 with both buffers, 5.1 KiB for NS = 40; Params
      544 / 864 bytes per parameter set). Its surface loops are written with `inner_for`,
      but later code reads surfaces another index wrote (the copy of the committed surfaces,
      `updateInnerSurface`), which needs a lane barrier the `VectorLanes` context does not
      have: until the lane-parallel corrector is designed and benchmarked, the material
      runs one thread per point. Material state in the kernels: two buffers per batch
      (committed and trial, the per-point layout of the geometry cache), commit swaps them,
      nothing to revert; the stage action updates the parameter sets on the device;
      implicit stages with Linear and a history material are refused (OpenSees commits a
      stress inconsistent with the strain there); the recomputed operator tangent falls
      back to the stored one for history materials. Single elements run against stored
      OpenSees output (`tests/system/soil_column/nonlinear_check.sh`). Element output: the
      VTKHDF recorder writes `stress3D6` and `strain3D6` as OpenSees does (each element's
      average over its points of the material's getStress / getStrain after the commit,
      Brick's and SSPbrick's averaging rules), computed on the device by an element-output
      kernel and copied with the node data. Steps 4 and 5 done (2026-11-09): Femora's
      retry and substep recovery (schema 0.5; ModifiedNewton as OpenSees runs it, the
      stored tangent kept across steps with -factoronce and formed afresh after a failed
      attempt or a new step size), and Femora's example column, exported by Femora, on the
      first 2 s of its shaking against stored OpenSees output
      (`tests/system/femora_example/`): the same retries, fields within 1.1e-11 on the CPU
      and 5.6e-5 on the GPU. Event sequences compare retries and substeps; active surfaces
      are not compared yet (docs/todo.md). Compacted-corrector kernel done (2026-11-09).
      PressureIndependMultiYield done (2026-11-09): bit-identical to OpenSees along 34
      strain paths; per-point moduli and surfaces (paramScaling at the switch) in the
      state, 30 + 8 NS doubles per buffer (2.97 KiB per point for NS = 20 with both
      buffers); stored system cases (docs/deviations.md).
   3. Verification against OpenSees: strain paths at the Gauss-point level (the OpenSees
      material called directly, no domain), then single elements, then the system test.
   4. System test: Femora's nonlinear layered soil column (its own 1x1x25 column, staged
      elastic gravity, the switch to plastic, shaking with FrequencySweep from 1.9 to
      3.9 s stored as a trimmed record), FemoraX against OpenSees on CPU and GPU, explicit
      and implicit where the example allows. Tolerances: exact event sequences (active
      surfaces, substeps, retries) plus value tolerances set from the first runs and
      justified in docs/deviations.md; how often GPU and CPU take different plastic paths
      is reported.
   5. Femora's retry and substep recovery, tested on a case that needs it.
   OpenSees references for nonlinear materials stay tiny (material paths; the 1x1x25
   column, 1-2 s): run once, stored in the repository, compared against by the tests;
   OpenSees never runs in the normal check. Bigger nonlinear models run only in FemoraX.
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
