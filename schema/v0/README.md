# FemoraX model file, schema version 0.5

Status: **0.0, 0.1 and 0.2 approved 2026-11-05; 0.3 approved 2026-11-06; 0.4 approved 2026-11-07; 0.5 approved 2026-11-09**. Every change bumps the version (CLAUDE.md rule 9); see
"Versioning".

Scope: what milestone 1 needs to run Femora's layered elastic soil column
(`examples/site_response/layered_elastic_soil_column.py`) in both forms:

- **1a, explicit:** stdBrick (lumped) and SSPbrick, ElasticIsotropic, fix, equalDOF (and the
  laminar boundary, which Femora writes as equalDOF), per-region Rayleigh damping,
  UniformExcitation with a Path time series, ExplicitDifference and CentralDifference
  (diagonal case), node and VTKHDF recorders.
- **1b, implicit:** the same, plus consistent mass, the gravity stage, Newmark with the
  Linear and Newton algorithms, convergence tests, and `loadConst`/`setTime` actions.
  The schema describes these now so 1b needs no format change; the 1a reader rejects them
  with "available in milestone 1b".
- **2, nonlinear soil column (0.4):** the PressureDependMultiYield and
  PressureIndependMultiYield materials, the `update_material_stage` action, and the VTKHDF
  element responses `stress3D6` and `strain3D6`. The reader accepts them; a build runs
  them once each is ported and verified (milestone 2), and until then refuses them as
  "not available yet".

Everything else stops the reader with a message that names the component and suggests the
OpenSees backend (see "Unsupported components").

Reference versions: Femora `79bd4f6`, OpenSees `955ceb0` (`docs/dependencies.md`).

---

## 1. Versioning

`schema_version` is a pair `[major, minor]`.

- **Minor** bump: a backward-compatible addition, such as a new optional attribute or
  dataset, or a new accepted value of a string attribute. A reader accepts every file with
  its own major version and a minor version up to its own; older minor versions read the
  same as before.
- **Major** bump: any breaking change (a renamed, removed or newly required item, or a
  change of meaning). A reader rejects other major versions with a message naming both
  versions.
- Each major version has its own directory, `schema/v<major>/`; this README is updated in
  place for minor versions, and the history below records what each minor version added.

| Version | Change |
|---|---|
| 0.0 | first version |
| 0.1 | VTKHDF recorder: optional `elements` dataset (the element tags to record); `region` stays rejected |
| 0.2 | equalDOF: optional `command` dataset (which `equalDOF` command each row comes from) |
| 0.4 | nD materials `PressureDependMultiYield` and `PressureIndependMultiYield` (with the optional `pairs` dataset); process step `update_material_stage`; VTKHDF responses `stress3D6`, `strain3D6` |
| 0.5 | Femora's retry and substep recovery defined: `max_retries`, `num_sublevels`, `num_substeps` (attributes since 0.0) with their rules |
| 0.3 | analysis: `analysis_type` `"Static"` with the `LoadControl` integrator (`load_increment`, optional `num_iter`, `min_increment`, `max_increment`); `algorithm` `"ModifiedNewton"` (optional `modified_newton_initial`, `modified_newton_factor_once`); optional FemoraX solver attributes `solver_tolerance`, `solver_max_iterations`, `preconditioner` |

## 2. Conventions

- **File:** HDF5, written with the file format of HDF5 1.10 or older (h5py:
  `h5py.File(path, "w", libver=("earliest", "v110"))`), so FemoraX's HDF5 1.10 library can
  read it. Any file name (`.femorax.h5` suggested). One file holds the whole model; the
  reader never opens other files (time series values are stored inline).
- **Strings:** UTF-8 attributes, fixed or variable length (h5py's default `str` is fine).
- **Booleans:** `uint8`, 0 or 1.
- **Integers:** tags and indices are `int32` (OpenSees tags are `int`); counts are `int64`.
- **Reals:** `float64` everywhere (rule 4).
- **Arrays:** row-major, as h5py/numpy write them. `coords[N,3]` means N rows of 3. The
  reader converts to FemoraX's structure-of-arrays layout; the file does not need to.
- **Tags:** positive `int32`, unique within their kind (node, element, material, time
  series, pattern), exactly as in the OpenSees Tcl file Femora would write, so results can
  be compared tag by tag. FemoraX renumbers internally and maps results back to these
  tags. Groups that are only containers (element groups, damping regions) are named by the
  exporter, not by Femora tags; Femora's tag and name are kept as metadata.
- **DOF numbers:** 1-based, as in OpenSees (`dof = 1` is x translation for a 3-DOF node).
- **Ordered children:** groups whose order matters (`/process`) name their children with
  zero-padded indices, `000000`, `000001`, ...; the reader sorts by name.
- **Strictness:** unknown groups, datasets or attributes are errors, not warnings, so an
  exporter/reader mismatch is caught at once. An item added in a later minor version than
  the file declares is also an error. Optional items are marked *(optional)*; everything
  else is required.
- **Metadata:** attributes `femora_tag` (int32) and `femora_name` (string) are allowed,
  *optional*, on every group that comes from a Femora object. They are informational and
  never used to resolve references.
- **Paths** (recorder output files) are used as written; relative paths resolve against
  the directory FemoraX runs in, as OpenSees does. `$pid` in a path is replaced by the
  partition id (always `0` in v0), as in Femora's Tcl.

---

## 3. Layout

```
/                                  attrs: schema_version, ndm, generator, model_name, ...
/nodes/                            tags, coords, ndf
/materials/nd/<tag>/               one group per nD material
/elements/<id>/                    one group per Femora element definition
/mass/                             (optional) nodal masses
/constraints/sp/fix/               (optional) one row per fixed DOF
/constraints/mp/equal_dof/         (optional) one row per tied DOF pair
/damping/rayleigh/<index>/         (optional) per element region, in file order
/time_series/<tag>/                (optional)
/patterns/<tag>/                   (optional)
/recorders/<name>/                 (optional)
/process/<index>/                  ordered analysis steps and actions
```

### 3.1 Root attributes

| Attribute | Type | Meaning |
|---|---|---|
| `schema_version` | `[2]` int32 | `[0, 5]` (a reader of 0.5 also reads 0.0 to 0.4) |
| `ndm` | int32 | spatial dimension; must be `3` in v0 |
| `generator` | string | e.g. `"femora 0.9.1"`; informational |
| `model_name` | string | Femora `model_name`; informational |
| `units` | string *(optional)* | informational only; FemoraX is unit-agnostic |

### 3.2 `/nodes`

| Dataset | Shape, type | Meaning |
|---|---|---|
| `tags` | `[N]` int32 | node tags |
| `coords` | `[N, ndm]` float64 | coordinates, full precision (never rounded) |
| `ndf` | `[N]` int32 | DOFs per node (`node ... -ndf <n>`); any positive value |

Femora source: `assembled_mesh.points`, `point_data["ndf"]` (ghost-node sentinels
`>= 1000` resolved with `GhostNodeElement.resolve_ndf`), tags
`start_node_tag + point index`.

### 3.3 `/materials/nd/<tag>`

One group per material; the group name is the material tag (the OpenSees tag, which
element groups reference). `femora_name` metadata may hold Femora's `user_name`.

| Attribute | Type | Meaning |
|---|---|---|
| `type` | string | `"ElasticIsotropic"`, `"PressureDependMultiYield"` *(since 0.4)* or `"PressureIndependMultiYield"` *(since 0.4)* |
| `E`, `nu`, `rho` | float64 | `nDMaterial ElasticIsotropic <tag> E nu rho` |

**PressureDependMultiYield** *(since 0.4)*: `nDMaterial PressureDependMultiYield <tag> nd rho
refShearModul refBulkModul frictionAng peakShearStra refPress pressDependCoe PTAng contrac
dilat1 dilat2 liquefac1 liquefac2 liquefac3 [noYieldSurf [gamma Gs ...] e cs1 cs2 cs3 pa c]`,
as Femora's `pressure_depend_multi_yield` writes it (`liquefac3` is OpenSees'
`liquefactionParam4`). Attribute names are Femora's and OpenSees' Tcl names. Not in the
schema: OpenSees' `hv` and `pv` (Femora never writes them; OpenSees' defaults 0 and 1, the
volumetric-plasticity branch off).

| Item | Type | Meaning |
|---|---|---|
| attr `nd` | int32 | `3` (`2`, plane strain, is unsupported: no plane elements) |
| attrs `rho`, `refShearModul`, `refBulkModul`, `frictionAng`, `peakShearStra`, `refPress`, `pressDependCoe`, `PTAng`, `contrac`, `dilat1`, `dilat2`, `liquefac1`, `liquefac2`, `liquefac3` | float64 | as in the Tcl command (`refPress` positive; FemoraX applies OpenSees' sign convention internally) |
| attr `noYieldSurf` | int32 *(optional)* | number of yield surfaces, default 20; negative: `abs(noYieldSurf)` user-defined surfaces from `pairs` |
| `pairs` | `[K, 2]` float64 *(only with noYieldSurf < 0)* | `K = abs(noYieldSurf)` rows of (shear strain gamma > 0, G/Gmax in (0, 1]) |
| attrs `e`, `cs1`, `cs2`, `cs3`, `pa`, `c` | float64 *(optional)* | defaults of OpenSees' Tcl parser: 0.6, 0.9, 0.02, 0.7, 101, 0.3 (Femora writes the same defaults) |

Rules (invalid otherwise): every value finite; `rho >= 0`; `refShearModul`, `refBulkModul`,
`peakShearStra`, `refPress`, `pa` positive; `0 < frictionAng < 90`; `0 < PTAng <= 90`;
`pressDependCoe`, `e`, `cs1`, `cs2`, `cs3`, `c` not negative; `noYieldSurf` non-zero with
`abs(noYieldSurf) < 40` (Femora's rule). `PTAng > frictionAng` with automatic surfaces is
accepted with a warning: OpenSees sets `PTAng = frictionAng`, and so does FemoraX.

**PressureIndependMultiYield** *(since 0.4)*: `nDMaterial PressureIndependMultiYield <tag> nd rho
refShearModul refBulkModul cohesi peakShearStra frictionAng refPress pressDependCoe
noYieldSurf [gamma Gs ...]`, as Femora's `pressure_independ_multi_yield` writes it (OpenSees'
parser requires all of them).

| Item | Type | Meaning |
|---|---|---|
| attr `nd` | int32 | `3` |
| attrs `rho`, `refShearModul`, `refBulkModul`, `cohesi`, `peakShearStra`, `frictionAng`, `refPress`, `pressDependCoe` | float64 | as in the Tcl command |
| attr `noYieldSurf` | int32 | number of yield surfaces; negative: user-defined surfaces from `pairs` |
| `pairs` | `[K, 2]` float64 *(only with noYieldSurf < 0)* | as for PressureDependMultiYield |

Rules: as above, with `cohesi >= 0`, `0 <= frictionAng < 90`, and `cohesi > 0` when
`frictionAng = 0` (OpenSees: fatal).

### 3.4 `/elements/<id>`

One group per **Femora element definition** (one `model.element.brick.std(...)` call): all
elements in it share type, material and parameters. `<id>` is chosen by the exporter (any
unique name; zero-padded indices suggested). Femora's element class tag
(`cell_data["ElementTag"]`) and name may be stored as `femora_tag` / `femora_name`
metadata. FemoraX forms its own batches from these groups.

| Item | Shape, type | Meaning |
|---|---|---|
| attr `type` | string | `"stdBrick"` or `"SSPbrick"` |
| attr `material` | int32 | tag of a `/materials/nd/` group |
| attr `body_force` | `[3]` float64 | `b1 b2 b3` |
| attr `lumped` | uint8 | stdBrick only: `1` if `-lumped`; absent or `0` means consistent mass. SSPbrick always lumps (the attribute must be absent) |
| `tags` | `[M]` int32 | element tags |
| `connectivity` | `[M, 8]` int32 | node tags in OpenSees order for the type |

Body force semantics (both elements, from `Brick.cpp` and `SSPbrick.cpp`): while no
`eleLoad` pattern acts on the element, `b` is applied as a **constant** body force in every
step, independent of any time series. v0 has no `eleLoad`, so `b` is always applied.

### 3.5 `/mass` *(optional)*

One row per nonzero nodal mass DOF (`mass <node> m1 m2 ...`).

| Dataset | Shape, type |
|---|---|
| `node` | `[K]` int32 |
| `dof` | `[K]` int32 (1-based) |
| `value` | `[K]` float64 |

Femora source: `point_data["Mass"]`. Absent when every nodal mass is zero (the soil
column's case).

### 3.6 `/constraints`

Both use one row per DOF, so any number of DOFs per node needs no padding.

`/constraints/sp/fix` *(optional)*: `fix <node> <flags...>` becomes one row per flag that is 1.

| Dataset | Shape, type |
|---|---|
| `node` | `[K]` int32 |
| `dof` | `[K]` int32 (1-based) |

`/constraints/mp/equal_dof` *(optional)*: `equalDOF <retained> <constrained> <dofs...>`
becomes one row per DOF.

| Dataset | Shape, type | Meaning |
|---|---|---|
| `retained` | `[K]` int32 | retained node |
| `constrained` | `[K]` int32 | constrained node |
| `dof` | `[K]` int32 | 1-based; the same DOF on both nodes, as in OpenSees |
| `command` | `[K]` int32 *(optional, since 0.2)* | index of the `equalDOF` command the row comes from: rows with the same value are one command (one OpenSees MP constraint), and commands are numbered in the order they appear in the Tcl file, from 0. All rows of one command have the same retained and constrained node. Absent: each (retained, constrained) pair is one command, which is how Femora writes them |

The laminar boundary is not a separate item: `laminar_boundary()` emits plain equalDOF
pairs (first node at each elevation retained, the others constrained), so it arrives here.

Rules checked by the reader: every node exists; `dof <= ndf` of each node involved;
no duplicate rows; the rows of one `command` share their retained and constrained node.

Rules checked when the model is compiled (`docs/deviations.md`, "Constraint handling"):
a joined group with a fixed DOF is fixed as a whole, and four patterns that OpenSees'
Transformation handler silently mishandles are refused with a message saying what is wrong
and how to fix the model: a node named by more than one `equalDOF` command, a chain through
a constrained node with a larger tag, a partial chain, and a fix on a constrained DOF whose
retained DOF is free. Chains in ascending tag order over all DOFs are accepted.

### 3.7 `/damping/rayleigh/<index>` *(optional)*

Rayleigh damping per element region, as Femora writes
`region <tag> -ele <tags...> -rayleigh alphaM betaK betaKinit betaKcomm`. Children are
ordered (`000000`, `000001`, ...) in the order the `region` commands appear in the Tcl
file. `frequency_rayleigh` is exported with its computed `alphaM` and `betaK` and zeros for
the other two. Femora's region tag and the damping's `user_name` may be kept as
`femora_tag` / `femora_name` metadata.

| Item | Shape, type | Meaning |
|---|---|---|
| attr `alphaM`, `betaK`, `betaKinit`, `betaKcomm` | float64 | coefficients |
| `elements` | `[M]` int32 | element tags in the region |

Element damping force, as OpenSees `Element::getRayleighDampingForces`:
`alphaM*M*v + (betaK*K_current + betaKinit*K_initial + betaKcomm*K_committed)*v`, with the
element's own mass matrix (lumped or consistent). An element in no region has no damping.
An element in several regions takes the coefficients of the last region in file order,
matching OpenSees, where the last `region` command wins. The reader prints a warning that
names the regions and the number of elements affected.

Not in v0: node regions (`-node`, nodal Rayleigh), the global `rayleigh` command, and
other damping types.

### 3.8 `/time_series/<tag>` *(optional)*

| Item | Shape, type | Meaning |
|---|---|---|
| attr `type` | string | `"Path"` |
| attr `factor` | float64 | `-factor` (default 1.0) |
| attr `dt` | float64 | uniform spacing (`-dt`); exactly one of `dt` or dataset `time` |
| `time` | `[K]` float64 | non-uniform times (`-time` / `-fileTime`) |
| `values` | `[K]` float64 | values (`-values` / `-filePath`), stored inline |
| attr `use_last` | uint8 *(optional)* | `-useLast` |
| attr `prepend_zero` | uint8 *(optional)* | `-prependZero` |

OpenSees uses two classes here: `PathSeries` (uniform `dt`) and `PathTimeSeries` (`time`
array). FemoraX ports both, with their interpolation and end-of-record behaviour.

### 3.9 `/patterns/<tag>` *(optional)*

Defining a pattern here does not activate it; a `pattern` step in `/process` does, at that
point in the timeline (Femora's `process.add_step(pattern)`).

| Attribute | Type | Meaning |
|---|---|---|
| `type` | string | `"UniformExcitation"` |
| `dof` | int32 | direction, 1-based |
| `accel` | int32 | time series tag (`-accel`) |
| `vel0` | float64 *(optional)* | `-vel0` |
| `factor` | float64 *(optional)* | `-fact` |

Effective load `-M * r * factor * a_g(t)` through each element's mass, as OpenSees'
`addInertiaLoadToUnbalance`, plus nodal masses from `/mass`. Responses are relative to the
ground, as in OpenSees.

### 3.10 `/recorders/<name>` *(optional)*

Like patterns, a recorder starts recording when a `recorder` step in `/process` activates
it. `<name>` is any unique string chosen by the exporter.

**Node recorder** (`recorder Node`):

| Item | Shape, type | Meaning |
|---|---|---|
| attr `type` | string | `"Node"` |
| attr `file` | string | `-file`; plain text in OpenSees' exact layout (large output belongs in VTKHDF) |
| attr `response` | string | `"disp"`, `"vel"` or `"accel"` |
| attr `time` | uint8 *(optional)* | `-time`: first column is the time |
| attr `dT` | float64 *(optional)* | `-dT` |
| attr `precision` | int32 *(optional)* | `-precision` (OpenSees default 6) |
| attr `close_on_write` | uint8 *(optional)* | `-closeOnWrite` |
| `nodes` | `[K]` int32 | `-node` (a `-nodeRange` is expanded by the exporter) |
| `dofs` | `[D]` int32 | `-dof` |
| `time_series` | `[D]` int32 *(optional)* | `-timeSeries`, one per DOF, for absolute response |

**VTKHDF recorder** (`recorder vtkhdf`):

| Item | Shape, type | Meaning |
|---|---|---|
| attr `type` | string | `"VTKHDF"` |
| attr `file` | string | file name as Femora writes it (with `$pid`) |
| attr `dT` | float64 *(optional)* | `-dT` |
| attr `rTolDt` | float64 *(optional)* | `-rTolDt` |
| attr `responses` | string | space-separated; any of `disp vel accel` (node data) and, since 0.4, `stress3D6 strain3D6` (element data: each element's stress or strain averaged over its integration points, as OpenSees' `stress3D6` / `strain3D6` element responses) |
| `elements` | `[M]` int32 *(optional, since 0.1)* | record only these elements (and their nodes); the exporter expands Femora's `-region` into its element tags. Absent: the whole model |
| attr `region` | int32 *(optional)* | not supported: the file has no region tags to refer to (element groups and damping regions are exporter-named). The reader rejects it; use `elements` |

The output layout follows OpenSees' `VTKHDF_Recorder.cpp`, so `femora.results` reads it
unchanged.

### 3.11 `/process/<index>`

The ordered timeline, one child group per step, in Femora `process.add_step` order. Every
step has attrs `kind` (string) and `label` (string, *optional*, Femora's description).

| `kind` | Other attributes | Meaning |
|---|---|---|
| `analysis` | see below | run a transient analysis stage |
| `pattern` | `tag` int32 | activate `/patterns/<tag>` |
| `recorder` | `name` string | activate `/recorders/<name>` |
| `set_time` | `time` float64 | `setTime` |
| `load_const` | `time` float64 *(optional)* | `loadConst [-time t]` (1b) |
| `wipe_analysis` | | `wipeAnalysis` (no effect on state) |
| `update_material_stage` *(since 0.4)* | `stage` int32 (0 elastic, 1 plastic); dataset `materials` `[K]` int32 | `updateMaterialStage -material <tag> -stage <stage>` for each listed material, in order (Femora's `update_material_stage_to_elastic` / `_to_plastic` actions list every stage-capable material). Each tag must name a PressureDependMultiYield or PressureIndependMultiYield material, at most once |

`analysis` step attributes:

| Attribute | Type | Meaning |
|---|---|---|
| `name` | string | Femora analysis name |
| `analysis_type` | string | `"Transient"`, or `"Static"` *(since 0.3)* |
| `dt` | float64 | step size (Transient only) |
| `num_steps` | int32 | Transient: exactly one of `num_steps` or `final_time`; Static: required (`analyze num_steps`) |
| `final_time` | float64 | step while `time < final_time`, exactly as Femora's `while {[getTime] < final_time}` loop. Time advances by adding `dt` to the domain time in double precision at each step, in the same place and order as the OpenSees integrator, so the number of steps and the time stamps match OpenSees bit for bit |
| `integrator` | string | Transient: `"ExplicitDifference"`, `"CentralDifference"` or `"Newmark"`; Static: `"LoadControl"` *(since 0.3)* |
| `gamma`, `beta` | float64 | Newmark only |
| `newmark_form` | string *(optional)* | Newmark `-form`: `"D"` (default), `"V"`, `"A"` |
| `load_increment` | float64 | LoadControl only *(since 0.3)*: `integrator LoadControl <incr>` |
| `num_iter` | int32 *(optional)* | LoadControl `<numIter>` (default 1) |
| `min_increment`, `max_increment` | float64 *(optional)* | LoadControl `<minIncr> <maxIncr>` (default: `load_increment`) |
| `algorithm` | string | `"Linear"` (required for the explicit integrators), `"Newton"` or `"ModifiedNewton"` *(since 0.3)* |
| `modified_newton_initial`, `modified_newton_factor_once` | uint8 *(optional, since 0.3)* | ModifiedNewton `-initial`, `-factoronce`. As OpenSees, matrix-free: the stored Gauss-point tangent (and the preconditioner built from it) is formed once per step after the first unbalance; `-initial` puts the initial stiffness in the stiffness term; `-factoronce` keeps the tangent across steps until a step fails its convergence test or the step size changes (Femora issues the algorithm again then) |
| `constraint_handler` | string | `"Transformation"`; `"Plain"` only when there are no MP constraints |
| `numberer` | string | recorded, ignored (FemoraX numbers its own equations) |
| `system` | string | recorded, ignored (explicit: lumped diagonal; 1b: Eigen) |
| `test_type` | string *(optional)* | `"NormUnbalance"`, `"NormDispIncr"`, `"EnergyIncr"` (1b) |
| `test_tol` | float64 *(optional)* | |
| `test_max_iter` | int32 *(optional)* | |
| `max_retries`, `num_sublevels`, `num_substeps` | int32 *(optional; defined since 0.5)* | Femora's retry and substep recovery, below |
| `initialize` | uint8 *(optional)* | `initialize` after `analysis` |
| `allow_unstable_dt` | uint8 *(optional)* | FemoraX only: warn instead of stop when `dt` exceeds the stable step |
| `solver_tolerance` | float64 *(optional, since 0.3)* | FemoraX only: relative tolerance of each Krylov solve, in (0, 1). Absent: FemoraX chooses (1e-12) |
| `solver_max_iterations` | int32 *(optional, since 0.3)* | FemoraX only: maximum Krylov iterations per solve. Absent: FemoraX chooses (the number of equations, between 200 and 10000) |
| `preconditioner` | string *(optional, since 0.3)* | FemoraX only: `"auto"` (default: FemoraX chooses; Jacobi today) or `"jacobi"`. The solver itself (CG or GMRES) follows from the declared tangent symmetry and is never chosen in the file |

Femora's retry and substep recovery (defined in 0.5; the three attributes exist since 0.0
and keep this meaning in every 0.x file). An implicit step fails when its convergence test
is not met within `test_max_iter` iterations or a linear solve fails. A failed step goes
back to the committed state and time (and the solver's initial guess) and:

1. is retried at the same size, up to `max_retries` times (absent: 0);
2. then, in Transient stages while its level is below `num_sublevels` (absent: no
   substeps), is replaced by `num_substeps` steps of `dt / num_substeps` one level down,
   each committed and recorded and each with the same recovery; the domain time adds each
   substep's size in double precision, as OpenSees does;
3. else stops the run, naming the step, the time and the last failure.

Rules (Femora's own): `max_retries >= 0`; `num_sublevels` and `num_substeps` together,
`num_sublevels >= 1`, `num_substeps >= 2`, Transient stages only. Static stages retry only
(LoadControl rescales its increment by the failed attempt's iterations, as OpenSees does).
Explicit stages ignore all three: a Linear explicit step never fails. A retry differs from
the failed attempt only through the tangent: with ModifiedNewton -factoronce it is formed
afresh (as OpenSees resets factorOnce after a failed test); with Newton, or ModifiedNewton
per step, a retry repeats the attempt. The stage summary counts retries and substepped
steps.

Users and exporters need not set any of the three solver attributes: absent, FemoraX chooses. They exist for
studies and for cases where the automatic choice needs overriding.

Rules for explicit stages (decided for milestone 1a):

- `ExplicitDifference`: every element must have lumped mass (stdBrick with `lumped = 1`,
  or SSPbrick). Consistent mass is rejected: OpenSees would solve with the full mass
  matrix, so the explicit result would differ.
- `CentralDifference`: lumped mass and `betaK = betaKinit = betaKcomm = 0` in every damping
  region, so the OpenSees system matrix `M/dt^2 + C/(2*dt)` is diagonal. Anything else is
  rejected until 1b adds the solver.
- At the start of each explicit stage, `dt` is checked against the stable step, including
  stiffness-proportional damping (`docs/design.md`, "Stable time step"). With a history
  material (PressureDependMultiYield), it is checked again every 100 steps, because the
  soil stiffens with confinement; `allow_unstable_dt` turns either stop into a warning.

---

## 4. Unsupported components

The reader stops on these, naming the component and suggesting the OpenSees backend:

- any element type other than `stdBrick` and `SSPbrick`, any material other than
  `ElasticIsotropic`, `PressureDependMultiYield` and `PressureIndependMultiYield` (and those
  two with `nd = 2`), any section or transformation;
- `ndm != 3`;
- other time series (`Linear`, `Constant`, `Trig`, ...), patterns (`Plain`, `MultipleSupport`,
  `H5DRM`), loads (`load`, `eleLoad`, `sp` in patterns);
- MP constraints other than equalDOF (`rigidDiaphragm`, `rigidLink`, ...), node regions,
  global `rayleigh`, other damping types;
- recorders other than Node and VTKHDF, Node recorder output other than `-file`
  (`-xml`, `-binary`, `-tcp`), VTKHDF responses other than `disp vel accel stress3D6 strain3D6`;
- analyses other than `Transient` and `Static` (`VariableTransient`, ramped `dt_min`/`dt_max`),
  static integrators other than `LoadControl` (`DisplacementControl`, ...), other
  integrators, algorithms, handlers;
- other process actions (`remove recorders`, `remove loadPattern`, `updateParameter`,
  arbitrary Tcl from `model.actions.tcl`);
- partitioned models (`num_partitions > 1`); v0 runs one partition.

---

## 5. Example: the 1a soil column

The explicit variant of Femora's layered elastic column (5 mesh parts, 1 x 1 x 18 m,
`lumped=True`, ExplicitDifference, no gravity stage). Values abbreviated.

```
/                       schema_version=[0,1] ndm=3 generator="femora ..."
/nodes/tags             [N]           int32
/nodes/coords           [N,3]         float64
/nodes/ndf              [N]           int32   all 3
/materials/nd/1         type="ElasticIsotropic" E=377000.0 nu=0.3 rho=2.0285...
                        femora_name="dense_ottawa_lower_material"
  ... one per layer (5)
/elements/000000        type="stdBrick" material=1 body_force=[0,0,-19.9] lumped=1
/elements/000000/tags   [2]           int32
/elements/000000/connectivity [2,8]   int32
  ... one per layer (5)
/constraints/sp/fix/node, dof          base nodes x 3 DOFs
/constraints/mp/equal_dof/retained, constrained, dof     laminar ties, 3 DOFs per pair
/damping/rayleigh/000000 alphaM=... betaK=... betaKinit=0 betaKcomm=0
                        femora_tag=1 femora_name="soil_damping"
/damping/rayleigh/000000/elements [M]
/time_series/1          type="Path" factor=9.81 ; time [K], values [K]
/patterns/2             type="UniformExcitation" dof=1 accel=1
/recorders/site         type="VTKHDF" file="results/site_response$pid.vtkhdf" dT=0.01
                        responses="accel disp vel"
/process/000000         kind="pattern"  tag=2
/process/000001         kind="recorder" name="site"
/process/000002         kind="set_time" time=0.0
/process/000003         kind="analysis" name="frequency_sweep" analysis_type="Transient"
                        integrator="ExplicitDifference" algorithm="Linear"
                        constraint_handler="Transformation" dt=... final_time=50.0
```

---

## 6. Cross-validation with OpenSees

System tests run the same Femora model through both backends. The FemoraX file holds
full-precision coordinates, and the Tcl file for OpenSees is exported with enough decimals
that both see the same geometry: `export_to_tcl(decimals=30)`. Femora passes each
coordinate through Python's `round(x, decimals)` and prints the shortest repr; with 30
decimal places every coordinate with |x| >= 1e-12 comes back unchanged (checked on 200,000
random values), while 17 decimals already loses digits below |x| = 0.1. Each system test
also asserts that the coordinates in the Tcl file equal those in the FemoraX file bit for
bit. Femora's default `decimals=5` would give OpenSees rounded
coordinates and is not used for comparisons.

## 7. Decisions

Recorded at approval (2026-11-05):

1. Coordinates are full precision; comparison tests export Tcl with `decimals=30` and
   check that both files hold identical coordinates.
2. Container groups (element groups, damping regions) are named by the exporter; Femora
   tags and names are metadata only.
3. Overlapping damping regions: the last region wins, as in OpenSees, with a warning.
4. Strict validation, with `[major, minor]` versions (section 1).
5. Node recorder output is plain text in OpenSees' exact layout; large output goes to
   VTKHDF.
6. Body forces are applied as OpenSees does (constant, every step, while no `eleLoad` acts).
7. Time accumulates exactly as in OpenSees.
