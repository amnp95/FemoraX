# FemoraX

GPU-native finite element solver for dynamic geotechnical and soil-structure analysis.
FemoraX is the solver backend for Femora (separate repo: GeotechUW/Femora). Femora builds
the model; FemoraX runs it on GPU or CPU through Kokkos.

Goal: the same element and material formulations as OpenSees, the same results, at GPU speed.

The full architecture is in the design document, imported here:
@docs/design.md

## What Femora is

Femora (github.com/GeotechUW/Femora) is a Python library from the same team that builds
OpenSees models. A Femora model goes: materials, sections and element templates → mesh parts
(volume, line, surface) → an assembler that merges the parts into one mesh (optionally
partitioned with METIS) → constraints, damping, loads and interfaces → recorders → an ordered
`process` timeline of analysis stages and actions (for example gravity, switch material stage
to plastic, then transient). Today Femora exports a Tcl file and runs OpenSees, and reads
results back from VTKHDF with `femora.results`.

FemoraX becomes Femora's second backend: Femora writes a FemoraX model file instead of Tcl,
and FemoraX writes VTKHDF that `femora.results` reads unchanged.

## Reference code (read-only)

Two clones sit next to this repository. Read them freely; never edit them, never import from
them, and never copy them into this repository.

- `../Femora/`: how each component is described, and how it maps to OpenSees commands.
  Start with its Tcl export code and the generated API knowledge files
  (built by `agent/build_knowledge.py`).
- `../OpenSees/`: the source of every element and material to port. Record the commit hash
  used for each port.

Sessions start with: `claude --add-dir ../Femora --add-dir ../OpenSees`

## Non-negotiable rules

1. **Same answers as OpenSees.** Every element and material port is verified against the
   original OpenSees class (material strain-path tests, element patch tests, system tests
   on Femora examples). A port is not done until its verification test passes.
2. **The physics layer is framework-free.** Code in `src/physics/` is plain C++20 functions
   marked `FEMORAX_HD` (host + device). No Kokkos types, no heap allocation, no virtual
   functions, no static or global mutable state, no exceptions, fixed-size arrays only.
3. **Kokkos lives only in the launch layer** (`src/exec/`). Native CUDA or HIP code is allowed
   only in `src/native/`, only when a profile shows a portable kernel is too slow, and always
   with the portable version kept as a fallback.
4. **FP64 everywhere** in physics, state and assembly. Mixed precision only inside solver
   preconditioners, behind an explicit option, and inside machine-learned materials, which
   convert their inputs and outputs to and from FP64 (docs/design.md).
5. **FemoraX never imports Femora.** Femora is a test-only dependency used for cross-validation.
6. **Solvers sit behind `femorax::LinearSolver`.** No solver-library types outside `src/solvers/`.
7. **Every assembly path has a deterministic mode** (graph coloring). Validation tests use it.
   The atomic path is allowed for speed.
8. **When porting from OpenSees:** name the source file and OpenSees version or commit in a
   comment at the top, keep variable names close to the original so the two can be reviewed
   side by side, and record any deliberate difference in `docs/deviations.md`.
9. **Ask before changing the model file schema** (`schema/`). Bump `schema_version` on every
   change: the minor version for a backward-compatible addition (a new optional field), the
   major version for a breaking change. The reader accepts every older minor version of its
   own major version.

## Layout

```
FemoraX/
  CLAUDE.md
  cmake/       dependency pins, warnings, physics purity check
  docs/        design.md, deviations.md, support-table.md, dependencies.md
  schema/      versioned model-file specification (HDF5)
  src/
    physics/   elements/, materials/, sections/   (framework-free)
    exec/      Kokkos launch layer: batches, assembly, time loop
    solvers/   LinearSolver interface and backends
    io/        model reader, recorders (VTKHDF writer)
    native/    optional CUDA/HIP specializations
    app/       the `femorax` command-line program
  python/      bindings -> the `femorax` Python package
  tests/
    unit/      per-material and per-element tests against reference data
    reference/ OpenSees driver programs and generated reference data
    system/    Femora example comparisons (OpenSees vs FemoraX)
  benchmarks/  harness, per-kernel benchmarks, check.py, per-machine baselines
```

## Build and test

CPU build (Kokkos Serial + OpenMP). Kokkos 5.1.0 and GoogleTest 1.17.0 are fetched and
pinned by `cmake/FemoraXDeps.cmake`; versions and hashes are in `docs/dependencies.md`.

```
cmake --preset cpu-debug            # or cpu-release, cpu-asan (ASan + UBSan)
cmake --build --preset cpu-debug -j8
ctest --preset cpu-debug
```

- Build trees go to `build/<preset>/`. Presets build with `-Werror`.
- The unit-test binary also runs as a whole with 1 and 8 OpenMP threads
  (`unit_all.threads1`, `unit_all.threads8`).
- `physics_purity.*` tests enforce rule 2: a source scan (`cmake/CheckPhysicsPurity.cmake`)
  and a standalone compile of every physics header with `-fno-exceptions -fno-rtti` and no
  Kokkos on the include path.
- Benchmarks: `python3 benchmarks/check.py` (see "Performance" and `benchmarks/README.md`).
- Validate a model file: `build/cpu-debug/src/app/femorax check model.femorax.h5`
  (exit 0 valid, 3 unsupported component, 4 invalid). Example models:
  `.venv/bin/python schema/v0/make_example.py <dir>`.
- Reader tests use fixtures written by `tests/io/make_fixtures.py` (one file per rule,
  listed in its manifest); add a fixture for every new reader rule.
- CUDA and HIP builds: not set up yet (separate build trees, `-DKokkos_ENABLE_CUDA=ON`).

## Performance

Performance is the top priority after correctness. These rules apply to every kernel:

1. **Compile-time sizes in every element template.** `NNODE`, `NODE_NDF`, `NDOF`, `NGP`,
   `NSTATE` and the material's `NSTRAIN` are `static constexpr`; every local array in a
   kernel has a compile-time size. No runtime-sized loops over nodes, DOFs or points.
2. **One fused gather-compute-scatter kernel per batch.** Each element is gathered,
   computed and scattered in the same kernel; no separate gather or scatter passes, and no
   global per-element temporaries for `fe` or `Ke`.
3. **Thread layout chosen per element type, by benchmark.** Each element offers its work
   per Gauss point as pointwise physics functions (docs/design.md, "Element kernels"), and
   the launch layer offers several layouts built from them (one thread per element, a team
   per element with one thread per Gauss point, team-computed 3x3 tangent blocks). The
   benchmark picks the default per element-material pair and backend. Physics code never
   knows about threads.
3a. **Never hold a full element tangent in one thread** in implicit assembly; compute and
   scatter it as node-pair blocks.
4. **32-bit indices** for nodes, elements, DOFs, equations and assembly maps. 64-bit only
   for counts that can pass 2^31 (such as nonzero offsets of a large matrix), checked in the
   compile step.
5. **Node and element renumbering** (RCM or a space-filling curve) in the compile step,
   before batches and maps are built, so neighbouring threads touch neighbouring memory.
6. **Optimize only what the benchmarks show is slow.** Tests must pass before and after an
   optimization, and the commit message gives the before and after numbers.
7. **Every new kernel comes with a benchmark** that states its compulsory memory traffic
   (`benchmarks/README.md`).

Run `python3 benchmarks/check.py` (cpu-release) on every change before committing. A
regression over the tolerance blocks the commit unless the slowdown is intended; then the
commit message explains it and the baseline is updated. Never update a baseline to hide a
regression.

## Development hardware

The GPU laptops have not arrived. Develop and test on the Kokkos Serial and OpenMP backends
first. Every kernel must pass its tests on CPU before it is run on GPU.

## Conventions

- C++20, CMake, namespace `femorax`, clang-format.
- Structure-of-arrays storage: arrays indexed `[component * n + i]` so neighbouring threads
  read neighbouring memory.
- FemoraX is unit-agnostic, like OpenSees.
- Tests: GoogleTest. Python bindings: nanobind, added after the CLI runs models.
- Python tools and tests use the virtual environment in `.venv/` (numpy, h5py; later
  `../Femora` installed editable for the system tests). Create it with
  `python3 -m venv .venv && .venv/bin/pip install numpy h5py`.
- CPU direct solver (milestone 1b): Eigen `SimplicialLDLT`. No GPL solver libraries.

## Workflow

- For a new module or any architectural change: propose a plan first and wait for approval.
- One element or material per change, together with its tests and benchmarks.
- One commit per finished step; tests and the benchmark check pass before each commit.
- Never mark a port finished with skipped or loosened tests. If a tolerance must change,
  explain why in `docs/deviations.md`.
- When a component becomes supported, update `docs/support-table.md`.

## Current milestone

Milestone 1a: layered elastic soil column with explicit central difference, end to end on the
CPU backend (see design.md, Roadmap). Update this line when the milestone changes.
