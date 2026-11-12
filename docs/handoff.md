# Handoff: where FemoraX stands (updated 2026-11-11)

A note for a new Claude Code session (for example on TACC) that does not have the earlier
conversation. `CLAUDE.md`, `docs/design.md` (Roadmap), `docs/todo.md` and the git log are the
full record; this file adds what they do not say yet. Read it after `CLAUDE.md`.

## Working rules agreed with the user (beyond CLAUDE.md)

- `../Femora` and `../OpenSees` are read-only. Femora reaches the venv through a `.pth` file.
- Ask before any schema change. One commit per step, only after `scripts/check.sh` passes on
  every preset; commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Nonlinear OpenSees references stay tiny, run once, stored in the repository; OpenSees never
  runs in the normal check.
- Never update a benchmark baseline to hide a regression. Benchmarks run on TACC only.
- Long runs and their results go to a permanent folder (for example
  `/home/aminp/femorax_runs/` on the laptop, `$SCRATCH` or `$WORK` on TACC), never to `/tmp`
  or a session scratch folder: a 5.5 GB basin result was lost that way.
- Do not edit files while `scripts/check.sh` runs; grep its log for `FAILED`.

## Current work (2026-11-11): Uniform damping

Approved with the user and taken **before H5DRM streaming**. The plan, the decisions and every
finding are in `docs/design.md`, "Frequency-independent ("Uniform") damping"; the user's
target hardware is in Claude Code's memory (NVIDIA/AMD GPUs and CPUs; Intel GPUs only need to
build and run).

Done (all on master, each after `scripts/check.sh`):
- `ff72e25` step 1: OpenSees' UniformDamping ported (`physics/damping/uniform_damping.hpp`),
  against stored OpenSees records within 1.3e-14 (OpenSees' command doubles the ratio: the
  class works with the loss factor 2 zeta).
- `ea651a6` step 2a: FemoraX's fit (`compile/uniform_damping_fit.hpp`): 5 % gives 5 % across
  the band (OpenSees' fit drops to about 3.4 % at 10 Hz for 0.1-10 Hz), the default
  (`fit = "femorax"`, `"opensees"` reproduces OpenSees).
- `32c7882` step 2b-1: the damping force in solid elements (`physics/damping/element_damping.hpp`).
- `c12c2ed` the plan of 2b-2; `89d0a19` (i) the step context (time, step size, static or
  dynamic) reaches every material update, no change in results, benchmarks within noise.
- `9e01b2e` (ii)-a the wrapper `UniformDamped<Mat, NF>` (`physics/damping/uniform_damped.hpp`).
- `ee74746` (ii)-b damped batches from `io::Model::uniform` regions through the real force
  kernels (registry entries for ElasticIsotropic, buckets 8 and 32).
- `a015feb` (iii)-a the stable step: the damped step is the undamped one over
  `sqrt(1 + 2 eta sum alpha_i)` (0.793 for 5 % over 0.1-10 Hz).
- (iii)-b the stiffness correction (default with FemoraX's fit; docs/deviations.md): the
  frequency at the band's geometric mean is kept (x1.002 for 5 %), the damping stays within
  5 % of the target, gravity stresses are untouched; the stable step is 0.890 of the
  undamped one.
- (iv) PressureIndependMultiYield and PressureDependMultiYield wrapped (20-surface buckets,
  filter buckets 8 and 32): the material stage reaches them, a yielding point stops adding
  damping, every kernel variant gives the same bits. The compacted path is not wrapped.
- Step 3, implicit stages: no code change needed (the wrapper's tangent is consistent);
  Newmark + Newton through `run_model` matches a host recurrence within 2e-13, 1.6 Newton
  iterations per step.
- Step 4, schema 0.7 (2026-11-12): `/damping/uniform` read, checked and run; the converter
  reads `damping Uniform` and `region ... -damp`; the run summary reports the fit. `-fact`
  is refused as unsupported until implemented.

**Start here (2026-11-12):** the cheap stable-step re-check is committed (fixes 2 and 3:
`stability_may_change` / `STABILITY_CONSTANT` and `ScaledStability` in `contracts.hpp`,
`max_omega_sq_reference` / `max_omega_sq_bound` in stdBrick and SSPbrick; docs/design.md
"Stable time step"). Then:
- Found: the elastic basin with Uniform damping on all soil (regions 1-7;
  `$SCRATCH/femorax_runs/basin_site2_elastic/speed/uniform_all.femorax.h5`) is refused
  before shaking: "H5DRM pattern 1: element 11233 lies on the DRM boundary and its material
  has history state (it can yield)". The damped regions include DRM-layer elements (the
  open DRM decision). Fix the message (damped elastic cannot yield: name Uniform damping),
  and ask the user: leave the DRM layer undamped, or extend the DRM force kernel to damped
  elastic layers.
- Rerun the full plastic basin to confirm 2.61 ms/step is gone (fix 1, 7fa1e74).
- Remove the worktree: `git worktree remove $SCRATCH/fx_before`.

Next, in order:
1. Step 5: the remaining verification (plane-wave attenuation, the basin with Uniform;
   yielding is done) and benchmarks of the damped kernels.
2. Open: a damped DRM layer is refused today (DRM elements must have no state); decide what
   the DRM forces do with Uniform damping.

Other findings of 2026-11-11 (the user asked; not started):
- A mass-proportional-only (alphaM) absorber: tested, see "The basin example" below.
- Implicit plastic basin in FemoraX against the user's OpenSees run (same scheme, 96 cores
  each): displacements within 0.2 %, 12 min against 32 min.
- The 2-day OpenSees explicit runs were the basin-with-buildings study (beams, fiber sections,
  embedded nodes, dt 5e-5, 700,000 steps), not the pure basin; their stdout had 78 million
  H5DRM lines.

## Milestone 4 (DRM basin and dynamic pile-soil interaction): H5DRM

Done (commits `14de063`, `b4a32fa`, `5024299`):

1. Schema 0.6: the `H5DRM` pattern (`schema/v0/README.md`).
2. The DRM file reader (`io/drm_file`) and the compile step (`compile/drm`): OpenSees'
   coordinate transformation, nearest-station matching (first station on ties), the DRM
   elements and their boundary masks; DRM elements with a history material are refused
   ("valid only if the DRM layer stays linear").
3. The DRM forces on the device (`exec/drm_load`, `kernels::launch_drm_forces`):
   `Peff_b = -M_be a_e - K_be u_e`, `Peff_e = M_eb a_b + K_eb u_b` from the element's own
   operator; OpenSees' interpolation (`i1 = floor((t - tstart)/dt)`), the file's z components
   negated, no forces outside `[tstart, tend]`; factor other than 1 refused (OpenSees stores
   it but never applies it). `test_drm_free_field`: a box inside a DRM layer reproduces a
   free field to 2.2e-14 (explicit).

Next steps (in order; each one commit):

4. Stream the DRM histories in time blocks (double buffering: a reader thread, pinned host
   buffers, asynchronous copies, block size from a memory budget), with a test that block
   size 1 equals the whole history in memory. Today the whole history is on the device
   (the Hayward Site 2 file: 0.6 GB as FP64).
5. A tiny stored OpenSees DRM box reference (run once, stored), compared in the check.
6. H5DRM in implicit stages: `form_unbalance` applies the forces, not verified yet
   (docs/todo.md item 10): a free field from the same Newmark recurrence, and the step-5
   reference.
7. DRM output: one summary block per pattern at setup (file, stations matched, nodes,
   DRM elements, largest match distance, dt, samples, `[tstart, tend]`, device memory),
   nothing per step, one warning if the stage runs past `tend`; a line in
   `docs/deviations.md` that OpenSees' per-step H5DRM prints (several lines per step and per
   rank) are deliberately not reproduced. Agreed with the user, not done.

## The basin example (Femora `examples/domain_reduction_method/basin.py`)

The user wants Femora's basin run in FemoraX as written. No FemoraX exporter in Femora yet
(the user does not want one now): `tests/system/femora_example/tcl_to_femorax.py` converts
the Tcl that `basin.py` exports (it merges the 96 OpenSeesMP partitions).

- Run recipe: `/home/aminp/femorax_runs/run_basin.sh` on the laptop (export with
  `basin.build_model(drm_file=..., output_dir=..., material="elastic")`, convert with
  `--explicit 2e-4 --recorder results/basin_response_explicit.vtkhdf`, `femorax check`,
  `femorax run ... --vtkhdf-precision single`). The DRM file: `HaywardFault_SW4_DRM_Site2.h5drm`
  (300 MB, 4906 stations, 2552 samples, dt 0.0108 s, float32), from the user (on TACC:
  `/scratch/08189/amnp95/drm_inputs/`, the path `basin.py` uses).
- Elastic case, laptop RTX 3050 Ti (2026-11-09): 46,080 stdBrick, 50,421 nodes, 132,540
  equations; gravity (2 static stages) about 20 s; shaking ExplicitDifference, 137,000 steps
  of 2e-4 s (stable step 2.18e-4, set by the absorber's betaK = 0.015), 5.19 ms/step, 711 s;
  total 12 min 16 s; device memory 0.74 GB; VTKHDF every 0.02 s with all responses: 5.5 GB
  in single precision. Explicit needs lumped mass (the example has consistent mass), so it is
  not identical to the implicit example.
- Sanity checks of that run against the DRM file: boundary nodes follow the free field
  within 0.5-1.8 % of its peak (0.12-0.15 m); exterior nodes carry only the scattered field
  (about 1 %); the absorber surface moves at most 4e-4 m. No OpenSees comparison yet.
- Plastic case: the converter needs PressureIndependMultiYield materials (a few lines);
  then the same recipe with `material="plastic"`. Implicit case: after step 6 above.
- Threads on spr nodes: use all 112 cores (`OMP_NUM_THREADS=112 OMP_PROC_BIND=spread
  OMP_PLACES=threads`): the elastic basin's explicit step takes 1.37 ms with 112 threads and
  2.12 ms with 56 (2026-11-12, 3000-step runs, `flat_gather_cached`).
- Absorber experiment (Stampede3 pvc node, CPU presets, 2026-11-11): the elastic basin with
  the absorber region's Rayleigh damping (alphaM 2.36, betaK 0.015: 95 % at 0.2 and 20 Hz)
  replaced by mass-only damping (alphaM 23.9: 95 % at 2 Hz, betaK 0). The stable step grows
  10x (2.2e-3 against 2.2e-4); dt 1e-3, 27,400 steps in 52 s (1.91 ms/step), 2 min 15 s in
  all. Against the Rayleigh absorber: surface-centre displacement within 2.3 % of its peak,
  surface-centre acceleration peak 10.3 against 9.7 (20 % largest difference), a DRM
  exterior node (scattered field only) 14.5 % in displacement and twice the acceleration
  peak (1.72 against 0.86): mass damping falls as 1/f, so the layer reflects the high
  frequencies. No late-time ringing difference (soil RMS velocity equal at 20-27 s).
  Model `$SCRATCH/femorax_runs/basin_site2_elastic/massonly.femorax.h5`, script and plot
  `/work2/08189/amnp95/femorax_plots/compare_absorbers.py`,
  `absorber_rayleigh_vs_massonly.png`. Conclusion: fine for displacements and long periods
  at a tenth of the cost; for accelerations a graded layer or PML (docs/todo.md item 13).

## TACC

Decided with the user (2026-11-10): **development moves to Stampede3**; Vista later, for
ARM and Grace-Hopper.

- Why Stampede3: x86 like the laptop, so the stored OpenSees reference data and the
  bit-identity tests behave as on the laptop (on ARM, libm `exp`/`pow` may differ in the
  last bit), and OpenSees runs there (on Vista it compiles but crashes without a message;
  untried suspects: unsigned `char` on ARM (`-fsigned-char`), the Tcl setup, undefined
  behaviour; get a gdb backtrace on a `gg` node). Stampede3 has H100 nodes (`h100`, few,
  queues) for the CUDA build and `spr` / `icx` / `skx` CPU nodes. Intel GPU (`pvc`): not
  supported.
- Layout: everything in `/work2/08189/amnp95/dev/` (`$STOCKYARD/dev`, the same path on
  Vista and Stampede3): `FemoraX`, `Femora` (pinned commit `79bd4f67`), `OpenSees` (pinned
  `955ceb0a`), both checked out detached and read-only. Claude Code's configuration (sessions,
  memory, settings) in `/work2/08189/amnp95/claude-home` through
  `export CLAUDE_CONFIG_DIR=/work2/08189/amnp95/claude-home` in `~/.bashrc` on both machines;
  one Claude Code install per machine (ARM and x86 binaries). Never one session on both
  machines at once.
- Python: a venv per machine; Femora reached read-only through a `.pth` file as on the
  laptop (`docs/dependencies.md`), not `pip install -e` (it writes into the clone).
- Builds are per machine and node type (x86 and ARM objects do not mix).
- Done on Stampede3 (2026-11-10, `spr` node): `scripts/stampede3_env.sh`; the presets
  `s3-cpu-debug` (197 tests), `s3-cpu-release` (198, the laptop's full set) and
  `s3-cpu-ubsan` (197), all passing, through `scripts/check.sh`; the Intel OpenSees reference
  build (`scripts/stampede3_opensees.sh`, `tests/reference/README.md`), with the two live
  OpenSees comparisons passing. Pitfalls met: TACC sets `OMP_NUM_THREADS=1`, which `nproc`
  obeys (check.sh now counts cores without it); TACC's XALT preloads a library that
  AddressSanitizer refuses (the env script unloads it); the compute nodes use strict memory
  accounting (`vm.overcommit_memory = 2`), so ASan cannot reserve its shadow memory there at
  all: Stampede3 runs UBSan only, ASan stays on the laptop (and maybe Vista); a `module load`
  inside a pipe (`module load x | tail`) runs in a subshell and changes nothing.
- Next on Stampede3: a `s3-cuda-hopper90` preset and the tests on an `h100` node; the first
  Stampede3 benchmark baselines (`benchmarks/run_all.sh`); then milestone 4 (above): the
  plastic basin, steps 4-7, the implicit basin, and FemoraX against the user's 96-rank
  OpenSees basin run.

- `benchmarks/run_all.sh` already detects Vista and Stampede3 nodes and configures the
  build: Grace CPUs with `-DKokkos_ARCH_ARMV9_GRACE=ON`, Hopper GPUs with
  `-DKokkos_ARCH_HOPPER90=ON` (Kokkos 5.1.0 has both), Sapphire Rapids with
  `Kokkos_ARCH_SPR`. It builds without tests (`FEMORAX_BUILD_TESTS=OFF`) and has not been run
  yet. Before benchmarks on Vista: build with tests on a `gg` (Grace-Grace) and a `gh` node and run
  the unit and system tests once (CPU and GPU results agree to rounding only; GPU tests
  compare at a tolerance).
- Compute nodes may have no internet access: fetch the dependencies on a login node first
  (`benchmarks/run_all.sh --build-only`, or one configure). Claude Code itself runs on a login
  node; heavy work goes to `idev` sessions or jobs.
- The user's earlier rule: no job submission from Claude without asking.

## Laptop notes

- WSL keeps Linux in one disk file on C: (`C:\Users\aminp\AppData\Local\wsl\{77eeef0c-...}\ext4.vhdx`,
  53 GB for 37 GB used, set sparse on 2026-11-10). C: was nearly full: free space inside
  Linux comes back to C: only after `sudo fstrim -av` (sparse) or a `diskpart` compaction.
