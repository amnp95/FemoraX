# Handoff: where FemoraX stands (2026-11-10)

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
- Builds are per machine and node type (x86 and ARM objects do not mix): the laptop presets
  (`cpu-*`, `cuda-ampere86`) do not fit; add TACC presets (for example `s3-cpu-release`,
  `s3-cpu-asan`, `s3-cuda-hopper90`) and a TACC mode of `scripts/check.sh`.
- First tasks on Stampede3, in order:
  1. Build and test: the last commit passed only cpu-debug and cpu-release on the laptop
     (C: ran out of space); run the asan preset's tests on Stampede3 (and the CUDA tests on
     an `h100` node; the laptop's `cuda-ampere86` when it has space).
  2. Build `tests/reference` (the serial OpenSees and the drivers) and run the system tests
     that need OpenSees.
  3. Then milestone 4, step 4 (above).

- `benchmarks/run_all.sh` already detects Vista and Stampede3 nodes and configures the
  build: Grace CPUs with `-DKokkos_ARCH_ARMV9_GRACE=ON`, Hopper GPUs with
  `-DKokkos_ARCH_HOPPER90=ON` (Kokkos 5.1.0 has both), Sapphire Rapids with
  `Kokkos_ARCH_SPR`. It builds without tests (`FEMORAX_BUILD_TESTS=OFF`) and has not been run
  yet. There are no CMake presets for TACC; `scripts/check.sh` knows only the laptop presets.
  Before benchmarks there: build with tests on a `gg` (Grace-Grace) and a `gh` node and run
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
