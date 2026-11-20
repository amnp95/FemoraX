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
- The elastic basin with Uniform damping on all soil (regions 1-7): its DRM layer is damped
  too; supported since 2026-11-18 (the user decided the layer is soil and is damped).
- Confirmed (2026-11-12, Vista gg node, 144 threads, `flat_gather_cached`): the full
  plastic basin's explicit stage runs at 0.805 ms/step against 0.792 for the elastic basin
  (137,000 steps; whole runs 2 min 52 s and 2 min 41 s). Before the re-check fixes the
  plastic basin took 2.61 ms/step on Stampede3 (twice its elastic step). Logs:
  `run_vista.log` in each basin folder.
- Data on Vista: the DRM file and the basin models are in Vista's scratch at the same paths
  as on Stampede3 (`/scratch/08189/amnp95/drm_inputs`, `.../femorax_runs/basin_site2_*`;
  Stampede3's copies remain there). Scratch is purged after inactivity; to copy again, run
  `/work2/08189/amnp95/femorax_data/copy_from_stampede3.sh` on Stampede3, then move the files
  from /work to Vista's scratch.
- Push from Stampede3 (the repository is shared): Vista's SSH key is not on GitHub.

**Start here:** the typed-batch split is committed (2026-11-19); its GPU tests have not run
since the split (only the H200 build, 2 min 15 s): run check.sh on the next gh node.

**PML: on hold (the user, 2026-11-19):** the PML formulation needs to change before FemoraX
implements it; do not plan or port OpenSees' PML3D until the user brings the new
formulation. The absorber experiments (docs/todo.md item 13) show why an absorber better than
mass damping is needed; meanwhile the basin keeps its Rayleigh absorber.

**SW4's attenuation for Site 2 (2026-11-19):** the H5DRM file carries no Q values (its
metadata: the box, dt 0.0108 s, t to 27.5 s). Matching the box's soil damping to SW4's
(Uniform ratio 1/(2Q) per layer) removes the scattered field the medium mismatch makes; asked
the user for SW4's input (`attenuation`, `block`/`rfile` Qs, Qp). Without it: run the box
without the basin (SW4's layered medium only) at a few Uniform ratios and take the one with
the smallest motion outside the DRM layer (needs the site's layer velocities, check model.tcl).

**DRM frame (2026-11-19): done, schema 1.0** (docs/deviations.md, first H5DRM entry). FemoraX
reads a DRM file as it is (no z flip), turns the motions with the stations under a
transformation, and needs no `drmbox_*`; H5DRM patterns in 0.x files are refused with an
explanation. Models made before (the Site 2 runs in Stampede3's
`/scratch/08189/amnp95/femorax_runs/`) must be exported again as 1.0 with a DRM file in one
frame (the user's: `drm_inputs/site2_natural/Site2_DRMinput_1km/OpenSeesDRMinput.h5drm`).
Femora's exporter should write 1.0 and its OpenSees export warn about OpenSees' z flip.

**Local time stepping (2026-11-19): approved; step 1 started.** The literature review
(`docs/research/time_step_remedies/`) chose the stabilized leapfrog LTS of Grote, Michel &
Sauter; a numpy reproduction of their 1D test matches their critical steps
(`prototype/`). Next: the C++ host prototype and its tests (docs/design.md, "Local time
stepping", Verification): one level bit for bit, the eigenvalue sweep (the paper's 0.7104 h_c
for p = 2 without stabilization, none with nu = 0.01), the conserved modified energy over long
runs, second-order convergence (the paper's Gaussian on (0, 1)), then levels in the compile
step. Still to discuss with the user: the constraints their models need, their interface idea.
Done today besides: the DRM frame (schema 1.0), nodal masses damped by their region (item 21),
the mass and damping standard for structures (docs/design.md, "Structures").

**LTS (2026-11-19):** steps 1, 2 and 3a committed (b555557): the host reference scheme, the
levels, and the 3D brick column with a stiff inclusion. 3b done: the scheme in `run_model`
(`RunOptions::local_time_stepping`), two levels, undamped and stateless only, matching the host
reference within 7e-13 (docs/design.md). Next: 3c, the recursive multilevel scheme; then step 4,
damping (mass, stiffness, Uniform); step 5, the substeps' vector work and gather restricted to
the fine equations, and Kokkos Graph capture, with benchmarks.

**Embedded piles (2026-11-19):** literature review in docs/research/embedded_and_constraints/.
The user's own element, `EmbeddedBeamContact3D` (OpenSees C++, with a theory document, tests
and studies), is in /work2/08189/amnp95/uploads/embedded_beam/ (not in the repository): surface
patch coupling of a Hermite beam to Hex8 hosts, unilateral normal penalty, Coulomb friction
with cohesion, no mass. The user's evidence: bonded, it reproduces a conforming 3D solid pile
(merged nodes) where OpenSees' EmbeddedBeamInterface (point-wise penalty, used by Femora)
overpredicts the displacement. Seen in its runs: no true bonded mode (imitated with cohesion
and an interference gap, which pre-compresses the soil by kn |g0|: B2's 1.8 mm, stiffer than
the tie); with contact, the beam is about 17 % stiffer than the solid pile with contact (one
mesh only: a mesh series is still needed). Design consequence: perimeter bonding cannot be
plain elimination (more surface points than beam DOFs): a soft bonded surface law, or a
kinematic coupling slaving the beam node to its perimeter. Order agreed: beams first (after
LTS 3b), then rigid links / diaphragms, then embedded piles.

**Curated components and the performance gate (the user, 2026-11-19):** Femora supports only
the OpenSees components judged useful, so one model runs in both solvers; each new component
passes a gate first: its explicit stable step (declared; cost; isolable by LTS), GPU fit
(compile-time sizes, no divergent iterations), the mass standard, and a TACC GPU benchmark.
Candidates noted with their risks (docs/design.md, "What Femora can emit" for the agreed list):
u-p elements for liquefaction (water's fast P wave shrinks the step: LTS or reduced fluid
compressibility), zeroLength with PySimple1 / TzSimple1 / QzSimple1 springs and viscous
dashpots (zero mass, stiff springs), Steel02 and cyclic concretes (divergent branches), shells
(step ~ thickness^2), tetrahedra (locking, small elements), bearings / joints / MVLEM (on
request only).

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

Next steps (in order; each one commit; step 4 done):

4. Stream the DRM histories in time blocks. Done (2026-11-17, Vista gh): `DrmLoad` keeps a
   window of W samples on the device (W from a 256 MB budget, `RunOptions::
   drm_window_samples` to set it), moves it forward when the interpolation's samples leave
   it (consecutive windows share a sample), reads each window with one HDF5 read through an
   `io::DrmReader` that keeps the file open, and reads the next window ahead on a thread when
   HDF5 is thread-safe (`io::hdf5_thread_safe`; synchronously otherwise). The host-to-device
   copy of a window is synchronous (about 30 MB, sub-millisecond on GH200). Tests: the DRM
   box with W = 2 and W = 7 equals the whole history bit for bit; the Site 2 basin (5000
   steps, H200) writes bit-identical output; its run takes 4.7 s (elastic) and 10.6 s
   (plastic) instead of 8.7 and 14.5 s, setup 0.13 s, the history window 256 MB instead of
   0.6 GB. Before this step the setup read every node's history with its own file open
   (50 s; fixed first, commit f2e8f22).
4b. The damped DRM layer: step 1 (Rayleigh terms) done 2026-11-17, step 2 (Uniform damping:
   the element's own force with a DRM state per side) done 2026-11-18 (docs/design.md). Next:
   the basin with Uniform damping in all its soil against Rayleigh (Uniform verification 7);
   a file's velocity dataset when present.
   Basin with Uniform damping (5 % over 0.1-10 Hz) in all its soil, DRM layer included,
   absorber Rayleigh (2026-11-18, H200, full 27.4 s, 108.7 s against 44.7 s with Rayleigh):
   surface-centre displacement peak 0.136 -> 0.133 m, acceleration 9.69 -> 8.50 m/s2; surface
   acceleration spectra Uniform / Rayleigh 1.00 (0.2-1 Hz), 0.81 (1-3), 0.73 (3-6), 0.60
   (6-10): the basin's Rayleigh (alphaM 0.94, betaK 5.3e-4 in region 1) damps 15 % at 0.5 Hz
   but only 2-3 % at 3-15 Hz. DRM exterior node: acceleration peak 0.86 -> 1.18 m/s2 (+38 %):
   the DRM is exact only when the box's soil is the medium of the free field (SW4's, with its
   Q); Uniform's dispersion (x0.93-1.09 over the band) and 5 % differ from it, as Rayleigh
   does differently. Next: Uniform damping matched to SW4's Q for Site 2 (ask the user for
   the SW4 attenuation settings), then compare the scattered fields.
   Effect on the elastic Site 2 basin (full 27.4 s, H200; its soil regions, DRM layer
   included, have light Rayleigh damping): surface-centre displacement within 0.14 % of its
   peak, acceleration 1.2 %; a DRM exterior node (scattered field) 0.84 % / 0.91 % (its
   acceleration peak 0.859 -> 0.856); largest nodal change 3.4e-4 m at 10 s. The whole run
   now takes 44.7 s on the H200 (96.6 s before the DRM reader fix and streaming).
5. A tiny stored OpenSees DRM box reference (run once, stored), compared in the check.
6. H5DRM in implicit stages: verified 2026-11-19 (Newmark free-field box, 2.7e-14; the
   record-end rounding fixed). Was: `form_unbalance` applies the forces, not verified yet
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
