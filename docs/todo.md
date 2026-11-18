# Open items

Recorded 2026-11-09 from the review of milestone 2 so far. Not being worked on now.

1. **Event-sequence comparison.** The milestone 2 plan requires exact event sequences against
   OpenSees (active surfaces, loading/unloading, substeps, retries). Retries and substeps
   are compared since 2026-11-09 (Femora's example column); active surfaces and
   loading/unloading are not, and the flip behind the implicit drift at shaking step 45
   (`element_std_implicit`, `element_ssp_implicit`) has not been identified.
2. **Tolerance band from OpenSees' own sensitivity.** The implicit nonlinear tolerances rest
   on one run each, of single elements. Set them from how far OpenSees differs from itself
   under a rounding-level change (as the host replica did with the LU pivoting), and check
   how the band grows with model size (the 7x7x25 column: 3.3e-4 in acceleration).
3. **deviations.md: behaviour differences recorded only in design.md.** Implicit stages with
   the Linear algorithm and a history material are refused; the recomputed operator tangent
   falls back to the stored one for history materials.
4. **GPU branch flips.** Single SSPbrick elements flip plastic branches on the GPU in
   explicit runs (7e-7, 1.2e-6) but the 25-element column does not (6e-13). Find why
   (motion, soil, time step, or chance).
5. **Licence.** Code ported from OpenSees is a derivative work under the OpenSees licence
   (educational, research and non-profit use); settle FemoraX's licence before the first
   public release.
6. **Femora's absolute tolerances.** The example's NormUnbalance 1e-7 and ModifiedNewton
   -factoronce re-equilibration fail in OpenSees itself on a 1x1x25 column of 0.5 m
   elements and on a 7x7x25 column; worth raising with the Femora side.
7. **KrylovNewton** (or another accelerated Newton) for slow, linear convergence.
   PressureDependMultiYield's tangent comes from the committed state, so Newton and
   ModifiedNewton both converge linearly (about 0.7 per iteration in the re-equilibration
   of Femora's example: 26 to 31 iterations to NormUnbalance 1e-7). A Krylov-accelerated
   Newton would cut that; it would be a new algorithm and a deviation from OpenSees.
8. **Implicit stages are slow and fragile at scale.** Seen on the 30 x 30 x 25 and 60 x 60 x 25
   PressureDependMultiYield columns (2026-11-09): the implicit gravity and re-equilibration
   took 450 s of an 808 s CPU run (Jacobi CG, 150 to 330 iterations per solve); in the
   plastic re-equilibration Newton falls into a 2-cycle (CPU) or stalls (GPU) and
   ModifiedNewton -factoronce fails at step 2; and the absolute tolerances of Femora's
   stages (NormUnbalance 1e-7) do not scale with the model. Candidates: matrix-free
   multigrid (the CG cost), KrylovNewton (item 7), and an optional explicit relaxation stage
   (damped explicit dynamics run to equilibrium) in place of the implicit gravity and
   re-equilibration.
9. **Output size.** VTKHDF of every node and element every 0.05 s is 160 MB for 30 x 30 x 25
   and 632 MB for 60 x 60 x 25 over 2 s. Single-precision output is done (`femorax run
   --vtkhdf-precision single`: responses as float32, about half the size; femora.results
   reads it, `system.vtkhdf_single_precision`). Still open: recording a region only, and a
   model-file attribute for the precision (a schema addition) if Femora should set it.
10. **H5DRM in implicit stages.** `form_unbalance` applies the DRM forces at the load time, but
    only the explicit path is verified (`tests/unit/test_drm_free_field.cpp`). Verify with a
    free field computed by the same Newmark recurrence, and against the stored OpenSees DRM
    box (milestone 4, step 5).
11. **Intel GPU: an intermittent GPU memory fault in `bench.smoke`** (2026-11-11, preset
    `s3-sycl-pvc`, Data Center GPU Max 1550). Once, with 8 test processes sharing one GPU
    tile, the benchmark smoke run stopped with "Segmentation fault from GPU ... type
    NotPresent ... access Write" right after the `explicit_force_pdmy_ssp` cases; the same
    binary passed alone 4 times out of 4, and the whole suite otherwise passes. Either a rare
    out-of-bounds write or race in a kernel (which would matter on every backend) or the
    Intel runtime with several processes on one tile. Next: run the benchmark kernels one by
    one under Intel's memory checking (`ONEAPI_DEVICE_SELECTOR`, Level Zero validation layer)
    and the CUDA build under `compute-sanitizer --tool memcheck`. Until then `s3-sycl-pvc`
    is not part of `scripts/check.sh`.
12. **Constant-Q damping (Qs, Qp)** (now planned as Uniform damping, docs/design.md; step 1 done), as SW4 uses for the DRM inputs: a viscoelastic law with
    a few relaxation mechanisms per integration point (deviatoric with Qs, volumetric with
    Qp), which barely reduces the explicit stable step, unlike Rayleigh stiffness damping.
    Findings (2026-11-11, OpenSees `955ceb0`, free vibration of a 1 x 1 x 10 elastic column,
    `/work2/08189/amnp95/femorax_plots/damping_test/column.tcl`): OpenSees' `damping Uniform`
    (Huang, Sturt & Willford 2019: filters the element's internal force, one ratio, a band)
    gives 4.40 % for a 5 % target on stdBrick (Rayleigh: 5.01 %) and **nothing at all on
    SSPbrick** (only a "yet to be implemented" warning per element), which explains the
    user's "no effect". It multiplies the stiffness by 1 + sum 4 alpha_i xi / (2 + w_i dt),
    so with a high ratio over a wide band (the user tried 95 % for absorbers) the element
    stiffens several times and the run goes unstable: causal frequency-independent damping
    is for physical damping (a few percent), not for absorbing layers (PML, or graded
    mass-proportional layers, which do not reduce the explicit step).
    Direction discussed with the user (nothing decided): one memory-filter damping for every
    element, input as a ratio (or separate shear and volume ratios for solids, the Qs/Qp of
    SW4) and a band [f1, f2]: SW4-consistent for elastic soil under SW4 DRM input, Huang-like
    (filtering the actual stress or force) for beams and yielding materials, and a stable-step
    check with the stiffened response. Plan first (formulation, verification, schema
    addition).
13. **Absorbing layers without betaK.** A mass-only absorber gives the basin a 10x larger
    explicit step with displacements within 2.3 % of the Rayleigh absorber, but reflects the
    high frequencies (twice the acceleration peak at a DRM exterior node; docs/handoff.md,
    "The basin example"). Next: a graded mass-proportional layer (alphaM rising inward, a
    per-element-layer Rayleigh region, which Femora can already export) measured the same
    way, then PML (milestone 4) as the frequency-independent absorber.
14. **The periodic stable-step check costs about 110 ms on the 46,080-element basin** (spr,
    112 threads, 2026-11-12): every 100 steps, about 1.1 ms per step on average, so it doubles
    the step time of a model whose every batch has material state (the plastic basin: 2.61
    against 1.30 ms/step elastic). Since 2026-11-12 the check covers only the batches with
    state (a damped region no longer pays for the whole model). Done the same day: the
    re-check covers only batches whose stable step can change, with a bound from reference
    eigenvalues stored once per stage (docs/design.md, "Stable time step").
15. **A production build with fused multiply-add.** Measured on Vista (gg node, 144 threads,
    `flat_gather_cached`, 5000 explicit steps of the Site 2 basin, 2026-11-12): the exact
    build (`-ffp-contract=off`, the vista presets) against the same build with
    `-ffp-contract=fast`: elastic 0.792 / 0.773 against 0.737 / 0.717 ms/step, plastic
    0.816 / 0.816 against 0.753 / 0.742 ms/step, about 7-8 % faster; results differ at
    rounding (the plastic gravity stage: 370.1 GMRES iterations per step instead of 364.5).
    Discussed with the user: keep the exact build for check.sh and the OpenSees comparisons,
    add a production build (fused, perhaps tuned for the CPU) whose tests compare at a
    tolerance as the CUDA build's do, and take benchmarks from it. Not decided or done.
16. **Large models on the H200 (Vista gh, GH200, 2026-11-17).** `femorax_bench --filter
    explicit_force_pdmy` (laminar soil columns n^3, dense and loose Ottawa sand,
    PressureDependMultiYield, cached geometry, gather scatter): the best variant's force
    evaluation, elastic / mixed / fully plastic state:
    - stdBrick: 48^3 2.9 / 7.9 / 13.5 ms, 64^3 6.6 / 18.2 / 31.5 ms, 80^3 12.6 / 34.7 /
      61.7 ms: linear in the elements, about 24 ns (elastic) and 120 ns (plastic) per
      element; best: team_gather_ept4 (elastic, mixed), the compacted corrector (plastic).
    - SSPbrick: 48^3 0.6 / 1.4 / 1.9 ms, 64^3 1.2 / 2.8 / 3.8 ms, 80^3 2.2 / 5.1 / 7.4 ms:
      about 4 and 14 ns per element; best: flat_gather (with or without compaction).
    - Memory: 96^3 ran out at 89.7 GB (the benchmark holds the three states at once): about
      30 KB per plastic stdBrick element, so one plastic stdBrick model of about 3 million
      elements fits the H200 (elastic stdBrick: about 3 KB, 25-30 million).
    - The auto-tune picked team_gather_ept2 for the basin; plastic models gain up to 3x
      with ept4 or the compacted path: check that the auto-tune finds them on full runs.
    Next: a generator for large laminar soil boxes (gravity, the switch to plastic, base
    shaking, no DRM) to run full analyses at 1-20 million elements, with the setup time
    measured (about 50 s for the 46,080-element basin, likely the DRM file).
17. **Full runs on the H200 (Vista gh, 2026-11-17).**
    - Site 2 basin, full 27.4 s (137,000 explicit steps): plastic 53.6 s of stepping
      (0.391 ms/step), 1 min 55 s in all; elastic 41.1 s (0.300 ms/step), 1 min 37 s
      (Grace CPU: 2 min 52 s and 2 min 41 s). Forced variants on the plastic basin:
      team_gather_ept2 0.390, ept4 0.401, ept4 compacted 0.462, flat compacted 0.544 ms/step:
      the auto-tune's choice (ept2) is the best there (one plastic region). Setup about 50 s
      of each run: the DRM file (the soil boxes below set up in seconds).
    - Laminar soil boxes, PressureIndependMultiYield clay, no gravity, 0.3 g 2 Hz base
      shaking, 2000 explicit steps of 5e-4 (generator: /scratch/08189/amnp95/femorax_bench/
      make_box.py on Vista, an experiment, not in the repository):
      | box | step | GPU memory | compile + setup | whole run |
      |---|---|---|---|---|
      | 100^3 stdBrick (1 M) | 23.5 ms | 29.9 GB | 2.0 + 0.9 s | 55 s |
      | 100^3 SSPbrick (1 M) | 10.5 ms | 5.4 GB | 2.1 + 0.8 s | 50 s |
      | 200^3 SSPbrick (8 M) | 81.0 ms | 43.4 GB | 17.7 + 6.4 s | 382 s |
      Linear in the elements. Limits on one H200: about 3 M plastic stdBrick, about 16 M
      plastic SSPbrick. The 8 M run spends about 195 s outside stepping and setup. Not the
      auto-tune (2026-11-17: the 1 M SSPbrick box takes the same 48 s with the variant
      forced): the stable-step reference eigenvalues, 13 Jacobi solves per plastic SSPbrick
      element at the start of each explicit stage (about 22 s per million elements on the
      H200; stdBrick needs one). About 8 % of a 30 s record; later: a narrower factor grid
      (0.5-4, 7 points: about 2x) or a cheaper bound.
    - Fixed 2026-11-17: the basin's ~50 s setup was the DRM histories read with one HDF5
      file open per node and history (9,814 opens of 5.4 ms on Vista's scratch);
      `io::DrmReader` opens the file once: setup 5 s, the 5000-step basin runs in 8.7 s
      (elastic) and 14.5 s (plastic) instead of about 60 s.
18. **Uniform damping's cost on the GPU.** Site 2 basin with Uniform damping (5 % over
    0.1-10 Hz, 7 filters) on region 1 (872 elements, 1.9 % of the model), 20,000 explicit
    steps on the H200 (2026-11-17), twice each: 0.366 ms/step against 0.335 undamped (+9 %),
    about 34 ns per damped element per step against about 7 ns for an average undamped one
    (on Stampede3's CPUs the same region cost nothing measurable). On the GPU the filter
    state (66 doubles per Gauss point, read and written every step) matters; part of the
    30 us may be the separate batch's launch. Fully damped plastic soil boxes (1 M PIMY
    elements, every element Uniform-damped, 2000 steps, same day): stdBrick 23.5 -> 30.8
    ms/step (+31 %), 29.9 -> 35.2 GB; SSPbrick 10.5 -> 13.3 ms/step (+26 %), 5.4 -> 6.1 GB.
    For plastic soil a moderate cost. Elastic (ElasticIsotropic, 1 M stdBrick, the same box):
    2.72 -> 9.34 ms/step (3.4x), 2.45 -> 10.9 GB (4.4x): the filter state (66 doubles per
    Gauss point, both buffers) is most of the work and memory of a damped elastic element.
    Still about 10 min for a 30 s record of a 1 M damped elastic box on one H200. Options:
    SSPbrick (one point), buckets matched to the fit (7 filters in the bucket of 8), one state
    copy in explicit stages, filtering strains for elastic materials (no stored activation
    stress), team layouts for the damped batch.
