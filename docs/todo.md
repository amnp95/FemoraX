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
