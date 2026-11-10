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
   and 632 MB for 60 x 60 x 25 over 2 s. Options: single-precision VTKHDF output (optional;
   femora.results must read it), and recording a region only.
10. **H5DRM in implicit stages.** `form_unbalance` applies the DRM forces at the load time, but
    only the explicit path is verified (`tests/unit/test_drm_free_field.cpp`). Verify with a
    free field computed by the same Newmark recurrence, and against the stored OpenSees DRM
    box (milestone 4, step 5).
