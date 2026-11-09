# Open items

Recorded 2026-11-09 from the review of milestone 2 so far. Not being worked on now.

1. **Event-sequence comparison.** The milestone 2 plan requires exact event sequences against
   OpenSees (active surfaces, loading/unloading, substeps, retries). Today only Newton
   iteration counts are compared; the flip behind the implicit drift at shaking step 45
   (`element_std_implicit`, `element_ssp_implicit`) has not been identified.
2. **Tolerance band from OpenSees' own sensitivity.** The implicit nonlinear tolerances rest
   on one run each, of single elements. Set them from how far OpenSees differs from itself
   under a rounding-level change (as the host replica did with the LU pivoting), and check
   how the band grows with model size (the 7x7x25 column: 3.3e-4 in acceleration).
3. **deviations.md: behaviour differences recorded only in design.md.** Implicit stages with
   the Linear algorithm and a history material are refused; the recomputed operator tangent
   falls back to the stored one for history materials.
4. **support-table.md:** the PressureDependMultiYield note still lists element output as
   next (done in `80c5118`).
5. **GPU branch flips.** Single SSPbrick elements flip plastic branches on the GPU in
   explicit runs (7e-7, 1.2e-6) but the 25-element column does not (6e-13). Find why
   (motion, soil, time step, or chance).
6. **Licence.** Code ported from OpenSees is a derivative work under the OpenSees licence
   (educational, research and non-profit use); settle FemoraX's licence before the first
   public release.
7. **Femora's absolute tolerances.** The example's NormUnbalance 1e-7 and ModifiedNewton
   -factoronce re-equilibration fail in OpenSees itself on a 1x1x25 column of 0.5 m
   elements and on a 7x7x25 column; worth raising with the Femora side.
