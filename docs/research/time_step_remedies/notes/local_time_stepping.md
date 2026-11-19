# Multilevel / clustered local time stepping (LTS) for explicit second-order elastodynamics

Scope: LTS families for explicit (leapfrog / central-difference / Newmark-explicit) integration of
`M u'' + K u = f` with lumped M, as a remedy when a small fraction of elements is small or stiff.
Context for the choice: FemoraX (Kokkos, GPU, matrix-free, FP64), ExplicitDifference /
CentralDifference, Rayleigh damping, Uniform (filter) damping with memory state, multi-yield
plasticity, beams, rigid links, penalty interfaces, DRM.

Note on method: about 20 searches/fetches; many full texts (KIT CRC 1173 preprints, Basel edoc
preprints, the JCP 2017 Rietmann paper) could not be read (PDF extraction unavailable, host
unreachable or bot-blocked). Claims taken only from abstracts or search snippets are marked; claims
from my background knowledge that I could not verify in this session are put under Inferences or
Gaps and labelled "unverified".

## 1. How the main LTS schemes work

### Takeaway
Two broad families exist. (a) Leapfrog/Newmark-based LTS for second-order systems (Diaz & Grote
2009 and its descendants: multilevel, stabilized, Newmark-on-GPU), which split the DOFs by a 0/1
diagonal projection P, evaluate the coarse forces once per coarse step and take p fine substeps on
the fine DOFs with the coarse contribution frozen; it is energy-conserving and second order and is
the natural match to central difference with lumped mass. (b) Clustered LTS for first-order DG
(ADER-DG in SeisSol/EDGE), with rate-2 clusters exchanging time-integrated data; it is the most
mature at production scale but tied to ADER-DG. Structural mechanics has its own older line
(Belytschko subcycling, Smolinski/Daniel, asynchronous variational integrators, LS-DYNA
subcycling).

### Cited Findings
- **Diaz & Grote 2009** (SIAM J. Sci. Comput. 31(3), DOI [10.1137/070709414](https://doi.org/10.1137/070709414)): explicit LTS for second-order wave equations; with a symmetric FE discretization and (nearly) diagonal mass the scheme stays explicit and parallel and conserves a discrete energy exactly; starts from leapfrog and derives higher-order versions. — [zendy record](https://zendy.io/pdf-viewer/10.1137/070709414); [HAL record](https://hal-univ-pau.archives-ouvertes.fr/inria-00409233)
- **Algorithm of LTS-LF2(p)** (as presented by Grote & Mitkova): unknowns split `z = (I-P)z + Pz`, P diagonal 0/1 marking the unknowns of the locally refined region. Per coarse step Δt: compute `w = (I-P)R_n - A(I-P) z_n` once (coarse forces frozen at t_n); run p leapfrog-type substeps of size Δτ = Δt/p on an auxiliary variable starting from `q0 = 2 z_n`, each substep applying `A P` (touches only fine unknowns and their neighbours) with sources at `t_n ± τ_m`; finish with `z_{n+1} = -z_{n-1} + q_1`. Cost: one product with `A(I-P)` and about p products with `AP` per coarse step. The 4th-order LTS-LFME4 needs 3 coarse and 2p fine products. — [Grote & Mitkova, arXiv:1205.0654](https://arxiv.org/abs/1205.0654) (ar5iv rendering read)
- Stability of LTS-LF2(p): stable when `(Δt²/4) λ_max(A_p) ∈ (0,1)` where `A_p` is a modified operator depending on Δt and p; LTS-LF2 conserves a discrete energy with respect to `A_p` (Prop. 3.5). — [arXiv:1205.0654](https://arxiv.org/abs/1205.0654)
- **Multilevel extension** (Diaz & Grote, Basel preprint 2014, published as "Multi-level explicit local time-stepping methods for second-order wave equations"): recursive application of the LTS idea to hierarchical refinement; remains explicit and energy-conserving with high accuracy. — [HAL hal-01184090](https://hal-univ-pau.archives-ouvertes.fr/hal-01184090) (page bot-blocked; abstract via search snippet); [Basel preprint 14-10](https://edoc-vmtest.ub.unibas.ch/64976/1/Preprint-1410.pdf) (unreachable during this session)
- **Runge-Kutta-based LTS** (Grote, Mehlin & Mitkova, SIAM J. Sci. Comput. 2015): explicit LTS of arbitrarily high accuracy from classical or low-storage RK methods. **Almquist & Mehlin 2017** (SIAM J. Sci. Comput. 39(5):A2020–A2048): multilevel LTS of RK type, for continuous FE with mass lumping and SBP-SAT finite differences; accuracy and energy properties proved for any number of levels (details from a secondary summary). — [KIT CRC 1173 preprint 2016/16](https://www.waves.kit.edu/downloads/CRC1173_Preprint_2016-16.pdf); [Mehlin seminar abstract](https://user.it.uu.se/~perl/seminar_abstracts/sem_spring15/Mehlin.html); [Mehlin thesis](https://edoc-vmtest.ub.unibas.ch/38308/1/Thesis_Mehlin.pdf)
- **Rietmann, Grote, Peter & Schenk 2017**, "Newmark local time stepping on high-performance computing architectures", J. Comput. Phys. 334:308–326, DOI [10.1016/j.jcp.2016.11.012](https://doi.org/10.1016/j.jcp.2016.11.012): multilevel LTS-Newmark implemented in SPECFEM3D (continuous spectral-element), adapted to continuous FE, efficient for element-size contrasts above 100x, run on thousands of CPU cores and hundreds of GPUs. — [edoc.unibas.ch/59456](https://edoc.unibas.ch/59456); [researchr bibtex](https://researchr.org/publication/RietmannGPS17/bibtex)
- **Rietmann et al., IPDPS 2015**, "Load-Balanced Local Time Stepping for Large-Scale Wave Propagation": the recursive multilevel LTS creates strong load imbalance under standard partitioning; graph vs hypergraph partitioners compared on CPU and GPU clusters with SPECFEM3D. — [record](https://sindex.sdl.edu.sa/esploro/outputs/conferenceProceeding/Load-Balanced-Local-Time-Stepping-for-Large-Scale/9943772408331); [edoc.unibas.ch/41217](https://edoc.unibas.ch/41217)
- **Carle & Hochbruck / Hochbruck & Sturm (leapfrog-Chebyshev, LFC):** the linear part is advanced with Chebyshev polynomials (much weaker CFL than leapfrog at higher cost), the nonlinear part with standard leapfrog; standard LFC schemes "suffer from stability problems" and fail to reproduce the stability of the exact solution, so a stabilized modification is proved stable via generating functions. Carle & Hochbruck (2023, DG wave equations): leapfrog on elements satisfying a mild local CFL, LFC on the remaining (fine) elements; the polynomial degree and stabilization parameter must be chosen appropriately; stability and optimal convergence proved. Hochbruck & Scheifinger (2025, Friedrichs' systems): local time integration with a filter that is a scaled/shifted Chebyshev polynomial, CFL much weaker than leapfrog at only slightly higher cost. — [KIT CRC preprint 2023-2](https://www.waves.kit.edu/downloads/CRC1173_Preprint_2023-2.pdf); [KIT CRC preprint 2019-19](https://www.waves.kit.edu/downloads/CRC1173_Preprint_2019-19.pdf); [KIT CRC preprint 2018-17](https://www.waves.kit.edu/downloads/CRC1173_Preprint_2018-17.pdf); [On leap-frog-Chebyshev schemes (KIT)](https://publikationen.bibliothek.kit.edu/1000085527/16708333); [arXiv:2503.05282](https://arxiv.org/pdf/2503.05282) (content from search snippets; full texts not read)
- **SeisSol / EDGE clustered LTS (ADER-DG):** an "efficient and scalable clustered local time stepping scheme for ADER-DG" for elastic waves was introduced to SeisSol by Breuer et al., "Petascale local time stepping for the ADER-DG finite element method" (IPDPS 2016). Breuer & Heinecke 2022 (EDGE) revise it as a rate-2 scheme with a parameter λ setting cluster boundaries; theoretical speedups of 2.28x over global time stepping (λ=1) and 2.67x with tuned λ (+17.5 %); over 10x time-to-solution improvement on the La Habra case, 256–1,536 Frontera nodes, >95 % parallel efficiency (baseline of the 10x not stated in the abstract). — [arXiv:2202.10313](https://arxiv.org/abs/2202.10313)
- **Structural-mechanics subcycling:** Belytschko–Lu subcycling was proved unstable; Daniel's modified version is only "statistically stable" (a probability of stability for given step limits); only the Smolinski–Sleith algorithm was stable in the classical sense, at a cost in accuracy; Smolinski–Sleith–Belytschko (1996) was the first explicit multi-time-step algorithm with a rigorous stability analysis; Daniel attributed the problems to the central difference method itself; energy-conserving multi-time-step extensions of central difference were later proposed (Smolinski, Daniel). Subcycled Newmark (Daniel 1997): with ≥3 subcycles the trapezoidal rule's stability becomes conditional, tending to that of central difference. — [Daniel, Comput. Mech. 1998](https://link.springer.com/article/10.1007/s004660050248); [search records](https://www.bohrium.com/paper-details/a-study-of-the-stability-of-subcycling-algorithms-in-structural-dynamics/812697760359776258-3776); [McMaster thesis](https://macsphere.mcmaster.ca/handle/11375/29249) (snippets only)
- **Asynchronous variational integrators (AVI)**, Lew, Marsden, Ortiz & West, Arch. Rational Mech. Anal. 2003: each element takes its own time step, with no integer ratio required; designed for *nonlinear* elastodynamics; symplectic-momentum preserving, local energy and momenta conserved subject to solvability; nearly conserve energy locally and globally (Lew thesis). Follow-up: Sohan, Haneesh, Darve & Lew (2011) scalable explicit asynchronous integrators. — [Lagrange group page](https://lagrange.mechse.illinois.edu/pubs/LeMaOrWe2003/); [Lew thesis](https://thesis.caltech.edu/2077/); [Sohan et al. 2011, DOI 10.1002/nme.3236](https://zendy.io/title/10.1002/nme.3236)

### Inferences
- For a lumped-mass central-difference code, the Diaz–Grote LF-LTS (multilevel, stabilized per §2) is the closest match: same three-term recurrence, one level = plain leapfrog bit for bit, matrix-free (only `A P` and `A (I-P)` products, i.e. element force evaluations restricted to element sets), no assembled matrix. The ADER-DG cluster scheme does not transfer to continuous FE with lumped mass.
- Note an important structural feature of LF-LTS: the fine substeps apply `A P`, i.e. the full stiffness *rows* of the fine unknowns, which in a matrix-free code means evaluating every element touching a fine node (fine elements plus the first ring of coarse elements) at every substep, and the coarse elements' contribution to fine nodes is the frozen `w`. So element membership in "fine force evaluations" is a halo of the fine DOF set, not just the fine elements (inference from the `A P` formulation; Rietmann's continuous-FE adaptation addresses exactly this, details unverified).
- AVI is the only family found that was designed for nonlinear elastodynamics from the outset; it is attractive conceptually for plasticity but its per-element asynchronous event queue maps poorly to batched GPU kernels (inference).

### Gaps
- Exact Rietmann 2017 algorithm (Newmark velocity form, how the continuous-FE interface layer is defined, level assignment rule, measured speedups per level) not read; JCP full text inaccessible.
- Diaz & Grote multilevel paper: journal (believed CMAME 291 (2015) 240–265, unverified) and recursion details not confirmed.
- Carle & Hochbruck 2023 exact journal/DOI and the stability conditions not read.

## 2. Stability at level interfaces

### Takeaway
The original LF-LTS of Diaz & Grote can go unstable at certain discrete values of Δt (for fixed p), so its proven CFL condition depends on the smallest elements; Grote, Michel & Sauter (Math. Comp. 2021) fix this with a Chebyshev-damped polynomial (small parameter ν, e.g. 0.05) and prove a CFL condition that depends only on the coarse mesh. Structural subcycling schemes (Belytschko–Lu) can be unstable or only statistically stable; Smolinski–Sleith is stable at an accuracy cost.

### Cited Findings
- Grote, Mehlin & Sauter, "Convergence analysis of energy conserving explicit local time-stepping methods for the wave equation" (SIAM J. Numer. Anal. 56 (2018) 994–1021, per a citation; journal page not seen): optimal convergence of fully discrete LF-LTS with conforming FE, **but under a CFL condition depending on the smallest elements**. — [arXiv:1703.07965](https://arxiv.org/abs/1703.07965); [KIT record](https://publikationen.bibliothek.kit.edu/1000075943)
- Grote, Michel & Sauter, "Stabilized leapfrog based local time-stepping method for the wave equation", Math. Comp. (DOI [10.1090/mcom/3650](https://doi.org/10.1090/mcom/3650), from library record): LF-LTS "may become unstable at certain discrete values of the time step", so its constraint cannot be removed from the original method; similar instabilities occur in LFC methods without stabilization. — [arXiv:2005.13350](https://arxiv.org/abs/2005.13350); [library record](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_crossref_primary_10_1090_mcom_3650)
- Mechanism: for ν = 0 the LTS polynomial touches the stability bounds at p−1 points of tangency (no margin); ν > 0 keeps it strictly inside. The instabilities "matter mainly in special situations and for long-time simulations". — [ar5iv 2005.13350](https://ar5iv.labs.arxiv.org/html/2005.13350)
- Stabilized recurrence: δ = 1 + ν/p², ω = 2T_p'(δ)/T_p(δ), `P_{p,ν}(x) = 2(1 − T_p(δ − x/ω)/T_p(δ))`. Per coarse step: `z0 = u^n`, `w = A Π_c z0`; `z1 = z0 − ½(Δt/p)²(2p²/(ωδ))(w + A Π_f z0)`; for k = 1..p−1: `z_{k+1} = (1+β_k) z_k − β_k z_{k−1} − (Δt/p)²(2p²/ω) β_{k+1/2} (w + A Π_f z_k)` with `β_k = T_{k−1}(δ)/T_{k+1}(δ)`, `β_{k+1/2} = T_k(δ)/T_{k+1}(δ)`; `u^{n+1} = −u^{n−1} + 2 z_p`. ν = 0 gives the original LF-LTS; p = 1 gives leapfrog. — [ar5iv 2005.13350](https://ar5iv.labs.arxiv.org/html/2005.13350)
- Proven CFL: `C_cont C_inv² (Δt/h_c)² ≤ (2 + ν/p²) ω_{p,ν}/p²` (h_c = coarse mesh size); a simpler sufficient condition `C_cont C_inv² (Δt/h_c)² ≤ 4 e^{−ν}`. Coercivity constant ∝ ν/(p²(2+ν)²) degrades as ν → 0, but small fixed ν behaves robustly in experiments. v4 correction: ν ≤ 1/2 (previously ≤ 1); in practice ν = 0.05. Fully explicit, second order, energy-conserving, three-term recurrence. — [arXiv:2005.13350](https://arxiv.org/abs/2005.13350); [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.13350)
- Cost: if the fine space is small, cost is dominated by one `A Π_c` per global step; away from the interface the coarse region is an ordinary leapfrog step independent of p and ν. — [ar5iv 2005.13350](https://ar5iv.labs.arxiv.org/html/2005.13350)
- LFC-based LTS (Carle & Hochbruck 2023): stability requires appropriate choice of Chebyshev degree and stabilization parameter. — [KIT CRC preprint 2023-2](https://www.waves.kit.edu/downloads/CRC1173_Preprint_2023-2.pdf) (snippet)
- Structural subcycling: Belytschko–Lu unstable; Daniel's version statistically stable; Smolinski–Sleith classically stable but less accurate. — [Daniel 1998](https://link.springer.com/article/10.1007/s004660050248)

### Inferences
- Practical recipe for FemoraX: implement the *stabilized* LF-LTS with small ν (0.01–0.05) from the start; ν = 0 gives the original scheme, so both are one code path. With ν > 0 the coarse step is reduced only slightly (factor e^{−ν/2} in the sufficient bound, ≈ 2.5 % for ν = 0.05) — inference from the stated sufficient condition.
- With integer ratio p per level (e.g. 2), the level assignment should leave a safety margin (e.g. assign an element to the coarsest level whose step is ≤ 0.9× its stable step), since the instabilities sit at discrete Δt values near the limit (inference).

### Gaps
- No proven stability theory found for the multilevel stabilized variant with more than two levels (the paper mentions multilevel only for the original method).
- No stability result found for LF-LTS with ExplicitDifference's lagged half-step damping.

## 3. Accuracy

### Takeaway
LF-LTS and its stabilized form are second order in time with optimal FE convergence (proved), conserve a modified discrete energy in the undamped case, and reduce to leapfrog away from the interface. Higher-order variants exist (LFME4, RK-LTS, AB-LTS).

### Cited Findings
- Optimal convergence rates proved for fully discrete LF-LTS + conforming FE (under a fine-mesh CFL). — [arXiv:1703.07965](https://arxiv.org/abs/1703.07965)
- Stabilized LF-LTS: second order, optimal convergence under coarse-only CFL, discrete energy conserved. — [arXiv:2005.13350](https://arxiv.org/abs/2005.13350)
- Energy conserved for undamped leapfrog LTS; "no longer conserved" in the damped case (energy decays). — [arXiv:1205.0654](https://arxiv.org/abs/1205.0654)
- Multilevel RK-LTS: accuracy and energy properties proved for any number of levels (secondary summary). — [KIT CRC preprint 2016/16](https://www.waves.kit.edu/downloads/CRC1173_Preprint_2016-16.pdf)
- AVI: local energy and momenta conserved subject to solvability. — [Lagrange group page](https://lagrange.mechse.illinois.edu/pubs/LeMaOrWe2003/)

### Inferences
- The conserved energy is with respect to the modified operator `A_p`, so energy monitoring tests should compare the LTS-modified energy, not the plain `½v'Mv + ½u'Ku` (inference from the A_p formulation).
- A natural verification ladder for FemoraX: 1D bar with refined segment, second-order convergence against analytic solution, one level reproduces leapfrog to the bit (consistent with the existing design.md LTS plan).

### Gaps
- No quantitative interface reflection/accuracy numbers retrieved (spurious reflections at the coarse/fine interface).

## 4. Damping in LTS

### Takeaway
Published LTS theory covers linear viscous (mass-type, diagonal) damping: Grote & Mitkova's leapfrog/Crank–Nicolson LTS (second order, diagonal damping treated implicitly-but-explicitly) and Adams–Bashforth LTS of arbitrary order. Stiffness-proportional (Rayleigh βK) damping and memory-variable attenuation with LF-LTS have no rigorous treatment that I found; in DG codes (SeisSol, EDGE) anelastic attenuation is part of the ADER element-local update and works with clustered LTS.

### Cited Findings
- Damped wave equation with damping operator `D = M^{-1/2} M_σ M^{-1/2}` (essentially diagonal): LTS-LFCN2 treats D implicitly via `(I ± (Δt/2p) D)^{-1}`, still explicit since D is diagonal; second order; "cannot be readily extended beyond order two"; with p = 1 energy decays independently of σ under the undamped CFL; "very large values of σ will affect the CFL stability condition of any explicit method". — [arXiv:1205.0654](https://arxiv.org/abs/1205.0654)
- Adams–Bashforth LTS for damped wave equations, arbitrarily high order, written for `y' = B y`, `B = [[0, I], [−A, −D]]`; verified with continuous and DG discretizations. Grote & Mitkova, J. Comput. Appl. Math. (2013). — [arXiv:1109.4480](https://arxiv.org/abs/1109.4480v2)
- EDGE's next-generation LTS paper lists anelasticity together with clustered LTS as enhancements. — [arXiv:2202.10313](https://arxiv.org/abs/2202.10313)
- SPECFEM3D: attenuation is supported on CPU, CUDA and HIP, but LTS is listed CPU-only; no statement on combining LTS with attenuation. — [SPECFEM3D features](https://specfem3d.readthedocs.io/en/latest/features/)

### Inferences
- Mass-proportional (αM) Rayleigh damping maps directly onto LTS-LFCN2 (diagonal with lumped mass): each substep applies its own `(I ± Δτ/2 αM)` factors, i.e. FemoraX's CentralDifference diagonal treatment per level. ExplicitDifference's lagged half-step damping has no LTS analysis.
- Stiffness-proportional damping `βK v` is a force computed inside the element kernel from velocity; in LF-LTS it would be frozen for coarse elements like the stiffness force. This is an un-analysed modification; its stability must be derived/checked numerically (consistent with the design.md note that damping is "the research part").
- Uniform (filter) damping: the filters live per Gauss point of an element; if each element updates its filters with its own level's Δτ (StepContext per batch) the filter recursion is consistent per element, but how frozen coarse filter stresses interact with fine substeps is unanalysed. A safe first rule: no damped element on an interface ring at a different level, or treat damping forces of coarse elements as frozen during fine substeps (inference; needs a stability study).

### Gaps
- No paper found analysing LF-LTS (or Newmark-LTS) with stiffness-proportional Rayleigh damping or with memory-variable (generalized Maxwell / filter) attenuation in continuous FE.
- Whether SPECFEM3D's LTS runs with attenuation enabled could not be verified.

## 5. Nonlinear materials, history state, constraints, contact

### Takeaway
Nonlinearity is handled in the LFC-LTS line (semilinear: nonlinear part by plain leapfrog) and natively in AVI; SeisSol runs off-fault plasticity and dynamic rupture with its clustered LTS in production; LS-DYNA subcycling runs nonlinear crash models with contact. No rigorous theory was found for LF-LTS with history-dependent plasticity, multipoint/rigid constraints spanning levels, or penalty contact across levels.

### Cited Findings
- Leapfrog-Chebyshev for semilinear problems: Chebyshev for the linear part, standard leapfrog for the nonlinearity. — [KIT CRC preprints](https://www.waves.kit.edu/downloads/CRC1173_Preprint_2019-19.pdf) (snippet)
- AVI designed for nonlinear elastodynamics with element-wise steps. — [Lagrange group page](https://lagrange.mechse.illinois.edu/pubs/LeMaOrWe2003/)
- SeisSol: off-fault Drucker–Prager plasticity (Wollherr, Gabriel & Uphoff 2018, GJI 214:1556–1584, DOI [10.1093/gji/ggy213](https://doi.org/10.1093/gji/ggy213)), ~7 % extra cost; whether that paper used LTS is not stated in what I read. Dynamic rupture with LTS reported 13.6x over the previous implementation and 6.8x over optimized global time stepping (Uphoff et al., SC17 Sumatra paper, per search summary). — [LMU repository](https://epub.ub.uni-muenchen.de/67911); [Sumatra paper record](https://www.geophysik.uni-muenchen.de/en/research/publications/2309)
- LS-DYNA subcycling: groups of nodes/elements at different steps; S-rail ~9 % CPU saving, RAV4 ~45 %, results unchanged; combined with mass scaling (`*CONTROL_SUBCYCLE_MASS_SCALED_PART_{SET}`) because a few small elements otherwise demand large added mass; DT2MS < 0 scales only elements that need it. — [LS-DYNA Forum 2014 paper](https://dynamore.de/de/download/papers/2014-ls-dyna-forum/documents/entwicklervortraege/current-status-of-subcycling-and-multiscale-simulations-in-ls-dyna); [Ansys webinar 2022](https://lsdyna.ansys.com/wp-content/uploads/2025/11/221215_Webinar_Mass-Scaling_and_Subcycling.pdf); [DYNAmore PHS paper](https://dynalook.com/conferences/11th-european-ls-dyna-conference/crash-metal-failure/applications-of-multiscale-and-subcycling-methods-for-press-hardened-steel-parts-failure-assessment)

### Inferences
- Plasticity in LF-LTS: each element's material is updated only when that element's force is evaluated, at its own level's substeps, and commits at its own substeps (per-batch buffer swap). Coarse elements adjacent to fine nodes are evaluated once per coarse step (frozen `w`), which is exactly the semilinear "nonlinear part by leapfrog" pattern if the nonlinear forces are treated as part of the coarse frozen force. This is consistent and explicit, but no stability proof exists (inference).
- Constraints (equalDOF, rigid links, rigid diaphragms) couple DOFs; the simplest rule is that all DOFs of one constraint group (one equation after elimination) are in one level, and the level of an equation is the finest of all elements touching any DOF of the group — FemoraX's DOF elimination already merges them into one equation, so this follows naturally (inference).
- Penalty interfaces and stiff beams are the reason to use LTS at all in SSI: putting them and their attached soil node ring in the fine level is the intended use; no published LTS stability analysis for penalty contact was found.

### Gaps
- No published LF-LTS treatment of plasticity with history variables, MPC across levels, or penalty contact.
- LS-DYNA subcycling algorithmic details (power-of-two groups, interface treatment, contact) not read (dynalook TLS error).

## 6. GPU implementation, launch overhead, load balance, measured speedups

### Takeaway
Multilevel LTS creates load imbalance (fine levels are tiny and run many substeps) and many small kernel launches; SPECFEM's answer was level-aware partitioning (hypergraph) and SeisSol's is CUDA/HIP graph capture of all cluster kernels. Measured LTS speedups range from ~2–3x (theoretical, EDGE) to ~7–65x (SeisSol, case-dependent).

### Cited Findings
- SPECFEM3D: `LTS_MODE` in Par_file enables LTS for the Newmark scheme and re-distributes the mesh partition to balance elements of similar local step sizes; current feature table lists LTS as CPU-only (Newmark: CPU, CUDA, HIP). The JCP paper reported GPU runs (research version). — [SPECFEM3D features](https://specfem3d.readthedocs.io/en/latest/features/); [user manual tex](https://www.github.com/SPECFEM/specfem3d/blob/master/doc/USER_MANUAL/features.tex); [edoc 59456](https://edoc.unibas.ch/59456)
- Load imbalance is strong with standard partitioning; graph vs hypergraph partitioning compared (IPDPS 2015). — [record](https://sindex.sdl.edu.sa/esploro/outputs/conferenceProceeding/Load-Balanced-Local-Time-Stepping-for-Large-Scale/9943772408331)
- Rietmann thesis: GPU + LTS + scaling optimizations together accelerate applications by 30x or more (combined figure). — [USI thesis](https://susi.usi.ch/usi/documents/318539)
- SeisSol GPU: clusters with the highest update frequency "tend to be extremely small" and introduce significant kernel-launch overhead; CUDA graphs captured for all time clusters and macro-kernels (volume, local, neighbour integrals, ADER); clusters updated sequentially to resolve flux dependencies; graphs gave ~630 DP-TFLOP/s more on average in the [512, 4096] node range (thesis claim). `USE_GRAPH_CAPTURING` is on by default for NVIDIA/AMD builds. — [TUM thesis](https://mediatum.ub.tum.de/doc/1729783/1729783.pdf); [SeisSol GPU docs](https://seissol.readthedocs.io/en/latest/gpus.html); [LRZ oneAPI slides](https://doku.lrz.de/files/17826165/16942055/2/1685941563067/oneapi-workshop-lrz-seissol.pdf)
- SeisSol measured LTS speedups (per search summaries of SeisSol papers): ≈9.75x vs global stepping (GPU Earth-air paper), 13.6x / 6.8x (dynamic rupture), 30x (Palu) and 65x (Helsinki) in poroelastic/multiphysics setups. EDGE: 2.28x / 2.67x theoretical. — [arXiv:2202.10313](https://arxiv.org/abs/2202.10313); [TUM doc 1730250](https://mediatum.ub.tum.de/doc/1730250/1730250.pdf) (summaries; individual sources not individually opened)
- General: kernel batching in CUDA graphs for iterative/time-stepping codes has an optimal batch size balancing overhead of graph creation. — [arXiv:2501.09398](https://arxiv.org/pdf/2501.09398)

### Inferences
- For FemoraX with Kokkos: per level, one fused force launch over that level's (element) batches restricted to the fine set plus interface ring; with ratio 2 and L levels the finest runs 2^L launches per coarse step. Kokkos::Graph capture (CUDA/HIP graphs underneath) is the direct analogue of SeisSol's approach. Speedup bound ≈ (N·2^L)/(Σ_l N_l·2^l) on work; with 1–5 % structural elements at 2–20x smaller steps, the theoretical gain is roughly the full ratio if launch overhead is hidden (inference).
- Multi-GPU: SPECFEM's experience says partition by level-weighted cost (each element weighted by its substep count) or balance each level separately (inference from the IPDPS paper).

### Gaps
- No per-level GPU timings or launch-overhead numbers from SPECFEM found; SeisSol speedup numbers came via search summaries.

## 7. Maturity and production use

### Takeaway
LTS is production-grade in seismology DG codes (SeisSol, EDGE: clustered rate-2 LTS, GPU, at petascale) and available on CPU in SPECFEM3D (Newmark LTS); in structural explicit codes LS-DYNA offers subcycling (often combined with mass scaling) with modest reported savings. For continuous FE with lumped mass and nonlinear/damped physics, the theory stops at linear damping; practical extensions are engineering choices to be verified numerically.

### Cited Findings
- SeisSol: clustered LTS in production, GPU graph-captured, used for dynamic rupture and multiphysics runs. — [SeisSol GPU docs](https://seissol.readthedocs.io/en/latest/gpus.html); [arXiv:2202.10313](https://arxiv.org/abs/2202.10313)
- SPECFEM3D_Cartesian: LTS (Newmark) available, CPU-only per current feature table. — [SPECFEM3D features](https://specfem3d.readthedocs.io/en/latest/features/)
- LS-DYNA: subcycling with 9–45 % CPU savings in examples; mass scaling combination. — [LS-DYNA Forum 2014](https://dynamore.de/de/download/papers/2014-ls-dyna-forum/documents/entwicklervortraege/current-status-of-subcycling-and-multiscale-simulations-in-ls-dyna)
- Structural subcycling history shows real instability risks (Belytschko–Lu). — [Daniel 1998](https://link.springer.com/article/10.1007/s004660050248)

### Inferences
- Recommended variant for FemoraX (my synthesis): multilevel stabilized LF-LTS (Diaz–Grote recursion + Grote–Michel–Sauter Chebyshev damping, ratio 2, ν ≈ 0.05), formulated on the displacement three-term recurrence (equivalent to central difference with lumped mass); undamped first, αM via the LFCN2 diagonal treatment, then βK and Uniform damping and plasticity as frozen-coarse-force extensions verified numerically; constraint groups on one level; Kokkos::Graph per coarse step on GPU; level-weighted partitioning for multi-GPU. Keep mass scaling (LS-DYNA style) as a cheaper fallback for isolated tiny elements.

### Gaps
- No verified production use of LF-LTS in a structural/geotechnical FE code with plasticity, Rayleigh βK damping and MPCs was found.
