# Subcycling, multi-time-step and mixed implicit-explicit (IMEX) partitioning as remedies for a small stiff region limiting the explicit step

Scope: methods that let a small region of an explicit central-difference model (structure: beams, fibre sections, diaphragms, penalty pile-soil interfaces; dt ~ 1e-4 s) run at its own step, or implicitly, while the soil (millions to a billion hexahedra, dt 2e-4 to 2e-3 s) runs at a larger explicit step. Target: a GPU-native, matrix-free, FP64 Kokkos solver with lumped mass, Rayleigh damping and filter-based (Uniform) damping.

Note on sourcing: about 20 searches and fetches. Several primary papers are paywalled, so some properties come from abstracts or from later papers that review them (Karimi & Nakshatrala 2014, Gates/Matouš/Heath 2008). DOIs marked "(DOI from memory, not verified here)" were not confirmed by a fetched page in this session.

## 1. Subcycling and mixed-time-step methods: how they work, stability, accuracy, nonlinearity, damping

### Takeaway
There are two families.

1. **Nodal-partition subcycling** (Belytschko–Mullen, Neal–Belytschko, Smolinski, Daniel). It is cheap and purely explicit. The early versions had narrow bands of unstable time steps ("statistical stability"). Later energy-based variants are stable, but they lose accuracy at the interface, especially when the interface material is nonlinear.
2. **Dual-Schur / Lagrange-multiplier subdomain coupling** (Gravouil–Combescure "GC", Prakash–Hjelmstad "PH", Mahjoubi–Gravouil–Combescure, Karimi–Nakshatrala, Brun's HATI). It allows any Newmark integrator and any step ratio in each subdomain, with stability that can be proven by the energy method. Two costs come with it:
   - an interface problem (a condensation operator) that must be solved at every fine step;
   - energy dissipation at the interface (GC) or a loss of accuracy when step ratios are large.

A third line is the **energy-conserving local time stepping (LTS) of the leapfrog scheme** from the wave-propagation community (Diaz–Grote; Rietmann et al.'s Newmark-LTS in SPECFEM). It is the explicit-explicit method most proven on GPUs at scale.

### Cited findings

**Foundations and mixed (nodal-partition) methods**
- Belytschko and Mullen were the first to combine explicit and implicit integration with a **nodal** partition. Hughes and Liu introduced the mixed method with an **element** partition. — [Smolinski/Wu implicit multi-time-step paper (Comput. Mech.)](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s004660050365.pdf), via the [search result summary](https://link.springer.com/article/10.1007/s004660050248)
- Belytschko, Yen & Mullen, "Mixed methods for time integration", CMAME 17/18 (1979) 259–275 — [typeset record](https://typeset.io/papers/mixed-methods-for-time-integration-43gpci4gr7). The record was found, but its contents were not read. DOI 10.1016/0045-7825(79)90022-7 (DOI from memory, not verified here).
- Neal–Belytschko subcycling: Daniel showed it is only "statistically stable". It is unstable for certain time steps *below* the expected stability limit, in narrow bands. — [Gates, Matouš & Heath, IJNME 2008 (AMVI)](https://www3.nd.edu/~kmatous/Papers/IJNME_AMVI.pdf), as cited in the search snippets. The full text could not be fetched (the redirect returned 404).
- Smolinski's and Daniel's energy-conserving multi-time-step extensions of central difference avoid the statistical-stability problem. They can lose accuracy when the material at the time-step interface does not deform elastically, which is relevant to plastic soil next to a pile. — same source.
- Smolinski & Sleith built an explicit subcycling algorithm for second-order problems that was later proved stable, but it is less accurate than other algorithms. Smolinski also gave a constant-velocity subcycling variant that allows **non-integer step ratios and multiple nodal groups**, and studied its stability. — [search result summary of Smolinski papers](https://www.bohrium.com/paper-details/subcycling-integration-with-non-integer-time-steps-for-structural-dynamics-problems/813095649154170881-3896)
- Smolinski & Wu: an implicit multi-time-step method based on the trapezoidal rule, with a nodal partition (Comput. Mech.). — [Springer PDF](https://springerlink.bibliotecabuap.elogim.com/content/pdf/10.1007/s004660050365.pdf)

**The GC method and its successors (dual Schur / Lagrange multipliers)**
- **GC method** (Gravouil & Combescure). Subdomains use Newmark schemes and are coupled by Lagrange multipliers in dual Schur form.
  - It enforces **velocity continuity at the fine time steps**.
  - Interface velocities, and the Lagrange multiplier, are interpolated linearly within the coarse step.
  - It shows **excessive numerical damping** with subcycling.
  — [Karimi & Nakshatrala, J. Comput. Phys. 273 (2014) 671–705, arXiv:1305.6355](https://ar5iv.labs.arxiv.org/html/1305.6355); DOI [10.1016/j.jcp.2014.05.034](https://doi.org/10.1016/j.jcp.2014.05.034)
  - Original papers: Gravouil & Combescure, "Multi-time-step explicit–implicit method for non-linear structural dynamics", IJNME 50 (2001) 199–225, and Combescure & Gravouil, CMAME 191 (2002) 1129–1157 (citation details and DOIs from memory, not verified here).
- The GC integrator is unconditionally stable for synchronous stepping, but dissipates energy with asynchronous stepping. The cause is the interpolation of the interface kinematic constraint between subdomains. — [Notre Dame (Matouš group), Comput. Mech. 2016, asynchronous space-time](https://www3.nd.edu/~kmatous/Papers/CM_Asyn_ST_2016.pdf)
- **PH method** (Prakash & Hjelmstad, IJNME 61 (2004) 2183–2204):
  - It is FETI-based and enforces velocity continuity at the **coarse** step only.
  - All kinematic variables are interpolated linearly within the coarse step.
  - Interface forces are computed only at the largest step.
  - The authors claim it is unconditionally stable and energy preserving, where the original GC method is dissipative.
  — [Prakash & Hjelmstad, doi 10.1002/nme.1136](https://zendy.io/title/10.1002/nme.1136)
  - Karimi & Nakshatrala say PH is limited to two subdomains, gives more weight to the subdomain with the largest step, and dispute its claim of energy preservation. — [Karimi & Nakshatrala 2014](https://ar5iv.labs.arxiv.org/html/1305.6355)
- **Mahjoubi, Gravouil, Combescure (and Greffet):**
  - "Coupling subdomains with heterogeneous time integrators and incompatible time steps", Comput. Mech. 2009 — [MaRDI record](https://portal.mardi4nfdi.de/wiki/Publication:1034776).
  - "A monolithic energy conserving method to couple heterogeneous time integrators with incompatible time steps in structural dynamics", CMAME 200(9) (2011) 1069–1086: a dual-Schur monolithic method for subdomains with their own integrators and steps that can differ by large ratios. — [HAL hal-00560918](https://hal.archives-ouvertes.fr/hal-00560918)
  - Note: the brief names "Mahjoudi"; the published spelling is Mahjoubi.
- **Karimi & Nakshatrala (2014):**
  - Monolithic dual-Schur coupling with velocity continuity at **system** time levels, which gives differential-algebraic equations.
  - Allows any number of subdomains, each with its own Newmark scheme and step.
  - The energy method shows that velocities and accelerations stay bounded when each subdomain's step is below its own critical value.
  - The method generally does *not* preserve energy. "Subcycling need not always improve accuracy"; they give a criterion, computable during a run, for whether it helps.
  - Enforcing velocity continuity causes **drift** of the interface displacements and accelerations, which they bound.
  - The visible text did not state the order of accuracy with subcycling.
  — [arXiv:1305.6355](https://ar5iv.labs.arxiv.org/html/1305.6355)
- The thesis version notes that the coupling method does not conserve energy when there is subcycling. A first-order (transport) follow-up shows that subdomain stability alone does not guarantee stability of the coupled method. — [arXiv:1405.3230](https://ar5iv.arxiv.org/html/1405.3230), and the [search summary](https://core.ac.uk/works/19171027)
- Other members of the family named by Karimi & Nakshatrala: Pegon–Magonette (PM), a generalized-α extension (Bursi et al.), partitioned real-time methods, and Mahjoubi–Krenk state-space multi-time-step methods. — [Karimi & Nakshatrala 2014](https://ar5iv.labs.arxiv.org/html/1305.6355)
- Recent work:
  - Yuan et al. (IJNME 2024), a multi-time-step algorithm aimed at energy conservation and stability — [PDF](https://userweb.eng.gla.ac.uk/sondipon.adhikari/fulltext/journal/ft401.pdf)
  - Pulicani, Brun, Allain & Gravouil (COUPLED 2025), an energy-conserving GC-based multi-time-step fluid-structure coupling without spurious interface energy — [abstract](https://coupled2025.cimne.com/event/contribution/4cada59a-a8b0-11ef-80a4-000c29ddfc0c)

**Asynchronous variational integrators (AVI)**
- Each element has its own time step, explicit and fully decoupled. The integrators nearly conserve energy, locally and globally, and gave substantial speedups at equal accuracy against explicit Newmark. — [Lew, Caltech thesis 2003](https://thesis.caltech.edu/2077/); [Lew, Marsden, Ortiz & West, "Variational time integrators", IJNME 2004](https://typeset.io/pdf/variational-time-integrators-4yo4bvl311.pdf). AVI original paper: Arch. Ration. Mech. Anal. 167 (2003) 85–146 (citation from memory).
- Extended to multi-domain problems by Gates/Matouš/Heath, and to hyperelastic solids by Beneš/Matouš. Kale & Lew gave a parallel implementation. — [search summary](https://www3.nd.edu/~kmatous/Papers/CMAME_AMVI_PDE.pdf)
- Asynchronous variational contact mechanics — [arXiv:1007.3240](https://arxiv.org/pdf/1007.3240)

**Leapfrog / Newmark local time stepping (LTS)**
- Diaz & Grote (SIAM J. Sci. Comput. 31 (2009), [doi 10.1137/070709414](https://zendy.io/pdf-viewer/10.1137/070709414)): explicit LTS of arbitrarily high order, derived from leapfrog. With a symmetric FE discretization and an essentially diagonal mass, the scheme is explicit, inherently parallel and **exactly conserves a discrete energy**. A multilevel version followed. — [arXiv:1205.0654 review](https://arxiv.org/pdf/1205.0654); [Grote, Michel & Sauter](https://cdnjs.deepai.org/publication/stabilized-leapfrog-based-local-time-stepping-method-for-the-wave-equation)
- The original leapfrog LTS can be **linearly unstable at certain discrete values of dt**. Grote, Michel & Sauter (Math. Comp. 2021, [doi 10.1090/mcom/3650](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_crossref_primary_10_1090_mcom_3650)) proposed a stabilized version that keeps explicitness, second order, the three-term recurrence and energy conservation. — [arXiv:2005.13350](https://arxiv.org/abs/2005.13350)
- In the **damped** case energy is not conserved. Grote & Mitkova use Adams–Bashforth-based LTS for it, explicit and of arbitrarily high order. — [search summary of LTS literature](https://arxiv.org/pdf/1205.0654)
- Convergence analysis of the energy-conserving leapfrog LTS — [arXiv:1703.07965](https://arxiv.org/pdf/1703.07965)

### Inferences
- For a central-difference code with lumped mass, the **nodal-partition / LTS** route (Diaz–Grote / Rietmann) fits the existing kernel structure best. It needs no interface solve, keeps matrix-free element forces, and has known energy and stability theory, at least undamped and for linear elasticity.
- **GC-type dual-Schur coupling** needs a condensed interface operator. That is cheap for a small structure, but it is a dense or sparse solve on the host or device at every fine step.
- None of the cited explicit LTS theory covers Rayleigh stiffness damping, ExplicitDifference's lagged half-step velocity, or recursive-filter (Uniform) damping. The stability of those combinations must be derived and tested in-house. This matches FemoraX's own design note.
- Plasticity at the interface degrades the accuracy of energy-conserving subcycling (Smolinski/Daniel). Placing the level boundary inside elastic material, rather than in yielding soil around a pile, should reduce this. This is a design inference, not a published result.

### Gaps
- Exact order-of-accuracy statements for GC versus PH with subcycling: the full texts were not obtained. Commonly reported is that GC is first order at the interface, with dissipation proportional to the step ratio. This could not be verified here.
- Daniel's specific nonlinear subcycling papers (IJNME, 1997–2003) were not retrieved, so their exact titles and DOIs are unverified.
- No source found that treats **Rayleigh β-stiffness damping** or **frequency-independent filter damping** inside an LTS or subcycling scheme.

## 2. LS-DYNA and Abaqus/Explicit subcycling: what they do and their documented limitations

### Takeaway
Both commercial explicit codes offer element-group subcycling in power-of-two or integer ratios. Both document modest or case-dependent gains, and Abaqus documents many restrictions: constraints, rigid bodies, connectors and general contact merge zones; beams with exact rotary inertia cannot share interface nodes. LS-DYNA's subcycling has existed for decades but has been little used, which LS-DYNA developers attribute to robustness concerns and limited practical need. Mass scaling remains the production default for small elements.

### Cited findings
**LS-DYNA**
- LS-DYNA has had a subcycling algorithm for many years, and it "has not been used extensively for reasons pertaining to its robustness but also due to lack of its practical needs and potential". — [DYNAmore 2014 LS-DYNA Forum, "Current status of subcycling and multiscale simulations in LS-DYNA"](https://dynamore.de/de/download/papers/2014-ls-dyna-forum/documents/entwicklervortraege/current-status-of-subcycling-and-multiscale-simulations-in-ls-dyna); [13th Int. LS-DYNA Conference version](https://dynalook.com/conferences/13th-international-ls-dyna-conference/simulation/current-status-of-subcycling-and-multiscale-simulations-in-ls-dyna-r) (from search snippets; the page could not be fetched because of a certificate error)
- Rule: an element's step must not exceed the steps of the nodes in its connectivity. Steps are grouped in powers of two. *CONTROL_SUBCYCLE takes M = 1 to 64 (powers of two, default 1), the largest step for external forces. — same sources (snippets)
- Reported savings: about 45 % CPU on a RAV4 car model, with results unchanged by the subcycling option; about 9 % on an S-rail. — same 2014 presentation (snippets)
- *CONTROL_SUBCYCLE_MASS_SCALED_PART_{SET} was added in about 2014 to combine subcycling with mass scaling. — same
- Application to press-hardened steel parts — [11th European LS-DYNA conference](https://dynalook.com/conferences/11th-european-ls-dyna-conference/crash-metal-failure/applications-of-multiscale-and-subcycling-methods-for-press-hardened-steel-parts-failure-assessment)
- Ansys LS-DYNA webinar "Mass Scaling and Subcycling" (2022): the PDF was downloaded, but its text could not be extracted (no PDF tools available). — [webinar PDF](https://lsdyna.ansys.com/wp-content/uploads/2025/11/221215_Webinar_Mass-Scaling_and_Subcycling.pdf)
- Mass scaling, the usual alternative:
  - DT2MS < 0 adds mass to elements whose step is below |DT2MS|.
  - Selective mass scaling scales only the high frequencies, leaving rigid-body and low-order motion unaffected.
  - Load-curve-based mass scaling exists for inertia-dominated early phases.
  - Mass scaling is "memory and CPU intensive" (selective).
  — [Ansys LS-DYNA, "What is mass scaling?"](https://lsdyna.ansys.com/?p=13413); [d3view blog](https://blog.d3view.com/?p=2064)

**Abaqus/Explicit selective subcycling** (`*SUBCYCLING, ELSET=`; documented in the 2016 and 2017 guides and current through 2025) — [Abaqus 2025 Analysis Guide](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-selectivesubcycling.htm); [2017 mirror](https://abaqus-docs.mit.edu/2017/English/SIMACAEANLRefMap/simaanl-c-selectivesubcycling.htm); [2016 mirror](https://ceae-server.colorado.edu/v2016/books/usb/pt04ch11s07aus76.html)
- Zones are element sets, fixed for the analysis, and are implemented on **domain-level parallelization**: each zone is split into parallel domains.
- The non-subcycling domains advance at the largest stable increment. Subcycle sizes are integer divisors of it, so all domains meet at common time points.
- **Interface treatment:** interface nodes take their velocity from the non-subcycling zone, held constant during the subcycles, so the interface displacement varies linearly across the subcycles.
- **Zone merging:** zones are merged when they overlap, share nodes, have adjacent nodes in different zones, share a constraint equation, connector or rigid body, or when **general contact** is used. A merged zone takes the smallest increment.
- **Restrictions:**
  - Beams with exact rotary inertia cannot share nodes between zones.
  - Analytical rigid surfaces cannot be in general contact when zones are defined.
  - Some element types are excluded: Eulerian EC3D8R/EC3D8RT, C3D10/C3D10T, PD3D, PC3D.
  - Every parallel domain needs at least one deformable element.
- **Performance:** the gain requires many subcycles and *small* zones. "If a majority of elements in the model are in subcycling zones, there will not be much performance benefit." No speedup figures are given.
- **Accuracy:** truncation error from the large increment and the interface interpolation "may slightly alter results". Do not split a contact surface across zones.
- Output and mass scaling occur only at time points common to all domains. Damping is not discussed.

### Inferences
- Abaqus' interface rule (frozen coarse-side velocity, linear displacement) is a simple constant-velocity subcycling of the Smolinski type, not an energy-conserving one. Vendors accept a small truncation error.
- For SSI, the Abaqus restrictions matter directly:
  - Rigid diaphragms and rigid links would force the merging of zones that share them, so a whole floor ends up in one zone.
  - Penalty pile-soil interfaces behave like constraints or contact that straddle the zone boundary.
  - Beams with rotary inertia cannot sit on an interface.
  FemoraX would need to place the boundary so that every constraint group lies inside one level.

### Gaps
- No authoritative current LS-DYNA keyword-manual list of what subcycling does not support was obtained. The 2022 webinar content could not be extracted.
- No published speedups for subcycling in SSI or geotechnical models in commercial codes were found.

## 3. Mixed implicit-explicit partitions (Hughes & Liu) and implicit-explicit coupling for SSI, including co-simulation

### Takeaway
The Hughes–Liu element partition (1978) makes the implicit group unconditionally stable while the explicit elements' critical step governs the system. That suits an "implicit structure, explicit soil" split only if the soil step is not itself the limit. The modern SSI practice is **hybrid asynchronous** dual-Schur co-simulation (Brun and co-workers): an explicit spectral-element soil (EFISPEC3D) or explicit FE, coupled to an implicit nonlinear structure (Code_Aster), through an external interface solver. It is reported with step ratios up to about 1000 to 5000 and wall-time reductions of 2x or about 60 %.

### Cited findings
- Hughes & Liu, "Implicit-Explicit Finite Elements in Transient Analysis: Stability Theory", J. Appl. Mech. 45(2) (1978) 371–374, [doi 10.1115/1.3424304](https://next.sciencegate.app/document/10.1115/1.3424304). Unconditional stability can be achieved for the implicit elements, and the critical step of the explicit elements governs the system. The companion paper "Implementation and Numerical Examples" is pp. 375–378, [doi 10.1115/1.3424305](https://next.sciencegate.app/document/10.1115/1.3424305).
- Later: "Implicit-explicit finite elements in nonlinear transient analysis" and "Mixed-time implicit-explicit finite elements for transient analysis" (Belytschko, Liu and co-authors, early 1980s) — [record](https://www.bohrium.com/paper-details/implicit-explicit-finite-elements-in-nonlinear-transient-analysis/812415362602631169-3776); [record](https://www.bohrium.com/paper-details/mixed-time-implicit-explicit-finite-elements-for-transient-analysis/812415162152648704-3896). Only the titles were confirmed.
- **GC as explicit/implicit:** the GC method couples subdomains with different Newmark integrators through velocity continuity at the interface, and has been extended to explicit/implicit coupling. — [Yuan et al. IJNME](https://userweb.eng.gla.ac.uk/sondipon.adhikari/fulltext/journal/ft401.pdf)
- Chantrait, Rannou & Gravouil coupled the implicit code Z-set with the explicit code Europlexus, with step ratios up to **5000** for low-energy impact. — [Chantrait et al. (core.ac.uk)](https://core.ac.uk/download/334542684.pdf) (from the search snippet; the full text returned 403)
- **Earthquake engineering, reinforced concrete (Brun, Batti, Limam, Combescure):**
  - "Implicit/explicit multi-time step co-computations for predicting reinforced concrete structure response under earthquake loading", Soil Dyn. Earthq. Eng. 33 (2012) 19–37, applied to the full-scale SPEAR frame: about **2x** less computation time. — [HAL hal-00938628](https://hal.archives-ouvertes.fr/hal-00938628); [Structurae](https://structurae.net/de/fachliteratur/fachartikel/implicitexplicit-multi-time-step-co-computations-for-predicting-reinforced-concrete-structure-response-under-earthquake-loading)
  - Blast analysis of a reinforced concrete frame (Brun, Batti, Limam, Gravouil, 2012, pp. 41–59; journal not confirmed): step ratios up to **1000** against fully explicit runs, accurate global and local responses, **about 60 %** less computation time. — [record](https://manuscript.isc.ac/Inventory/10/1458313.htm)
  - Multi-time-step implicit methods for nonlinear reinforced concrete (14th WCEE, 2008). — [WCEE paper](https://iitk.ac.in/nicee/wcee/article/14_14-0080.PDF)
- **SSI co-simulation of spectral and finite elements:**
  - Zuchowski, Brun & De Martin, "Co-simulation coupling spectral/finite elements for 3D soil/structure interaction problems", Comptes Rendus Mécanique (2018). The implicit FE code (Code_Aster) is coupled to the explicit SEM code (EFISPEC3D) by domain decomposition in transient dynamics, with an external coupling program handling the interface problem; it uses the "Hybrid Asynchronous Time Integrator". — [EM-consulte](https://www.em-consulte.com/it/article/1208666/co-simulation-coupling-spectralfinite-elements-for)
  - Brun, De Martin & Richart, "Hybrid asynchronous SEM/FEM co-simulation for seismic nonlinear analysis of concrete gravity dams", Computers & Structures (2021): the SE code handles the far field and the FE code the near field; mortar coupling in space; hybrid explicit/implicit asynchronous time integration. — [record](https://vufind-test.katalog.k.utb.cz/SummonRecord/FETCH-LOGICAL-c398t-75cc46c4d392c1f868a57ee4eb34e0ce4362b1a26170660f32f5850fa50f84903); [EPFL Infoscience](https://infoscience.epfl.ch/record/284404)
  - HATI (Heterogeneous Asynchronous Time Integrator), M2P 2023 (Brun, Chau, De Martin, Richart): non-conforming interface meshes, different schemes and step sizes per subdomain. — [M2P 2023 abstract](https://m2p2023.cimne.com/event/contribution/3b5c7668-9e65-11ed-9820-000c29ddfc0c)

### Inferences
- **Cost structure of an implicit structure inside explicit soil** (GC/HATI): every coarse (soil) step needs
  1. a solve of the structure's implicit system, factorizable once if the structure is linear, or Newton if it is nonlinear;
  2. an interface condensation operator H = sum_i C_i K_eff,i^-1 C_i^T, of size equal to the number of interface DOFs, solved for the Lagrange multipliers;
  3. the free and linked solution corrections.
  For a structure that is a small fraction of the model, the implicit part is small, but in the FemoraX setting it would be a direct or iterative solve of beam/fibre elements on the device or host. That conflicts with FemoraX's matrix-free, no-direct-solver path, unless the structural subdomain is small enough for a dense or banded factorization.
- For FemoraX, the structure's small step (1e-4 s) is only 2x to 20x below the soil's. Implicit integration of the structure would allow a structure step equal to the soil step, but force-based fibre beams already iterate internally, and an implicit Newton on them at every soil step may cost more than 2x to 20x explicit substeps. Explicit LTS (subcycling) is likely the cheaper and simpler route here. Implicit-explicit coupling pays off when the ratio is in the hundreds or thousands (as in the reported blast and impact cases).
- **Penalty interfaces** (embedded pile-soil) are stiff. With an implicit structure they would cross the partition boundary. In GC form they would need either to be on the implicit side, including the adjacent soil element, or to be replaced by Lagrange multipliers.

### Gaps
- No quantitative comparison was found of implicit-structure/explicit-soil co-simulation against fully explicit runs for building–soil SSI with fibre beams. Only reinforced-concrete frame (SPEAR) and dam cases were found.
- OpenSees-based partitioned SSI co-simulations (for example OpenSees with SPECFEM, or OpenFresco-style coupling) were not researched in this pass.

## 4. Reported speedups, implementation complexity and suitability for GPUs and matrix-free codes

### Takeaway
Reported gains are modest in commercial crash codes (about 9–45 %) and large in specialized research codes: 2x to 60 % for implicit/explicit reinforced-concrete co-computation, and step ratios up to 1000 to 5000 in GC-based codes. The only demonstrated large-scale **GPU** multi-rate explicit method is Newmark local time stepping in SPECFEM3D (Rietmann, Grote, Peter, Schenk 2017). It handles element-size contrasts above 100x on CPU and GPU clusters, and needed a dedicated load-balancing partitioner.

### Cited findings
- Rietmann, Grote, Peter & Schenk, "Newmark local time stepping on high-performance computing architectures", J. Comput. Phys. 334 (2017) 308–326, doi 10.1016/j.jcp.2016.11.012. A multilevel LTS-Newmark scheme in a continuous (spectral) FE seismic code, handling element-size contrasts above 100x, run on large CPU and GPU clusters. — [dblp](https://dblp.org/rec/journals/jcphy/RietmannGPS17); [unibas edoc](https://edoc.unibas.ch/41217)
- Companion paper "Load-Balanced Local Time Stepping for Large-Scale Wave Propagation" (IPDPS 2015): partitioning for the same LTS on CPU and GPU clusters. — [record](https://sindex.sdl.edu.sa/esploro/outputs/conferenceProceeding/Load-Balanced-Local-Time-Stepping-for-Large-Scale/9943772408331)
- Rietmann's USI thesis (2015): GPU, LTS and scaling optimizations together accelerate existing applications by 30x or more. This is a combined figure, not LTS alone. — [USI thesis](https://susi.usi.ch/usi/documents/318539)
- CSCS (2017) news on high-resolution seismic wave propagation in space and time with LTS — [CSCS](https://www.cscs.ch/science/earth-env-science/2017/high-resolution-propagation-of-seismic-waves-in-space-and-time). LTS was also used for high-resolution dynamic-rupture simulations — [AGU 2016 abstract](https://ui.adsabs.harvard.edu/abs/2016AGUFM.T23C2951P)
- LS-DYNA subcycling: about 45 % (RAV4) and about 9 % (S-rail) savings — [DYNAmore 2014](https://dynamore.de/de/download/papers/2014-ls-dyna-forum/documents/entwicklervortraege/current-status-of-subcycling-and-multiscale-simulations-in-ls-dyna)
- Abaqus: a benefit only when zones are small and the subcycles many — [Abaqus guide](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-selectivesubcycling.htm)
- GC-type explicit/implicit: about 60 % time saved (blast, ratio up to 1000); about 2x (SPEAR earthquake); ratio up to 5000 (Z-set and Europlexus) — sources in section 3
- AVI: "substantial speedups at equal accuracy" versus explicit Newmark (no number in the snippet) — [Lew thesis](https://thesis.caltech.edu/2077/)

### Inferences
- **Ideal speedup bound for FemoraX.** Let f be the fraction of elements in the fine level and r = dt_soil / dt_struct. Relative to a single step at dt_struct, the cost ratio is about 1 / (1 - f + f·r)·r, that is, speedup ≈ r / (1 + f(r-1)).
  - With f = 1e-3 (structure) and r = 2–20, the speedup is about r, so 2x to 20x before overheads.
  - Overheads: on GPUs, each substep of a tiny fine level is a kernel launch (about 5–10 µs) plus synchronization. For small fine levels, launch latency dominates, so Kokkos Graphs or fused level launches are needed (the same concern as FemoraX's own LTS design note).
  - For billion-element runs on many GPUs, the fine level is concentrated on a few ranks. Without level-aware partitioning, those ranks become stragglers. Rietmann's IPDPS 2015 load-balanced partitioning addresses exactly this.
- **Matrix-free compatibility.** Explicit LTS and nodal subcycling need only per-level force evaluations and per-level equation lists, and fit a batch-per-(element, material, level) design. GC/dual-Schur methods need an interface operator, which is only small when the interface is small. Implicit-explicit partitions need a linear solver on the implicit part, which conflicts with the matrix-free, no-direct-solver design unless the implicit part is small and handled by a dedicated (dense or banded) solver.
- **Determinism.** LTS keeps fixed operation orders per level, so FemoraX's bit-identity rules can carry over.
- **Implementation complexity, ranked:** constant-velocity subcycling (Abaqus-like) < leapfrog/Newmark LTS (Diaz–Grote, Rietmann) < GC/HATI dual-Schur with an interface solve < full IMEX with Newton on fibre beams and coupling.

### Gaps
- No published LTS-only speedups on GPUs with a structural (beam) fine level, nor any with nonlinear soil plasticity.
- No published treatment of recursive-filter (frequency-independent) damping under LTS.

## 5. Which methods are used in production today, and their practical limitations

### Takeaway
In commercial explicit structural codes, **mass scaling** (including selective mass scaling) is the everyday production remedy. Subcycling exists in LS-DYNA (*CONTROL_SUBCYCLE) and Abaqus/Explicit (selective subcycling) but is niche, with merging and contact restrictions and case-dependent gains. In seismology, **LTS is in production** in SPECFEM3D (CPU and GPU). In SSI research, GC/HATI hybrid asynchronous co-simulation (Code_Aster with EFISPEC3D, Z-set with Europlexus) is used, but not as a commercial product.

### Cited findings
- LS-DYNA: subcycling is little used (robustness and limited need); mass scaling is controlled by DT2MS, with selective and load-curve variants. — [DYNAmore 2014](https://dynamore.de/de/download/papers/2014-ls-dyna-forum/documents/entwicklervortraege/current-status-of-subcycling-and-multiscale-simulations-in-ls-dyna); [Ansys LS-DYNA mass scaling](https://lsdyna.ansys.com/?p=13413)
- Abaqus/Explicit selective subcycling restrictions (merging on constraints, rigid bodies and general contact; beams with rotary inertia; analytical rigid surfaces) — [Abaqus guide](https://docs.software.vt.edu/abaqusv2025/English/SIMACAEANLRefMap/simaanl-c-selectivesubcycling.htm)
- SPECFEM3D LTS (Newmark-LTS, CPU and GPU) — [Rietmann et al. 2017](https://dblp.org/rec/journals/jcphy/RietmannGPS17)
- Code_Aster with EFISPEC3D (HATI) for SSI and dams — [Zuchowski et al. 2018](https://www.em-consulte.com/it/article/1208666/co-simulation-coupling-spectralfinite-elements-for); [Brun et al. 2021](https://infoscience.epfl.ch/record/284404)

### Inferences
- **Practical limitations that bear on FemoraX's SSI models:**
  1. Constraint groups (rigid diaphragm, rigid link, equalDOF, embedded penalty) must lie inside one level, or the level boundary must be handled by coupling.
  2. Beams with rotary inertia and penalty interfaces at the level boundary are documented trouble spots (Abaqus).
  3. Nonlinear (plastic) material at the interface reduces accuracy (Smolinski/Daniel).
  4. Damping beyond mass-proportional damping is untreated in the LTS literature.
  5. Load balance across GPUs is needed.
  6. Small fine levels suffer from kernel-launch overhead.
- **Pragmatic recommendation**, an inference consistent with FemoraX's existing LTS plan:
  1. Implement 2-level (power-of-2) leapfrog LTS (Diaz–Grote style, with the Grote–Michel–Sauter stabilization) for an all-explicit run.
  2. Put the whole structure, its constraint groups and the embedded-interface elements in the fine level.
  3. Keep the soil coarse.
  4. Verify the damping terms separately.
  5. Keep implicit-explicit (GC/HATI) as a later option, only if a structural component needs dt far smaller than the soil's (step ratios of 100 or more), where an implicit solve of a small structure pays off.

### Gaps
- Could not verify whether Abaqus' or LS-DYNA's subcycling preserves energy or stability with stiffness-proportional damping; the documentation reviewed does not mention damping.
- Whether any production SSI or geotechnical code (PLAXIS, FLAC3D, OpenSees, Real-ESSI) offers subcycling: not researched in this pass. FLAC's density scaling is analogous to mass scaling.
