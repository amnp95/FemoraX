# Interface and contact modeling for GPU-native, matrix-free, explicit dynamic SSI (FemoraX)

Scope note: about 20 searches and fetches. The Abaqus and MDPI pages returned HTTP 403, so the
statements from them come from search snippets only. Statements marked "(OpenSees source,
955ceb0)" come from reading the local OpenSees clone at commit 955ceb0a5, not from a web page.
Statements marked [UNVERIFIED] are background knowledge that I could not check against a source
in this session. They are listed so the writer knows to leave them out or check them.

## 1. Interface representations (zero-thickness, thin-layer, node-to-node / node-to-segment, mortar, embedded springs, continuum)

### Takeaway
Production geotechnical codes use two families for piles: (a) a meshed volume pile with
zero-thickness or node-to-face interfaces that carry Coulomb friction and gap (FLAC3D, PLAXIS
volume piles, Abaqus), and (b) embedded beams joined to a soil mesh that does not conform to them,
through springs or penalty ties (PLAXIS embedded beam, OpenSees ASDEmbeddedNodeElement, the
"embedded interface" workflow of the FemoraX authors). Embedded beams work well for axial and
serviceability pile-group behaviour. They do not fully capture lateral interface behaviour and
they show traction oscillations. Volume piles with zero-thickness interfaces are the more exact
option.

### Cited Findings
- Commercial packages give two options for piles: volume elements with interface elements
  between pile and soil, or the embedded beam approach — [TU Delft thesis](https://repository.tudelft.nl/file/File_a6c17457-9f56-4480-b8f8-1082eca22926?preview=1)
- A proposed embedded beam model "is unable to capture lateral interface behaviour equivalent
  to a full 3D model with interfaces" — [TU Delft thesis](https://repository.tudelft.nl/file/File_cfa50983-29ec-4623-b5fc-d98cb90391ad) (search snippet; which of the TU Delft files says it is uncertain)
- Validation of embedded beam formulations (2020/2021): at working loads the skin traction
  profiles of both embedded beam formulations fit solid FE results qualitatively well, but skin
  traction oscillations appear even with linear elastic soil. Newer formulations with an explicit
  interaction surface (Turello et al.) aim to reduce mesh dependence — [Practice-Oriented Validation of Embedded Beam Formulations, Processes 9(10):1739](https://mdpi.com/2227-9717/9/10/1739) (from the search summary; the page was not opened)
- In PLAXIS the embedded beam is not meshed with solids. Springs with axial, lateral and base
  resistance join it to the soil, and skin resistance is input per unit length — [PLAXIS forum](https://communities.bentley.com/products/geotech-analysis/f/plaxis-soilvision-forum/213190/embedded-beam-row-skin-resistance-input-value-clarification); [Bentley blog](https://blog.bentley.com/?p=46763) (embedded beams give "particularly good results for serviceability states of pile groups under axial loading")
- PLAXIS zero-thickness interfaces: the key parameter is the strength reduction factor R_inter.
  The interface's "virtual thickness" (a zone with reduced soil properties) changes the
  displacements and should not be too large — [Novi Sad conference paper 2016](https://www.gf.uns.ac.rs/~zbornik/doc/NS2016.035.pdf)
- Goodman, Taylor & Brekke (1968) joint element, the origin of zero-thickness interfaces. Desai's
  thin-layer element is a thin solid element. Its shear stiffness comes from laboratory tests, its
  normal stiffness from the layer plus the adjoining solids, and a thickness-to-adjacent-element
  ratio of 0.01-0.1 is recommended — [Sharma & Desai, J. Eng. Mech. 118(12), 1992](https://ascelibrary.com/doi/10.1061/%28ASCE%290733-9399%281992%29118%3A12%282442%29); [Desai et al., IJNAG 8(1) 1984](https://zendy.io/title/10.1002/nag.1610080103); [Coutinho et al., "Simple zero thickness kinematically consistent interface elements"](https://www.academia.edu/128247912/Simple_zero_thickness_kinematically_consistent_interface_elements)
- Goodman's element is kinematically inconsistent (pointed out by Kaliakin & Li). Conventional
  thin solid elements, including the thin-layer model, have such inconsistencies too — [Coutinho et al.](https://www.academia.edu/128247912/Simple_zero_thickness_kinematically_consistent_interface_elements)
- FLAC3D interfaces are one-sided, node-to-face contacts. Two triangular interface elements are
  defined per quadrilateral zone face. Each interface node has a representative area. Every step
  computes the normal penetration and the relative shear velocity at each node. Normal force =
  kn × penetration × area. The shear force is incremental (ks), capped by cohesion plus friction on
  the effective normal force. Tensile and shear bond strengths are optional. Dilation is applied
  during slip. Pore pressure is used in the effective stress by default. Contacts are detected and
  forces transferred only at interface nodes — [FLAC3D Theory: Interfaces, formulation](https://docs.itascacg.com/pfc700/flac3d/docproject/source/theory/interfaces/formulation.html)
- FLAC3D stiffness choice: kn = ks ≈ 10 × the equivalent stiffness of the stiffest neighbouring
  zone, max[(K + 4G/3)/Δz_min]. Use "the lowest stiffness consistent with small interface
  deformation", because high values slow the solution. If the physical stiffness is below 10× the
  zone stiffness, the physical value can be used — [FLAC3D: choice of properties](https://docs.itascacg.com/pfc700/flac3d/docproject/source/theory/interfaces/choiceofproperties.html). (The worked-example numbers in the search snippet are internally inconsistent: "15 GPa" against 150e9/0.5. Check the page before quoting them.)
- Real-ESSI (Jeremić group) has several interface/contact element types. A nuclear power plant
  SSSI study uses a zero-length, stress-based, dry contact element with nonlinear hardening shear
  (StressBasedSoftContact_NonLinHardShear), whose normal force "changes gradually upon contact and
  becomes zero upon detachment", plus viscous parameters for the energy lost when the gap opens
  and closes — [arXiv:2310.10251](https://arxiv.org/pdf/2310.10251); [Real-ESSI Simulator](https://sokocalo.engr.ucdavis.edu/~jeremic/Real_ESSI_Simulator); [Jeremić et al. lecture notes (chapter on contact elements, dynamic normal contact verification)](https://sokocalo.engr.ucdavis.edu/~jeremic/LectureNotes/Jeremic_et_al_CompMech_LectureNotes.pdf)
- Opening and closing of the soil-foundation interface can amplify the response of reactor-building
  components above 10 Hz (SMiRT 27, 2024), via the same search — [arXiv:2310.10251 / related](https://arxiv.org/pdf/2310.10251) (attribution from the search summary)
- OpenSees options that Femora exposes: `zeroLengthContactASDimplex` (2-node penalty, Mohr-Coulomb,
  arbitrary normal, different DOF counts per node: solid to beam/shell/u-p) — [OpenSees manual](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/elements/zeroLengthContactASDimplex.html). Also `zeroLengthContact2D/3D` (node-to-node, non-symmetric tangent) and `zeroLengthInterface2D` (node-to-segment) — [OpenSees docs](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/elements/zeroLength.html) (from the search summary)
- The FemoraX authors' own pile workflow (Pakzad, Arduino, Zhang, Taciroglu, PanAm 2024,
  arXiv:2509.05675) uses "Embedded interface elements" to connect structures and soil, with DRM
  and PML — [arXiv:2509.05675](https://arxiv.org/abs/2509.05675)
- ASDEmbeddedNodeElement is a penalty tie of a beam node to its host soil element (3 or 4 host
  nodes, optional `-rot` and `-p`). The default penalty is `K = 1.0e18`, scaled as `iK = K * sqrt(V)`
  with V the host element's measure; K_e = iK BᵀB (OpenSees source, 955ceb0:
  `SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp` lines 222, 849-852). Femora's
  `EmbeddedNodeInterface` uses `K=1e10` by default and finds host elements with a KD-tree
  (Femora `src/femora/components/interface/embedded_node_interface.py`).

### Inferences
- For FemoraX's targets (piles, pile groups, mats), a sensible ladder is: (1) tied embedded beams
  (ASDEmbeddedNode style), which are linear, constant and parallelize like elements; (2) embedded
  beams with a nonlinear shaft and tip law (t-z / q-z / p-y-like springs between the beam node and
  the interpolated soil point). This is the PLAXIS embedded beam idea, and it keeps a non-conforming
  soil mesh; (3) volume piles or foundations with zero-thickness node-to-face interfaces (FLAC3D
  style: kn, ks, Coulomb, tension cutoff, dilation) on conforming meshes for detailed studies. All
  three are "element-like" (fixed connectivity, fixed-size state), which fits the existing batch,
  gather and coloring machinery. General node-to-segment or mortar contact with search is needed
  only for large sliding or re-contact over distances comparable to the element size, which is
  rare for piles and foundations in earthquake SSI (small-sliding assumption).
- Zero-length elements between coincident nodes (zeroLengthContactASDimplex) need doubled nodes
  and a conforming mesh. FLAC3D's node-to-face pairing gives the same physics with tolerance for
  non-matching meshes. A node-to-face "small sliding" pairing fixed at the start of the stage keeps
  the connectivity static on the GPU.
- Treat p-y/t-z springs as a calibrated empirical alternative only for beam-on-springs models.
  With a continuum soil they double-count soil compliance.

### Gaps
- I did not find a quantitative head-to-head dynamic benchmark (embedded beam vs volume pile with
  interfaces) for seismic loading.
- Mortar and segment-to-segment methods (Wriggers, *Computational Contact Mechanics*; Laursen,
  *Computational Contact and Impact Mechanics*; Puso & Laursen 2004 mortar) were not consulted in
  this session [UNVERIFIED]. Their known strengths (patch test passing, smooth pressure on
  non-matching meshes) are background knowledge only.
- The Real-ESSI contact element formulations (exact normal and friction laws) were not read in full.

## 2. Explicit contact: penalty vs kinematic, stable step, friction, ZeroLengthContactASDimplex

### Takeaway
Penalty contact adds stiffness and lowers the stable step unless the penalty is tied to the nodal
mass and the step (LS-DYNA SOFT=1: k ∝ m/Δt²). Kinematic predictor-corrector contact (Abaqus/Explicit)
does not change the stable step. Bi-penalty (stiffness + mass penalty) can keep the contact-free
step. OpenSees' ZeroLengthContactASDimplex is a sound penalty Mohr-Coulomb law (IMPL-EX for implicit
robustness), but it has zero mass and a default implicit tangent bug, and its user-given Kn must be
bounded for explicit runs.

### Cited Findings
- LS-DYNA SOFT=1 ("soft constraint"): k = max[SLSFAC·SFS·k0, SOFSCL·k1]. k0 comes from the material and
  the element size (SOFT=0, the default). k1 comes from the nodal masses and the time step and is
  ∝ 1/Δt²: halving the step quadruples k1. It is recommended for soft-on-stiff contact and dissimilar
  mesh densities, and gives 1-2 orders of magnitude higher stiffness for foam on metal. SOFT=1 checks
  the SOFT=0 stiffness frequency against the stable step and warns. A run can still go unstable when
  the SOFT=0 value is the larger one. DTSTIF overrides the step used in k1 — [LS-DYNA KB: SOFT option](https://lsdyna.ansys.com/?p=13360); [Contact stiffness calculation](https://lsdyna.ansys.com/?p=13140); [Contact stiffness influenced by time step](https://lsdyna.ansys.com/?p=13341); [LSTC FAQ contact.soft1](https://ftp.lstc.com/anonymous/outgoing/jday/faq/contact.soft1)
- Smaller time steps also shorten the bucket-sort (contact search) interval and the incremental
  penetrations — [LS-DYNA KB](https://lsdyna.ansys.com/?p=13341)
- Abaqus/Explicit kinematic contact is a predictor/corrector: the model is advanced without contact,
  then accelerations are corrected where overclosure occurs, so it "has no influence on the stable
  time increment". Penalty enforcement adds stiffness. Abaqus "automatically accounts for the effect
  of the penalty stiffnesses in the automatic time incrementation, although this effect is usually
  small". The default penalty is a scaled representative stiffness of the underlying elements.
  Softened kinematic contact allows penetration to satisfy a pressure-overclosure law. Contact pairs
  default to kinematic, general contact always uses penalty — [Abaqus 2024 docs: Contact constraint enforcement methods in Abaqus/Explicit](https://docs.software.vt.edu/abaqusv2024/English/SIMACAEITNRefMap/simaitn-c-expcontactconstraints.htm) (403 on fetch; from search snippets); [Abaqus 2017 mirror](https://abaqus-docs.mit.edu/2017/English/SIMACAEITNRefMap/simaitn-c-expcontactconstraints.htm)
- Bi-penalty: a stiffness penalty alone lowers the critical step. A mass penalty alone does not, but
  is less accurate. Choosing the ratio of stiffness to mass penalty keeps the maximum eigenfrequency
  (and the critical step) unchanged, with accuracy comparable to the stiffness penalty — [Hetherington thesis, Sheffield 2014](https://etheses.whiterose.ac.uk/5303/1/JEH-Thesis_online-20140228.pdf); [Kopačka et al., COMPDYN 2013](https://webarchiv.lib.cas.cz:443/wayback/20150411041150/http://www.it.cas.cz/files/u2130/2013_Kopacka_COMPDYN2013.pdf); [Kolman et al., Engineering Mechanics 2019](https://engmechx.it.cas.cz/improc/2019/041_Kolman_082.pdf)
- In explicit cohesive-zone + penalty contact simulations, a large penalty "reduces the critical time
  step and can excite high-frequency oscillations in the contact forces". Remedies listed: bi-penalty,
  mass redistribution, singular-mass adjustments — [arXiv:2511.14323 (2025)](https://arxiv.org/abs/2511.14323)
- Penalty forces have a stable step that drops as the stiffness rises, so very stiff penalties are
  "untenable". Asynchronous variational contact mechanics uses a barrier made of nested quadratic
  penalty layers, each integrated at its own step, with momentum and energy conservation — [Vouga, Harmon, Tamstorf, Grinspun, arXiv:1007.3240](https://arxiv.org/pdf/1007.3240); [Harmon et al., arXiv:0907.0706](https://arxiv.org/pdf/0907.0706)
- Wolff & Bucher (2013), explicit asynchronous collision integrators for unilateral contact with
  friction: penalty methods allow penetration and may affect the critical step — [PMC3745265](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745265/)
- FLAC3D adds interface stiffnesses to the gridpoint stiffnesses on both sides for numerical
  stability (its explicit scheme uses them for the step / density scaling) — [FLAC3D formulation](https://docs.itascacg.com/pfc700/flac3d/docproject/source/theory/interfaces/formulation.html); a recommendation to keep kn, ks ≤ ~10× zone stiffness — [choice of properties](https://docs.itascacg.com/pfc700/flac3d/docproject/source/theory/interfaces/choiceofproperties.html)
- ZeroLengthContactASDimplex: syntax `$n1 $n2 $Kn $Kt $mu <-orient nx ny nz> <-intType 0|1>`. Penalty
  normal and tangential, Mohr-Coulomb with mu = tan φ. IMPL-EX (Oliver, Huespe & Cante, CMAME
  197(21-24):1865-1889, 2008) extrapolates the internal variables linearly during Newton iterations,
  so the element is linear in the trial strain, and corrects implicitly at commit. Authors: O. D. Akan
  (IUSS), M. Petracca (ASDEA) — [OpenSees manual](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/elements/zeroLengthContactASDimplex.html)
- Observations from reading the OpenSees source (955ceb0, `SRC/element/zeroLength/ZeroLengthContactASDimplex.cpp`):
  - `getMass()` returns zero (line 447-452). In an explicit stage the element adds stiffness but no
    mass, so its frequency is set by the nodes' other masses.
  - The initial gap `gap0` is fixed at the first update from the coordinates and displacements
    (lines 313-323). Normal stress SN = Kn·eps_n. Compression only (PC = 1 when SN ≤ 0). The friction
    limit is cres = -mu·SN. There is no cohesion, tension strength or dilation.
  - `time_factor` is forced to 1.0 (line ~1000). The IMPL-EX extrapolation does not use the real step
    ratio (the comment says this is deliberate, because of the pseudo-time in load control).
  - Probable bug: in the implicit plastic tangent, `double dE1`/`dE2` are re-declared inside the
    `if (UTsqnorm > eps)` block (lines ~1077-1083). The outer values stay 0, so dDdE1 = dDdE2 = 0 and
    C(1,1) = C(2,2) = 0 during plastic sliding (the tangential tangent is overwritten to zero). The
    tangent is approximate, but the residual is not affected.
  - Probable bug in `recvSelf`: `gap0(1) = ddata(39)` where `ddata(29)` is sent (line 613). This
    affects only parallel/database restart.
  - `getInitialStiff` gives Kn and Kt only if the initial gap is closed (lines 430-434).
  (These are my own reading. They have not been reported upstream or confirmed by a test.)

### Inferences
- For FemoraX's explicit stage with lumped mass, a penalty interface between nodes i and j with
  stiffness k adds a mode with ω² ≤ k(1/m_i + 1/m_j). A rule Kn ≤ c·m_eff/Δt² (LS-DYNA SOFT=1 style, c
  small) keeps the stable step, and it fits FemoraX's element-eigenvalue bound if the interface
  element declares `max_omega_sq` with a share of its nodes' masses (the design's mass-partition
  rule). A FLAC3D-style "10× neighbouring zone stiffness" penalty is roughly on the order of the
  soil's own ω_max² per unit mass, so it costs about a factor sqrt(1 + 10·area-factor) in step. The
  formula is not verified. It is to be computed per element in `critical_dt`.
- OpenSees default embedded penalty (1e18·sqrt(V)) is unusable in explicit runs. Femora's 1e10 may
  also set the step. FemoraX should derive the tie penalty from the mass (or use DOF elimination for
  tied, non-sliding embedding: an exact multi-point constraint u_beam = Σ N_a u_a, which costs no
  stable step and is what "tied" really means) and report the step it implies.
- Kinematic (predictor-corrector) contact keeps the step but needs a global or per-pair correction
  pass after the force evaluation. With lumped mass and node-to-face pairs fixed for a stage, the
  correction is local (a small solve per contact node and face). This is a GPU-friendly second
  kernel. Penalty is simpler (it is just another element in the fused force kernel) and is the
  natural first version.
- IMPL-EX is an implicit-stage device. In explicit central difference every law is already
  evaluated explicitly from the trial displacement, so the IMPL-EX option is irrelevant there.
  A port should implement the Backward-Euler law exactly (fixed-size state: eps_commit, shear(2),
  cres, rs, xs, PC), match OpenSees where it is correct, and record the tangent bug in
  `docs/deviations.md`.

### Gaps
- The exact LS-DYNA k0/k1 formulas (SOFT=0: k = SFS·K·A²/V for solids; SOFT=1: k1 = 0.5·SOFSCL·m/Δt²)
  are background knowledge [UNVERIFIED in this session]. They are in the LS-DYNA Theory Manual
  (contact chapter).
- Taylor & Flanagan's PRONTO-3D contact (1987-1989, the "defense node" / master-slave symmetric
  pass) and its predictor-corrector equations were not found online [UNVERIFIED].
- Abaqus default penalty scale factors (in my memory, penalty stiffness defaults are scaled so the
  step drops only a few percent) could not be confirmed (403).

## 3. GPU and HPC contact: detection, enforcement, partitions, determinism

### Takeaway
Public evidence of production GPU explicit contact is thin. LS-DYNA's documented GPU work is in
implicit linear algebra. Research codes report GPU contact (crash, finite particle), mostly with
atomic force assembly and grid or linked-list search. For SSI with small sliding, FemoraX can avoid
the hard part (dynamic global search) by fixing contact pairs per stage. Then contact forces use the
same deterministic gather as element forces.

### Cited Findings
- LS-DYNA (2011): GPU use limited to the direct factorization of the implicit solver, because
  host-GPU communication was too slow for other phases. Contact and element processing take much of
  the time — [Progress on GPU implementation for LS-DYNA implicit mechanics, DYNAmore 2011](https://www.dynamore.de/de/download/papers/konferenz11/papers/session6-paper2.pdf); [Performance benefits of NVIDIA GPUs for LS-DYNA](https://www.dynamore.de/de/download/papers/konferenz11/papers/session6-paper1.pdf/view)
- A 2025/2026 paper (Kim, MDPI J. Manuf. Mater. Process. 10(6):197) says publicly documented GPU work
  in LS-DYNA is linear algebra in the implicit path, not the explicit loop. It says most GPU
  implementations "still straddle the host and the device". It describes a GPU-resident explicit
  large-deformation pipeline and lists "rigid-body contact search without a BVH broad phase" as a
  remaining limitation — [MDPI 10(6):197](https://www.mdpi.com/2504-4494/10/6/197/htm) (403 on fetch; from search snippets; third-party claim)
- Ansys material on MPP explicit crash contact (2025) covers CPU-side settings and modular contact
  for highly parallel runs. No GPU explicit contact was found — [Ansys LS-DYNA simulation methods 2025](https://lsdyna.ansys.com/wp-content/uploads/2025/12/SimulationMethods_Birner_Wainscott_Ziegenhagel_Ansys.pdf); [HPC Day Ansys](https://lsdyna.ansys.com/wp-content/uploads/2025/11/HPC_Day_Ansys.pdf)
- GPU crashworthiness system: contact searching is the most time-consuming part of a body-in-white
  model. Atomic operations assemble the contact forces. GPU results match CPU results, much faster — [Zhejiang Univ. Sci. A, doi:10.1631/jzus.A2200311](https://link.springer.com/article/10.1631/jzus.A2200311); [summary](https://www.bohrium.com/paper-details/a-high-performance-crashworthiness-simulation-system-based-on-gpu/814635472415358976-3934)
- GPU contact for 2D large-deformation finite particle method: a linked-list structure for
  parallel global and local search. Speed-ups 28.5 (total) and 77.4 (contact) against single-core
  Abaqus/Explicit — [Wang et al., CMES 129(2):595-626, 2021, doi:10.32604/cmes.2021.017321](https://techscience.com/CMES/v129n2/44811)
- Sandia: parallel domain-decomposed contact global search with MPI plus on-rank Kokkos threading
  (ACME lineage). Attaway et al.'s PRONTO3D parallel contact search with dynamic load balancing is
  the early reference — [OSTI 1335669](https://www.osti.gov/pages/servlets/purl/1335669)

### Inferences
- Determinism: a fixed-connectivity interface element (zero-length, node-to-face with pairs fixed
  per stage, embedded node) is just another batch. FemoraX's gather scatter (each element writes to
  its own slots, each equation sums its slots in a fixed order) and coloring apply unchanged, so the
  CLAUDE.md rule 7 deterministic mode is free. Only dynamic search (pairs changing in the time loop)
  breaks this, because slot lists would change. Re-pairing could happen at fixed intervals with a
  deterministic sort (by node id) and a rebuild of the gather lists, outside the inner loop.
- Load balance across partitions: interface elements are few and cheap compared with soil
  elements. Partition weights (as planned for LTS) can include them. An interface pair across two
  subdomains is handled like a constraint across subdomains (the owner of one side, the other side
  a ghost).
- Branchy state updates (stick/slip/open) are a small fraction of the work, so warp divergence
  inside an interface batch costs little in total time.

### Gaps
- Nothing found on OpenRadioss GPU contact, or on current (2025-2026) LS-DYNA GPU explicit status
  from an official source.
- No source found on deterministic (non-atomic) GPU contact force assembly specifically.

## 4. Local time stepping and subcycling across interfaces and contact

### Takeaway
No published analysis covers stabilized leapfrog LTS (Diaz-Grote / Grote-Michel-Sauter) across
nonlinear contact or penalty interfaces. Subcycling literature reports momentum-conservation and
stability issues at level interfaces. The safest course for FemoraX is the one the design already
states: put each interface and its attached elements wholly inside one level. Give the interface
its own stable step (penalty ω) so it pulls its neighbourhood into a fine level rather than setting
the global step. Asynchronous contact integrators (AVI) show that per-force time steps for
penalty layers can be made conservative. SeisSol runs fault (dynamic rupture) interfaces within
clustered LTS.

### Cited Findings
- Subcycling can lose momentum conservation at the step interface and has a separate "statistical
  stability" problem. Smolinski & Daniel's energy-conserving multi-timestep central difference
  avoids it but may lose accuracy when the interface material does not deform elastically.
  Gravouil-Combescure Lagrange-multiplier coupling is provably stable but dissipative — [Karimi & Nakshatrala, "On multi-time-step monolithic coupling algorithms for elastodynamics", arXiv:1305.6355](https://arxiv.org/pdf/1305.6355); [core.ac.uk 15001063](https://core.ac.uk/download/15001063.pdf)
- "Challenges still remain with respect to stability and accuracy at interfaces" in multi-time-
  stepping — [Oxford ORA paper](https://ora.ox.ac.uk/objects/uuid:fa716278-95b0-4048-aa60-27e11c41b78a/files/rs1784n240)
- AVIs give each force (not each element) its own step. Contact is resolved by a barrier of nested
  quadratic penalties, each at its own rate, keeping momentum and energy conservation. A kinetic data
  structure handles detection — [arXiv:1007.3240](https://arxiv.org/pdf/1007.3240)
- SeisSol: LTS covers the whole simulation including dynamic rupture (fault friction interfaces).
  Clustered DR-LTS gave 6.8× over optimized global time stepping (CPU, EGU 2018). GPU LTS efficiency
  was "mediocre" in 2022, improved in later work — [LMU Munich geophysics](https://www.geophysik.uni-muenchen.de/de/forschung/erdbeben-physik/anwendungen); [EGU2018-15363](https://meetingorganizer.copernicus.org/EGU2018/EGU2018-15363.pdf?pdf); [Next-generation LTS for ADER-DG, arXiv:2202.10313](https://arxiv.org/pdf/2202.10313); [Dorozhinskii thesis, TUM](https://mediatum.ub.tum.de/doc/1730250/1730250.pdf) (not opened)

### Inferences
- In FemoraX's LTS (`compile/lts_levels.hpp`), an interface element should be an element with its own
  `critical_dt` (from its penalty and its share of the nodal masses). The existing rule (each element
  at the coarsest level whose step ≤ 0.9 of its own, each equation at the finest of its elements)
  then puts the interface and its ring in the fine level automatically. This is the same as the
  existing statement that "penalty interfaces, with the soil attached, stay inside one level".
- The scheme's coarse/fine split is proven for linear operators. Stick-slip and gap opening are
  nonlinear, so keeping both sides of an interface in one level avoids splitting a nonlinear force
  across levels. Verification by long runs and energy balance (friction work plus elastic energy)
  is required.
- Mass-scaled penalties (k ≤ c·m/Δt_level²) per level let the interface run at its level's step
  without forcing a finer level. This trades penetration for cost and must be reported.

### Gaps
- No paper found on Grote-Michel-Sauter stabilized LTS with penalty contact or friction.
- How SeisSol's clustered LTS handles the two sides of a fault in different clusters (it is DG,
  flux-coupled) was not verified.

## 5. Production geotechnical codes for pile-soil interfaces in dynamic analysis

### Takeaway
FLAC3D (explicit, the closest analogue to FemoraX): node-to-face interfaces with kn/ks, Coulomb,
bonds, dilation and effective stress, and stiffness limited to about 10× the zone stiffness for speed.
PLAXIS: zero-thickness interfaces with R_inter, or embedded beams with skin and base resistance.
OpenSees: zeroLength contact elements and ASDEmbeddedNode penalty ties (what Femora exports).
Real-ESSI: a family of zero-length stress-based soft contact elements with hardening shear and gap
viscosity. Abaqus and LS-DYNA: general penalty or kinematic surface contact.

### Cited Findings
- FLAC3D — [formulation](https://docs.itascacg.com/pfc700/flac3d/docproject/source/theory/interfaces/formulation.html); [choice of properties](https://docs.itascacg.com/pfc700/flac3d/docproject/source/theory/interfaces/choiceofproperties.html); FLAC3D soil-pile models set kn, ks to 10× the neighbouring zone stiffness (citing Itasca 2012) — [Scribd document](https://www.scribd.com/document/725727965/interface) (secondary)
- PLAXIS — [Novi Sad 2016](https://www.gf.uns.ac.rs/~zbornik/doc/NS2016.035.pdf); [geoengineer.org: how to model piles in PLAXIS](https://geoengineer.org/news/how-to-model-piles-in-plaxis); [Processes 9:1739](https://mdpi.com/2227-9717/9/10/1739)
- OpenSees — [ZeroLengthContactASDimplex](https://opensees.github.io/OpenSeesDocumentation/user/manual/model/elements/zeroLengthContactASDimplex.html); ASDEmbeddedNodeElement (OpenSees source, 955ceb0)
- Real-ESSI — [arXiv:2310.10251](https://arxiv.org/pdf/2310.10251); [Real-ESSI](https://sokocalo.engr.ucdavis.edu/~jeremic/Real_ESSI_Simulator)
- Abaqus/Explicit — [contact enforcement](https://abaqus-docs.mit.edu/2017/English/SIMACAEITNRefMap/simaitn-c-expcontactconstraints.htm); LS-DYNA — [SOFT option](https://lsdyna.ansys.com/?p=13360)
- A recent pile dynamic SSI workflow (Pakzad et al.) uses embedded interface elements with DRM and
  PML — [arXiv:2509.05675](https://arxiv.org/abs/2509.05675)

### Inferences
- Femora today exports exactly two interface kinds (ASDEmbeddedNodeElement ties, optionally with
  contact options refused in some cases, and zeroLengthContactASDimplex). Supporting these first in
  FemoraX keeps compatibility level 1. A FLAC3D-like node-to-face interface (cohesion, tension
  cutoff, dilation, effective stress) would be a FemoraX-level-3 extension for volume piles and mats.

### Gaps
- The PLAXIS dynamic interface behaviour (whether interfaces carry viscous boundaries and how R_inter
  acts in cyclic loading) was not checked in official manuals.
- Published seismic benchmarks of pile-soil gapping (centrifuge tests with interface models in FE)
  were not covered in this session.
