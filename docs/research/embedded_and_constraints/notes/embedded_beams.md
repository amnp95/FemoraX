# Embedded beam / embedded pile formulations (beams in non-conforming 3D solid meshes)

Scope note: web access to most full-text PDFs failed in this session (binary PDF streams, bot
protection, 403s). Items marked **[unverified]** come from prior knowledge and were not confirmed
against a source here. The OpenSees and Femora findings come from reading the local source
clones directly (OpenSees commit `955ceb0a5`, `../OpenSees/SRC/...`; Femora `../Femora/src/femora/...`),
which is first-hand evidence.

## 1. Geotechnical embedded beams (Plaxis embedded beam / embedded pile, Turello-Pinto-Sanchez)

### Takeaway
The geotechnical "embedded beam" is a beam that crosses solid elements freely and is tied to the
soil through interface springs (skin traction vs. relative displacement, plus a tip spring), with
the soil displacement interpolated at the beam's points. The original line-interface version
(Plaxis EB) applies a singular line load to the soil and is mesh-dependent beyond shaft capacity;
the remedies are an elastic zone around the beam (Plaxis) and, better, an explicit interaction
surface at the pile radius (Turello et al. 2016, EB-I), which converges better but still shows
skin-traction oscillations studied by Granitzer & Tschuchnigg (2023).

### Cited Findings
- Plaxis' embedded beam creates an elastic zone in the surrounding soil automatically, with its size
  set by the pile radius; its purpose is numerical stability with fine meshes, and an earlier Plaxis
  paper describes it as avoiding premature failure near the beam. — [ISSMGE / Plaxis-related papers (search summary)](https://www.issmge.org/uploads/publications/1/141/2273.pdf); [Bentley wiki: practice-oriented validation](https://communities.bentley.com/products/geotech-analysis/w/wiki/56388/practice-oriented-validation-of-embedded-beam-formulations-in-geotechnical-engineering)
- In Plaxis 3D the embedded beam is a 3-noded line element that may cross 10-noded tetrahedral soil
  elements; the line element causes stress discontinuities and mesh dependence. In the existing
  point-interface embedded beam the skin traction is linked to the relative displacement by a
  material law, and the tip force has a user-defined ultimate value independent of the soil. — [TU Delft thesis (repository)](https://repository.tudelft.nl/file/File_a6c17457-9f56-4480-b8f8-1082eca22926?preview=1)
- A proposed foot (tip) interface gives a stiffer axial response; its stiffness came out about one
  order of magnitude lower than in the existing embedded beam when calibrated, and a general
  definition of the foot-interface stiffness "still needs research". — [TU Delft thesis](https://repository.tudelft.nl/file/File_a6c17457-9f56-4480-b8f8-1082eca22926?preview=1)
- Load-displacement curves of the original embedded beam are considerably mesh-sensitive at loads
  beyond the shaft capacity; Turello et al. (2016) introduced the embedded beam with interaction
  surface to overcome mesh dependence; Smulders, Hosseini & Brinkgreve (2019) extended it in Plaxis. — [search summary of Smulders et al. / NUMGE papers](https://www.issmge.org/uploads/publications/51/119/NUMGE2023-109.pdf)
- Turello, Pinto & Sánchez (2016), "Embedded beam element with interaction surface for lateral
  loading of piles", Int. J. Numer. Anal. Methods Geomech. 40(4):568-582, doi:10.1002/nag.2416:
  perfect adherence along the interaction surface; better convergence than a previously formulated
  embedded element. — [CONICET repository](https://ri.conicet.gov.ar/handle/11336/38239); [Bohrium record](https://www.bohrium.com/paper-details/embedded-beam-element-with-interaction-surface-for-lateral-loading-of-piles/814612348944777217-3967)
- Turello et al. (2017), a three-dimensional elasto-plastic interface that couples with the
  embedded beam element to model nonlinear beam-solid interaction, IJNAMG 41(6):859-879. — [CONICET repository](https://ri.conicet.gov.ar/handle/11336/78731)
- Turello, Sánchez, Blanco & Pinto (2018), "A variational approach to embed 1D beam models into 3D
  solid continua", Computers & Structures 206:145-168. — [library record](https://bvbvf1.bib-bvb.de/vufind/PrimoRecord/cdi_proquest_journals_2100381012/Details)
- Turello, Pinto & Sánchez (2019), "Analysis of lateral loading of pile groups using embedded beam
  elements with interaction surface", IJNAMG 43:272-. — [ADS record](https://ui.adsabs.harvard.edu/abs/2019IJNAM..43..272T/abstract)
- Granitzer & Tschuchnigg (2023), "Finite element formulations for implicit beam-to-solid coupling:
  numerical obstacles and solution strategies", NUMGE 2023, doi:10.53243/NUMGE2023-109: embedded
  beams model load-settlement well, but skin-traction profiles often oscillate; parametric studies on
  the embedded beam with interaction surface (improved version of Plaxis' interaction-line element)
  give numerical evidence for the cause and possible remedies. Further mesh refinement does not
  necessarily converge to the exact solution (EB). — [TU Graz record](https://tugraz.elsevierpure.com/en/publications/finite-element-formulations-for-implicit-beam-to-solid-coupling-n); [ISSMGE PDF](https://www.issmge.org/uploads/publications/51/119/NUMGE2023-109.pdf)
- Skin-traction oscillations appear even with linear elastic soil when embedded stiffnesses are
  uniform along the pile; at working loads both formulations fit solid FE skin tractions
  qualitatively well. Tschuchnigg & Schweiger (2015) is the reference for the EB line formulation
  and showed realistic pile load-settlement and skin-resistance mobilization. — [Granitzer & Tschuchnigg 2021, Processes 9(10):1739 (MDPI)](https://mdpi.com/2227-9717/9/10/1739) (via search summary; page returned 403)

### Inferences
- The geotechnical family is the closest to what FemoraX needs (piles with t-z/p-y-like
  interface laws), but it is formulated and validated for static/implicit Plaxis analyses; none of
  the sources found discusses explicit dynamics or the stable time step.
- The line-load singularity (Section 4) is the root cause of EB's mesh dependence; the
  interaction-surface variant spreads the coupling over the pile perimeter, which is the same idea as
  OpenSees' EmbeddedBeamInterface (points at radius r on the perimeter, Section 2).

### Gaps
- Original papers not read: Sadek & Shahrour (2004, Comput. Geotech., "A three dimensional embedded
  beam element for reinforced geomaterials") **[unverified]**; Engin, Septanika & Brinkgreve (2007,
  embedded pile in Plaxis) **[unverified]**; Dao (2011, TU Delft MSc thesis, embedded pile in Plaxis
  3D) **[unverified]**; Sluis (2012 thesis; 2014 Sluis/Besseling/Stuurwold embedded beam row in 2D)
  **[unverified]**; Tschuchnigg & Schweiger (2015), "The embedded pile concept - verification of an
  efficient tool for modelling complex deep foundations", Comput. Geotech. 63:244-254 **[unverified
  bibliographic details]**.
- Exact Plaxis interface law (shaft resistance types linear / multilinear / layer-dependent,
  interface stiffness factors ISF_RS, ISF_KF, ISF_RN) and how elastic-zone size is computed: not
  confirmed from the manual in this session.
- Quantitative mesh-dependence numbers (EB vs EB-I) not extracted (PDF unreadable).

## 2. OpenSees: ASDEmbeddedNodeElement and EmbeddedBeamInterface(L/P/EP), and how Femora uses them

### Takeaway
ASDEmbeddedNodeElement is a linear penalty tie of one node to a linear triangle/tetrahedron
(not a hexahedron), with the optional rotation tie `rot = curl(u)/2` of the host; default penalty
1e18 scaled by cbrt(V). Femora tetrahedralizes the hex soil to use it. Ghofrani-Turello-Arduino's
EmbeddedBeamInterface couples beam perimeter points at radius r (beam kinematics: centreline
displacement plus rotation times offset) to the 8-node brick interpolation, by Lagrange multipliers
(L), penalty (P) or an elastoplastic interface NDMaterial per point (EP).

### Cited Findings (from source, OpenSees `955ceb0a5`)
- `ASDEmbeddedNodeElement $tag $Cnode $Rnode1 $Rnode2 $Rnode3 <$Rnode4> <-rot> <-p> <-K $K> <-KP $KP>`;
  default `K = 1.0e18`, `KP = K` unless given; `-rot` and `-p` are exclusive. — `SRC/element/CEqElement/ASDEmbeddedNodeElement.cpp` lines 201-285 (Massimo Petracca, ASDEA)
- Retained support is 3 nodes (triangle) or 4 nodes (tetrahedron) only: `TRI_2D_*`, `TRI_3D_*`,
  `TET_3D_U/UR/UP`. — same file, `getTangentStiff` (lines 534-605) and 801-1440
- TET_3D_U: constraint rows `B = [-I | N1 I | N2 I | N3 I | N4 I]` (linear tet shape functions at
  the constrained node), stiffness `K = (m_K * cbrt(V)) B^T B`. TET_3D_UR adds three rows
  `R_C = 1/2 curl(u_host)` from `dN/dX` (constant in a linear tet), i.e. the small-strain
  infinitesimal rotation of the host, with the same penalty `m_K*cbrt(V)` (`KP` is used for the
  pressure DOF in the u-p variant). — same file lines 1245-1370
- The element is linear and geometrically small-displacement: `getInitialStiff = getTangentStiff`,
  force `F = K U` (total displacements), zero mass and damping, no state (`update`,
  `revertToLastCommit` empty). — same file lines 524-645
- Femora's `EmbeddedNodeInterface` (tied, `ASDEmbeddedNodeElement`) tetrahedralizes the assembled
  mesh, finds the containing tetrahedron of each constrained point, and writes one element per point;
  it can offset points along surface normals; alternatively `contact=True` writes a frictional
  `EmbeddedNodeContact` (Kn, Kt default 1e8, mu 0.5, implicit or IMPL-EX integration, optional
  area scaling). — `../Femora/src/femora/components/interface/embedded_node_interface.py` lines 25-180, 194-240, 563-620
- Femora's `EmbeddedBeamSolidInterface` (radius, n_peri, n_long, penalty_param 1e12, g_penalty)
  generates perimeter points around line elements and searches candidate solid cells (hex, tet,
  wedge, pyramid) per segment. — `../Femora/src/femora/components/interface/embedded_beam_solid_interface.py` lines 36-80, 270-300, 550-600
- `EmbeddedBeamInterfaceL` (Ghofrani, Turello, Arduino, U. Washington, 2017): stores per point beam
  (rho, theta), solid (xi, eta, zeta), area and length; creates Lagrange-multiplier nodes (6 DOF each,
  `m_Lambda(6*numBeamNodes)`); solid side uses 8-node brick shape functions; beam point kinematics
  `Hb = Bu - r (cos(theta) skew(c2) + sin(theta) skew(c3)) Bphi`, i.e. centreline displacement plus
  rotation times the radial offset. — `SRC/element/UWelements/EmbeddedBeamInterfaceL.cpp` lines 1-170, 930-990
- `EmbeddedBeamInterfaceP`: the same with a penalty parameter `m_ep` (stiffness and force scaled
  by it). `EmbeddedEPBeamInterface`: an `NDMaterial` copy per interface point (`setTrialStrain` with
  the relative displacement, `getTangent` as Cep), i.e. a nonlinear interface law. — `EmbeddedBeamInterfaceP.cpp` lines 98-102, 633-662; `EmbeddedEPBeamInterface.cpp` lines 99-214, 340, 755
- The generator `generateInterfacePoints` takes `-shape circle|square|rectangle|hexagon`, `-radius`,
  `-nP` (perimeter), `-nL` (longitudinal), `-penaltyParam`, and the coupling choice `-penalty`,
  `-lagrange`, `-AL`, `-gPenalty`, `-gAL`, `-contact`, `-gContact`, `-ElastoPlastic -mat`. — `SRC/element/UWelements/Tcl_generateInterfacePoints.cpp` lines 100-390

### Inferences
- ASDEmbeddedNodeElement's tie on a sub-tet of a hex is not consistent with the hex's trilinear
  field (the tet's linear interpolation of 4 of the 8 corners differs from the hex's), so the beam
  sees a slightly different soil field than the soil itself; a FemoraX version should interpolate
  with the 8-node hex shape functions directly.
- Stable step: a penalty spring k between an embedded node of mass m_b and host nodes adds
  roughly omega^2 ~ k (1/m_b + sum N_i^2/m_i). With K = 1e18 * cbrt(V) (or Femora's 1e10-1e12)
  this is orders of magnitude above the soil's omega^2, consistent with the user's experience that
  the tie sets the explicit step. A kinematic elimination (Abaqus-style, Section 3) has no such
  limit.
- The rotation tie `curl(u)/2` is exact only for linear (constant-gradient) host fields; on an
  8-node hex the curl varies within the element and the polar rotation of F (Section 4) is the
  objective large-rotation analogue.

### Gaps
- Published papers behind these elements: Ghofrani (2018, U. Washington PhD thesis) and any
  Petracca publication on ASDEmbeddedNodeElement **[unverified]**; OpenSees documentation pages not
  fetched in this session.
- Femora's `EmbeddedNodeContact` source (an OpenSees element outside the main tree?) not inspected.

## 3. Commercial codes: Abaqus *EMBEDDED ELEMENT, LS-DYNA *CONSTRAINED_BEAM_IN_SOLID

### Takeaway
Abaqus eliminates the embedded nodes' translational DOFs kinematically (exact tie to the host
interpolation), leaves beam rotations free, and adds the embedded elements' stiffness and mass on
top of the host (volume overlap). LS-DYNA offers a constraint form (with axial debonding through a
user bond-slip law, AXFOR/CDIR) and a penalty form (no AXFOR; penalty from the bulk moduli;
recommended for impact/blast, conserves kinetic energy).

### Cited Findings
- Abaqus: embedded nodes inside host elements have their common translational DOFs "constrained to
  the interpolated values" of the host; weights from the node's position; nodes on a face/edge tie
  only to that face/edge. — [Abaqus 2025 docs, Embedded elements](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-embeddedelement.htm)
- Abaqus: embedded elements may have rotational DOFs, "but these rotations are not constrained by the
  embedding"; hosts may not have rotational DOFs (except axisymmetric with twist); hosts cannot be
  embedded themselves. — [Abaqus 2025 docs](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-embeddedelement.htm); [Abaqus 2017 docs (MIT)](https://abaqus-docs.mit.edu/2017/English/SIMACAECSTRefMap/simacst-c-embeddedelement.htm)
- Abaqus: exterior tolerance default 0.05 x the average non-embedded element size (nodes within are
  projected onto the host, outside gives an error); roundoff tolerance zeroes small weights and
  redistributes them, "performed only at the start of the analysis and does not create any strain".
  Embedded nodes also tied by other kinematic constraints give an overconstraint error. — [Abaqus 2025 docs](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-embeddedelement.htm)
- Abaqus: "Additional mass and stiffness due to the embedded elements are added to the model" (no
  subtraction of host material). — [Abaqus 2025 docs](https://docs.software.vt.edu/abaqusv2025/English/SIMACAECSTRefMap/simacst-c-embeddedelement.htm)
- LS-DYNA *CONSTRAINED_BEAM_IN_SOLID: CDIR and AXFOR work only with the constraint-based coupling;
  penalty springs act between coupling points on the beam and the solid with stiffness from the bulk
  moduli of beam and solid, scaled by PSSF; the penalty form is recommended for impact/blast with
  large momentum transfer and conserves kinetic energy; it first appeared in a development version
  (not R9/R10). — [LSTC CBIS manual / directory](https://ftp.lstc.com/anonymous/outgoing/hao/cbis) (via search summary; the PDF fetch failed on its TLS certificate); [Ansys forum](https://innovationspace.ansys.com/knowledge/forums/topic/how-to-embed-beam-elements-representing-reinforcement-into-a-solid-lagrangian-part)
- LS-DYNA bond-slip: AXFOR < 0 calls a *DEFINE_FUNCTION or user routine `rebar_bondslip_get_force()`
  receiving slip (relative axial displacement of beam node vs. surrounding solid) and tributary length;
  CDIR = 1 releases the axial constraint; implicit runs also need a stiffness. — [LS-DYNA user group](https://groups.google.com/g/ls-dyna2/c/RZo-ko1bRGM/m/JUQo_gTYAgAJ); [CBIS directory](https://ftp.lstc.com/anonymous/outgoing/hao/cbis)

### Inferences
- Kinematic elimination is what fits FemoraX's DOF-elimination constraint machinery (design.md:
  equalDOF already by elimination): each beam node's translations become `u_b = sum N_i(xi_b) u_i` of
  the host hex, adding no stiffness and so no time-step penalty. The price: the beam's lumped mass
  maps to host DOFs as `T^T M_b T`, which is not diagonal (N_i N_j couplings); row-sum lumping
  `m_i += N_i m_b` keeps it diagonal (the usual choice, not confirmed from Abaqus/Explicit docs).
- Neither Abaqus nor LS-DYNA (per the sources found) ties beam rotations to the solid; for piles in
  soil the bending is carried by the beam with the soil acting through translations at many points,
  which needs beam nodes / coupling points dense enough relative to the soil mesh.
- An embedded-node elimination cannot represent slip; slip/gap/t-z laws need a penalty or an
  interface element (LS-DYNA's AXFOR route: elimination in the transverse directions, a law in the
  axial direction).

### Gaps
- Abaqus/Explicit and LS-DYNA documentation on how the constraint changes the stable step and mass
  lumping: not found.
- LS-DYNA *CONSTRAINED_LAGRANGE_IN_SOLID (fluid-structure coupling, penalty-based, mostly ALE) not
  researched in this session. Ansys Mechanical's "reinforcement" (MESH200/REINF264) not researched.

## 4. Mixed-dimensional beam-to-solid coupling (Popp / Meier group, 1D-3D Lagrange multipliers)

### Takeaway
Steinbrecher et al. define a Lagrange-multiplier line load along the beam centreline (overlapping
volumes), discretize it either point-wise (GPTS, prone to locking when beam elements are short
relative to solid elements) or as a mortar field (linear multipliers, robust), usually regularized
by a penalty; rotations are coupled in a later paper via the polar rotation of the solid's
deformation gradient. The 1D-3D model is valid only when solid elements are not smaller than the
beam diameter (line-load singularity).

### Cited Findings
- Steinbrecher, Mayr, Grill, Kremheller, Meier & Popp (2020), "A mortar-type finite element approach
  for embedding 1D beams into 3D solid volumes", Comput. Mech. 66(6):1377-1398,
  doi:10.1007/s00466-020-01907-0, arXiv:1912.04153: Lagrange-multiplier line load along the
  centreline; solid meshed without subtracting the beam volume. — [arXiv 1912.04153](https://ar5iv.labs.arxiv.org/html/1912.04153); [Springer](https://link.springer.com/10.1007/s00466-020-01907-0)
- In that paper: the overlap is a modelling error that grows with fibre volume fraction and
  fibre-to-matrix stiffness ratio; GPTS enforces the constraint at each Gauss point (3 constraints
  each) and is the Dirac-multiplier special case of mortar; penalty regularization scaled by a nodal
  matrix passes patch tests independent of element length; larger penalties lead to "an
  ill-conditioned tangential system matrix". — [arXiv 1912.04153](https://ar5iv.labs.arxiv.org/html/1912.04153)
- GPTS over-constrains when beam elements are short relative to solid elements ("contact locking",
  a discrete inf-sup violation); mortar with linear multipliers showed no locking at length ratios
  10 and 5, quadratic/cubic multipliers did at smaller ratios; segment-based integration is
  independent of the length ratio and reproduces the patch test to machine precision. — [arXiv 1912.04153](https://ar5iv.labs.arxiv.org/html/1912.04153)
- The exact solution of a line load in 3D (generalized Kelvin problem) "has a singularity ... not only
  in the stress field, but also in the displacement field"; convergence was O(h^2) on coarse meshes
  with a kink at a solid element size of about 0.12 m, after which error stopped decreasing (about
  0.5 % tip-displacement error vs. a 2D-3D reference); rule of thumb: solid elements not smaller
  than the beam diameter. — [arXiv 1912.04153](https://ar5iv.labs.arxiv.org/html/1912.04153)
- With positional coupling only, the rotation about the beam axis is unrestrained and can make the
  system singular (handled by Dirichlet conditions or deflation); torsion-free beams avoid it. — [arXiv 1912.04153](https://ar5iv.labs.arxiv.org/html/1912.04153)
- A further study finds the mortar discretization with linear Lagrange-multiplier basis the better
  choice for its robustness with respect to the penalty parameter. — [arXiv 2101.02527 (search summary)](https://ar5iv.labs.arxiv.org/html/2101.02527)
- Steinbrecher, Popp & Meier (2022), "Consistent coupling of positions and rotations for embedding 1D
  Cosserat beams into 3D solid volumes", Comput. Mech. 69(3):701-732, doi:10.1007/s00466-021-02111-4,
  arXiv:2107.11151: three translational plus three rotational constraints; solid triads from the
  polar decomposition of F (shown L2-optimal); objective relative-rotation constraints; embedded mortar
  scheme for both; consistent spatial convergence. — [arXiv 2107.11151](https://arxiv.org/abs/2107.11151); [UniBw news](https://www.unibw.de/imcs-en/news/new-publication-on-beam-to-solid-coupling)
- Steinbrecher, Hagmeyer, Meier & Popp, beam-to-surface (1D-2D) coupling with exact angular momentum
  conservation for non-zero surface-normal distance, arXiv:2210.16010. — [arXiv 2210.16010](https://www.arxiv.org/pdf/2210.16010)
- Khristenko, Wohlmuth et al. (fibre-reinforced materials): 1D beams in 3D matrix derived from a full
  surface-to-volume Lagrange-multiplier coupling, plus a tensor multiplier for cross-section
  incompressibility; multiplier split into mean and harmonic parts around the perimeter; two-step
  static condensation. — [Siegen poster](https://www.mb.uni-siegen.de/nm/forschung/paper/short_fiber_poster.pdf); [Siegen slides](https://www.mb.uni-siegen.de/nm/mitglieder/ehemalige/krueger/pdf/shortfibercoupling.pdf)
- Heltai & Zunino, "Reduced Lagrange multiplier approach for non-matching coupling of mixed-
  dimensional domains" (arXiv:2303.10600, 2023): general framework, well-posedness and robustness
  w.r.t. the inclusion's smallest length, inf-sup stability, interplay of mesh size, multiplier-space
  dimension and inclusion size. — [arXiv 2303.10600](https://arxiv.org/abs/2303.10600)

### Inferences
- For FemoraX (explicit, matrix-free, lumped mass): a true Lagrange multiplier / mortar method needs a
  saddle-point solve per step, not compatible with central difference; the penalty-regularized mortar
  is explicit-friendly but brings the penalty time-step limit back. The useful lessons are the
  integration (segment-based along the beam across hex boundaries), linear multiplier/penalty weights
  (avoid locking), and the mesh rule (soil elements >= pile diameter for a line model; otherwise use
  perimeter/surface coupling such as EB-I / EmbeddedBeamInterface).
- Rotational coupling via polar rotation of F is the objective choice when beam rotations must follow
  the soil (fibres); for piles the bending is normally carried by the pile and resisted by distributed
  translational springs, so rotation coupling is secondary (Abaqus and LS-DYNA also leave it free).

### Gaps
- Kremheller/Meier/Popp follow-ups (e.g. beam-to-solid with contact, BACI/4C implementation details)
  and any explicit-dynamics use of these schemes: not found.
- Validation of the Popp/Meier schemes against pile load tests: none found (their targets are fibre
  composites and biomechanics).

## 5. Cross-cutting comparison (exactness, rotations, nonlinearity, mesh dependence, time step, cost, maturity)

### Takeaway
Three implementable families for FemoraX: (a) kinematic elimination of beam-node translations into
the hex interpolation (Abaqus-style; exact, no time-step cost, no slip); (b) penalty / interface
springs between beam points (centreline or perimeter) and the hex interpolation, with linear or
nonlinear (t-z, p-y, friction) laws (Plaxis EB/EB-I, OpenSees EmbeddedBeamInterfaceP/EP, LS-DYNA
penalty; time-step cost set by the spring stiffness, which is physical for soft t-z/p-y laws and
huge for "tie" penalties); (c) Lagrange multiplier / mortar (Popp/Meier, EmbeddedBeamInterfaceL;
exact but a saddle point, implicit only).

### Cited Findings
- Penalty: larger values reduce the constraint violation but ill-condition the tangent. — [arXiv 1912.04153](https://ar5iv.labs.arxiv.org/html/1912.04153)
- LS-DYNA's penalty form derives its stiffness from the materials' bulk moduli (a material-based,
  not arbitrary, penalty) and is the variant suggested for explicit impact. — [CBIS directory](https://ftp.lstc.com/anonymous/outgoing/hao/cbis)
- ASDEmbeddedNodeElement default penalty 1e18 x cbrt(V) (OpenSees source, Section 2).
- Line-model mesh dependence: EB mesh-sensitive beyond shaft capacity; interaction surface reduces it;
  1D-3D line coupling valid only for solid elements no smaller than the beam diameter. — [NUMGE 2023](https://tugraz.elsevierpure.com/en/publications/finite-element-formulations-for-implicit-beam-to-solid-coupling-n); [arXiv 1912.04153](https://ar5iv.labs.arxiv.org/html/1912.04153)

### Inferences
- Stable step: for an interface spring of stiffness k per coupling point (linear in tributary area
  for t-z/p-y laws), the coupling's omega^2 bound is about k / m_eff with m_eff the smaller of the
  beam's and the host's lumped masses at that point; a soil-derived interface stiffness (of order
  G times tributary length) gives an omega^2 comparable to the soil's own, while tie penalties of
  1e10-1e18 do not. Elimination plus a separate (soft) slip spring in the axial direction (the
  LS-DYNA AXFOR pattern) keeps the step and still allows t-z behaviour; for lateral p-y behaviour a
  soft perimeter interface (EB-I / EmbeddedBeamInterface-EP pattern) is the analogue.
- Kinematic elimination makes the beam translations host-interpolated: the pile's distributed
  inertia and stiffness add to the host equations (T^T K_b T applied matrix-free, element by element)
  and the pile can then only deform as the trilinear soil field allows between beam nodes inside one
  hex (a stiffening when the soil mesh is coarse relative to the pile's bending wavelength).
- Overlap (beam volume not subtracted from soil): all formulations found accept it; for piles with
  E_pile/E_soil large it is a small error (consistent with the Popp group's statement), though the
  soil mass inside the pile is double counted unless the beam density is reduced by rho_soil.
- Matrix-free / GPU fit: all three families are per-coupling-point gather (8 host nodes + 2 beam
  nodes) compute scatter, fitting FemoraX's kernel pattern; mortar/Lagrange forms additionally need
  multiplier DOFs and a non-SPD saddle-point operator (GMRES), which the solvers can handle in implicit
  stages only.

### Gaps
- No source found that measures the explicit stable step of any embedded-beam formulation, or that
  uses embedded beams with explicit central difference in soil-structure interaction.
- No published benchmark found that compares embedded-beam formulations under dynamic (seismic)
  loading of piles; the validations found are static pile load tests (Tschuchnigg & Schweiger 2015;
  Granitzer & Tschuchnigg 2021) and lateral pile-group loading (Turello et al. 2019).
