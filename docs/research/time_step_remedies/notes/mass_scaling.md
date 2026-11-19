# Mass scaling methods for explicit dynamics (conventional, selective, reciprocal, rotary inertia)

Scope: what each method does, how mature it is, what it costs per step, and how well it fits a
GPU-native, matrix-free, FP64 explicit solver (FemoraX: central difference with lumped mass,
hex8 soil plus beams, diaphragms and penalty interfaces). Where I could not open a paper,
I say so; numbers are given only where a source showed them.

## Conventional (nonuniform) mass scaling: LS-DYNA and Abaqus/Explicit practice, accuracy, acceptable added mass

### Takeaway
Conventional mass scaling adds diagonal mass only to the elements whose stable step is below a
target. It is free per step and trivial to implement, but it lowers *every* frequency of the
scaled region, including rigid-body and low modes. Vendors give no official percentage
limit for dynamic analyses: the guidance is "check added mass and energies". One vendor
example shows that a total added mass of about 1 % already gave 19 % peak force error in
a dynamic impact.

### Cited Findings
- LS-DYNA: mass scaling is set in `*CONTROL_TIMESTEP` via DT2MS. A negative DT2MS adds mass only to elements whose step is below |DT2MS|, and this is the recommended form. The amount of added mass is set by |DT2MS| and is not multiplied by TSSFAC. Added mass is reported in d3hsp, messag and glstat. MS1ST=1 restricts scaling to the first cycle, and DT2MSLC lets the target vary in time — [LS-DYNA "Mass scaling" how-to](https://lsdyna.ansys.com/mass-scaling/); [DYNAmore webinar 2022](https://lsdyna.ansys.com/wp-content/uploads/2025/11/221215_Webinar_Mass-Scaling_and_Subcycling.pdf)
- DYNAmore describes conventional mass scaling as "a broadsword" that "scales down all eigenfrequencies indiscriminately". Selective scaling is "a scalpel" that only scales down the highest eigenfrequency — [DYNAmore webinar, Hallén 2022](https://lsdyna.ansys.com/wp-content/uploads/2025/11/221215_Webinar_Mass-Scaling_and_Subcycling.pdf)
- Quantitative example from the same webinar: an NHTSA Honda Accord front bumper impact (dynamic, contact force compared; base dt 1.8e-7 s set by spotweld solids; 421 s on 16 MPP cores). The table shows CMS and SMS side by side:

  | dt (s) | added mass | CMS time | CMS max / mean error (RMS contact force) | SMS time | SMS max / mean error |
  |---|---|---|---|---|---|
  | 4.5e-7 | 0.013 % | 181 s | 2.7 / 0.3 % | 227 s | 2.7 / 0.3 % |
  | 1.0e-6 | 0.8 % | 80 s | 18.9 / 4.3 % | 116 s | 4.7 / 0.66 % |
  | 1.5e-6 | 4.2 % | 60 s | 94.3 / 20 % | 76 s | 10.5 / 2.7 % |
  | 2.0e-6 | 11.6 % | 46 s | 154 / 33.6 % | 61 s | 19.7 / 5.6 % |

  — [DYNAmore webinar 2022, slides 15 and 18](https://lsdyna.ansys.com/wp-content/uploads/2025/11/221215_Webinar_Mass-Scaling_and_Subcycling.pdf)
- Abaqus/Explicit offers fixed mass scaling (once at step start) and variable mass scaling (`*VARIABLE MASS SCALING`, periodically during the step). Both can target a desired element stable increment (DT) for an element set. Variable scaling is "most useful when the stiffness properties that control the stable time increment change drastically" — [Abaqus 2017 docs, mass scaling](https://abaqus-docs.mit.edu/2017/English/SIMACAEANLRefMap/simaanl-c-massscaling.htm); [*VARIABLE MASS SCALING](https://abaqus-docs.mit.edu/2017/English/SIMACAEKEYRefMap/simakey-r-variablemassscaling.htm)
- Abaqus ties its scaling guidance mainly to quasi-static analyses, where kinetic energy must stay small. For dynamic use it points to a benchmark, "Impact of a copper rod", which I did not open — [Abaqus 6.6 docs](https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usb/pt04ch11s07aus63.html)
- Practitioner rules (not official): kinetic/internal energy below 5 % for quasi-static runs ([Eng-Tips](https://eng-tips.com/threads/how-to-judge-mass-scaling-results.445951)). A training write-up used an 8 % added-mass cap ([Skill-Lync student project](https://www.skill-lync.com/student-projects/week-8-mass-scaling-26)). Neither is a vendor standard.
- Theory (Voet, Sande, Buffa 2024): conventional scaling with factor α changes each eigenfrequency by a ratio between 1 and √α, so low modes of the scaled region are not protected — [arXiv:2410.23816](https://arxiv.org/abs/2410.23816)
- Macek & Aubert 1995 proposed a "mass penalty technique to control the critical time increment" (EESD 24(10):1315–1331), an early structured form of this idea — cited in [Voet et al. 2024 references](https://arxiv.org/html/2410.23816v2)

### Inferences
- Scaling law for a target step: the stable step of an axial or solid element goes as L·√(ρ/E), so reaching dt_target from dt_e needs a mass factor (dt_target/dt_e)². Going from 1e-4 to 1e-3 s takes a factor of 100; going to 2e-4 s takes a factor of 4. (This is standard Courant algebra, not taken from one of the sources.)
- For SSI, conventional scaling of the *structure* is dangerous for the quantities of interest. Column and beam mass enters storey inertia and the structure's own periods. Even if column mass is small next to floor (diaphragm) masses, a factor of 4–100 on member mass shifts the structural periods and adds inertial force on the foundation. Scaling small *soil* bricks near a structure adds mass directly to the SSI kinematic and inertial interaction. The bumper table suggests that sub-percent *total* added mass can still give large local errors.
- The cost in FemoraX would be nil: a per-node lumped-mass edit in the compile step, plus a report of added mass by region. It is the obvious first option, but only with mandatory reporting, such as the added-mass fraction per region and the change of the structure's first periods.

### Gaps
- No official LS-DYNA or Abaqus percentage limit for *dynamic* analyses was found. The commonly quoted "a few percent" rule could not be traced to a primary source.
- I did not open the Abaqus copper-rod benchmark or the LSTC "mass_scaling" FAQ text (the TLS certificate failed).

## Selective mass scaling (SMS): methods, speedups, low-mode preservation, cost of the non-diagonal solve

### Takeaway
SMS adds a non-diagonal, element-level mass that does not change total (translational) mass or
rigid-body inertia, and lowers only the highest element frequencies. Reported step increases
are 10–30x (up to 50x in forming) while low modes stay largely intact. The price is a sparse
linear solve with the scaled mass every step, done by Jacobi-preconditioned CG. Olovsson &
Simonsson report about 13 iterations at 10x scaling and 24 at 50x. Vendors report a per-step
cost of up to about +50 %, and recommend SMS only on the small parts that limit the step.

### Cited Findings
- Olovsson, Simonsson & Unosson 2005, "Selective mass scaling for explicit finite element analyses", IJNME 63(10):1436–1445, [doi:10.1002/nme.1293](https://doi.org/10.1002/nme.1293). Non-diagonal mass terms are added that lower the high eigenfrequencies. The critical step increased "up to 30 times" in two examples. The scaled mass's inverse needs extra memory — [abstract via Bohrium](https://www.bohrium.com/paper-details/selective-mass-scaling-for-explicit-finite-element-analyses/812283546260471809-3991). An earlier version is Olovsson, Unosson & Simonsson 2004, "Selective mass scaling for thin walled structures modeled with tri-linear solid elements", Comput. Mech. 34:134–136 ([cited in Voet et al.](https://arxiv.org/html/2410.23816v2)).
- Olovsson & Simonsson 2006, "Iterative solution technique in selective mass scaling", Commun. Numer. Meth. Engng 22(1):77–82. It solves M a = f iteratively every step instead of inverting M. The search snippet of the paper's table reports these figures. At scaling factor 10: 13.1 average CG iterations per step, 1346 steps instead of 4268, speedup 2.0. At factor 50: 24.0 iterations per step, 621 steps, speedup 3.4 — [LSTC copy of the paper](https://ftp.lstc.com/anonymous/outgoing/support/FAQ_docs/olovsson2006_selective_mass_scaling.pdf). Caution: I saw these numbers only in the search snippet; I could not open the PDF to confirm them.
- LS-DYNA implementation: `IMSCL` in `*CONTROL_TIMESTEP`. IMSCL=1 applies SMS everywhere; IMSCL<0 applies it to a part set, which is recommended to limit CPU time. Mass is added up to the DT2MS target. The mass matrix is no longer diagonal, so a sparse system is solved every step by (preconditioned) CG. SMS mass is reported in the same "added mass" variable as CMS mass, so kinetic energy must be checked directly. It is incompatible with some constraint types (CONTACT_TIED_xxx, CONSTRAINED_SPOTWELD) and works with MAT_RIGID and nodal rigid bodies — [LS-DYNA SMS how-to](https://lsdyna.ansys.com/?p=3558); [LS-DYNA mass scaling how-to](https://lsdyna.ansys.com/mass-scaling/)
- LS-DYNA examples: a 250k-tet model on 16 MPP CPUs took 15,874 steps under both CMS and SMS. CPU time was 997 s (SMS) against 647 s (CMS), so SMS costs about 54 % more per step. Speedup over no scaling was 9.3 (SMS) and 14.3 (CMS). The steering-wheel example ran 0.7 h with SMS at dt = 1e-6 against 5.5 h with CMS at 1e-7. SMS is "very effective" when applied to "limited regions with very small critical timesteps" — [LS-DYNA SMS how-to](https://lsdyna.ansys.com/?p=3558)
- In the bumper case (table above), SMS at dt = 2e-6 (11x the base step) kept the mean error at 5.6 %, against 33.6 % for CMS, at about 33 % more wall time than CMS at the same dt — [DYNAmore webinar 2022](https://lsdyna.ansys.com/wp-content/uploads/2025/11/221215_Webinar_Mass-Scaling_and_Subcycling.pdf)
- LS-DYNA rotational SMS: `RMSCL` extends the projection to rotational rigid-body modes, which further reduces the inertia for problems with rotation. A 2011 DYNAmore presentation called it CPU-intensive and its practical use unproven at that time — [Borrvall, DYNAmore (search summary)](https://lsdyna.ansys.com/wp-content/uploads/2025/02/borrvall.pdf). I could not open the PDF text.
- Altair Radioss "Advanced Mass Scaling" (AMS): a non-diagonal mass that keeps total mass constant (it raises the diagonal and offsets this with off-diagonal terms), so translational kinetic energy is not inflated. Targets are 10–20x the conventional step, up to 50x in manufacturing; one example ran a 10x step for a 3x total speedup. The mass-matrix work can reach 50 % of the total cost in highly nonlinear models. The CG solver's default tolerance is 1e-3 (formerly 1e-4) — [Altair Radioss 2025 help](https://2025.help.altair.com/2025/hwsolvers/rad/topics/solvers/rad/increase_speed_maintain_accuracy_ams_r.htm)
- Cocchetti, Pagani & Perego 2013, "Selective mass scaling and critical time-step estimate for explicit dynamics analyses with solid-shell elements", Computers & Structures 127:39–52. SMS by a linear transformation of the element DOFs, plus an analytical estimate of the element's highest frequency, hence the critical step. The 2015 follow-up, "Selective mass scaling for distorted solid-shell elements in explicit dynamics: optimal scaling factor and stable time step estimate", IJNME 101(9):700–731, handles distorted elements. After scaling, the step is governed by the smallest in-plane dimension rather than the thickness, *and the mass matrix stays diagonal*. Its own caveat: the scaling is in effect a change of rotational inertia, which can distort kinetic energy when rotation dominates — [Polimi preprint 2015](https://re.public.polimi.it/bitstream/11311/884363/6/11311-884363_Perego.pdf); [2013 record](https://re.public.polimi.it/handle/11311/783916); bibliographic data from [Voet et al.](https://arxiv.org/html/2410.23816v2)
- Tkachuk & Bischoff 2013, "Variational methods for selective mass scaling", Comput. Mech. 52:563–570 ([MaRDI record](https://portal.mardi4nfdi.de/wiki/Variational_methods_for_selective_mass_scaling)). Tkachuk & Bischoff 2014, "Local and global strategies for optimal selective mass scaling", Comput. Mech. 53(6):1197–1207 ([Springer s00466-013-0961-5, as indexed](https://link.springer.com/article/10.1007/s00466-013-0961-5)). Both derive scaled mass matrices from penalized Hamilton principles (bibliographic data via [Voet et al.](https://arxiv.org/html/2410.23816v2)).
- Related variationally based SMS: Askes, Nguyen & Tyas 2011, "Increasing the critical time step: micro-inertia, inertia penalties and mass scaling", Comput. Mech. 47:657–667. Stoter, Nguyen, Hiemstra & Schillinger 2022, "Variationally consistent mass scaling for explicit time-integration schemes of lower- and higher-order FEM", CMAME 399:115310 ([arXiv:2201.10475](https://arxiv.org/abs/2201.10475)). Ye et al. 2017, SMS for shear waves in nearly incompressible materials, IJNME 109(2):155–173. Hoffmann, Tkachuk, Bischoff & Oesterle 2023 (COMPDYN), element-technology-based SMS for thin-walled solids. Krauß et al. 2024, "Intrinsically selective mass scaling with hierarchic plate formulations", CMAME 432:117430. All references as listed in [Voet et al.](https://arxiv.org/html/2410.23816v2)
- Theory (Voet, Sande & Buffa, EPFL, arXiv 2024): for Olovsson's element SMS with parameter β, the eigenfrequency ratio is at most √(1+8β/7) and the condition number of the scaled mass is at most 1+8β/7. Hoffmann's method is bounded by √(1+9β/2). Raising the step and keeping the scaled mass well conditioned (that is, few CG iterations) "are conflicting goals". Olovsson's method is "quite inaccurate for lower frequencies"; Hoffmann's preserves them better. Stiffness-proportional SMS (M + μK) needs K re-formed in nonlinear problems, which is "prohibitively expensive". Nearly all SMS methods give a non-diagonal mass and "defeat the spirit of explicit dynamics": they pay only if the saved steps outweigh the solves. The authors point to modifying the inverse mass (or the stiffness) directly as the promising route — [arXiv:2410.23816](https://arxiv.org/abs/2410.23816)
- González & Park, "Large-step explicit time integration via mass matrix tailoring", IJNME, [doi:10.1002/nme.6282](https://doi.org/10.1002/nme.6282) (Zendy lists 2019, Voet et al. list 121(8):1647–1664, 2020). It keeps the low-frequency content and replaces unwanted high mesh frequencies with a user-chosen cutoff, at the element, substructure or global level, without changes to element routines — [Zendy abstract](https://zendy.io/title/10.1002/nme.6282)

### Inferences
- **Fit to FemoraX's matrix-free GPU path.** The Olovsson-type element SMS mass is M_L + Σ_e λ_e(M̃_e), a sum of element terms. Its product M_s·v is therefore an element-by-element kernel, the same shape as FemoraX's existing matrix-free operator (gather, an element matvec, then a deterministic scatter), with a diagonal (Jacobi) preconditioner. The solve is CG on an SPD, well-conditioned operator (condition number at most 1+8β/7 for Olovsson). At about 13–24 iterations per step for 10–50x, each iteration costs one mass-operator application on the scaled set plus two global reductions. FemoraX's own CG and deterministic dot products (src/solvers/) already provide all of this.
- **Restrict it to the step-limiting region.** In SSI the elements that limit the step (structural members, small bricks next to foundations) are a tiny fraction of a billion-element soil mesh. If SMS is restricted to the equations those elements touch (unscaled equations keep the lumped diagonal, so the system decouples into a diagonal block plus a small SPD block), the CG works only on the small block. Its cost would then be negligible next to the soil force kernel, except for launch and latency overhead. On GPUs, about 20 small CG iterations per step means about 40–60 tiny kernel launches plus reductions. That overhead could rival the soil force kernel; the design document's LTS section raises the same concern. This needs a benchmark, not a guess.
- **SMS on hex8 soil itself.** An 8-node trilinear brick has no rotational DOFs, so element SMS lowers the highest element modes (hourglass-like and dilatational short-wavelength modes) while the rigid translations keep their mass. Wave propagation up to about 10–20 Hz with roughly 8–10 elements per wavelength sits in the low part of the element spectrum. The Voet bound suggests moderate β (step gain of about 2–3x) keeps the low modes close, but published SMS verification is almost all crash/forming (impact, shells, quasi-static), not wave propagation or site response.
- **Beams.** LS-DYNA RMSCL and Cocchetti's thin-solid scaling both, in effect, target rotational inertia. For beam-columns, rotary-inertia scaling (next section) is the cheap diagonal version of the same idea.

### Gaps
- I could not open Olovsson & Simonsson 2006 to confirm the iteration counts, the convergence tolerance or the preconditioner type. The figures above come from the search snippet.
- No source found on SMS for wave propagation or site response (frequency-domain accuracy of scaled hex8 meshes at given points per wavelength).
- No GPU implementation of SMS with CG was found in the literature.
- I found no paper by "Frank, Tkachuk et al." on SMS. The name may refer to a different author combination (for example Hoffmann/Tkachuk/Bischoff/Oesterle 2023, or Krauß et al. 2024); unverified.

## Reciprocal (directly constructed sparse inverse) mass matrices

### Takeaway
A reciprocal mass matrix (RMM) builds a sparse approximation of M⁻¹ directly, element by
element, so the explicit update stays a sparse matrix-vector product with no linear solve.
It can be combined with selective scaling (a "selectively scaled RMM"). Reported gains are
moderate. Without scaling, the step gain over the lumped mass is at most about 50 %
according to one survey (one benchmark shows a 48 % lower highest frequency). With Allman
shells the net speedup was 25 %. Large factors need selective scaling on top, and the
construction for hexahedra with constraints and boundaries is less settled than for
simplex elements.

### Cited Findings
- Tkachuk & Bischoff 2015, "Direct and sparse construction of consistent inverse mass matrices: general variational formulation and application to selective mass scaling", IJNME 101(6):435–469, [doi:10.1002/nme.4805](https://doi.org/10.1002/nme.4805). A modified Hamilton principle with displacement and momentum as independent fields. Biorthogonal momentum bases allow the momentum unknowns to be eliminated without inversion, giving the RMM directly. Variational scaling (a penalized Hamilton principle with an extra velocity field) then gives a selectively scaled RMM. The examples use simplex elements — [Zendy abstract](https://zendy.io/title/10.1002/nme.4805)
- Schaeuble, Tkachuk & Bischoff 2017, "Variationally consistent inertia templates for B-spline- and NURBS-based FEM: inertia scaling and customization", CMAME 326:596–621. Schaeuble, Tkachuk & Bischoff 2018, "Time step estimates for explicit dynamics with reciprocal mass matrices", Computers & Structures 202:74–84 — [references in Voet et al.](https://arxiv.org/html/2410.23816v2). Schäuble's 2019 Stuttgart dissertation, "variationally consistent inertia templates", covers the consistency requirements on ansatz spaces, raising the critical step, tailoring the inertia (for example for low-frequency accuracy), and making RMMs more practical — [Uni Stuttgart elib](https://elib.uni-stuttgart.de/items/b0758432-3fd0-46c8-af2f-347b9dc4c43c/full)
- González, Kolman, Cho, Felippa & Park 2018, "Inverse mass matrix via the method of localized Lagrange multipliers", IJNME 113(2):277–295. A kernel inverse mass matrix is built from standard element mass matrices, and boundary conditions are enforced by localized Lagrange multipliers; the accuracy is close to the consistent mass. The 2017 COMPDYN version notes that this avoids the biorthogonal construction, which needs special handling at free edges and boundaries — [idus.us.es record](https://idus.us.es/signposting/describedby/f4bcfed6-ef6f-4ca8-b31a-17da8b30dd4a); [COMPDYN 2017 (core.ac.uk)](https://core.ac.uk/download/222572567.pdf)
- González, Kopačka, Kolman, Cho & Park 2019, isogeometric inverse mass via localized Lagrange multipliers, IJNME 117(9):939–966 ([Voet et al. refs](https://arxiv.org/html/2410.23816v2)). González, Kopačka, Kolman & Park 2021 (IJNME): localized multipliers and a direct sparse inverse mass for bodies in contact, with element-by-element mass scaling to enlarge the stable step — [idus.us.es record](https://idus.us.es/signposting/describedby/60088f7d-e545-467b-9f9e-442d17c92323)
- Tkachuk, Kolman, González, Bischoff, Kopačka et al. 2020/21, "Reciprocal mass matrices and a feasible time step estimator for finite elements with Allman's rotations", IJNME. The standard solid-element RMM construction "fails for Allman-type interpolation". RMM plus their step estimator gave "a moderate speed-up of 25 %" over the diagonal mass — [Uni Stuttgart elib](https://elib.uni-stuttgart.de/items/a26af4f5-c0a3-404f-adef-b6316fbb74bc/full) (abstract via search; full text blocked)
- A Tkachuk–Bischoff paper on variationally selectively scaled RMMs reports a 48 % reduction of the highest frequency relative to the lumped mass in one benchmark (element type not confirmed) — [search snippet, core.ac.uk](https://core.ac.uk/download/79659389.pdf)
- Voet et al. 2024: RMM-based methods "reportedly cannot increase the step size beyond about 50 %" (without further scaling). Their eigenvalue tools do not cover RMMs. They name direct modification of the inverse mass as the promising direction — [arXiv:2410.23816](https://arxiv.org/html/2410.23816v2)
- A 2025 Comput. Mech. paper notes that Cocchetti et al. scaled hexahedral element mass matrices locally, and that scaling the *consistent* mass may not even reach the critical step of the lumped mass — [Springer, Comput. Mech. 2025](https://link.springer.com/article/10.1007/s00466-025-02611-7) (seen only as a search snippet; title and authors not confirmed)
- Higher-order sparse inverse mass matrices: Cimrman, Kolman, González & Park, IJNME 125(11) — [search result listing](https://arxiv.org/pdf/2402.14956v2) (not opened). Geometric construction for tetrahedra: Pitassi, Trevisan & Specogna, [arXiv:2012.01094](https://arxiv.org/abs/2012.01094)
- Lombardo & Askes 2013, "Lumped mass finite element implementation of continuum theories with micro-inertia", IJNME 96(7):448–466. Micro-inertia terms (which raise the critical step) are moved into the residual so the system matrix stays lumped — [Loughborough record](https://dspace.lboro.ac.uk/2134/13580). This is a different route than an RMM, but related.

### Inferences
- **Fit to FemoraX.** An RMM is the most GPU-friendly of the non-diagonal options. The update a = C_RMM·(f_ext − f_int) is one element-by-element (or sparse) product per step, with no iteration and no global reduction. It fits FemoraX's fused gather-compute-scatter kernels and deterministic gather. Constraints (equalDOF, diaphragms, DOF elimination) would need the RMM built on the eliminated DOF space, or the localized-Lagrange-multiplier approach of González et al., which was designed for exactly that partitioned setting.
- The unscaled RMM's step gain (about 1.25–1.5x) is too small to bridge the 2–20x gap between structural members and soil. Its value lies in the *selectively scaled* RMM, which is still research code without a verified hex8 plus beam implementation in production.
- Applying a non-diagonal inverse to every soil DOF doubles or triples the memory traffic of the update on a billion-element mesh. Restricting it to the region near the structure keeps the soil's lumped diagonal.

### Gaps
- No verified RMM construction for trilinear hexahedra combined with 3D beam elements and multipoint constraints was found. Most published examples are simplex, IGA or shells.
- No production code (LS-DYNA, Abaqus, Radioss) was found that uses an RMM. Its maturity is academic.
- I could not open full texts to give the exact step gains per element type.

## Rotary inertia scaling for beams and shells

### Takeaway
In explicit codes with lumped mass, the rotational inertia of beam and shell nodes is commonly
raised artificially so that the bending and rotational modes stop limiting the step and the
axial or membrane wave speed governs. The idea is old and standard, but I could not open the
primary textbook derivations, so the exact factors below are unverified.

### Cited Findings
- When rotational DOFs are used with lumped masses, "a larger stable time step is usually obtained by artificially increasing the rotational inertia" — [Cocchetti et al. 2015 preprint](https://re.public.polimi.it/bitstream/11311/884363/6/11311-884363_Perego.pdf). The same paper warns that a scaling acting on rotational inertia can introduce kinetic-energy errors when rotation dominates.
- A 2021/22 Stuttgart IJNME paper on RMMs for Allman elements states that the rotary inertia scaling in a lumped mass matrix should be adjusted to the element's stabilization stiffness — [Uni Stuttgart NME6583 (search snippet)](https://elib.uni-stuttgart.de/bitstream/11682/14428/1/NME_NME6583.pdf)
- Elastic beam critical step when flexure governs: MOOSE/BlackBear documents a formula built from the shear factor, G, A, ρ and I (after Krieg 1973, Wright 1998, Junior & Rivera 2015). The code takes the minimum of the axial/shear and flexural estimates, and attributes the general definition to Hughes (1987) — [BlackBear docs](https://blackbear.inl.gov/modules/solid_mechanics/CriticalTimeStepMath.html)
- LS-DYNA beam formulations (Hughes-Liu; Belytschko-Schwer resultant and integrated) have their own time-step rules — [ANSYS BEAM161 help](https://mm.bme.hu/~gyebro/files/ans_help_v182/ans_elem/Hlp_E_BEAM161.html); Ansys Explicit Dynamics applies an extra beam time-step safety factor — [Ansys course](https://innovationspace.ansys.com/courses/courses/topics-in-explicit-dynamics-analysis/lessons/time-step-calculation-for-beams/)
- LS-DYNA RMSCL (rotational selective mass scaling), as above — [Borrvall/DYNAmore (search summary)](https://lsdyna.ansys.com/wp-content/uploads/2025/02/borrvall.pdf)

### Inferences
- For the example column (steel, c = √(E/ρ) ≈ 5100–5200 m/s, L = 0.6 m), the axial limit is L/c ≈ 1.2e-4 s, as the brief states. Rotary inertia scaling cannot go below this axial limit: it only removes the flexural and rotational limits, which for a lumped Timoshenko beam with small rotational mass can be much lower. So for FemoraX, rotary inertia scaling is necessary to *reach* about 1e-4 s, but it is not enough to reach the soil's 2e-4 to 2e-3 s.
- Going beyond the axial limit with mass alone needs translational scaling of the members, a factor (dt/1.2e-4)² (4x for 2.4e-4 s, about 70x for 1e-3 s). The other options are LTS or subcycling (in the design document, and LS-DYNA's `*CONTROL_SUBCYCLE`, where steps are power-of-2 multiples: [DYNAmore webinar](https://lsdyna.ansys.com/wp-content/uploads/2025/11/221215_Webinar_Mass-Scaling_and_Subcycling.pdf)), or fewer, longer beam elements. A 3 m column as one element (L/c ≈ 6e-4 s) is still adequate for frame response with an exact-stiffness elastic or force-based beam.
- Rotary inertia scaling is benign for frame response when rotational node inertia is physically negligible next to floor masses, but it changes high-order bending modes and wave transmission along piles. A pile in soil is the case to verify, because there the beam's rotation couples to soil bricks through embedded penalty interfaces.

### Gaps
- I could not access Belytschko, Liu & Moran (*Nonlinear Finite Elements for Continua and Structures*) or Hughes (1987) to quote the standard rotary-inertia scaling factor and its effect on the beam's critical step.
- No quantitative study was found on how much rotary inertia scaling changes the frequencies of piles or frames.

## Interaction with damping and nonlinear materials

### Takeaway
No source directly addresses Rayleigh damping with scaled mass. The notes below are inferences.
On nonlinear materials, the main reported points are: variable or adaptive scaling is used
when stiffness changes (Abaqus), stiffness-proportional SMS is impractical for nonlinear
problems (Voet), and SMS keeps energy balances closer to unscaled runs (LS-DYNA).

### Cited Findings
- Abaqus variable mass scaling recomputes the scaling periodically when "the stiffness properties that control the stable time increment change drastically" — [Abaqus docs](https://abaqus-docs.mit.edu/2017/English/SIMACAEANLRefMap/simaanl-c-massscaling.htm)
- Stiffness-proportional SMS (M + μK) needs K re-formed in nonlinear problems, which is "prohibitively expensive" — [Voet et al. 2024](https://arxiv.org/html/2410.23816v2)
- LS-DYNA example 3: SMS gave lower kinetic energy than CMS and an energy ratio near 1, like a run with no scaling; users should monitor internal energy, kinetic energy and energy ratio — [LS-DYNA SMS how-to](https://lsdyna.ansys.com/?p=3558)
- Radioss AMS: the mass-matrix cost can reach 50 % of the total in highly nonlinear models — [Altair help](https://2025.help.altair.com/2025/hwsolvers/rad/topics/solvers/rad/increase_speed_maintain_accuracy_ams_r.htm)

### Inferences
- Rayleigh αM·M: if M is the scaled mass, mass-proportional damping grows with the added mass, so the damping force rises by the scaling factor at the scaled nodes. With SMS, αM·M_s is non-diagonal and enters the CG solve. FemoraX should form the αM term with the *physical* lumped mass (as a separately stored diagonal). This keeps the specified damping ratio of the physical system and the diagonal structure. The "Rayleigh damping with nodal masses" todo in FemoraX's recent commits is a related decision.
- Stiffness-proportional damping and Uniform damping act through stresses and do not involve M, so they are unaffected by mass scaling. Their effect on the stable step (the c = α + βω² term in the design document's dt_crit formula) still applies, with ω computed from the scaled mass.
- Nonlinear soil: SMS mass is fixed (geometric), while the element stiffness changes with confinement and yielding. FemoraX already re-checks the stable step every N steps (`stable_step_check_interval`), and a scaled region must be included in that re-check. Conventional DT2MS-style scaling computed at stage start may need updating. Abaqus's variable scaling is the model for that, but changing the mass mid-run alters the momentum.
- Gravity stages: mass scaling has no effect on static or quasi-static gravity results, but it changes ground-motion inertia loads (UniformExcitation: f = −M·1·a_g). With SMS, M_s·1 = M_L·1 if the scaling preserves rigid translations (Olovsson, AMS: total mass constant), so base-excitation loads are unchanged. With conventional scaling they grow.

### Gaps
- No published study of mass scaling combined with Rayleigh or frequency-independent damping in earthquake or SSI analysis was found.

## Maturity, production use and reported limitations

### Takeaway
Conventional scaling is universal in commercial codes. SMS has been in production in LS-DYNA
(since V971 R4.2.1) and Radioss AMS for over 15 years, but mainly for crash, forming and
quasi-static work, applied to small parts. Reciprocal mass matrices remain academic. A 2024
theoretical review is sceptical that non-diagonal methods pay off in general.

### Cited Findings
- LS-DYNA SMS requires V971 R4.2.1 or later. It is incompatible with CONTACT_TIED_xxx and CONSTRAINED_SPOTWELD, CONTACT_SPOTWELD switches to the offset formulation, and MAT_SPOTWELD_DA cannot be used on SMS parts — [LS-DYNA SMS how-to](https://lsdyna.ansys.com/?p=3558)
- DYNAmore conclusion: "Too much selective mass-scaling is CPU intensive". Subcycling and multiscale (power-of-2 multiples of the smallest step) are presented as the complement. In the bumper example, subcycling gave 309–358 s against 421 s, and the multiscale run 1290 s against 1859 s — [DYNAmore webinar 2022](https://lsdyna.ansys.com/wp-content/uploads/2025/11/221215_Webinar_Mass-Scaling_and_Subcycling.pdf)
- Radioss AMS: the Courant condition still applies, so stability must be checked at the target step. Compatibility with RBE2/RBE3 came in v14 and with moving rigid walls in v13 — [Altair help](https://2025.help.altair.com/2025/hwsolvers/rad/topics/solvers/rad/increase_speed_maintain_accuracy_ams_r.htm)
- "Mass scaling can irremediably degrade solutions"; non-diagonal methods are competitive only if the saved steps outweigh the solve cost — [Voet et al. 2024](https://arxiv.org/html/2410.23816v2)

### Inferences
- Constraint compatibility is the recurring production limitation (tied contact, spotwelds, rigid elements). For FemoraX, rigid diaphragms and links and embedded penalty interfaces are the analogues, and any SMS or RMM must be built on the post-elimination equation space.
- Ranking for FemoraX's case (structure limits the step at about 1e-4 s, soil at 2e-4 to 2e-3 s), by cost and risk:
  1. Rotary inertia scaling of beams: diagonal, free, needed anyway to reach the axial limit.
  2. Coarser beam discretization or exact-stiffness elements.
  3. LTS or subcycling: no change to the physics, as in the design document's plan.
  4. Conventional scaling only for tiny regions, with reporting.
  5. SMS on the step-limiting block via a small matrix-free CG.
  6. Selectively scaled RMM: research.

  This ranking is my synthesis, not a source's.

### Gaps
- No published use of SMS or RMMs in geotechnical earthquake or SSI codes was found.
- No GPU or Kokkos implementation of SMS or RMM was found.
