# The nonlinear (PressureDependMultiYield) reference cases: name, make_nonlinear.py
# arguments after the output directory, and the comparison tolerance on host backends and
# on GPUs (docs/deviations.md, "Nonlinear system tests"). Sourced by nonlinear_check.sh and
# nonlinear_references.sh.
#
# Every case re-equilibrates with Newton (make_nonlinear.py --equilibration newton), not
# Femora's ModifiedNewton -factoronce: FemoraX runs ModifiedNewton as Newton
# (docs/deviations.md), so OpenSees runs the same algorithm here and the comparison stays at
# rounding. The effect of the deviation itself is measured on Femora's example
# (femora_example/).
NONLINEAR_CASES=(
  "element_std_implicit|1 1 1 0.5 5e-3 --element stdBrick --soil dense --equilibration newton|1e-5|5e-4"
  "element_ssp_implicit|1 1 1 0.5 5e-3 --element SSPbrick --soil dense --equilibration newton|5e-5|2e-4"
  "element_ssp_explicit|1 1 1 0.5 5e-4 --element SSPbrick --soil dense --shaking explicit --equilibration newton|1e-12|5e-6"
  "element_ssp_loose_explicit|1 1 1 0.5 5e-4 --element SSPbrick --soil loose --shaking explicit --equilibration newton|1e-12|5e-6"
  "column4_ssp_explicit_vtkhdf|1 1 4 0.5 5e-4 --element SSPbrick --soil layered --shaking explicit --vtkhdf 0.01 --equilibration newton|1e-12|5e-6"
  # 25 SSPbricks (12.5 m, dense below, loose above), 2 s of explicit shaking (8000 steps),
  # nodes recorded every 0.01 s, VTKHDF every 0.05 s. Here OpenSees' own ModifiedNewton
  # -factoronce re-equilibration also fails (NormUnbalance 1e-7, at t = 30.02).
  "column25_ssp_explicit|1 1 25 2.0 2.5e-4 --element SSPbrick --soil layered --shaking explicit --vtkhdf 0.05 --record-dt 0.01 --equilibration newton|1e-12|5e-6"
)
