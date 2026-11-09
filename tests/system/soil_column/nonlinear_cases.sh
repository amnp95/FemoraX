# The nonlinear (PressureDependMultiYield) reference cases: name, make_nonlinear.py
# arguments after the output directory, and the comparison tolerance on host backends and
# on GPUs (docs/deviations.md, "Nonlinear system tests"). Sourced by nonlinear_check.sh and
# nonlinear_references.sh.
NONLINEAR_CASES=(
  "element_std_implicit|1 1 1 0.5 5e-3 --element stdBrick --soil dense|1e-6|5e-4"
  "element_ssp_implicit|1 1 1 0.5 5e-3 --element SSPbrick --soil dense|5e-5|2e-4"
  "element_ssp_explicit|1 1 1 0.5 5e-4 --element SSPbrick --soil dense --shaking explicit|1e-12|5e-6"
  "element_ssp_loose_explicit|1 1 1 0.5 5e-4 --element SSPbrick --soil loose --shaking explicit|1e-12|5e-6"
  "column4_ssp_explicit_vtkhdf|1 1 4 0.5 5e-4 --element SSPbrick --soil layered --shaking explicit --vtkhdf 0.01|1e-12|5e-6"
)
