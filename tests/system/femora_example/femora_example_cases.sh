# Femora's nonlinear layered soil column (make_femora_example.py) on a window of its motion:
# the case's arguments after the output directory, and its tolerances (docs/deviations.md,
# "Femora's nonlinear soil column"): displacement, stress and strain relative to their
# largest values; the surface PGA and 5% response spectrum relative to their peaks. Host
# backends, then GPUs. Sourced by femora_example_check.sh and femora_example_reference.sh.
FEMORA_EXAMPLE_ARGS="--start 1.9 --duration 2.0"
FEMORA_EXAMPLE_TOL_HOST="1e-9"
FEMORA_EXAMPLE_TOL_GPU="5e-4"
