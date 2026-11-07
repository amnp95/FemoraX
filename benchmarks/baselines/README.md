# Baselines (TACC only)

One file per TACC machine, node type and backend, written by
`python3 benchmarks/check.py --no-build --preset tacc-<machine>-<node> --update-baseline`
after a `benchmarks/run_all.sh` run (benchmarks/README.md). No laptop baselines: laptop
timings are not representative.
