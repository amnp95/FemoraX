# Benchmarks

Performance is the top priority after correctness. Every change is checked against a
per-machine baseline, so slowdowns are caught at the change that causes them.

## Run

```
cmake --preset cpu-release                # once
python3 benchmarks/check.py               # builds femorax_bench, runs it, compares
```

`check.py` exits 0 when no gated benchmark is slower than the baseline by more than the
tolerance (default 10% on the minimum time over 20 runs), and 1 on a regression. Gated cases
are the paths that run: each backend's default layout, the deterministic (colored) form of
it, and the STREAM reference. Other layout variants are measured and reported, marked
"not gated", because they decide `src/exec/layout_defaults.hpp` but do not run in
production (on CPU the team-layout times vary by 40-60% between runs). A regressed case is
re-run once and the faster result kept, so a single noisy run does not fail the check. It
refuses Debug builds. Results of the last run are in `build/<preset>/bench/last.json`.

Useful options: `--filter <text>`, `--reps N`, `--tolerance 0.05`, `--no-build`,
`--baseline <file>`, `--update-baseline`.

Threads are pinned (`OMP_PROC_BIND=spread`, `OMP_PLACES=threads`) unless already set.
Close heavy programs before a run; on a laptop, keep it on mains power.

## What is measured

For each benchmark, per call (a kernel launch) or per step (a whole time step):

- **time**: minimum, median and maximum over the timed runs, each bracketed by fences, so
  asynchronous device launches are timed completely;
- **achieved bandwidth**: `bytes / minimum time`, where `bytes` is the **compulsory
  traffic** counted from the data structures (every array the kernel must read or write,
  once). It is a model, not a hardware counter: it leaves out cache misses and
  write-allocate traffic, so a kernel near the triad bandwidth is close to the hardware
  limit;
- **%triad**: achieved bandwidth as a fraction of `stream/triad` on the same run, the
  practical memory-bandwidth ceiling of the machine. Bandwidth-bound kernels (most explicit
  FEM kernels) are judged by this number;
- **GFLOP/s**, when the benchmark counts its floating-point operations.

## Baselines

`baselines/<host>-<cpu>-<space><threads>.json`, one per machine and backend, committed.
A baseline is created on the first run on a new machine.

Update a baseline (`--update-baseline`) only:

- when a change makes things faster, or
- when a change is knowingly slower (more physics, a correctness fix), with the reason and
  the before/after numbers in the commit message.

Never update a baseline to make a regression disappear.

## Adding a benchmark

Every new kernel comes with its benchmark in the same change.

1. Write `bench_<area>.cpp` with a `register_<area>(Registry&, const Options&)` function
   that adds one `Benchmark` per case: a unique `group/case` name, `unit` (`call` or
   `step`), the compulsory `bytes` (and `flops`, `items` when known), and a `run`
   callable that launches the kernel once. The kernel itself lives in `src/exec/`;
   benchmark code never uses Kokkos directly.
2. Declare the function in `harness.hpp`, call it in `bench_main.cpp`, and add the file to
   `CMakeLists.txt`.
3. Size the problem so its arrays do not fit in cache; give `--quick` a small size for the
   `bench.smoke` test.
4. Run `check.py --filter <group> --update-baseline` to add the new cases to the baseline.

Current:

- `stream/*`: copy and triad, the bandwidth reference.
- `explicit_force/<n>^3/<layout>_<scatter>[_ept<k>]`: stdBrick + ElasticIsotropic internal
  and Rayleigh damping force on an n x n x n Femora-style soil column (renumbering on), in
  every layout and scatter mode; team variants for each elements-per-team that fits the
  backend's scratch. The fastest variant per backend goes into
  `src/exec/layout_defaults.hpp`.

Planned for milestone 1a: SSPbrick, the explicit step (time per step and DOFs per second),
and the recorders' per-step cost.
