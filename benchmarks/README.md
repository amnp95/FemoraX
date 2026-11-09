# Benchmarks

Performance is the top priority after correctness. Every change is checked against a
per-machine baseline, so slowdowns are caught at the change that causes them.

## Where and when

Benchmarks never block a commit and never run in `scripts/check.sh`. Performance is
measured on TACC only (Vista, Stampede3); laptop timings are not representative (thermal
throttling, weak FP64 on consumer GPUs). The suite still builds everywhere, and the
`bench.smoke` test runs it once on tiny models as a correctness check.

## Run on TACC

`benchmarks/run_all.sh` detects the machine (Vista or Stampede3) and the node type, loads
the modules, builds FemoraX in `build/tacc-<machine>-<node>`, runs the full suite on the
40^3, 64^3 and 96^3 soil columns, and writes `benchmarks/results/<machine>-<node>-<date>-
<commit>/` (`meta.json` with the commit, machine, node, CPU, GPU, modules and date, and one
result file per size). It submits no jobs.

```
# 1. Get a compute node (interactive):
idev -p gh-dev -N 1 -n 1 -t 01:00:00            # Vista Grace-Hopper (H200)
idev -p gg -N 1 -n 1 -t 01:00:00                # Vista Grace-Grace (CPU)
idev -p h100 -N 1 -n 1 -t 01:00:00              # Stampede3 H100 nodes
idev -p spr -N 1 -n 1 -t 01:00:00               # Stampede3 Sapphire Rapids (CPU)
# 2. Build and run (on the compute node, so the build targets its CPU and GPU):
cd $WORK/FemoraX
benchmarks/run_all.sh
benchmarks/run_all.sh --no-build --sizes "40 128" --reps 10   # rerun, other sizes

# 3. Compare with the baseline of the same machine and node type (or record one):
python3 benchmarks/check.py --no-build --preset tacc-vista-gh
python3 benchmarks/check.py --no-build --preset tacc-vista-gh --update-baseline
```

Modules default to `gcc cuda cmake hdf5` (GPU nodes) or `gcc cmake hdf5` (CPU nodes);
override with `FEMORAX_MODULES="gcc/13.2.0 cuda/12.6 cmake hdf5"`. The configure step
downloads Kokkos and GoogleTest; if the compute node has no internet access, run
`benchmarks/run_all.sh --build-only` once on a login node first (it fetches the sources;
the build on the compute node then reuses them). Check the names with
`module spider` on the machine. Intel GPU (Stampede3 `pvc`) nodes are not supported yet.

`check.py` compares one run (40^3 by default) with `baselines/<host>-<cpu>-<space><threads>.json`
for the same machine; baselines are recorded on TACC only. It reports each case relative
to the machine state measured in the same run (below), and exits 1 when a gated case is
slower than the tolerance (default 10%), as information: nothing blocks a commit.

**Machine state.** The FP64 peak (`peak/fp64_fma`) and STREAM triad (`stream/triad`)
references run before and after the benchmarks; their mean time against the baseline is
the machine's state. Each benchmark's time is divided by the slowdown of its bounding
reference (peak for compute-bound, triad for memory-bound, from the roofline of the
baseline machine or the benchmark's declared `bound`), and the report shows the raw change
and the change relative to the reference ("vs ref"). A case over the tolerance is re-run
once, after a pause (`--cooldown`, default 30 s), next to freshly measured references.

Gated cases are the paths that run in production: each backend's default layout and its
deterministic form, and the default operator and solvers. Other variants are reported,
marked "not gated", because they decide the defaults.

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

## Baselines (TACC only)

`baselines/<host>-<cpu>-<space><threads>.json`, one per TACC machine, node type and
backend, committed. A baseline is created on the first run on a new machine.

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
   callable that launches the kernel once. Solver kernels live in `src/exec/`; reference
   kernels used only by benchmarks (STREAM, FP64 peak, timing) live in
   `benchmarks/kernels/`, the only benchmark code that uses Kokkos directly.
2. Declare the function in `harness.hpp`, call it in `bench_main.cpp`, and add the file to
   `CMakeLists.txt`.
3. Size the problem so its arrays do not fit in cache; give `--quick` a small size for the
   `bench.smoke` test.
4. Run `check.py --filter <group> --update-baseline` to add the new cases to the baseline.

Current:

- `stream/*`: copy and triad, the bandwidth reference.
- `solvers/dot`, `solvers/axpy`: the solver vector kernels on 2^25 entries (the
  deterministic chunked dot product, and axpy), judged by %triad.
- `solvers/spmv/<n>^3`: CSR y = A x on the n x n x n soil column's pattern (one thread
  per row), judged by %triad. Not gated: assembled operators exist only in tests; it is
  the reference for the matrix-free operator.
- `operator/<n>^3/<scatter>_<geometry>_<tangent>`: the matrix-free implicit operator
  y = K_eff x (stiffness, Rayleigh damping, consistent mass) in every scatter, geometry and
  tangent-source variant; the default is gated. Compare with `solvers/spmv/<n>^3`.
- `solvers/cg20/<n>^3`: 20 CG iterations with Jacobi on a Newmark step (dt = 1e-3) with
  the default operator: the implicit solver's cost per solve; `solvers/gmres20/<n>^3`, the
  same with GMRES(30), the cost of a non-symmetric stage; `solvers/cg20_chebyshev4/<n>^3`,
  CG with the degree-4 Chebyshev-Jacobi preconditioner.
- `explicit_force/<n>^3/<layout>_<scatter>[_ept<k>]`: stdBrick + ElasticIsotropic internal
  and Rayleigh damping force on an n x n x n Femora-style soil column (renumbering on), in
  every layout and scatter mode; team variants for each elements-per-team that fits the
  backend's scratch. The fastest variant per backend goes into
  `src/exec/force_settings.cpp`.
- `explicit_force_ssp/<n>^3/<variant>` and `operator_ssp/<n>^3/<variant>`: the same with
  SSPbrick (one integration point, lumped mass, the hourglass term `Kstab` from the element
  constants, cached with the geometry, or recomputed). FLOPs are not counted for SSPbrick
  yet (`bound` = compute). Laptop orientation, 40^3, gather scatter: explicit force cached
  11.9 ms (stdBrick 19.1 ms) on 8 threads, 1.67 ms (stdBrick 6.2 ms) on the RTX 3050 Ti;
  recomputed 47 ms / 10.7 ms (the stabilization's setup runs in every call); operator
  7.6 ms / 1.10 ms (stdBrick, consistent mass: 18.9 ms / 4.8 ms). Not baselines.
- `explicit_force_pdmy[_ssp]/<n>^3/<mix>/<variant>`: PressureDependMultiYield (dense below,
  loose above, plastic stage) under a shear field, `elastic` (no point yields), `mixed`
  (the upper part yields: 56% of the points) and `plastic` (every point); the gather
  cached variants, fused and with the compacted plastic corrector
  (`src/exec/compacted_kernels.hpp`). The run prints the number of plastic points per mix.
  Compulsory traffic adds the material state (committed read, trial written) and, for the
  compacted path, 48 doubles per point written and read, plus a flag and a list entry.
  Laptop orientation, 16^3 (32768 points for stdBrick), RTX 3050 Ti, fused team ept4
  against compacted team ept4: elastic 8.5 / 9.1 ms, mixed 18.3 / 22.3 ms, plastic
  34.1 / 40.3 ms; SSPbrick (flat or ept8) 1.9 / 1.8, 4.1 / 4.5, 5.5 / 6.3 ms. On this card
  the compacted path does not pay: the fused team layout already runs one thread per point,
  and compaction adds a scan, a count copied to the host, the per-point results and a
  second predictor pass in the corrector. CPU (8 threads): no gain either, as expected
  without warp divergence. The question for TACC is the H100/H200 balance. Not baselines.

Planned: the explicit step (time per step and DOFs per second) and the recorders' per-step
cost.
