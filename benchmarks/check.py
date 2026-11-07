#!/usr/bin/env python3
"""Run the FemoraX benchmarks and compare them with this machine's baseline.

    python3 benchmarks/check.py                    # build cpu-release, run, compare
    python3 benchmarks/check.py --update-baseline  # accept the current numbers
    python3 benchmarks/check.py --filter stream    # run a subset

Exit status: 0 when nothing is slower than the baseline by more than the tolerance,
1 on a regression, 2 on a usage or build error. Standard library only.
See benchmarks/README.md.

The gate compares each benchmark with the baseline relative to the machine's state: the
FP64 peak and STREAM triad references are measured in the same run (before and after the
benchmarks), and a benchmark's time is divided by how much slower the matching reference
is than at baseline (compute-bound cases by the peak, memory-bound by the triad, from the
roofline of the baseline machine). Thermal throttling or a power mode slows the
references as much as the kernels, so only real code slowdowns fail the gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "benchmarks" / "baselines"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def show(path: Path) -> str:
    """Path relative to the repository when it is inside it, absolute otherwise."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def machine_key(machine: dict) -> str:
    return "-".join(
        [
            slug(machine["host"]),
            slug(machine["cpu"])[:40],
            f'{slug(machine["execution_space"])}{machine["concurrency"]}',
        ]
    )


def run_bench(exe: Path, out: Path, args: argparse.Namespace, filt: str | None) -> dict:
    env = dict(os.environ)
    # Pinned threads give repeatable host timings (and silence Kokkos' warning).
    env.setdefault("OMP_PROC_BIND", "spread")
    env.setdefault("OMP_PLACES", "threads")
    cmd = [str(exe), "--json", str(out), "--reps", str(args.reps), "--warmup", str(args.warmup)]
    if filt:
        cmd += ["--filter", filt]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        sys.exit(2)
    return json.loads(out.read_text())


def by_name(run: dict) -> dict[str, dict]:
    return {r["name"]: r for r in run["results"]}


REFERENCES = ("stream/triad", "peak/fp64_fma")
REF_FILTER = ",".join(REFERENCES)


def machine_state(before: dict, after: dict) -> dict[str, float]:
    """Mean minimum time of each reference over the runs before and after the benchmarks."""
    b, a = by_name(before), by_name(after)
    return {n: 0.5 * (b[n]["min_s"] + a[n]["min_s"]) for n in REFERENCES if n in b and n in a}


def reference_for(r: dict, base: dict) -> str:
    """The reference that bounds a benchmark on the baseline machine (roofline)."""
    bb = by_name(base)
    triad = bb.get("stream/triad", {}).get("gbytes_per_s", 0.0)
    peak = bb.get("peak/fp64_fma", {}).get("gflops", 0.0)
    if r.get("flops", 0) > 0 and r.get("bytes", 0) > 0 and triad > 0 and peak > 0:
        intensity = r["flops"] / r["bytes"]       # flop per byte
        balance = peak / triad                   # flop per byte the machine sustains
        return "peak/fp64_fma" if intensity > balance else "stream/triad"
    return "peak/fp64_fma" if r.get("flops", 0) > 0 and r.get("bytes", 0) == 0 else "stream/triad"


def machine_factor(r: dict, base: dict, state: dict[str, float]) -> float:
    """How much slower the machine is now than at baseline, for this benchmark (1 = same)."""
    ref = reference_for(r, base)
    bb = by_name(base)
    if ref in state and ref in bb and bb[ref]["min_s"] > 0:
        return state[ref] / bb[ref]["min_s"]
    return 1.0


def normalized_change(r: dict, b: dict, base: dict, state: dict[str, float]) -> float:
    return r["min_s"] / (b["min_s"] * machine_factor(r, base, state)) - 1.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--preset", default="cpu-release", help="CMake preset (build/<preset>)")
    ap.add_argument("--tolerance", type=float, default=0.10,
                    help="allowed slowdown of the minimum time, as a fraction (default 0.10)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--baseline", type=Path, default=None, help="override the baseline file")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write the current results as the baseline")
    ap.add_argument("--no-build", action="store_true", help="do not rebuild femorax_bench")
    args = ap.parse_args()

    build_dir = ROOT / "build" / args.preset
    if not args.no_build:
        b = subprocess.run(["cmake", "--build", "--preset", args.preset, "--target", "femorax_bench"],
                           cwd=ROOT, capture_output=True, text=True)
        if b.returncode != 0:
            sys.stderr.write(b.stdout + b.stderr)
            return 2
    exe = build_dir / "benchmarks" / "femorax_bench"
    if not exe.exists():
        sys.stderr.write(f"{exe} not found; configure with: cmake --preset {args.preset}\n")
        return 2

    out_dir = build_dir / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    refs_before = run_bench(exe, out_dir / "refs_before.json", args, REF_FILTER)
    current = run_bench(exe, out_dir / "last.json", args, args.filter)
    refs_after = run_bench(exe, out_dir / "refs_after.json", args, REF_FILTER)
    state = machine_state(refs_before, refs_after)
    # The reported reference rows are the mean machine state, not the single in-run sample.
    cur_by = by_name(current)
    for n in REFERENCES:
        if n in by_name(refs_before):
            r = dict(by_name(refs_before)[n])
            r["min_s"] = state.get(n, r["min_s"])
            r["gbytes_per_s"] = r["bytes"] / r["min_s"] * 1e-9 if r["min_s"] > 0 else 0.0
            r["gflops"] = r["flops"] / r["min_s"] * 1e-9 if r["min_s"] > 0 else 0.0
            cur_by[n] = r
    current["results"] = sorted(cur_by.values(), key=lambda r: r["name"])

    build_type = current["machine"]["build_type"]
    if build_type not in ("Release", "RelWithDebInfo"):
        sys.stderr.write(f"refusing to compare a {build_type} build; use a Release preset\n")
        return 2

    key = machine_key(current["machine"])
    baseline_path = args.baseline or BASELINE_DIR / f"{key}.json"

    if args.update_baseline or not baseline_path.exists():
        if baseline_path.exists() and args.filter:
            # Merge a partial run into the existing baseline.
            base = json.loads(baseline_path.read_text())
            merged = by_name(base)
            merged.update(by_name(current))
            current["results"] = sorted(merged.values(), key=lambda r: r["name"])
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2) + "\n")
        what = "updated" if args.update_baseline else "created (no baseline existed)"
        print(f"baseline {what}: {show(baseline_path)}")
        print_report(current, None, args.tolerance, {})
        return 0

    base = json.loads(baseline_path.read_text())
    regressed = compare(current, base, args.tolerance, state)
    if regressed:
        # Re-run only the regressed cases once, keeping the faster result, so one noisy run
        # does not fail the check.
        rerun = run_bench(exe, out_dir / "rerun.json", args, ",".join(regressed))
        fresh = by_name(rerun)
        cur = by_name(current)
        for name in regressed:
            if name in fresh and fresh[name]["min_s"] < cur[name]["min_s"]:
                cur[name] = fresh[name]
        current["results"] = list(cur.values())
        (out_dir / "last.json").write_text(json.dumps(current, indent=2) + "\n")
        regressed = compare(current, base, args.tolerance, state)

    print(f"baseline: {show(baseline_path)}")
    bb = by_name(base)
    for n in REFERENCES:
        if n in state and n in bb:
            print(f"machine state: {n} {state[n] / bb[n]['min_s'] - 1.0:+.1%} against the baseline")
    print_report(current, base, args.tolerance, state)
    if regressed:
        print(f"\nREGRESSION: {', '.join(regressed)} slower than baseline by more than "
              f"{args.tolerance:.0%} relative to the machine state")
        return 1
    print("\nno regressions")
    return 0


def compare(current: dict, base: dict, tol: float, state: dict[str, float]) -> list[str]:
    """Gated cases slower than the baseline by more than tol, relative to the machine state.
    The references themselves define the state and are not gated."""
    b = by_name(base)
    return [r["name"] for r in current["results"]
            if r.get("gated", True) and r["name"] in b and r["name"] not in REFERENCES
            and normalized_change(r, b[r["name"]], base, state) > tol]


def print_report(current: dict, base: dict | None, tol: float, state: dict[str, float]) -> None:
    cur = by_name(current)
    triad = cur.get("stream/triad", {}).get("gbytes_per_s", 0.0)
    peak = cur.get("peak/fp64_fma", {}).get("gflops", 0.0)
    b = by_name(base) if base else {}
    print(f"{'benchmark':44s} {'min [us]':>11s} {'GB/s':>8s} {'%triad':>7s} {'GFLOP/s':>8s}"
          f" {'%peak':>6s} {'base [us]':>11s} {'raw':>8s} {'vs ref':>8s}")
    for name in sorted(cur):
        r = cur[name]
        frac = f"{100 * r['gbytes_per_s'] / triad:6.1f}%" if triad > 0 and r["bytes"] else "     - "
        pk = f"{100 * r['gflops'] / peak:5.1f}%" if peak > 0 and r["flops"] else "    - "
        if name in b:
            raw = r["min_s"] / b[name]["min_s"] - 1.0
            if name in REFERENCES:
                tail = f" {b[name]['min_s'] * 1e6:11.2f} {raw:+7.1%} {'':>8s}  (machine reference)"
            else:
                norm = normalized_change(r, b[name], base, state)
                ref = "peak" if reference_for(r, base) == "peak/fp64_fma" else "triad"
                gated = r.get("gated", True)
                flag = ("  <-- slower" if gated else "  (slower, not gated)") if norm > tol else ""
                if not gated and not flag:
                    flag = "  (not gated)"
                tail = f" {b[name]['min_s'] * 1e6:11.2f} {raw:+7.1%} {norm:+7.1%} {ref}{flag}"
        else:
            tail = f" {'new':>11s}"
        print(f"{name:44s} {r['min_s'] * 1e6:11.2f} {r['gbytes_per_s']:8.2f} {frac} "
              f"{r['gflops']:8.2f} {pk}{tail}")
    for name in sorted(set(b) - set(cur)):
        print(f"{name:36s} (in baseline, not run)")


if __name__ == "__main__":
    sys.exit(main())
