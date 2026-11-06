#!/usr/bin/env python3
"""Compare FemoraX and OpenSees node-recorder output, value by value.

    .venv/bin/python tests/system/soil_column/compare.py <dir> [--femorax PREFIX] [--tol TOL]

Reads <dir>/opensees_{disp,vel,accel}.txt and <dir>/<PREFIX>_{disp,vel,accel}.txt (PREFIX
defaults to femorax). Checks that both recorded the same steps at the same times, then for
each response reports the largest difference relative to that response's largest magnitude,
and where it occurs. Exit status 1 if any relative difference exceeds TOL (default 1e-12,
the system-test tolerance in docs/deviations.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--femorax", default="femorax")
    ap.add_argument("--tol", type=float, default=1e-12)  # docs/deviations.md, system tests
    args = ap.parse_args()

    worst = 0.0
    print(f"{'response':9s} {'steps':>6s} {'values':>10s} {'max |OpenSees|':>15s} {'max |diff|':>11s} "
          f"{'relative':>10s}  where (step, column)")
    for resp in ("disp", "vel", "accel"):
        a = np.loadtxt(args.dir / f"opensees_{resp}.txt", ndmin=2)
        b = np.loadtxt(args.dir / f"{args.femorax}_{resp}.txt", ndmin=2)
        if a.shape != b.shape:
            print(f"{resp}: shapes differ, OpenSees {a.shape} vs FemoraX {b.shape}")
            return 1
        if not np.array_equal(a[:, 0], b[:, 0]):
            k = int(np.argmax(a[:, 0] != b[:, 0]))
            print(f"{resp}: record times differ first at row {k}: {a[k, 0]!r} vs {b[k, 0]!r}")
            return 1
        va, vb = a[:, 1:], b[:, 1:]
        scale = np.abs(va).max()
        d = np.abs(va - vb)
        k = np.unravel_index(int(np.argmax(d)), d.shape)
        rel = float(d.max() / scale) if scale > 0 else 0.0
        worst = max(worst, rel)
        print(f"{resp:9s} {a.shape[0]:6d} {va.size:10d} {scale:15.6e} {d.max():11.3e} {rel:10.3e}  {k}")
    print(f"largest relative difference: {worst:.3e}")
    if args.tol is not None and worst > args.tol:
        print(f"FAIL: exceeds the tolerance {args.tol:.1e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
