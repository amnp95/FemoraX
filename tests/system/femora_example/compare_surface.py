#!/usr/bin/env python3
"""Compare the surface acceleration of two runs by its peak and its response spectrum.

    .venv/bin/python tests/system/femora_example/compare_surface.py <opensees_surface_accel.txt>
        <femorax_surface_accel.txt> <motion.acc> <motion.time> <factor> [--tol TOL]

The node recorders give the x acceleration relative to the base (UniformExcitation); the
absolute acceleration adds the base motion, factor * the Path series at the record's times.
Compared: the peak absolute acceleration (PGA at the surface), and the pseudo-spectral
acceleration at 5% damping over 100 periods from 0.02 to 5 s (an elastic oscillator
integrated exactly for an acceleration linear between samples, Nigam and Jennings 1969),
each relative to the OpenSees peak. Step-by-step accelerations depend on the convergence
test of the implicit stage (a displacement tolerance of 1e-5 m is up to 40 m/s^2 of
acceleration at dt = 0.001), so they are not compared directly.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def absolute(path: str, motion_t: np.ndarray, motion_a: np.ndarray, factor: float) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(path)
    t, rel = d[:, 0], d[:, 1]
    return t, rel + factor * np.interp(t, motion_t, motion_a)


def spectrum(t: np.ndarray, a: np.ndarray, periods: np.ndarray, xi: float = 0.05) -> np.ndarray:
    """Pseudo-spectral acceleration: omega^2 max|u| of u'' + 2 xi w u' + w^2 u = -a(t)."""
    dt = float(np.mean(np.diff(t)))
    if np.max(np.abs(np.diff(t) - dt)) > 1e-9:
        raise SystemExit("compare_surface: the record needs a uniform time step")
    out = np.empty(len(periods))
    for i, T in enumerate(periods):
        w = 2 * np.pi / T
        wd = w * np.sqrt(1 - xi * xi)
        e = np.exp(-xi * w * dt)
        s, c = np.sin(wd * dt), np.cos(wd * dt)
        # Nigam-Jennings coefficients (piecewise-linear excitation).
        a11 = e * (xi / np.sqrt(1 - xi * xi) * s + c)
        a12 = e / wd * s
        a21 = -w / np.sqrt(1 - xi * xi) * e * s
        a22 = e * (c - xi / np.sqrt(1 - xi * xi) * s)
        t1 = (2 * xi * xi - 1) / (w * w * dt)
        t2 = 2 * xi / (w ** 3 * dt)
        b11 = e * ((t1 + xi / w) * s / wd + (t2 + 1 / (w * w)) * c) - t2
        b12 = -e * (t1 * s / wd + t2 * c) - 1 / (w * w) + t2
        b21 = e * ((t1 + xi / w) * (c - xi / np.sqrt(1 - xi * xi) * s)
                   - (t2 + 1 / (w * w)) * (wd * s + xi * w * c)) + 1 / (w * w * dt)
        b22 = -e * (t1 * (c - xi / np.sqrt(1 - xi * xi) * s) - t2 * (wd * s + xi * w * c)) - 1 / (w * w * dt)
        u = v = 0.0
        peak = 0.0
        for k in range(len(a) - 1):
            u, v = (a11 * u + a12 * v + b11 * a[k] + b12 * a[k + 1],
                    a21 * u + a22 * v + b21 * a[k] + b22 * a[k + 1])
            peak = max(peak, abs(u))
        out[i] = w * w * peak
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("opensees")
    ap.add_argument("femorax")
    ap.add_argument("motion_acc")
    ap.add_argument("motion_time")
    ap.add_argument("factor", type=float)
    ap.add_argument("--tol", type=float, default=1e-12)
    args = ap.parse_args()
    mt = np.loadtxt(args.motion_time).ravel()
    ma = np.loadtxt(args.motion_acc).ravel()
    ta, aa = absolute(args.opensees, mt, ma, args.factor)
    tb, ab = absolute(args.femorax, mt, ma, args.factor)
    errors = []
    if len(ta) != len(tb) or not np.array_equal(ta, tb):
        print(f"surface: record times differ ({len(ta)} and {len(tb)} records)")
        return 1
    pga_a, pga_b = np.abs(aa).max(), np.abs(ab).max()
    periods = np.logspace(np.log10(0.02), np.log10(5.0), 100)
    sa, sb = spectrum(ta, aa, periods), spectrum(tb, ab, periods)
    d_pga = abs(pga_a - pga_b) / pga_a
    d_sa = float(np.abs(sa - sb).max() / sa.max())
    k = int(np.argmax(sa))
    print(f"surface PGA: OpenSees {pga_a:.6e}, FemoraX {pga_b:.6e} m/s^2, relative difference {d_pga:.3e}")
    print(f"surface spectrum (5%, 100 periods 0.02-5 s): peak {sa[k]:.6e} m/s^2 at T = {periods[k]:.3f} s, "
          f"largest difference {d_sa:.3e} of the peak")
    if d_pga > args.tol:
        errors.append(f"PGA difference {d_pga:.3e} > {args.tol:.1e}")
    if d_sa > args.tol:
        errors.append(f"spectrum difference {d_sa:.3e} > {args.tol:.1e}")
    for e in errors:
        print("  FAIL " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
