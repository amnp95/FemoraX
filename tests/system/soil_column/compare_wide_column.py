#!/usr/bin/env python3
"""Compare a wide soil column (n x n x nz, laminar sides, uniform base motion) with the stored
1 x 1 x nz reference of the same layers: every vertical line of nodes must move as the single
column does.

    .venv/bin/python tests/system/soil_column/compare_wide_column.py <reference.vtkhdf> <wide.vtkhdf> [--tol TOL]
        [--nodes <reference dir> <wide dir>]

Through femora.results: for every frame of the reference, every node of the wide model is
compared with the reference node at the same height (displacement, velocity, acceleration),
and every element with the reference element of the same layer (stress3D6, strain3D6), each
relative to the reference's largest magnitude of that response. With --nodes, the node
recorders' text files (opensees_<r>.txt in the reference directory, femorax_<r>.txt in the
wide one; three vertical lines: a corner, the middle of a side, the centre) are compared
row by row. The wide run may be shorter than the reference: the frames and rows it has
are compared.

Exit status 1 if a difference exceeds TOL (default 1e-12).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    from femora.results import open_results

    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("wide")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--nodes", nargs=2, metavar=("REFDIR", "WIDEDIR"))
    args = ap.parse_args()
    errors = []
    if args.nodes:
        for r in ("disp", "vel", "accel"):
            a = np.loadtxt(f"{args.nodes[0]}/opensees_{r}.txt")
            b = np.loadtxt(f"{args.nodes[1]}/femorax_{r}.txt")
            n = len(b)
            if n == 0 or n > len(a) or not np.allclose(a[:n, 0], b[:, 0], rtol=0, atol=1e-12):
                errors.append(f"{r}: record times differ")
                continue
            scale = np.abs(a[:n, 1:]).max()
            rel = float(np.abs(a[:n, 1:] - b[:, 1:]).max() / scale)
            print(f"  node lines {r:5s} {n} records, largest difference {rel:.3e}")
            if rel > args.tol:
                errors.append(f"node lines {r}: {rel:.3e} > {args.tol:.1e}")
    with open_results(args.reference) as ra, open_results(args.wide) as rb:
        ta, tb = np.asarray(ra.times), np.asarray(rb.times)
        if len(tb) == 0 or len(tb) > len(ta) or not np.allclose(ta[:len(tb)], tb, rtol=0, atol=1e-12):
            print(f"frames differ: {len(ta)} reference, {len(tb)} wide")
            return 1
        ta = ta[:len(tb)]
        ma, mb = ra.mesh(), rb.mesh()
        za, zb = np.round(ma.points[:, 2], 9), np.round(mb.points[:, 2], 9)
        levels = {z: int(np.flatnonzero(za == z)[0]) for z in np.unique(za)}
        missing = set(np.unique(zb)) - set(levels)
        if missing:
            print(f"wide model has heights the reference does not: {sorted(missing)[:5]}")
            return 1
        node_ref = np.array([levels[z] for z in zb])
        # elements: by the height of their centre
        def centres(m):
            cells = m.cells.reshape(-1, 9)[:, 1:]
            return np.round(m.points[cells, 2].mean(axis=1), 9)
        ca, cb = centres(ma), centres(mb)
        clev = {z: int(np.flatnonzero(ca == z)[0]) for z in np.unique(ca)}
        cell_ref = np.array([clev[z] for z in cb])
        reader_a, reader_b = ra.readers[0], rb.readers[0]
        print(f"{mb.n_points} nodes on {len(levels)} levels, {mb.n_cells} elements in {len(clev)} layers, "
              f"{len(ta)} frames")
        for resp in ra.available_point_responses:
            if resp == "nodeTags":
                continue
            worst, scale, where = 0.0, 0.0, None
            for k in range(len(ta)):
                fa = reader_a.point_frame(resp, k)
                fb = reader_b.point_frame(resp, k)
                scale = max(scale, float(np.abs(fa).max()))
                d = np.abs(fb - fa[node_ref]).max(axis=1)
                i = int(d.argmax())
                if d[i] > worst:
                    worst, where = float(d[i]), (k, i)
            rel = worst / scale if scale > 0 else 0.0
            print(f"  {resp:13s} max |reference| {scale:.6e}  largest difference over all nodes {rel:.3e}"
                  f"  (frame {where[0]}, node at z = {mb.points[where[1], 2]:.3f})" if where else "")
            if rel > args.tol:
                errors.append(f"{resp}: {rel:.3e} > {args.tol:.1e}")
        for resp in ra.available_cell_responses:
            if resp == "eleTags":
                continue
            worst, scale = 0.0, 0.0
            for k in range(len(ta)):
                fa = reader_a.cell_frame(resp, k)
                fb = reader_b.cell_frame(resp, k)
                scale = max(scale, float(np.abs(fa).max()))
                worst = max(worst, float(np.abs(fb - fa[cell_ref]).max()))
            rel = worst / scale if scale > 0 else 0.0
            print(f"  {resp:13s} max |reference| {scale:.6e}  largest difference over all elements {rel:.3e}")
            if rel > args.tol:
                errors.append(f"{resp}: {rel:.3e} > {args.tol:.1e}")
    for e in errors:
        print("  FAIL " + e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
