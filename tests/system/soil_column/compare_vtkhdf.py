#!/usr/bin/env python3
"""Compare a FemoraX VTKHDF file with the OpenSees one of the same model.

    .venv/bin/python tests/system/soil_column/compare_vtkhdf.py <opensees.vtkhdf> <femorax.vtkhdf> [--tol TOL]

Two checks:

1. Layout (h5py): the same groups, datasets and attributes, with the same element types,
   ranks, chunk shapes and maximum shapes; integer datasets (mesh, offsets, counts, tags)
   and the Version/Type/NSteps attributes equal.
2. femora.results: both files opened with `open_results`; the same times, steps, points,
   cells, responses, mesh (points, cells, cell types), and for every point response every
   frame equal to TOL relative to the response's largest magnitude (default 1e-12, the
   system-test tolerance in docs/deviations.md; float datasets other than responses must
   be identical).

Exit status 1 on any difference.
"""

from __future__ import annotations

import argparse
import sys

import h5py
import numpy as np


def layout(f: h5py.File) -> dict:
    items = {}

    def visit(name, obj):
        attrs = {k: obj.attrs[k] for k in obj.attrs}
        if isinstance(obj, h5py.Dataset):
            items[name] = ("dataset", obj.dtype.str, obj.ndim, obj.chunks, obj.maxshape, attrs)
        else:
            items[name] = ("group", attrs)

    items["/"] = ("group", {k: f.attrs[k] for k in f.attrs})
    f.visititems(visit)
    return items


def same_attrs(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, (bytes, str)) or isinstance(vb, (bytes, str)):
            if (va.decode() if isinstance(va, bytes) else va) != (vb.decode() if isinstance(vb, bytes) else vb):
                return False
        elif not np.array_equal(np.asarray(va), np.asarray(vb)):
            return False
    return True


def check_layout(pa: str, pb: str) -> list[str]:
    errors = []
    with h5py.File(pa, "r") as fa, h5py.File(pb, "r") as fb:
        la, lb = layout(fa), layout(fb)
        for name in sorted(set(la) | set(lb)):
            if name not in la or name not in lb:
                errors.append(f"{name}: only in {'OpenSees' if name in la else 'FemoraX'}")
                continue
            a, b = la[name], lb[name]
            if a[0] != b[0]:
                errors.append(f"{name}: {a[0]} vs {b[0]}")
            elif a[0] == "group":
                if not same_attrs(a[1], b[1]):
                    errors.append(f"{name}: attributes differ: {a[1]} vs {b[1]}")
            else:
                if a[1:5] != b[1:5]:
                    errors.append(f"{name}: dtype/rank/chunks/maxshape {a[1:5]} vs {b[1:5]}")
                if not same_attrs(a[5], b[5]):
                    errors.append(f"{name}: attributes differ")
                da, db = fa[name], fb[name]
                if da.shape != db.shape:
                    errors.append(f"{name}: shape {da.shape} vs {db.shape}")
                elif da.dtype.kind in "iu" and not np.array_equal(da[()], db[()]):
                    errors.append(f"{name}: integer values differ")
                elif da.dtype.kind == "f" and not name.startswith("VTKHDF/PointData/") \
                        and not name.startswith("VTKHDF/CellData/") and not np.array_equal(da[()], db[()]):
                    errors.append(f"{name}: values differ (max {np.abs(da[()] - db[()]).max():.3e})")
    return errors


def check_results(pa: str, pb: str, tol: float, only: set[str] | None = None) -> tuple[list[str], list[str]]:
    from femora.results import open_results

    errors, report = [], []
    with open_results(pa) as ra, open_results(pb) as rb:
        if ra.number_of_steps != rb.number_of_steps:
            errors.append(f"steps {ra.number_of_steps} vs {rb.number_of_steps}")
            return errors, report
        if not np.array_equal(ra.times, rb.times):
            errors.append("times differ")
        if ra.available_point_responses != rb.available_point_responses:
            errors.append(f"responses {ra.available_point_responses} vs {rb.available_point_responses}")
        ma, mb = ra.mesh(), rb.mesh()
        if ma.n_points != mb.n_points or ma.n_cells != mb.n_cells:
            errors.append(f"mesh {ma.n_points}/{ma.n_cells} vs {mb.n_points}/{mb.n_cells}")
            return errors, report
        if not np.array_equal(ma.points, mb.points):
            errors.append("mesh points differ")
        if not np.array_equal(ma.cells, mb.cells) or not np.array_equal(ma.celltypes, mb.celltypes):
            errors.append("mesh cells differ")
        reader_a, reader_b = ra.readers[0], rb.readers[0]
        for resp in ra.available_point_responses:
            fa = np.stack([reader_a.point_frame(resp, k) for k in range(ra.number_of_steps)])
            fb = np.stack([reader_b.point_frame(resp, k) for k in range(rb.number_of_steps)])
            if fa.shape != fb.shape:
                errors.append(f"{resp}: shape {fa.shape} vs {fb.shape}")
                continue
            if fa.dtype.kind in "iu":
                if not np.array_equal(fa, fb):
                    errors.append(f"{resp}: values differ")
                report.append(f"{resp:13s} {str(fa.shape):16s} identical" if np.array_equal(fa, fb) else "")
                continue
            scale = np.abs(fa).max()
            rel = float(np.abs(fa - fb).max() / scale) if scale > 0 else 0.0
            skipped = only is not None and resp not in only
            report.append(f"{resp:13s} {str(fa.shape):16s} max |OpenSees| {scale:.6e}  relative diff {rel:.3e}"
                          + ("  (reported only)" if skipped else ""))
            if rel > tol and not skipped:
                errors.append(f"{resp}: relative difference {rel:.3e} > {tol:.1e}")
        if ra.available_cell_responses != rb.available_cell_responses:
            errors.append(f"cell responses {ra.available_cell_responses} vs {rb.available_cell_responses}")
        for resp in ra.available_cell_responses:
            if resp == "eleTags":
                continue
            fa = np.stack([reader_a.cell_frame(resp, k) for k in range(ra.number_of_steps)])
            fb = np.stack([reader_b.cell_frame(resp, k) for k in range(rb.number_of_steps)])
            if fa.shape != fb.shape:
                errors.append(f"{resp}: shape {fa.shape} vs {fb.shape}")
                continue
            scale = np.abs(fa).max()
            rel = float(np.abs(fa - fb).max() / scale) if scale > 0 else 0.0
            skipped = only is not None and resp not in only
            report.append(f"{resp:13s} {str(fa.shape):16s} max |OpenSees| {scale:.6e}  relative diff {rel:.3e}"
                          + ("  (reported only)" if skipped else ""))
            if rel > tol and not skipped:
                errors.append(f"{resp}: relative difference {rel:.3e} > {tol:.1e}")
        # A history read through the high-level interface, as a user would.
        p = ra.nearest_point(tuple(ma.points[-1]), tolerance=1e-9)
        q = rb.nearest_point(tuple(mb.points[-1]), tolerance=1e-9)
        ha = ra.point_history("acceleration", p, component="x")
        hb = rb.point_history("acceleration", q, component="x")
        if ha.shape != hb.shape:
            errors.append("point_history shapes differ")
    return errors, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("opensees")
    ap.add_argument("femorax")
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--only", default="", help="comma-separated responses to compare; the others are reported")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None
    errors = check_layout(args.opensees, args.femorax)
    print(f"layout: {'identical' if not errors else f'{len(errors)} difference(s)'}")
    for e in errors:
        print("  " + e)
    rerrors, report = check_results(args.opensees, args.femorax, args.tol, only)
    print("femora.results:")
    for line in report:
        if line:
            print("  " + line)
    for e in rerrors:
        print("  FAIL " + e)
    return 1 if errors or rerrors else 0


if __name__ == "__main__":
    sys.exit(main())
