#!/usr/bin/env python3
"""Check FemoraX's single-precision VTKHDF output against the double-precision file of the
same run.

    .venv/bin/python tests/system/soil_column/compare_single_precision.py <double.vtkhdf> <single.vtkhdf>

`femorax run --vtkhdf-precision single` writes the response datasets (PointData
displacement / velocity / acceleration, CellData stress3D6 / strain3D6) as float32 and
leaves everything else as in the double file. Checks: the same datasets, attributes, ranks,
chunks and maximum shapes; the response datasets float32 and each value equal to the
double value rounded to float32 (round to nearest); every other dataset identical, type
included; and femora.results opens the single file with the same times, mesh and
responses. Exit status 1 on any difference.
"""

from __future__ import annotations

import sys

import h5py
import numpy as np

RESPONSES = {"VTKHDF/PointData/displacement", "VTKHDF/PointData/velocity", "VTKHDF/PointData/acceleration",
             "VTKHDF/CellData/stress3D6", "VTKHDF/CellData/strain3D6"}


def items(f: h5py.File) -> dict:
    out = {}
    f.visititems(lambda n, o: out.__setitem__(n, o))
    return out


def main() -> int:
    pd, ps = sys.argv[1], sys.argv[2]
    errors = []
    seen = 0
    with h5py.File(pd, "r") as fd, h5py.File(ps, "r") as fs:
        a, b = items(fd), items(fs)
        if a.keys() != b.keys():
            errors.append(f"different objects: {sorted(a.keys() ^ b.keys())}")
        for name in sorted(a.keys() & b.keys()):
            x, y = a[name], b[name]
            if set(x.attrs) != set(y.attrs) or any(not np.array_equal(x.attrs[k], y.attrs[k]) for k in x.attrs):
                errors.append(f"{name}: attributes differ")
            if not isinstance(x, h5py.Dataset):
                continue
            if (x.ndim, x.chunks, x.maxshape, x.shape) != (y.ndim, y.chunks, y.maxshape, y.shape):
                errors.append(f"{name}: shape or chunks differ")
                continue
            if name in RESPONSES:
                seen += 1
                if x.dtype != np.float64 or y.dtype != np.float32:
                    errors.append(f"{name}: types {x.dtype} / {y.dtype}, expected float64 / float32")
                elif not np.array_equal(x[()].astype(np.float32), y[()]):
                    errors.append(f"{name}: values are not the double values rounded to float32")
            elif x.dtype != y.dtype or not np.array_equal(x[()], y[()]):
                errors.append(f"{name}: differs")
    if seen == 0:
        errors.append("no response dataset in the files")

    from femora.results import open_results
    frames = 0
    with open_results(pd) as rd, open_results(ps) as rs:
        frames = rs.number_of_steps
        if rd.number_of_steps != frames or not np.array_equal(rd.times, rs.times):
            errors.append("femora.results: steps or times differ")
        if (rd.available_point_responses != rs.available_point_responses
                or rd.available_cell_responses != rs.available_cell_responses):
            errors.append("femora.results: responses differ")
        a_, b_ = rd.readers[0], rs.readers[0]
        for resp in rd.available_point_responses:
            for k in range(frames):
                if not np.array_equal(np.asarray(a_.point_frame(resp, k)).astype(np.float32),
                                      np.asarray(b_.point_frame(resp, k)).astype(np.float32)):
                    errors.append(f"femora.results: {resp} frame {k} differs")
                    break
        for resp in rd.available_cell_responses:
            if resp == "eleTags":
                continue
            for k in range(frames):
                if not np.array_equal(np.asarray(a_.cell_frame(resp, k)).astype(np.float32),
                                      np.asarray(b_.cell_frame(resp, k)).astype(np.float32)):
                    errors.append(f"femora.results: {resp} frame {k} differs")
                    break
    for e in errors:
        print("FAIL", e)
    if not errors:
        print(f"single precision: {seen} response datasets float32 and correctly rounded; the rest identical; "
              f"femora.results reads all {frames} frames")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
