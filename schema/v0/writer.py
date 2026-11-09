"""Reference writer for FemoraX model files, schema v0 (schema/v0/README.md).

A model is a tree of plain dicts:

    node = {"attrs": {name: value}, "data": {name: array}, "groups": {name: node}}

write(path, tree) writes it with the types the schema requires and refuses anything else,
so a mistake shows up here rather than as a reader error. Values must already have their
schema type: numpy int32 / uint8 / float64 scalars or arrays, or str.

Used by the examples and the reader tests; Femora's exporter can follow the same rules.
"""

from __future__ import annotations

import h5py
import numpy as np

SCHEMA_VERSION = (0, 6)

_ALLOWED = (np.int32, np.uint8, np.float64)


def i32(x):
    """int32 scalar or array."""
    return np.asarray(x, dtype=np.int32) if np.ndim(x) else np.int32(x)


def f64(x):
    """float64 scalar or array."""
    return np.asarray(x, dtype=np.float64) if np.ndim(x) else np.float64(x)


def u8(x):
    """uint8 scalar (booleans)."""
    return np.uint8(1 if x else 0)


def node(attrs=None, data=None, groups=None) -> dict:
    return {"attrs": dict(attrs or {}), "data": dict(data or {}), "groups": dict(groups or {})}


def _check_value(where: str, value) -> None:
    if isinstance(value, str):
        return
    dtype = getattr(value, "dtype", None)
    if dtype is None or dtype.type not in _ALLOWED:
        raise TypeError(f"{where}: {type(value).__name__} {getattr(value, 'dtype', '')} is not "
                        "a schema type (int32, uint8, float64 or str)")


def _write_node(h: h5py.Group, tree: dict, path: str) -> None:
    for name, value in tree.get("attrs", {}).items():
        _check_value(f"{path} attribute {name}", value)
        h.attrs[name] = value
    for name, value in tree.get("data", {}).items():
        _check_value(f"{path}/{name}", value)
        h.create_dataset(name, data=value)
    for name, child in tree.get("groups", {}).items():
        _write_node(h.create_group(name), child, f"{path}/{name}")


def write(path, tree: dict) -> None:
    """Write a model tree, in the HDF5 1.10 file format FemoraX's library reads."""
    with h5py.File(path, "w", libver=("earliest", "v110")) as f:
        _write_node(f, tree, "")


def root_attrs(model_name: str, generator: str = "femorax schema/v0/writer.py") -> dict:
    return {
        "schema_version": i32(list(SCHEMA_VERSION)),
        "ndm": i32(3),
        "generator": generator,
        "model_name": model_name,
    }
