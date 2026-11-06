#!/usr/bin/env python3
"""Write the model-reader test fixtures and their manifest.

    .venv/bin/python tests/io/make_fixtures.py out_dir

Each case starts from a valid model (schema/v0/make_example.py), applies one change with
h5py, and records what the reader must do with it:

    valid         reads, and passes check_supported
    warning       reads with a warning containing the text
    invalid       ModelError whose message contains the text
    unsupported   UnsupportedComponent whose message contains the text

manifest.tsv: name <TAB> expectation <TAB> text <TAB> file. CTest runs this script as a
fixture before the reader tests (tests/CMakeLists.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "schema" / "v0"))
import make_example  # noqa: E402
import writer as w  # noqa: E402


# ---------------------------------------------------------------------------
# Small h5py helpers for the mutations
# ---------------------------------------------------------------------------
def replace(group: h5py.Group, name: str, data) -> None:
    del group[name]
    group.create_dataset(name, data=data)


def first_element_group(f: h5py.File) -> h5py.Group:
    return f["elements"][sorted(f["elements"])[0]]


def explicit_step(f: h5py.File) -> h5py.Group:
    for name in sorted(f["process"]):
        g = f["process"][name]
        if g.attrs["kind"] == "analysis":
            return g
    raise RuntimeError("no analysis step")


def add_time_series_dt(f: h5py.File) -> None:
    g = f["time_series"]["1"]
    g.attrs["dt"] = np.float64(0.005)


# ---------------------------------------------------------------------------
# Cases: (name, base variant, expectation, text, mutation)
# ---------------------------------------------------------------------------
def m_overlap(f):
    r = f["damping"]["rayleigh"]
    g = r.create_group("000001")
    for k, v in (("alphaM", 0.5), ("betaK", 0.001), ("betaKinit", 0.0), ("betaKcomm", 0.0)):
        g.attrs[k] = np.float64(v)
    g.create_dataset("elements", data=np.array([1, 2, 3], dtype=np.int32))


def m_central_mass_only(f):
    step = explicit_step(f)
    step.attrs["integrator"] = "CentralDifference"
    f["damping"]["rayleigh"]["000000"].attrs["betaK"] = np.float64(0.0)


def m_sspbrick(f):
    g = first_element_group(f)
    g.attrs["type"] = "SSPbrick"
    del g.attrs["lumped"]


def m_minimal(f):
    for name in ("constraints", "damping", "time_series", "patterns", "recorders"):
        del f[name]
    for name in list(f["process"]):
        del f["process"][name]


def m_version(major, minor):
    def m(f):
        f.attrs["schema_version"] = np.array([major, minor], dtype=np.int32)
    return m


def m_dup_node(f):
    tags = f["nodes"]["tags"][...]
    tags[1] = tags[0]
    replace(f["nodes"], "tags", tags)


def m_conn_missing(f):
    g = first_element_group(f)
    c = g["connectivity"][...]
    c[0, 0] = 9999
    replace(g, "connectivity", c)


def m_conn_repeat(f):
    g = first_element_group(f)
    c = g["connectivity"][...]
    c[0, 1] = c[0, 0]
    replace(g, "connectivity", c)


def m_dup_element(f):
    names = sorted(f["elements"])
    g2 = f["elements"][names[1]]
    t = g2["tags"][...]
    t[0] = 1
    replace(g2, "tags", t)


def m_ndf_mismatch(f):
    ndf = f["nodes"]["ndf"][...]
    ndf[0] = 6
    replace(f["nodes"], "ndf", ndf)


def m_fix_dof(f):
    d = f["constraints"]["sp"]["fix"]["dof"][...]
    d[0] = 4
    replace(f["constraints"]["sp"]["fix"], "dof", d)


def m_fix_dup(f):
    g = f["constraints"]["sp"]["fix"]
    n, d = g["node"][...], g["dof"][...]
    replace(g, "node", np.append(n, n[0]).astype(np.int32))
    replace(g, "dof", np.append(d, d[0]).astype(np.int32))


def m_eq_self(f):
    g = f["constraints"]["mp"]["equal_dof"]
    c = g["constrained"][...]
    c[0] = g["retained"][0]
    replace(g, "constrained", c)


def m_eq_twice(f):
    g = f["constraints"]["mp"]["equal_dof"]
    r, c, d = g["retained"][...], g["constrained"][...], g["dof"][...]
    other_master = int(r[-1])
    replace(g, "retained", np.append(r, other_master).astype(np.int32))
    replace(g, "constrained", np.append(c, c[0]).astype(np.int32))
    replace(g, "dof", np.append(d, d[0]).astype(np.int32))


def m_rayleigh_missing(f):
    g = f["damping"]["rayleigh"]["000000"]
    e = g["elements"][...]
    e[0] = 999
    replace(g, "elements", e)


def m_rayleigh_name(f):
    r = f["damping"]["rayleigh"]
    r.move("000000", "first")


def m_ts_decreasing(f):
    g = f["time_series"]["1"]
    t = g["time"][...]
    t[5] = t[3]
    replace(g, "time", t)


def m_ts_length(f):
    g = f["time_series"]["1"]
    replace(g, "values", g["values"][:-1])


def m_pattern_series(f):
    f["patterns"]["2"].attrs["accel"] = np.int32(9)


def m_recorder_node(f):
    g = f["recorders"]["surface_accel"]
    n = g["nodes"][...]
    n[0] = 9999
    replace(g, "nodes", n)


def m_process_pattern(tag):
    def m(f):
        for name in sorted(f["process"]):
            g = f["process"][name]
            if g.attrs["kind"] == "pattern":
                g.attrs["tag"] = np.int32(tag)
                return
    return m


def m_pattern_twice(f):
    names = sorted(f["process"])
    width = len(names[0])
    g = f["process"].create_group(f"{len(names):0{width}d}")
    g.attrs["kind"] = "pattern"
    g.attrs["tag"] = np.int32(2)


def m_both_steps(f):
    explicit_step(f).attrs["num_steps"] = np.int32(10)


def m_gamma_explicit(f):
    explicit_step(f).attrs["gamma"] = np.float64(0.5)


def m_ssp_lumped(f):
    first_element_group(f).attrs["type"] = "SSPbrick"


def m_bool_2(f):
    first_element_group(f).attrs["lumped"] = np.uint8(2)


def m_material_name(f):
    f["materials"]["nd"].move("1", "01")


def m_int64_attr(f):
    f.attrs["ndm"] = np.int64(3)


def m_float32_coords(f):
    replace(f["nodes"], "coords", f["nodes"]["coords"][...].astype(np.float32))


def m_coords_shape(f):
    replace(f["nodes"], "coords", f["nodes"]["coords"][:, :2])


def m_unknown_attr(f):
    f["nodes"].attrs["colour"] = "blue"


def m_unknown_dataset(f):
    first_element_group(f).create_dataset("extra", data=np.zeros(3))


def m_no_nodes(f):
    del f["nodes"]


def m_no_version(f):
    del f.attrs["schema_version"]


def m_explicit_newton(f):
    explicit_step(f).attrs["algorithm"] = "Newton"


def m_set_attr(path, name, value):
    def m(f):
        f[path].attrs[name] = value
    return m


def m_new_root_group(name):
    def m(f):
        f.create_group(name)
    return m


def m_consistent(f):
    del first_element_group(f).attrs["lumped"]


def m_central_betak(f):
    explicit_step(f).attrs["integrator"] = "CentralDifference"


def m_vtkhdf_region(f):
    f["recorders"]["site"].attrs["region"] = np.int32(1)


def m_rigid_link(f):
    f["constraints"]["mp"].create_group("rigid_link")


def m_uniaxial(f):
    f["materials"].create_group("uniaxial")


def m_modal(f):
    f["damping"].create_group("modal")


def m_process_kind(kind):
    def m(f):
        names = sorted(f["process"])
        width = len(names[0])
        g = f["process"].create_group(f"{len(names):0{width}d}")
        g.attrs["kind"] = kind
    return m


S = "/process/{explicit}"  # placeholder resolved below

CASES = [
    # --- valid -----------------------------------------------------------------------
    ("explicit", "explicit", "valid", "", None),
    ("central_difference_mass_damping", "explicit", "valid", "", m_central_mass_only),
    ("sspbrick_group", "explicit", "valid", "", m_sspbrick),
    ("minimal", "explicit", "valid", "", m_minimal),
    ("overlapping_regions", "explicit", "warning", "share 3 element(s)", m_overlap),
    # --- invalid ---------------------------------------------------------------------
    ("not_hdf5", None, "invalid", "as an HDF5 file", None),
    ("no_version", "explicit", "invalid", "missing / attribute 'schema_version'", m_no_version),
    ("major_version", "explicit", "invalid", "this reader reads 0.x files", m_version(1, 0)),
    ("newer_minor", "explicit", "invalid", "newer than this reader", m_version(0, 1)),
    ("int64_attribute", "explicit", "invalid", "attribute 'ndm' must be int32, found int64",
     m_int64_attr),
    ("float32_coords", "explicit", "invalid", "/nodes/coords must be float64, found float32",
     m_float32_coords),
    ("coords_shape", "explicit", "invalid", "must have shape [n, 3], found [104, 2]",
     m_coords_shape),
    ("unknown_attribute", "explicit", "invalid", "unknown item(s) in /nodes: attribute 'colour'",
     m_unknown_attr),
    ("unknown_dataset", "explicit", "invalid", "unknown item(s) in /elements/000000: 'extra'",
     m_unknown_dataset),
    ("no_nodes", "explicit", "invalid", "missing group /nodes", m_no_nodes),
    ("duplicate_node", "explicit", "invalid", "node 1 appears more than once", m_dup_node),
    ("missing_connectivity_node", "explicit", "invalid", "node 9999 does not exist",
     m_conn_missing),
    ("repeated_element_node", "explicit", "invalid", "uses a node more than once", m_conn_repeat),
    ("missing_material", "explicit", "invalid", "material 77 does not exist",
     m_set_attr("elements/000000", "material", np.int32(77))),
    ("duplicate_element", "explicit", "invalid", "element 1 appears more than once",
     m_dup_element),
    ("ndf_mismatch", "explicit", "invalid", "needs 3 DOFs at node 1, which has ndf 6",
     m_ndf_mismatch),
    ("fix_dof_range", "explicit", "invalid", "DOF 4 is out of range for node", m_fix_dof),
    ("fix_duplicate", "explicit", "invalid", "appears more than once", m_fix_dup),
    ("equal_dof_self", "explicit", "invalid", "is tied to itself", m_eq_self),
    ("equal_dof_twice", "explicit", "invalid", "is constrained more than once", m_eq_twice),
    ("rayleigh_missing_element", "explicit", "invalid", "element 999 does not exist",
     m_rayleigh_missing),
    ("rayleigh_child_name", "explicit", "invalid", "zero-padded indices", m_rayleigh_name),
    ("series_dt_and_time", "explicit", "invalid",
     "exactly one of attribute 'dt' and dataset 'time'", add_time_series_dt),
    ("series_decreasing_time", "explicit", "invalid", "must not decrease", m_ts_decreasing),
    ("series_length", "explicit", "invalid", "time and values differ in length", m_ts_length),
    ("pattern_missing_series", "explicit", "invalid", "time series 9 does not exist",
     m_pattern_series),
    ("recorder_missing_node", "explicit", "invalid", "node 9999 does not exist",
     m_recorder_node),
    ("process_missing_pattern", "explicit", "invalid", "pattern 42 does not exist",
     m_process_pattern(42)),
    ("pattern_activated_twice", "explicit", "invalid", "pattern 2 is activated twice",
     m_pattern_twice),
    ("steps_and_final_time", "explicit", "invalid",
     "exactly one of 'num_steps' and 'final_time'", m_both_steps),
    ("gamma_on_explicit", "explicit", "invalid", "applies only to the Newmark integrator",
     m_gamma_explicit),
    ("sspbrick_lumped_attribute", "explicit", "invalid", "SSPbrick always lumps",
     m_ssp_lumped),
    ("boolean_2", "explicit", "invalid", "must be 0 or 1, found 2", m_bool_2),
    ("material_tag_name", "explicit", "invalid", "must be a positive integer tag",
     m_material_name),
    ("explicit_newton", "explicit", "invalid", "requires the Linear algorithm",
     m_explicit_newton),
    # --- unsupported -----------------------------------------------------------------
    ("element_type", "explicit", "unsupported", 'element type "SSPquad"',
     m_set_attr("elements/000000", "type", "SSPquad")),
    ("material_type", "explicit", "unsupported", 'nD material type "PressureDependMultiYield"',
     m_set_attr("materials/nd/1", "type", "PressureDependMultiYield")),
    ("material_kind", "explicit", "unsupported", 'material kind "uniaxial"', m_uniaxial),
    ("ndm_2", "explicit", "unsupported", "ndm = 2", m_set_attr("/", "ndm", np.int32(2))),
    ("sections_group", "explicit", "unsupported", "unsupported component: /sections",
     m_new_root_group("sections")),
    ("rigid_link", "explicit", "unsupported", 'MP constraint "rigid_link"', m_rigid_link),
    ("modal_damping", "explicit", "unsupported", 'damping type "modal"', m_modal),
    ("linear_series", "explicit", "unsupported", 'time series type "Linear"',
     m_set_attr("time_series/1", "type", "Linear")),
    ("plain_pattern", "explicit", "unsupported", 'pattern type "Plain"',
     m_set_attr("patterns/2", "type", "Plain")),
    ("node_response", "explicit", "unsupported", 'Node recorder response "reaction"',
     m_set_attr("recorders/surface_accel", "response", "reaction")),
    ("vtkhdf_region", "explicit", "unsupported", "VTKHDF recorder -region", m_vtkhdf_region),
    ("hht_integrator", "explicit", "unsupported", 'integrator "HHT"',
     lambda f: explicit_step(f).attrs.__setitem__("integrator", "HHT")),
    ("tcl_action", "explicit", "unsupported", 'process step "tcl"', m_process_kind("tcl")),
    ("explicit_consistent_mass", "explicit", "unsupported", "consistent mass matrix",
     m_consistent),
    ("central_difference_betak", "explicit", "unsupported", "stiffness-proportional damping",
     m_central_betak),
    ("implicit_newmark", "implicit", "unsupported", "Newmark (/process/000000) is available "
     "from milestone 1b", None),
]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, base, expect, text, mutate in CASES:
        path = out / f"{name}.femorax.h5"
        if base is None:
            path.write_text("this is not an HDF5 file\n")
        else:
            w.write(path, make_example.soil_column(base))
            if mutate is not None:
                with h5py.File(path, "r+") as f:
                    mutate(f)
        rows.append(f"{name}\t{expect}\t{text}\t{path.name}")
    (out / "manifest.tsv").write_text("\n".join(rows) + "\n")
    print(f"{len(rows)} fixtures in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
