#!/usr/bin/env python3
"""Write the constraint cases for the dof_numbering driver and the FemoraX DOF-numbering test.

    .venv/bin/python tests/reference/make_constraint_cases.py tests/reference/data/constraints/cases

Each case is a text file:
    node <tag> <ndf> <x> <y> <z>
    fix <tag> <dof>                      (1-based DOF)
    equal <retained> <constrained> <dof> [<dof> ...]
    expect valid | expect invalid <text the FemoraX error must contain>

The soil columns build the laminar boundary exactly as Femora's
`MP_ConstraintManager.laminar_boundary` does: at each elevation inside the bounds, the
nodes on the side surface (not the top or bottom face) are grouped, the first one is
retained and every other one is tied to it in DOFs 1 2 3. The base is fixed. Columns of
1x1, 2x2 and 3x2 elements in plan give corner, edge and face-interior side nodes, plus
interior nodes that are not tied.
"""

from __future__ import annotations

import sys
from pathlib import Path


def grid(nx, ny, nz, dx=1.0, dy=1.0, dz=0.5, z0=-10.0):
    nodes = []  # (tag, x, y, z, i, j, k)
    tag = 1
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                nodes.append((tag, i * dx, j * dy, z0 + k * dz, i, j, k))
                tag += 1
    return nodes


def column(nx, ny, nz, laminar_includes_base=False):
    nodes = grid(nx, ny, nz)
    lines = [f"node {t} 3 {x!r} {y!r} {z!r}" for (t, x, y, z, *_r) in nodes]
    for (t, _x, _y, _z, _i, _j, k) in nodes:
        if k == 0:
            lines += [f"fix {t} {d}" for d in (1, 2, 3)]
    first_level = 0 if laminar_includes_base else 1
    for k in range(first_level, nz + 1):
        side = [t for (t, _x, _y, _z, i, j, kk) in nodes
                if kk == k and (i in (0, nx) or j in (0, ny))]
        for slave in side[1:]:
            lines.append(f"equal {side[0]} {slave} 1 2 3")
    return lines, nodes


def cases():
    out = {}
    for nx, ny, nz in [(1, 1, 25), (2, 2, 6), (3, 2, 4)]:
        lines, _ = column(nx, ny, nz)
        out[f"column_{nx}x{ny}"] = lines + ["expect valid"]

    # One equalDOF command per DOF instead of one command with three DOFs. OpenSees makes
    # one MP_Constraint per command and its handler keeps only the first per constrained
    # node, so only DOF 1 is tied. Schema 0.2's `command` field records the command
    # boundaries, and FemoraX refuses the model.
    lines, nodes = column(1, 1, 3)
    split = []
    for ln in lines:
        if ln.startswith("equal "):
            _e, r, c, *dofs = ln.split()
            split += [f"equal {r} {c} {d}" for d in dofs]
        else:
            split.append(ln)
    out["split_equal_dof_commands"] = split + ["expect invalid equalDOF commands"]

    # A node tied to two different retained nodes (laminar plus an extra tie).
    lines, nodes = column(1, 1, 3)
    top = [t for (t, *_r, k) in nodes if k == 3]
    out["two_retained_nodes"] = lines + [f"equal {top[0] - 4} {top[1]} 1"] + [
        "expect invalid is constrained by 2 equalDOF commands"]

    lvl = {k: [t for (t, *_r, kk) in nodes if kk == k] for k in range(4)}

    # A chain in ascending tag order, all DOFs in order: OpenSees resolves it correctly.
    lines, nodes = column(1, 1, 3)
    out["chain"] = lines + [f"equal {lvl[2][0]} {lvl[3][0]} 1 2 3", "expect valid"]

    # A chain whose intermediate node is constrained and has a larger tag than the node
    # tied to it: level-2's master (tag 9) is tied to a level-3 slave (tag 14, itself tied
    # to 13). OpenSees processes nodes in ascending tag order, so node 9 reads node 14's
    # IDs before node 14's are set.
    lines, nodes = column(1, 1, 3)
    out["chain_reversed_tags"] = lines + [f"equal {lvl[3][1]} {lvl[2][0]} 1 2 3",
                                          "expect invalid has a larger tag"]

    # A partial chain: the inner tie covers DOF 1 only, so the retained node's transformed
    # ID layout no longer lines up with its nodal DOFs.
    lines, nodes = column(1, 1, 3)
    out["partial_chain"] = lines + [f"equal {lvl[2][0]} {lvl[3][0]} 2",
                                    "expect invalid partial chain"]

    # Laminar bounds that include the fixed base: the retained base node is fixed as well,
    # and OpenSees fixes the tied DOFs, the physically right answer.
    lines, _ = column(1, 1, 3, laminar_includes_base=True)
    out["laminar_includes_fixed_base"] = lines + ["expect valid"]

    # A fix on a constrained DOF whose retained node is free: OpenSees' addSP_Constraint
    # puts the -1 at a position the retained DOFs overwrite, so the fix is lost.
    lines, nodes = column(1, 1, 3)
    out["fix_on_constrained_dof"] = lines + [f"fix {lvl[2][1]} 1",
                                             "expect invalid the fix is silently lost"]

    # Fixed retained node: the tied nodes become fixed with it (valid in both programs).
    lines, nodes = column(1, 1, 3)
    m = [t for (t, *_r, k) in nodes if k == 2][0]
    out["fixed_retained_node"] = lines + [f"fix {m} 1"] + ["expect valid"]
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, lines in cases().items():
        (out / f"{name}.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(cases())} cases in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
