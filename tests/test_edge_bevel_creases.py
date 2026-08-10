"""`sharp` on an OPEN surface, and edge_bevel's silence when it prunes to nothing.

Both of these cost real time on a lofted moulding. `_dihedral_deg` reports 180 degrees for
an edge with one face, which is right for `crease` — you do want a rim held hard through a
subdivide — and useless for "the hard creases in this part": on an open lofted bowl the two
rims are 192 of the 246 edges `sharp` returns. `edge_bevel` then prunes every one of them,
returns a copy, and nothing anywhere says so.
"""
from __future__ import annotations

import math

import pytest

from mirage.meshlang import MeshProgram, _resolve_edges


def _open_bowl(rings=12, steps=48, crease_at=0.4):
    """A lofted bowl with a genuine crease ring part way down and two open rims."""
    verts, faces = [], []
    for j in range(rings):
        u = j / (rings - 1)
        # a flat flange, then a hard turn into a steep wall: the crease is at `crease_at`
        if u <= crease_at:
            r, z = 1.0 - 0.25 * (u / crease_at), 0.0
        else:
            t = (u - crease_at) / (1.0 - crease_at)
            r, z = 0.75 - 0.25 * t, -1.2 * t
        for i in range(steps):
            a = 2.0 * math.pi * i / steps
            verts.append((r * math.cos(a), r * math.sin(a), z))
    for j in range(rings - 1):
        a0, b0 = j * steps, (j + 1) * steps
        for i in range(steps):
            k = (i + 1) % steps
            faces.append([a0 + i, a0 + k, b0 + k, b0 + i])
    return MeshProgram().mesh(verts=verts, faces=faces, mark="bowl")


def test_sharp_returns_the_rims_of_an_open_surface_unless_interior_is_asked_for():
    mesh = _open_bowl().build()
    rims = [e for e in mesh.edges if len(mesh.edge_faces(e)) != 2]
    assert rims, "the fixture is supposed to be an OPEN surface"

    loose = _resolve_edges(mesh, {"by": "sharp", "angle": 20.0}, None)
    tight = _resolve_edges(mesh, {"by": "sharp", "angle": 20.0, "interior": True}, None)

    # every rim edge is in the loose set and none is in the interior one
    assert all(e in loose for e in rims)
    assert not any(e in tight for e in rims)
    assert len(tight) == len(loose) - len(rims)
    # and what is left is a real crease ring, not nothing
    assert len(tight) > 0
    assert all(len(mesh.edge_faces(e)) == 2 for e in tight)


def test_sharp_interior_is_a_no_op_on_a_closed_solid():
    """The flag must not change anything where there is no boundary — a cube's twelve."""
    cube = MeshProgram().cube(size=1.0).build()
    a = _resolve_edges(cube, {"by": "sharp", "angle": 30.0}, None)
    b = _resolve_edges(cube, {"by": "sharp", "angle": 30.0, "interior": True}, None)
    assert len(a) == len(b) == 12


def test_edge_bevel_warns_instead_of_silently_doing_nothing():
    """An incomplete crease ring prunes to empty. The geometry result is right; the
    silence is not, because the counts do not move and there is nothing to read."""
    prog = _open_bowl(crease_at=0.4)
    mesh = prog.build()
    # Pick a threshold that catches only part of the ring by selecting a handful of edges
    # by hand — a lone arc, which cannot separate its faces.
    ring = _resolve_edges(mesh, {"by": "sharp", "angle": 20.0, "interior": True}, None)
    assert len(ring) > 4
    partial = {"or": [{"by": "sharp", "angle": 20.0, "interior": True},
                      {"by": "boundary"}]}
    before = (len(mesh.verts), len(mesh.faces))
    with pytest.warns(RuntimeWarning, match="pruned"):
        out = _open_bowl(crease_at=0.4).edge_bevel({"by": "sharp", "angle": 179.0},
                                                   width=0.02).build()
    assert (len(out.verts), len(out.faces)) == before      # geometry unchanged, as documented
    assert partial                                          # selector stays well-formed


def test_edge_bevel_rounds_a_complete_interior_crease_ring():
    prog = _open_bowl(crease_at=0.4)
    before = prog.build()
    out = _open_bowl(crease_at=0.4).edge_bevel(
        {"by": "sharp", "angle": 20.0, "interior": True}, width=0.15).build()
    assert len(out.verts) > len(before.verts)
    assert len(out.faces) > len(before.faces)


def test_region_inset_makes_one_pad_where_per_face_makes_islands():
    """The difference the emboss operator turns on.

    A patch of a lofted surface is many quads. Insetting each one separately gives every
    quad its own border ring — geometrically fine, and it renders as stipple. The region
    form insets the patch's outline once, so an extrude after it lifts a single pad.
    """
    box = {"by": "box", "min": [-0.45, -0.45, -0.01], "max": [0.45, 0.45, 0.01]}
    grid = MeshProgram().grid(size_x=2.0, size_y=2.0, x_div=10, y_div=10, mark="plate")
    each = grid.inset(box, thickness=0.3).build()
    once = (MeshProgram().grid(size_x=2.0, size_y=2.0, x_div=10, y_div=10, mark="plate")
            .inset(box, thickness=0.3, region=True).build())
    plain = MeshProgram().grid(size_x=2.0, size_y=2.0, x_div=10, y_div=10, mark="plate").build()
    # per-face adds a ring per selected quad; region adds one ring for the whole patch
    assert len(each.faces) - len(plain.faces) > 3 * (len(once.faces) - len(plain.faces))
    assert len(once.faces) > len(plain.faces)
    # and only the patch's boundary vertices moved
    assert len(once.verts) - len(plain.verts) < len(each.verts) - len(plain.verts)
