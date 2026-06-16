import math

import pytest

import numpy as np

from mirage.kernel import (
    make_cube, make_cylinder_ngon, make_plane, catmull_clark, extrude_faces, inset_faces,
    bevel_faces, loop_cut, edge_bevel, delete_faces, bridge_faces, fill_holes,
    face_normal, faces_by_normal,
)
from mirage.meshlang import resolve_edges, ESel


def test_cube_topology():
    m = make_cube()
    m.validate()
    s = m.stats()
    assert (s["verts"], s["edges"], s["faces"]) == (8, 12, 6)
    assert s["euler"] == 2 and s["closed_manifold"]


def test_traversal_loops_and_radial():
    m = make_cube()
    assert len(m.face_verts(m.faces[0])) == 4              # quad face
    for e in m.edges:
        assert len(m.edge_faces(e)) == 2                   # manifold: 2 faces per edge


def test_catmull_clark_preserves_euler_and_manifold():
    m = make_cube()
    for level in range(1, 4):
        m = catmull_clark(m)
        m.validate()
        assert m.euler() == 2 and m.is_closed_manifold()
        assert len(m.faces) == 6 * 4 ** level              # quad split each level


def test_subdivision_rounds_corners():
    m = make_cube(2.0)
    corner0 = max(math.dist((0, 0, 0), v.co) for v in m.verts)  # cube corner ~1.732
    for _ in range(2):
        m = catmull_clark(m)
    corner1 = max(math.dist((0, 0, 0), v.co) for v in m.verts)
    assert corner1 < corner0 - 0.2                          # corners pulled inward (rounded)


def test_extrude_preserves_manifold_and_lifts():
    m = make_cube(1.0)
    top = faces_by_normal(m, "z", 1.0)
    assert len(top) == 1
    zmax0 = max(v.co[2] for v in m.verts)
    m2 = extrude_faces(m, top, 0.8)
    m2.validate()
    assert m2.euler() == 2 and m2.is_closed_manifold()      # still a closed solid
    assert len(m2.faces) > len(m.faces)                     # walls + cap added
    assert max(v.co[2] for v in m2.verts) > zmax0 + 0.5     # the cap lifted


def test_inset_then_extrude_stays_closed():
    m = make_cube(1.0)
    m = inset_faces(m, faces_by_normal(m, "z", 1.0), 0.35)
    m.validate()
    assert m.euler() == 2 and m.is_closed_manifold()
    m = extrude_faces(m, [m.faces[-1]], 0.5)                # extrude the inset inner face
    m.validate()
    assert m.euler() == 2 and m.is_closed_manifold()        # a raised boss, still closed


def test_bevel_chamfers_the_rim():
    m = make_cube(1.0)                                       # top face at z = +0.5
    top = faces_by_normal(m, "z", 1.0)
    depth = 0.2
    b = bevel_faces(m, top, width=0.25, depth=depth)
    b.validate()
    # same topology as an inset (a ring of quads + the inner face), still closed
    assert b.euler() == 2 and b.is_closed_manifold()
    assert b.stats() == inset_faces(m, top, 0.25).stats()   # bevel == inset + a normal offset
    inner = b.faces[-1]                                      # the lifted inner face
    cz = sum(v.co[2] for v in b.face_verts(inner)) / len(b.face_verts(inner))
    assert abs(cz - (0.5 + depth)) < 1e-9                    # raised by `depth` along +z


def test_loop_cut_cube_stays_closed():
    m = make_cube(1.0)
    seed = faces_by_normal(m, "y", -1.0)            # a side quad seeds the ring
    lc = loop_cut(m, seed, axis="z")                 # a horizontal band around z
    lc.validate()
    s = lc.stats()
    # 4 vertical edges bisected: +4 verts, the 4 side quads -> 8, caps unchanged
    assert (s["verts"], s["faces"]) == (12, 10)
    assert s["euler"] == 2 and s["closed_manifold"]


def test_loop_cut_cylinder_adds_a_ring():
    m = make_cylinder_ngon(8, 0.5, 1.0)
    seed = [f for f in m.faces if len(m.face_verts(f)) == 4][:1]  # a side quad
    lc = loop_cut(m, seed, axis="z")
    lc.validate()
    assert lc.euler() == 2 and lc.is_closed_manifold()
    assert lc.stats()["faces"] == m.stats()["faces"] + 8   # each of 8 side quads split


def _all_outward(m):
    ctr = np.mean([v.co for v in m.verts], axis=0)
    return all(np.dot(np.array(face_normal(m, f)),
                      np.mean([v.co for v in m.face_verts(f)], axis=0) - ctr) >= 0 for f in m.faces)


def test_edge_bevel_cube_is_chamfered_cube():
    c = make_cube(1.0)
    b = edge_bevel(c, resolve_edges(c, ESel.all()), width=0.2)
    b.validate()
    s = b.stats()
    # 6 shrunk faces + 12 chamfer quads + 8 corner triangles, 8 verts x 3 copies
    assert (s["verts"], s["faces"]) == (24, 26)
    assert s["euler"] == 2 and s["closed_manifold"]
    assert _all_outward(b)                                # winding stays outward-consistent


def test_edge_bevel_cylinder_sharp_stays_closed():
    m = make_cylinder_ngon(8, 0.5, 1.0)
    b = edge_bevel(m, resolve_edges(m, ESel.sharp(20)), width=0.12)
    b.validate()
    assert b.euler() == 2 and b.is_closed_manifold() and _all_outward(b)


def test_edge_bevel_partial_selection_is_safe():
    # the top 4 edges alone don't form complete vertex stars on a closed cube,
    # so the fixpoint prunes them -> a safe no-op (never a torn mesh).
    c = make_cube(1.0)
    top_edges = resolve_edges(c, ESel.on_face({"by": "normal", "axis": "z", "sign": 1.0}))
    # intersect with sharp to keep just the rim; still not a full star -> no-op
    b = edge_bevel(c, top_edges, width=0.2)
    b.validate()
    assert b.is_closed_manifold()  # valid regardless


def test_plane_is_an_open_mesh():
    p = make_plane(1.0)
    p.validate()                                     # structurally valid...
    assert not p.is_closed_manifold()                # ...but open (4 boundary edges)
    assert p.stats()["faces"] == 1


def test_delete_faces_opens_the_mesh():
    c = make_cube(1.0)
    opened = delete_faces(c, faces_by_normal(c, "z", 1.0))   # remove the top
    opened.validate()
    assert not opened.is_closed_manifold()           # a box open at the top
    assert opened.stats()["faces"] == 5


def test_delete_then_fill_recloses():
    c = make_cube(1.0)
    opened = delete_faces(c, faces_by_normal(c, "z", 1.0))
    closed = fill_holes(opened)
    closed.validate()
    assert closed.is_closed_manifold() and closed.euler() == 2 and _all_outward(closed)


def test_bridge_makes_a_closed_box():
    c = make_cube(1.0)
    sides = [f for f in c.faces if abs(face_normal(c, f)[2]) < 0.5]
    opened = delete_faces(c, sides)                  # top + bottom (2 disjoint quads)
    tube = bridge_faces(opened, list(opened.faces))  # bridge -> open tube
    tube.validate()
    assert tube.euler() == 0 and not tube.is_closed_manifold() and _all_outward(tube)
    box = fill_holes(tube)                           # cap -> closed box
    box.validate()
    assert box.is_closed_manifold() and box.euler() == 2 and _all_outward(box)


def test_bridge_rejects_adjacent_faces():
    # two faces sharing an edge cannot be bridged cleanly -> safe no-op (copy)
    c = make_cube(1.0)
    adj = [faces_by_normal(c, "z", 1.0)[0], faces_by_normal(c, "x", 1.0)[0]]
    out = bridge_faces(c, adj)
    assert out.stats() == c.stats()


def test_loop_cut_ngon_seed_is_noop():
    m = make_cylinder_ngon(6, 0.5, 1.0)
    cap = [f for f in m.faces if len(m.face_verts(f)) == 6][:1]  # an n-gon cap can't seed
    lc = loop_cut(m, cap, axis="z")
    assert lc.stats() == m.stats()


def test_bevel_zero_depth_equals_inset():
    m = make_cube(1.0)
    top = faces_by_normal(m, "z", 1.0)
    bev = bevel_faces(m, top, width=0.3, depth=0.0)
    ins = inset_faces(m, top, 0.3)
    bev.validate()
    bz = sorted(round(v.co[2], 9) for v in bev.verts)
    iz = sorted(round(v.co[2], 9) for v in ins.verts)
    assert bz == iz                                         # depth=0 is a plain inset


# --- regression tests for bugs found by the adversarial verification workflow --- #
def test_extrude_closed_region_is_valid():
    m = make_cube(2.0)
    m2 = extrude_faces(m, list(m.faces), 0.5)               # extrude ALL 6 faces (closed region)
    m2.validate()
    assert m2.euler() == 2 and m2.is_closed_manifold()      # was euler=10 with 8 orphan verts


def test_extrude_corner_region_no_orphans():
    m = make_cube(2.0)
    m2 = extrude_faces(m, [m.faces[1], m.faces[3], m.faces[4]], 1.0)  # 3 faces at a corner
    m2.validate()
    assert m2.euler() == 2                                  # interior corner vert no longer orphaned


def test_extrude_opposite_faces_no_cancellation():
    m = make_cube(2.0)
    sel = faces_by_normal(m, "z", 1.0) + faces_by_normal(m, "z", -1.0)  # normals cancel
    m2 = extrude_faces(m, sel, 1.0)
    m2.validate()                                          # per-vertex normals -> no zero-area faces
    assert m2.euler() == 2 and m2.is_closed_manifold()


def test_inset_thickness_clamped():
    m = make_cube(2.0)
    m2 = inset_faces(m, faces_by_normal(m, "z", 1.0), 1.5)  # >1 would bowtie; clamped to (0,1)
    m2.validate()
    assert m2.is_closed_manifold()


def test_validate_catches_orphan_vertex():
    m = make_cube(1.0)
    m.add_vert([5, 5, 5])                                  # a loose, unreferenced vertex
    with pytest.raises(AssertionError):
        m.validate()
