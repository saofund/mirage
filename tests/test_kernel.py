import math

import pytest

import numpy as np

from mirage.kernel import (
    make_cube, make_cylinder_ngon, make_plane, make_grid, catmull_clark, extrude_faces,
    inset_faces, bevel_faces, loop_cut, edge_bevel, delete_faces, bridge_faces, fill_holes,
    solidify, face_normal, faces_by_normal,
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


def test_edge_bevel_top_loop_mixed():
    # the 4 top edges form a closed LOOP -> round just the top rim, sides stay sharp
    c = make_cube(1.0)
    top = resolve_edges(c, ESel.on_face({"by": "normal", "axis": "z", "sign": 1.0}))
    b = edge_bevel(c, top, width=0.2)
    b.validate()
    assert b.euler() == 2 and b.is_closed_manifold() and _all_outward(b)
    assert b.stats()["faces"] == 10            # 4 unchanged sides + top + 4 chamfers + the inset top? -> 10
    # the bottom verts are untouched (no selected edge there)
    assert any(abs(v.co[2] + 0.5) < 1e-9 and abs(v.co[0]) == 0.5 for v in b.verts)


def test_edge_bevel_lonely_edges_are_safe_noop():
    # 4 pairwise-disjoint vertical edges are lone cuts at every vertex -> pruned
    c = make_cube(1.0)
    out = edge_bevel(c, resolve_edges(c, ESel.axis("z")), width=0.2)
    out.validate()
    assert out.stats() == c.stats()            # nothing cleanly bevelable -> unchanged


def test_edge_bevel_subdivided_stays_closed():
    # subdivided meshes have >=3-sector vertices: each needs a corner face or the
    # mesh tears open (the stress-test bug). Must stay a closed 2-manifold.
    sc = catmull_clark(make_cube(1.0))
    for sel in (resolve_edges(sc, ESel.sharp(30)), resolve_edges(sc, ESel.on_face({"by": "normal", "axis": "z", "sign": 1.0}))):
        b = edge_bevel(sc, sel, width=0.06)
        b.validate()
        assert b.euler() == 2 and b.is_closed_manifold()


def test_edge_bevel_inset_rim_loop_stays_closed():
    c = make_cube(1.0)
    ip = inset_faces(c, faces_by_normal(c, "z", 1.0), 0.3)
    b = edge_bevel(ip, resolve_edges(ip, ESel.sharp(30)), width=0.1)
    b.validate()
    assert b.euler() == 2 and b.is_closed_manifold()


def test_edge_bevel_open_mesh_never_crashes():
    # boundary vertices are pruned -> beveling an open mesh is a safe no-/partial-op
    c = make_cube(1.0)
    ob = delete_faces(c, faces_by_normal(c, "z", 1.0))     # open box
    for esel in (ESel.all(), ESel.sharp(30)):
        b = edge_bevel(ob, resolve_edges(ob, esel), width=0.1)
        b.validate()                                        # must not crash, must validate


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


# --- solidify's rim band ---------------------------------------------------- #

def _tags(f):
    return f.attrs.get("tags", [])


def test_solidify_rim_mark_tags_only_the_walls():
    # A 3x3 grid shells into 9 outer + 9 inner faces and 12 wall quads (one per boundary
    # edge). Only the walls may carry the rim mark — if it leaked onto the inner shell the
    # whole underside would paint as laminate.
    m = solidify(make_grid(1.0, x_div=3, y_div=3), 0.05, mark="new", rim_mark="edge")
    m.validate()
    assert m.is_closed_manifold()
    rim = [f for f in m.faces if "edge" in _tags(f)]
    assert len(rim) == 12                                   # the grid's 12 boundary edges
    assert all("new" in _tags(f) for f in rim)              # the rim is still part of `mark`
    assert len([f for f in m.faces if "new" in _tags(f)]) == 9 + 12   # inner shell + walls


def test_solidify_rim_is_thin_and_vertical():
    # The rim band is the CUT EDGE: it should be `thickness` tall and stand across the
    # sheet, not lie in it. This is what makes it read as the laminate stripe.
    m = solidify(make_grid(1.0, x_div=2, y_div=2), 0.04, rim_mark="edge")
    rim = [f for f in m.faces if "edge" in _tags(f)]
    assert rim
    for f in rim:
        assert abs(face_normal(m, f)[2]) < 1e-6            # normal is horizontal: a wall
        zs = [v.co[2] for v in m.face_verts(f)]
        assert max(zs) - min(zs) == pytest.approx(0.04, abs=1e-9)


def test_solidify_rim_mark_survives_subdivision():
    # The point of marking rather than selecting: tags are inherited by child faces, so the
    # band is still addressable at the limit surface, where no query could find it.
    p = solidify(make_grid(1.0, x_div=3, y_div=3), 0.05, rim_mark="edge")
    sub = catmull_clark(p, levels=2)
    sub.validate()
    rim = [f for f in sub.faces if "edge" in _tags(f)]
    assert len(rim) == 12 * 4 ** 2                          # each wall quad -> 16 children
    assert 0 < len(rim) < len(sub.faces)


def test_solidify_without_rim_mark_is_unchanged():
    a = solidify(make_grid(1.0, x_div=3, y_div=3), 0.05, mark="new")
    b = solidify(make_grid(1.0, x_div=3, y_div=3), 0.05, mark="new", rim_mark=None)
    assert [v.co for v in a.verts] == [v.co for v in b.verts]
    assert [_tags(f) for f in a.faces] == [_tags(f) for f in b.faces]


# --- semi-sharp creases (DeRose/Kass/Truong 1998) --------------------------- #

def _reach(m):
    """How far the mesh still reaches along the body diagonal: 1.5 for an exact unit cube,
    dropping as Catmull-Clark rounds its corners off.

    A bounding box is useless here — the middle of a flat face never moves under either
    rule, so a cube's bbox reads 0.5 whether its edges are razor sharp or fully rounded.
    Corner reach is what actually responds to sharpness.
    """
    return max(v.co[0] + v.co[1] + v.co[2] for v in m.verts)


def _diag(m, a, b):
    """Reach along one face-diagonal — how hard the rim between axes `a` and `b` is."""
    return max(v.co[a] + v.co[b] for v in m.verts)


def _crease_all(m, w):
    for e in m.edges:
        e.crease = w
    return m


def test_uncreased_subdivision_rounds_the_cube_off():
    # The baseline the crease rules exist to defeat: plain Catmull-Clark pulls a cube's
    # corners far inside its control cage (1.5 -> ~0.75, i.e. halfway to a sphere).
    m = catmull_clark(make_cube(1.0), levels=3)
    m.validate()
    assert m.is_closed_manifold()
    assert _reach(m) < 0.80


def test_crease_holds_the_cube_exactly():
    # Every edge sharp for at least as many levels as we subdivide => the limit surface IS
    # the cage, to the last bit. This is what lets a subdivided shell keep a crisp rim.
    m = catmull_clark(_crease_all(make_cube(1.0), 3.0), levels=3)
    m.validate()
    assert m.is_closed_manifold()
    assert _reach(m) == pytest.approx(1.5, abs=1e-12)
    assert len(m.faces) == 6 * 4 ** 3


def test_crease_decays_one_level_at_a_time():
    # Sharpness is measured in LEVELS, not a flag. Subdivided three times, weight 1/2/3
    # must come out strictly ordered — each extra level of sharpness holds the corner
    # further out — and all of them above the uncreased case.
    soft = _reach(catmull_clark(make_cube(1.0), levels=3))
    w1 = _reach(catmull_clark(_crease_all(make_cube(1.0), 1.0), levels=3))
    w2 = _reach(catmull_clark(_crease_all(make_cube(1.0), 2.0), levels=3))
    w3 = _reach(catmull_clark(_crease_all(make_cube(1.0), 3.0), levels=3))
    assert soft < w1 < w2 < w3
    assert w3 == pytest.approx(1.5, abs=1e-12)   # weight == levels -> still exact


def test_fractional_crease_interpolates():
    # A weight in (0,1) BLENDS the smooth and sharp rules rather than switching between
    # them — the point of the semi-sharp scheme over Hoppe's integer sharpness.
    soft = _reach(catmull_clark(make_cube(1.0), levels=2))
    half = _reach(catmull_clark(_crease_all(make_cube(1.0), 0.5), levels=2))
    hard = _reach(catmull_clark(_crease_all(make_cube(1.0), 2.0), levels=2))
    assert soft < half < hard


def test_crease_is_local_to_the_creased_edges():
    # Crease only the four edges bounding the +z face. The rim between +x and +z is held
    # hard, while the vertical edges — untouched — round off as if nothing were creased.
    # On a plain subdivided cube those two rims are equal by symmetry, so any gap between
    # them is the crease doing exactly as much as it was asked and no more.
    plain = catmull_clark(make_cube(1.0), levels=2)
    assert _diag(plain, 0, 2) == pytest.approx(_diag(plain, 0, 1))

    m = make_cube(1.0)
    for e in m.edges:
        if e.v1.co[2] > 0.4 and e.v2.co[2] > 0.4:          # the four edges of the top face
            e.crease = 4.0
    sub = catmull_clark(m, levels=2)
    sub.validate()
    assert sub.is_closed_manifold()
    assert _diag(sub, 0, 2) > _diag(plain, 0, 2) + 0.3     # the creased top rim is held
    assert _diag(sub, 0, 1) < _diag(sub, 0, 2) - 0.2       # the vertical edges are not


def test_crease_children_inherit_and_mesh_stays_valid():
    # The decay has to survive the rebuild: after one level the children of a weight-3
    # edge must carry weight 2, or multi-level subdivision would silently lose the crease.
    one = catmull_clark(_crease_all(make_cube(1.0), 3.0), levels=1)
    one.validate()
    creased = [e.crease for e in one.edges if e.crease > 0]
    assert creased and all(c == pytest.approx(2.0) for c in creased)
    assert len(creased) == 24        # each of the cube's 12 edges split into two children


def test_zero_crease_matches_plain_subdivision():
    # A crease of 0 must be exactly the old code path — no drift for every existing op-log.
    plain = catmull_clark(make_cube(1.0), levels=2)
    zeroed = catmull_clark(_crease_all(make_cube(1.0), 0.0), levels=2)
    assert [v.co for v in plain.verts] == [v.co for v in zeroed.verts]
