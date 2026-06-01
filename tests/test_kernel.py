import math

import pytest

from mirage.kernel import make_cube, catmull_clark, extrude_faces, inset_faces, faces_by_normal


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
