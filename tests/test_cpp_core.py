"""Differential test: the native C++ kernel (mirage_core) vs the Python kernel.

The Python kernel is the truth oracle. Every C++ operator must produce a mesh
whose invariants (verts/edges/faces/euler/closed_manifold) match Python exactly,
so the C++ port can be refactored freely without regressing. Skipped if the
pybind11 module hasn't been built (configure core/ with -Dpybind11_DIR=... and
build the _mirage_core target).
"""
import os
import sys

import pytest

_PYD = os.path.join(os.path.dirname(__file__), "..", "core", "build", "Release")
if os.path.isdir(_PYD):
    sys.path.insert(0, _PYD)

cpp = pytest.importorskip("_mirage_core")

from mirage.kernel import (  # noqa: E402
    make_cube, make_cylinder_ngon, catmull_clark, extrude_faces, inset_faces,
)


def _s(stats):
    return (stats["verts"], stats["edges"], stats["faces"], stats["euler"], stats["closed_manifold"])


def _py(mesh):
    return _s(mesh.stats())


def _cpp(mesh):
    mesh.validate()
    return _s(mesh.stats())


def _topf(m):
    return max(m.faces, key=lambda f: sum(v.co[2] for v in m.face_verts(f)) / len(m.face_verts(f)))


def test_cube_matches():
    assert _cpp(cpp.make_cube(1.0)) == _py(make_cube(1.0))


def test_cylinder_matches():
    assert _cpp(cpp.make_cylinder(8, 0.5, 1.0)) == _py(make_cylinder_ngon(8, 0.5, 1.0))


def test_catmull_clark_matches():
    assert _cpp(cpp.catmull_clark(cpp.make_cube(1.0))) == _py(catmull_clark(make_cube(1.0)))


def test_catmull_clark_twice_matches():
    assert _cpp(cpp.catmull_clark(cpp.catmull_clark(cpp.make_cube(1.0)))) \
        == _py(catmull_clark(catmull_clark(make_cube(1.0))))


def test_extrude_top_matches():
    c = make_cube(1.0)
    assert _cpp(cpp.extrude_top(cpp.make_cube(1.0), 0.5)) == _py(extrude_faces(c, [_topf(c)], 0.5))


def test_inset_top_matches():
    c = make_cube(1.0)
    assert _cpp(cpp.inset_top(cpp.make_cube(1.0), 0.3)) == _py(inset_faces(c, [_topf(c)], 0.3))


def test_inset_then_extrude_boss_matches():
    c = make_cube(1.0)
    ins = inset_faces(c, [_topf(c)], 0.3)
    py_boss = extrude_faces(ins, [ins.faces[-1]], 0.5)
    cpp_boss = cpp.extrude_top(cpp.inset_top(cpp.make_cube(1.0), 0.3), 0.5)
    assert _cpp(cpp_boss) == _py(py_boss)


@pytest.mark.parametrize("sides", [3, 6, 12, 24, 32])
def test_cylinder_various_sides(sides):
    assert _cpp(cpp.make_cylinder(sides, 0.5, 1.0)) == _py(make_cylinder_ngon(sides, 0.5, 1.0))
