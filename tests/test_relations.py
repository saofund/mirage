import numpy as np
import pytest

from mirage import Session
from mirage.relations import (
    extract_relations, world_aabb, describe, place_on, place_inside, align_tops,
)

VIEW = {"lookat": [0, 0, 0.55], "distance": 2.0, "azimuth": 90, "elevation": -20}


def _desk() -> Session:
    s = Session(name="d")
    s.add_box("table", position=[0, 0, 0.25], size=[1.2, 0.8, 0.5], dynamic=False)
    s.add_box("a", position=[-0.3, 0, 0.55], size=[0.2, 0.2, 0.1])
    s.add_box("b", position=[0.3, 0, 0.55], size=[0.2, 0.2, 0.1])
    return s


def test_world_aabb_box():
    s = Session(name="t")
    s.add_box("x", position=[1, 2, 3], size=[0.4, 0.6, 0.8])
    lo, hi = world_aabb(s.scene.get_entity("x"))
    assert np.allclose(lo, [0.8, 1.7, 2.6])
    assert np.allclose(hi, [1.2, 2.3, 3.4])


def test_on_supports_and_direction():
    g = extract_relations(_desk().scene, view=VIEW)
    rels = {(r["type"], r["a"], r["b"]) for r in g["relations"]}
    assert ("on", "a", "table") in rels and ("on", "b", "table") in rels
    assert ("supports", "table", "a") in rels
    assert ("left_of", "a", "b") in rels  # a at x=-0.3 is to the viewer's left of b


def test_place_on_rests_centered_on_top():
    s = _desk()
    s.place_on("a", "b")  # put a on b
    A, B = describe(s.scene, "a"), describe(s.scene, "b")
    assert abs(A["lo"][2] - B["hi"][2]) < 1e-6           # a's bottom == b's top
    assert abs(A["center"][0] - B["center"][0]) < 1e-6   # centered on b


def test_place_inside_yields_inside_relation():
    s = Session(name="c")
    s.add_box("box", position=[0, 0, 0.2], size=[0.5, 0.5, 0.4], dynamic=False)
    s.add_sphere("ball", position=[0, 0, 2], radius=0.05)
    s.place_inside("ball", "box")
    g = extract_relations(s.scene, view=VIEW)
    assert any(r["type"] == "inside" and r["a"] == "ball" and r["b"] == "box" for r in g["relations"])


def test_align_tops():
    s = _desk()
    s.place_on("a", "table")  # raise a above b first
    s.align_tops(["a", "b"])
    assert abs(describe(s.scene, "a")["hi"][2] - describe(s.scene, "b")["hi"][2]) < 1e-6


def test_relations_are_logged_and_replayable():
    s = _desk()
    s.place_on("a", "b")
    assert [e["op"] for e in s.get_log()][-1] == "place_on"
    assert s.replay().scene.get_position("a") == pytest.approx(s.scene.get_position("a"))


def test_set_of_mark_render():
    pytest.importorskip("mujoco")
    img, boxes = _desk().set_of_mark(view=VIEW, width=160, height=120)
    assert img.shape == (120, 160, 3)
    assert {"table", "a", "b"} <= set(boxes)
