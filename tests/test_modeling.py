import pytest

trimesh = pytest.importorskip("trimesh")

from mirage.modeling import Part, l_bracket, flanged_pipe, perforated_plate
from mirage import Session


def test_library_parts_are_watertight_solids():
    for p in (l_bracket(), flanged_pipe(), perforated_plate()):
        st = p.stats()
        assert st["watertight"]
        assert st["volume"] > 0 and st["faces"] > 0


def test_boolean_cut_reduces_volume():
    solid = Part("s").box([1, 1, 0.3]).build().volume
    drilled = Part("d").box([1, 1, 0.3]).cylinder(0.2, 1.0, op="cut").build().volume
    assert drilled < solid - 1e-4


def test_parametric_rebuild_changes_geometry():
    a = perforated_plate(nx=5, ny=3).build()
    b = perforated_plate(nx=9, ny=5).build()
    assert a.faces.shape[0] != b.faces.shape[0]
    assert b.volume < a.volume  # more holes removes more material


def test_program_is_serializable_and_roundtrips():
    p = l_bracket()
    q = Part.from_features(p.features, name="copy")
    assert q.stats()["volume"] == pytest.approx(p.stats()["volume"])


def test_add_part_into_scene_is_a_mesh_entity_and_logged():
    s = Session(name="m")
    s.add_part("bracket", l_bracket(), position=[0, 0, 0])
    e = s.scene.get_entity("bracket")
    assert e.geometry.kind == "mesh"
    assert "bbox_lo" in e.geometry.params  # real bounds baked for the relation layer
    assert [op["op"] for op in s.get_log()][-1] == "add_part"
