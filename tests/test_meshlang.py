import pytest

from mirage.meshlang import MeshProgram, Sel, SelectorEmpty, resolve
from mirage.kernel import make_cube


def test_build_cube_program():
    m = MeshProgram().cube(1.0).build()
    m.validate()
    assert m.stats()["faces"] == 6 and m.is_closed_manifold()


def test_inset_extrude_via_tags():
    m = (MeshProgram().cube(1.0)
         .inset(on=Sel.normal("z", 1), thickness=0.3, mark="ring")
         .extrude(on=Sel.tag("ring"), distance=0.5, mark="boss")
         .assert_(closed_manifold=True, euler=2)).build()
    m.validate()
    assert m.euler() == 2 and m.is_closed_manifold()


def test_last_created_chaining():
    p = MeshProgram().cube(1.6)
    p.inset(on=Sel.normal("z", 1), thickness=0.26)
    p.extrude(on=Sel.last(), distance=0.45)        # last_created = the inset inner face
    for _ in range(3):
        p.inset(on=Sel.last(), thickness=0.26)
        p.extrude(on=Sel.last(), distance=0.45)
    m = p.build()
    m.validate()
    assert m.euler() == 2 and m.is_closed_manifold()


def test_json_roundtrip_is_deterministic():
    p = (MeshProgram().cube(1.0)
         .inset(on=Sel.normal("z", 1), thickness=0.3, mark="r")
         .extrude(on=Sel.tag("r"), distance=0.4))
    assert MeshProgram.from_json(p.to_json()).build().stats() == p.build().stats()


def test_selector_empty_raises_with_diagnostics():
    p = MeshProgram().cube(1.0).extrude(on=Sel.tag("does_not_exist"))
    with pytest.raises(SelectorEmpty) as exc:
        p.build()
    assert "bbox" in exc.value.diagnostics


def test_parametric_edit_changes_geometry():
    base = MeshProgram().cylinder(sides=24, radius=0.4, height=0.9)
    edited = MeshProgram.from_json(base.to_json())
    edited.ops[0]["sides"] = 8
    assert edited.build().stats()["faces"] != base.build().stats()["faces"]
    assert edited.build().is_closed_manifold()


def test_tags_survive_subdivision():
    m = (MeshProgram().cube(1.0)
         .tag(on=Sel.normal("z", 1), name="lid")
         .subdivide(levels=1)).build()
    from mirage.meshlang import _tags
    assert any("lid" in _tags(f) for f in m.faces)   # tag propagated to child quads


def test_selector_extreme_picks_top_face():
    assert len(resolve(make_cube(1.0), Sel.extreme("z", "max"))) == 1


def test_last_created_after_scale():
    # regression: scale/translate/tag set last_tag but never stamped it onto the
    # faces, so `last_created` on the NEXT op matched 0 (goblet-flare repro).
    m = (MeshProgram().cylinder(sides=8, radius=0.5, height=0.4)
         .extrude(on=Sel.extreme("z", "max"), distance=0.2)
         .scale(on=Sel.last(), by=[2, 2, 1])
         .extrude(on=Sel.last(), distance=0.2)        # was SelectorEmpty before the fix
         .assert_(closed_manifold=True, euler=2)).build()
    m.validate()
    assert m.is_closed_manifold() and m.euler() == 2
