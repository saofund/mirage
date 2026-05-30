import pytest

from mirage import (
    Scene, Entity, Transform, Geometry, Material, PhysicsBody, Camera, Light,
    Engine, EntityNotFound, DuplicateName, InvalidName,
)


def _box(name="b", **kw):
    return Entity(
        name=name,
        transform=Transform(position=[1, 2, 3]),
        geometry=Geometry(kind="box", params={"size": [1, 1, 1]}),
        material=Material(base_color=[0.2, 0.4, 0.6, 1.0]),
        physics=PhysicsBody(kind="dynamic", mass=2.0),
        **kw,
    )


def test_json_roundtrip():
    scene = Scene(name="t")
    scene.add(_box(tags=["target"]))
    restored = Scene.from_json(scene.to_json())
    e = restored.get_entity("b")
    assert restored.name == "t"
    assert e.transform.position == [1, 2, 3]
    assert e.geometry.params == {"size": [1, 1, 1]}
    assert e.material.base_color == [0.2, 0.4, 0.6, 1.0]
    assert e.physics.mass == 2.0
    assert e.tags == ["target"]
    # round-trip is lossless: a re-serialized copy diffs clean
    assert scene.diff(restored) == {}


def test_camera_and_light_roundtrip():
    scene = Scene(name="cl")
    scene.add(Camera(name="cam", width=320, height=240, modalities=["rgb", "depth"]))
    scene.add(Light(name="sun", kind="sun", intensity=3.0))
    restored = Scene.from_json(scene.to_json())
    cam = restored.get_camera("cam")
    assert (cam.width, cam.height) == (320, 240)
    assert cam.modalities == ["rgb", "depth"]
    assert restored.get_light("sun").kind == "sun"
    assert restored.get_light("sun").intensity == 3.0


def test_usd_export_and_load(tmp_path):
    scene = Scene(name="exp")
    scene.add(_box())
    text = scene.to_usda()
    assert "World" in text and "def " in text
    p = tmp_path / "scene.usda"
    scene.export(str(p))
    loaded = Scene.load(str(p))
    assert scene.diff(loaded) == {}


def test_crud_edit_remove_rename():
    scene = Scene(name="crud")
    scene.add(_box())
    scene.set_transform("b", position=[5, 5, 5])
    assert scene.get_position("b") == [5, 5, 5]
    scene.set_material("b", roughness=0.1)
    assert scene.get_entity("b").material.roughness == 0.1
    scene.set_velocity("b", linear=[0, 0, 1])
    assert scene.get_entity("b").physics.linear_velocity == [0, 0, 1]
    scene.rename("b", "c")
    assert scene.entity_names() == ["c"]
    scene.remove("c")
    assert scene.entity_names() == []


def test_typed_errors():
    scene = Scene(name="err")
    scene.add(_box())
    with pytest.raises(DuplicateName):
        scene.add(_box())
    with pytest.raises(EntityNotFound):
        scene.get_entity("missing")
    with pytest.raises(InvalidName):
        scene.add(Entity(name="has spaces"))


def test_null_physics_gravity_pulls_down():
    scene = Scene(name="g")
    scene.add(Entity(name="b", physics=PhysicsBody(kind="dynamic")))
    Engine(scene=scene).step(dt=0.1, steps=1)
    assert scene.get_position("b")[2] < 0  # fell under gravity


def test_static_body_does_not_move():
    scene = Scene(name="s")
    scene.add(Entity(name="floor", physics=PhysicsBody(kind="static")))
    Engine(scene=scene).step(dt=0.1, steps=10)
    assert scene.get_position("floor") == [0.0, 0.0, 0.0]


def test_render_through_camera():
    scene = Scene(name="r")
    scene.add(_box())
    scene.add(Camera(name="cam"))
    result = Engine(scene=scene).render("cam")
    assert result.camera == "cam"
    assert "1 entities" in result.summary
