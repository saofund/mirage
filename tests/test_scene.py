from mirage import Scene, Entity, Transform, Geometry, PhysicsBody, Engine


def test_json_roundtrip():
    scene = Scene(name="t")
    scene.add(
        Entity(
            name="b",
            transform=Transform(position=[1, 2, 3]),
            geometry=Geometry(kind="box", params={"size": [1, 1, 1]}),
            physics=PhysicsBody(kind="dynamic", mass=2.0),
        )
    )
    restored = Scene.from_json(scene.to_json())
    assert restored.name == "t"
    assert restored.entities["b"].transform.position == [1, 2, 3]
    assert restored.entities["b"].geometry.params == {"size": [1, 1, 1]}
    assert restored.entities["b"].physics.mass == 2.0


def test_null_physics_gravity_pulls_down():
    scene = Scene(name="g")
    scene.add(Entity(name="b", physics=PhysicsBody(kind="dynamic")))
    engine = Engine(scene=scene)
    engine.step(dt=0.1, steps=1)
    assert scene.entities["b"].transform.position[2] < 0  # fell under gravity


def test_static_body_does_not_move():
    scene = Scene(name="s")
    scene.add(Entity(name="floor", physics=PhysicsBody(kind="static")))
    engine = Engine(scene=scene)
    engine.step(dt=0.1, steps=10)
    assert scene.entities["floor"].transform.position == [0.0, 0.0, 0.0]
