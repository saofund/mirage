"""Minimal end-to-end loop on the dependency-free null backends.

    python examples/falling_box.py

(Run after ``pip install -e .``, or with ``src/`` on PYTHONPATH.)
"""
from mirage import Engine, Scene, Entity, Transform, Geometry, PhysicsBody, Camera, Light


def main() -> None:
    scene = Scene(name="falling_box")
    scene.add(
        Entity(
            name="box",
            transform=Transform(position=[0.0, 0.0, 10.0]),
            geometry=Geometry(kind="box", params={"size": [1, 1, 1]}),
            physics=PhysicsBody(kind="dynamic", mass=1.0),
        )
    )
    scene.add(
        Entity(
            name="ground",
            geometry=Geometry(kind="plane", params={"size": [50, 50]}),
            physics=PhysicsBody(kind="static"),
        )
    )
    scene.add(Light(name="sun", kind="sun", transform=Transform(position=[0, 0, 20])))
    scene.add(Camera(name="cam", transform=Transform(position=[0, -8, 4])))

    engine = Engine(scene=scene)
    for _ in range(5):
        engine.step(dt=0.1, steps=1)
        z = scene.entities["box"].transform.position[2]
        print(f"t={engine.time:.1f}s  box.z={z:.3f}")

    print(engine.render("cam").summary)


if __name__ == "__main__":
    main()
