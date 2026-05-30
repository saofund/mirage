"""Case 02 — Collision, contact & toppling.

A neatly stacked tower of cubes authored in Mirage; a heavy sphere is launched
into it (initial velocity authored on the body). MuJoCo resolves the contacts:
the tower scatters and the debris settles. Demonstrates collision detection,
friction, restitution and resting contact — none of which the null integrator
could do. Saves a frame sequence + an animated GIF.

    uv run python examples/cases/02_collision_stack.py
"""
import numpy as np
from mirage import Scene, Entity, Transform, Geometry, Material, PhysicsBody, Light
from mirage.mujoco_backend import MujocoSim
from _util import outdir, save_png, save_gif

CUBE = 0.3  # full edge length


def build_scene() -> Scene:
    s = Scene(name="collision_tower")
    s.add(Entity(name="ground", geometry=Geometry(kind="plane", params={"size": [6, 6]}),
                 physics=PhysicsBody(kind="static")))
    palette = [[0.85, 0.3, 0.25, 1], [0.95, 0.7, 0.2, 1], [0.3, 0.7, 0.4, 1],
               [0.25, 0.5, 0.85, 1], [0.6, 0.4, 0.8, 1]]
    for i in range(5):  # stack of 5 cubes, touching
        s.add(Entity(
            name=f"cube{i}",
            transform=Transform(position=[0.0, 0.0, CUBE / 2 + i * CUBE]),
            geometry=Geometry(kind="box", params={"size": [CUBE, CUBE, CUBE]}),
            material=Material(base_color=palette[i]),
            physics=PhysicsBody(kind="dynamic", mass=0.5),
        ))
    # projectile: heavy sphere flying in along +x at the tower's mid-height
    s.add(Entity(
        name="wrecker",
        transform=Transform(position=[-2.2, 0.0, 0.75]),
        geometry=Geometry(kind="sphere", params={"radius": 0.2}),
        material=Material(base_color=[0.15, 0.15, 0.18, 1.0], metallic=0.6),
        physics=PhysicsBody(kind="dynamic", mass=5.0, linear_velocity=[6.5, 0.0, 1.2]),
    ))
    s.add(Light(name="sun", kind="sun"))
    return s


def main() -> None:
    out = outdir("02_collision_stack")
    scene = build_scene()
    sim = MujocoSim.from_scene(scene)

    view = dict(lookat=[0.0, 0.0, 0.5], distance=4.2, azimuth=110, elevation=-12)
    heights0 = [sim.body_pos(f"cube{i}")[2] for i in range(5)]
    frames, max_contacts = [], 0
    for k in range(120):                 # ~2 s at 60 fps render cadence
        sim.step_for(1 / 60)
        max_contacts = max(max_contacts, sim.ncontact)
        frames.append(sim.render(640, 480, **view)["rgb"])
        if k in (0, 30, 60, 119):
            save_png(frames[-1], out / f"frame_{k:03d}.png")

    save_gif(frames, out / "collision.gif", fps=30)
    spread = float(np.std([sim.body_pos(f"cube{i}")[:2] for i in range(5)]))
    print(f"initial cube heights: {[round(h,2) for h in heights0]}")
    print(f"peak simultaneous contacts: {max_contacts}")
    print(f"final horizontal scatter (std of xy): {spread:.3f}  (tower toppled: {spread>0.15})")
    print(f"wrote {out/'collision.gif'} + frame_*.png")


if __name__ == "__main__":
    main()
