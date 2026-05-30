"""Case 01 — Modeling + multimodal rendering.

Author a tabletop scene with Mirage's USD-native API (primitives, materials,
lights), then render it through MuJoCo as **RGB + depth + segmentation** — the
three sensor modalities a synthetic-data / perception pipeline needs. Also dumps
the USD source of truth.

    uv run python examples/cases/01_modeling_render.py
"""
from mirage import Scene, Entity, Transform, Geometry, Material, PhysicsBody, Light
from mirage.mujoco_backend import MujocoSim
from _util import outdir, save_png, colorize_depth, colorize_seg


def build_scene() -> Scene:
    s = Scene(name="tabletop")
    s.add(Entity(name="ground", geometry=Geometry(kind="plane", params={"size": [4, 4]}),
                 physics=PhysicsBody(kind="static")))
    # a small still-life of primitives, each a distinct material
    s.add(Entity(name="red_box", transform=Transform(position=[-0.45, 0.0, 0.20]),
                 geometry=Geometry(kind="box", params={"size": [0.4, 0.4, 0.4]}),
                 material=Material(base_color=[0.85, 0.20, 0.18, 1.0], roughness=0.6),
                 physics=PhysicsBody(kind="dynamic")))
    s.add(Entity(name="blue_ball", transform=Transform(position=[0.1, 0.25, 0.18]),
                 geometry=Geometry(kind="sphere", params={"radius": 0.18}),
                 material=Material(base_color=[0.15, 0.45, 0.9, 1.0], metallic=0.3),
                 physics=PhysicsBody(kind="dynamic")))
    s.add(Entity(name="green_cyl", transform=Transform(position=[0.5, -0.15, 0.25]),
                 geometry=Geometry(kind="cylinder", params={"radius": 0.13, "height": 0.5}),
                 material=Material(base_color=[0.2, 0.75, 0.35, 1.0]),
                 physics=PhysicsBody(kind="dynamic")))
    s.add(Entity(name="amber_box", transform=Transform(position=[0.15, -0.5, 0.12]),
                 geometry=Geometry(kind="box", params={"size": [0.24, 0.24, 0.24]}),
                 material=Material(base_color=[0.95, 0.7, 0.15, 1.0]),
                 physics=PhysicsBody(kind="dynamic")))
    s.add(Light(name="sun", kind="sun", color=[1.0, 0.97, 0.9]))
    return s


def main() -> None:
    out = outdir("01_modeling_render")
    scene = build_scene()
    sim = MujocoSim.from_scene(scene)
    sim.step_for(0.4)  # let the objects settle onto the ground

    view = dict(lookat=[0.05, -0.1, 0.15], distance=2.6, azimuth=130, elevation=-22)
    imgs = sim.render(960, 720, modalities=("rgb", "depth", "segmentation"), **view)

    save_png(imgs["rgb"], out / "rgb.png")
    save_png(colorize_depth(imgs["depth"]), out / "depth.png")
    save_png(colorize_seg(imgs["segmentation"]), out / "segmentation.png")
    (out / "scene.usda").write_text(scene.to_usda(), encoding="utf-8")

    print(f"entities: {scene.entity_names()}")
    print(f"wrote {out/'rgb.png'}, depth.png, segmentation.png, scene.usda")
    print(f"rgb mean={imgs['rgb'].mean():.1f}  depth[min,max]="
          f"{imgs['depth'][imgs['depth']>0].min():.2f},{imgs['depth'].max():.2f}")


if __name__ == "__main__":
    main()
