"""Case 10 — Studio rendering: a clean PBR-style render preset.

The ``studio`` preset adds a sky gradient, soft shadows, a reflective floor,
glossy materials derived from each object's metallic/roughness, and MSAA — a big
jump over the flat default. Renders the same still-life in 'basic' vs 'studio' to
compare, plus a studio shot of a modeled metal assembly.

    uv run python examples/cases/10_studio_render.py
"""
from mirage import Session
from mirage.modeling import l_bracket, flanged_pipe, perforated_plate
from mirage.mujoco_backend import MujocoSim
from mirage.imaging import save_png
from _util import outdir


def still_life(s: Session) -> None:
    s.add_plane("ground", size=[4, 4])
    s.add_box("red_box", position=[-0.45, 0.0, 0.2], size=[0.4, 0.4, 0.4], color=[0.85, 0.2, 0.18])
    s.set_material("red_box", metallic=0.0, roughness=0.7)            # matte
    s.add_sphere("steel_ball", position=[0.25, 0.18, 0.2], radius=0.2, color=[0.75, 0.77, 0.82])
    s.set_material("steel_ball", metallic=0.95, roughness=0.12)       # polished metal
    s.add_cylinder("gold_can", position=[0.5, -0.3, 0.2], radius=0.13, height=0.4, color=[0.95, 0.78, 0.25])
    s.set_material("gold_can", metallic=0.9, roughness=0.25)          # gold
    s.add_box("plastic", position=[-0.12, -0.42, 0.12], size=[0.24, 0.24, 0.24], color=[0.2, 0.55, 0.85])
    s.set_material("plastic", metallic=0.0, roughness=0.5)            # plastic
    s.add_light("sun", kind="sun")


def main() -> None:
    out = outdir("10_studio_render")
    view = dict(lookat=[0, 0, 0.18], distance=2.6, azimuth=125, elevation=-18)

    s = Session(name="stilllife")
    still_life(s)
    save_png(MujocoSim.from_scene(s.scene, quality="basic").render(960, 720, **view)["rgb"], out / "stilllife_basic.png")
    save_png(MujocoSim.from_scene(s.scene, quality="studio").render(960, 720, **view)["rgb"], out / "stilllife_studio.png")
    print("wrote stilllife_basic.png vs stilllife_studio.png")

    a = Session(name="assembly")
    a.add_plane("ground", size=[4, 4])
    a.add_part("plate", perforated_plate(nx=5, ny=3), position=[0, 0, 0], color=[0.55, 0.57, 0.62])
    a.set_material("plate", metallic=0.6, roughness=0.45)
    a.add_part("bracket", l_bracket(holes=2), position=[0, 0, 0], color=[0.82, 0.5, 0.2])
    a.set_material("bracket", metallic=0.7, roughness=0.35)
    a.add_part("pipe", flanged_pipe(), position=[0, 0, 0], color=[0.6, 0.65, 0.72])
    a.set_material("pipe", metallic=0.85, roughness=0.22)
    a.place_on("bracket", "plate")
    a.place_beside("pipe", "bracket", side="right", gap=0.12)
    save_png(a.render_studio(view=dict(lookat=[0.1, 0, 0.15], distance=1.9, azimuth=125, elevation=-20),
                             width=960, height=720), out / "assembly_studio.png")
    print("wrote assembly_studio.png")
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
