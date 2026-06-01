"""Launch MuJoCo's native interactive desktop viewer on a Mirage/MuJoCo scene.

    uv run python examples/launch_native_viewer.py          # Franka Panda (real robot)
    uv run python examples/launch_native_viewer.py blocks   # a physics playground

Controls: drag = orbit, scroll = zoom, double-click a body then Ctrl+drag = push it,
space = pause/resume. Close the window to exit.
"""
import os
import sys

from mirage.mujoco_backend import MujocoSim


def panda() -> MujocoSim:
    from robot_descriptions import panda_mj_description
    base = os.path.dirname(panda_mj_description.MJCF_PATH)
    scene_xml = os.path.join(base, "scene.xml")
    path = scene_xml if os.path.exists(scene_xml) else panda_mj_description.MJCF_PATH
    return MujocoSim.from_mjcf_path(path)


def blocks() -> MujocoSim:
    import numpy as np
    from mirage import Session
    s = Session(name="playground")
    s.add_plane("ground", size=[4, 4])
    colors = [[0.9, 0.3, 0.25], [0.95, 0.7, 0.2], [0.3, 0.75, 0.4],
              [0.25, 0.55, 0.9], [0.6, 0.4, 0.85], [0.95, 0.45, 0.7]]
    rng = np.random.default_rng(3)
    for i in range(12):
        c = colors[i % len(colors)]
        x, y, z = float(rng.uniform(-0.7, 0.7)), float(rng.uniform(-0.7, 0.7)), 1.0 + 0.25 * i
        if i % 3 == 0:
            s.add_box(f"box{i}", position=[x, y, z], size=[0.3, 0.3, 0.3], color=c)
        elif i % 3 == 1:
            s.add_sphere(f"ball{i}", position=[x, y, z], radius=0.16, color=c)
        else:
            s.add_cylinder(f"cyl{i}", position=[x, y, z], radius=0.13, height=0.34, color=c)
    s.add_light("sun", kind="sun")
    return MujocoSim.from_scene(s.scene)


def studio() -> MujocoSim:
    from mirage import Session
    from mirage.modeling import flanged_pipe
    s = Session(name="studio")
    s.add_plane("ground", size=[5, 5])
    s.add_box("red_box", position=[-0.5, 0.1, 0.2], size=[0.4, 0.4, 0.4], color=[0.85, 0.2, 0.18])
    s.set_material("red_box", roughness=0.7)
    s.add_sphere("steel_ball", position=[0.25, 0.2, 0.2], radius=0.2, color=[0.75, 0.77, 0.82])
    s.set_material("steel_ball", metallic=0.95, roughness=0.12)
    s.add_cylinder("gold_can", position=[0.55, -0.25, 0.2], radius=0.13, height=0.4, color=[0.95, 0.78, 0.25])
    s.set_material("gold_can", metallic=0.9, roughness=0.25)
    s.add_box("blue_box", position=[-0.12, -0.45, 0.12], size=[0.24, 0.24, 0.24], color=[0.2, 0.55, 0.85])
    s.set_material("blue_box", roughness=0.5)
    s.add_part("pipe", flanged_pipe(), position=[-0.78, -0.5, 0.0], color=[0.6, 0.65, 0.72])
    s.set_material("pipe", metallic=0.8, roughness=0.25)
    return MujocoSim.from_scene(s.scene, quality="studio")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "panda"
    sim = {"panda": panda, "blocks": blocks, "studio": studio}.get(which, panda)()
    import mujoco.viewer
    print(f"launching native MuJoCo viewer: '{which}' (nq={sim.model.nq}) — close the window to exit")
    mujoco.viewer.launch(sim.model, sim.data)


if __name__ == "__main__":
    main()
