"""Case 07 — Web viewport.

Author a scene with Mirage's API, roll out a physics trajectory, and export a
self-contained three.js page that reconstructs the scene and plays the motion
back. Open the printed ``index.html`` in any browser (drag to orbit). This is the
human-facing GUI; the agent drives via the API and inspects via render PNGs.

    uv run python examples/cases/07_web_viewport.py
"""
from mirage import Session
from mirage.mujoco_backend import MujocoSim
from mirage.viewport import WebViewport, trajectory_from_sim
from _util import outdir


def main() -> None:
    out = outdir("07_web_viewport")
    s = Session(name="viewport_demo")
    s.add_plane("ground", size=[4, 4])
    s.add_box("red", position=[-0.4, 0.0, 1.3], color=[0.9, 0.3, 0.2])
    s.add_sphere("blue", position=[0.3, 0.0, 1.7], color=[0.2, 0.5, 0.9])
    s.add_cylinder("green", position=[0.0, 0.4, 1.0], color=[0.2, 0.7, 0.4])

    sim = MujocoSim.from_scene(s.scene)
    frames = trajectory_from_sim(sim, s.scene, steps=150, dt=1 / 60)

    index = WebViewport(s.scene, frames=frames, title="Mirage demo").write(out, inline=True)
    print(f"entities: {s.list()['entities']}; trajectory frames: {len(frames)}")
    print(f"wrote {index} (self-contained) — open it directly in a browser")


if __name__ == "__main__":
    main()
