"""Case 03 — Hinge joints / articulation (the double pendulum).

Two links joined by two **hinge joints** (铰链), released from horizontal. The
double pendulum is the textbook chaotic system, so it is a sharp test of the
joint dynamics. Saves an animated GIF and a matplotlib plot of both joint angles
over time (visibly chaotic / non-periodic).

    uv run python examples/cases/03_pendulum_joints.py
"""
import numpy as np
from mirage.mujoco_backend import MujocoSim
from _util import outdir, save_png, save_gif

MJCF = """
<mujoco model="double_pendulum">
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light directional="true" pos="0 -2 4" dir="0 0.5 -1"/>
    <body name="upper" pos="0 0 2.2">
      <joint name="j1" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0.7 0 0" size="0.045" rgba="0.85 0.3 0.25 1"/>
      <body name="lower" pos="0.7 0 0">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.7 0 0" size="0.04" rgba="0.25 0.5 0.85 1"/>
        <site name="tip" pos="0.7 0 0" size="0.05" rgba="1 1 0.3 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> None:
    out = outdir("03_pendulum_joints")
    sim = MujocoSim.from_mjcf(MJCF)

    view = dict(lookat=[0.4, 0.0, 1.8], distance=3.4, azimuth=90, elevation=0)
    ts, a1, a2, frames = [], [], [], []
    steps_per_frame = max(1, round((1 / 60) / sim.timestep))
    for _ in range(360):                 # ~6 s
        sim.step(steps_per_frame)
        ts.append(sim.time)
        a1.append(float(sim.joint("j1").qpos[0]))
        a2.append(float(sim.joint("j2").qpos[0]))
        frames.append(sim.render(640, 480, **view)["rgb"])

    save_gif(frames, out / "pendulum.gif", fps=30)
    for k in (0, 120, 240, 359):
        save_png(frames[k], out / f"frame_{k:03d}.png")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(ts, np.unwrap(a1), label="joint 1", color="#d94d40")
    ax.plot(ts, np.unwrap(a2), label="joint 2", color="#3f7fd9")
    ax.set_xlabel("time (s)"); ax.set_ylabel("angle (rad)")
    ax.set_title("Double pendulum — hinge joint angles"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "joint_angles.png", dpi=110); plt.close(fig)

    swing = float(np.ptp(np.unwrap(a2)))
    print(f"simulated {ts[-1]:.2f}s, {len(frames)} frames")
    print(f"joint-2 angle range (peak-to-peak): {swing:.2f} rad  (large => articulated swinging)")
    print(f"wrote {out/'pendulum.gif'}, joint_angles.png, frame_*.png")


if __name__ == "__main__":
    main()
