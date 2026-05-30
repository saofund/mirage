"""Case 04 — Robot motion: an actuated 3-DOF arm.

A 3-link arm with three **actuated hinge joints** (position actuators) sweeps
forward under closed-loop control and rakes a stack of cubes off a pedestal —
articulated robot motion + actuation + contact-rich interaction. Records the
end-effector trajectory and the cubes' displacement; saves a GIF + key frames.

    uv run python examples/cases/04_robot_arm.py
"""
import numpy as np
from mirage.mujoco_backend import MujocoSim
from _util import outdir, save_png, save_gif

MJCF = """
<mujoco model="arm_sweep">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.25 0.35 0.45" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>
  </asset>
  <default>
    <joint damping="3"/>
    <geom friction="1 0.1 0.01"/>
  </default>
  <worldbody>
    <light directional="true" pos="-1 -1 4" dir="0.2 0.2 -1"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid"/>
    <geom name="pedestal" type="box" pos="0.6 0 0.3" size="0.12 0.22 0.3" rgba="0.4 0.4 0.45 1"/>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" pos="0 0 0.15" size="0.08 0.15" rgba="0.25 0.25 0.3 1"/>
      <body name="link1" pos="0 0 0.3">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.05" rgba="0.85 0.45 0.2 1"/>
        <body name="link2" pos="0 0 0.5">
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.4" size="0.045" rgba="0.9 0.6 0.2 1"/>
          <body name="link3" pos="0 0 0.4">
            <joint name="j3" type="hinge" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.04" rgba="0.95 0.75 0.2 1"/>
            <site name="ee" pos="0 0 0.3" size="0.04" rgba="1 0.2 0.2 1"/>
          </body>
        </body>
      </body>
    </body>
    <body name="cubeA" pos="0.6 0 0.66"><freejoint name="cubeA"/><geom type="box" size="0.06 0.06 0.06" rgba="0.2 0.8 0.3 1" mass="0.1"/></body>
    <body name="cubeB" pos="0.6 0 0.78"><freejoint name="cubeB"/><geom type="box" size="0.06 0.06 0.06" rgba="0.3 0.6 0.9 1" mass="0.1"/></body>
    <body name="cubeC" pos="0.6 0 0.90"><freejoint name="cubeC"/><geom type="box" size="0.06 0.06 0.06" rgba="0.9 0.3 0.6 1" mass="0.1"/></body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="150" ctrlrange="-3.14 3.14"/>
    <position name="a2" joint="j2" kp="90" ctrlrange="-3.14 3.14"/>
    <position name="a3" joint="j3" kp="50" ctrlrange="-3.14 3.14"/>
  </actuator>
</mujoco>
"""

CUBES = ["cubeA", "cubeB", "cubeC"]


def main() -> None:
    out = outdir("04_robot_arm")
    sim = MujocoSim.from_mjcf(MJCF)

    view = dict(lookat=[0.45, 0.0, 0.55], distance=2.7, azimuth=90, elevation=-8)
    start = {c: sim.body_pos(c).copy() for c in CUBES}
    ee_path, frames = [], []
    T_sweep = 0.9
    for k in range(150):                 # 2.5 s
        # ramp the shoulder forward to sweep through the stack; keep arm straight
        j1 = min(1.0, 1.0 * sim.time / T_sweep)
        sim.data.ctrl[:3] = [j1, 0.0, 0.0]
        sim.step_for(1 / 60)
        ee_path.append(sim.site_pos("ee").copy())
        frames.append(sim.render(640, 480, **view)["rgb"])
        if k in (0, 45, 75, 149):
            save_png(frames[-1], out / f"frame_{k:03d}.png")

    save_gif(frames, out / "arm_sweep.gif", fps=30)
    disp = {c: float(np.linalg.norm(sim.body_pos(c) - start[c])) for c in CUBES}
    ee = np.array(ee_path)
    knocked = sum(d > 0.1 for d in disp.values())
    print(f"end-effector traveled {np.linalg.norm(ee[-1]-ee[0]):.2f} m; "
          f"x-range [{ee[:,0].min():.2f},{ee[:,0].max():.2f}]")
    print(f"cube displacements: { {c: round(d,2) for c,d in disp.items()} }")
    print(f"cubes knocked off (>0.1 m): {knocked}/3")
    print(f"wrote {out/'arm_sweep.gif'} + frame_*.png")


if __name__ == "__main__":
    main()
