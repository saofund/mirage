"""Case 06 — A real robot: the Franka Emika Panda (7-DOF).

Loads the Panda from the MuJoCo Menagerie (via ``robot_descriptions``) through
Mirage's MuJoCo backend, then drives its position actuators along a reach
trajectory. Demonstrates loading an industry-standard articulated robot and
actuating its joints. Falls back to a self-contained arm if the asset is absent.

    uv run python examples/cases/06_franka_panda.py
"""
import os
import numpy as np
from mirage.mujoco_backend import MujocoSim
from _util import outdir, save_png, save_gif

_FALLBACK = """
<mujoco model="fallback_arm">
  <option gravity="0 0 -9.81"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <worldbody>
    <light directional="true" pos="0 -1 3" dir="0 0.3 -1"/>
    <geom type="plane" size="3 3 .1" rgba=".3 .35 .4 1"/>
    <body pos="0 0 0.2"><joint name="j1" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 .5" size=".05" rgba=".85 .45 .2 1"/>
      <body pos="0 0 .5"><joint name="j2" type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0 0 .4" size=".045" rgba=".9 .6 .2 1"/></body>
    </body>
  </worldbody>
  <actuator><position joint="j1" kp="80"/><position joint="j2" kp="60"/></actuator>
</mujoco>
"""


def _load():
    try:
        from robot_descriptions import panda_mj_description
        base = os.path.dirname(panda_mj_description.MJCF_PATH)
        scene_xml = os.path.join(base, "scene.xml")
        path = scene_xml if os.path.exists(scene_xml) else panda_mj_description.MJCF_PATH
        return MujocoSim.from_mjcf_path(path), os.path.basename(path)
    except Exception as e:  # pragma: no cover - asset/network dependent
        print(f"(robot_descriptions unavailable: {type(e).__name__}; using fallback arm)")
        return MujocoSim.from_mjcf(_FALLBACK), "fallback_arm"


def main() -> None:
    out = outdir("06_franka_panda")
    sim, src = _load()
    nu, nq = sim.model.nu, sim.model.nq
    print(f"loaded {src}  (nq={nq}, nu={nu})")

    lo = sim.model.actuator_ctrlrange[:, 0]
    hi = sim.model.actuator_ctrlrange[:, 1]
    limited = sim.model.actuator_ctrllimited.astype(bool)

    home = np.zeros(nu)
    home[:min(7, nu)] = sim.data.qpos[:min(7, nu)]
    target = home.copy()
    reach = np.array([0.6, -0.4, 0.0, -2.2, 0.0, 2.0, 0.8])
    target[:min(7, nu)] = reach[:min(7, nu)]

    view = dict(lookat=[0.2, 0.0, 0.4], distance=2.2, azimuth=135, elevation=-20)
    frames, T = [], 140
    for k in range(T):
        a = min(1.0, k / (T * 0.6))
        ctrl = (1 - a) * home + a * target
        sim.data.ctrl[:nu] = np.where(limited, np.clip(ctrl, lo, hi), ctrl)
        sim.step_for(1 / 60)
        frames.append(sim.render(640, 480, **view)["rgb"])
        if k in (0, 45, 90, T - 1):
            save_png(frames[-1], out / f"frame_{k:03d}.png")

    save_gif(frames, out / "panda.gif", fps=30)
    q = np.array(sim.data.qpos[:min(7, nq)])
    travel = float(np.linalg.norm(q - home[:min(7, nq)]))
    print(f"arm joint travel: {travel:.2f} rad; final qpos: {np.round(q, 2).tolist()}")
    print(f"wrote {out/'panda.gif'} + frame_*.png")


if __name__ == "__main__":
    main()
