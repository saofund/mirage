# Mirage demo cases

Presentable, end-to-end scenarios exercising the stack from **modeling → rendering
→ collision → joints → robot motion → synthetic data**. Each is self-contained and
writes real artifacts (images / GIFs / a labeled dataset) under `outputs/<case>/`.

Run them with the project venv:

```bash
uv run python examples/cases/01_modeling_render.py
uv run python examples/cases/02_collision_stack.py
uv run python examples/cases/03_pendulum_joints.py
uv run python examples/cases/04_robot_arm.py
uv run python examples/cases/05_synthetic_data.py
uv run python examples/cases/06_franka_panda.py
uv run python examples/cases/07_web_viewport.py
```

Requires the `usd`, `mujoco`, and `demos` extras (`uv pip install -e ".[usd,mujoco,demos]"`).

| # | Case | Exercises | Key outputs | Success looks like |
|---|------|-----------|-------------|--------------------|
| 01 | `01_modeling_render` | Mirage USD authoring + MuJoCo render: **RGB / depth / segmentation** | `rgb.png`, `depth.png`, `segmentation.png`, `scene.usda` | non-blank RGB; depth heatmap shows the objects; each object a distinct seg color |
| 02 | `02_collision_stack` | **Collision, friction, restitution, resting contact** | `collision.gif`, `frame_*.png` | a wrecking sphere scatters the 5-cube tower; `tower toppled: True` |
| 03 | `03_pendulum_joints` | **Hinge joints / articulation** (double pendulum) | `pendulum.gif`, `joint_angles.png`, `frame_*.png` | links swing chaotically; angle plot is non-periodic; large joint-2 travel |
| 04 | `04_robot_arm` | **Actuated hinge joints + robot motion + contact** | `arm_sweep.gif`, `frame_*.png` | the arm sweeps forward and knocks ≥1 cube off the pedestal |
| 05 | `05_synthetic_data` | **Domain randomization + multimodal sensors + auto-labels** | `annotations.json`, `contact_sheet.png`, `rgb_*.png`, `boxes_*.png` | 9 varied scenes; tight 2D boxes around each object; valid COCO JSON |
| 06 | `06_franka_panda` | **Real robot**: 7-DOF Franka Panda (Menagerie) with actuated joints | `panda.gif`, `frame_*.png` | the Panda arm drives along a reach trajectory (~3 rad joint travel) |
| 07 | `07_web_viewport` | **Web viewport**: three.js scene + trajectory playback | `index.html`, `scene.json`, `frames.json` | open `index.html` in a browser; orbit the scene; objects play back the sim |

All physics/rendering runs through MuJoCo behind Mirage's `PhysicsBackend` /
`RenderBackend` interfaces; the scene model is OpenUSD (`mirage.scene.Scene`).
