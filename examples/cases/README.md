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

Cases **08–18** go deeper on modeling and scale: `08` spatial relations · `09`
parametric modeling · `10` studio render · `11` the modeling kernel · `12` the
meshlang op-log · `13` a goblet from scratch · `14` mesh repair · `15` the shared
op-log (GUI ↔ AI) · `16` hard-surface & open-mesh operators · **`17` a whole scene
at scale, path-traced** (`uv run python examples/cases/17_city_scene.py`; also
`--bench` for the scaling stress test and `--hero` for the gallery render) ·
**`18` a whole interior composed by the native `place` op** (`uv run python
examples/cases/18_interior_scene.py`; `--hero` for the gallery still, `--film` for the
making-of, `--oplog` to dump the legible scene op-log).

Cases 01–10 run physics/rendering through MuJoCo behind Mirage's `PhysicsBackend` /
`RenderBackend` interfaces (scene model = OpenUSD, `mirage.scene.Scene`); 11–18 are
the op-log modeling kernel; 17 and 18 also drive the `mirage_render` path tracer (18
composes the whole scene from a legible op-log of `place` ops).
