---
name: mirage
description: >-
  Drive the Mirage AI-native 3D engine. Use when asked to model a 3D object,
  compose or render a scene, run a physics sim, or produce a render/turntable
  with Mirage — anything touching its MCP tools (new_model, apply_mesh_op,
  render_model, add_box, step, render, …), its op-log (meshlang MeshProgram),
  or its renderers (mirage_render path tracer, mirage_viewer GUI).
---

# Driving Mirage

Mirage is an AI-native 3D modeling engine. **One legible op-log is the single
source of truth**, and a human (the native GUI) and an AI (these MCP tools) are
co-equal operators on it. You author by **emitting operations**, read back
**structured state**, and **look at renders** — never by editing raw indices.

## 0. Setup (once)

The repo ships a project-scoped `.mcp.json`, so **Claude Code auto-detects the
`mirage` server** — run `/mcp` and approve it. If tools aren't present:

```bash
uv pip install -e ".[usd,mujoco,mcp,demos]"   # or: pip install -e ".[usd,mujoco,mcp,demos]"
python -m mirage.mcp_server                    # stdio server; also the `mirage-mcp` script
```

For another MCP client, register the same command (`uv run python -m
mirage.mcp_server`, cwd = repo root). Photoreal stills and the live GUI are
native C++ and are built separately (both land in `core/build/Release/`):

```bash
cmake -S core -B core/build -DMIRAGE_BUILD_VIEWER=ON   # -DMIRAGE_BUILD_VIEWER omit -> render only
cmake --build core/build --config Release              # -> mirage_render.exe, mirage_viewer.exe
```

## 1. Which surface? Pick by task

- **Model one object** (a goblet, a bracket, a jet) → the **mesh op-log** tools.
  This is Mirage's sharpest edge and where its beauty comes from.
- **Compose a scene** of many objects + physics (a stack, a robot, a dataset) →
  the **scene** tools (USD + MuJoCo).

They are different engines (see §4). Don't reach for the scene layer to model a
detailed object, or for the op-log to lay out a hundred objects.

## 2. Model authoring — the op-log loop

The core loop, **one op at a time**:

1. `new_model` (optionally seed a primitive).
2. `apply_mesh_op` — append **one** meshlang op and rebuild. `auto_repair`
   silently fixes low-risk slips (tag typo, too-tight tol, scalar scale) and
   reports them; an intent-changing mistake rolls back with a `diagnostic` and
   ranked `suggestions`.
3. Read between steps: `get_mesh_state` (program + invariants + tags + lint),
   `lint_mesh_program` (silent traps that build but lose intent),
   `diagnose_mesh_op` (dry-run an uncertain op before committing it).
4. **Look:** `render_model` (studio still) or `render_mesh_marked` (each face
   tagged `F{id}` — use this to *see* which faces to select next).
5. `undo_mesh_op` to back out; `save_mesh_program` / `load_mesh_program` to
   persist — that JSON is the **same file the native `mirage_viewer` GUI
   Loads/Saves**, so you and a human share one op-log. `export_gltf` to leave.

**Selection is always a query, never an index.** Ops take `on: {"by": …}`:
`normal` (face facing ±axis), `tag`, `material`, `extreme` (topmost/…),
`last_created`, `connected`, `box`, `area`, `curvature`, combined with
`and/or/not`. Example op dicts:

```json
{"op": "cube", "size": 1.2}
{"op": "inset",   "on": {"by": "normal", "axis": "z", "sign": 1}, "thickness": 0.3}
{"op": "extrude", "on": {"by": "last_created"}, "distance": 0.5}
{"op": "material","on": {"by": "last_created"}, "color": [1,0.78,0.34], "metallic": 1, "roughness": 0.18}
```

Operators: primitives (`cube` `cylinder` `plane` `uv_sphere` `cone` `torus`
`grid` `profile` `mesh`), edits (`extrude` `inset` `bevel` `edge_bevel`
`loop_cut` `bridge` `fill` `delete` `solidify`), global (`mirror` `array`
`bisect` `spin`=lathe `screw`=helix `subdivide` `boolean`=BSP CSG), and
annotation (`tag` `material` `translate` `scale` `assert`). Same op-log builds
**byte-identically** in the Python kernel and the C++ core (differential-tested).

## 3. Scene composition — many objects + physics

Author: `add_box` `add_sphere` `add_cylinder` `add_plane` `add_mesh`
`add_camera` `add_light`. Edit: `move` `set_transform` `set_material`
`set_velocity` `remove` `rename`. Intent-level layout: `place_on` `place_beside`
`place_inside` `stack` `align_tops`, and `relations` for the on/in/next-to graph.
Simulate & see: `step` (MuJoCo), `render` (returns a PNG), `ground` (Set-of-Mark
overlay). Reproduce: `get_scene`/`set_scene`, `save_scene`/`load_scene` (USD),
`get_log`/`replay_log`.

## 4. Performance — the rules that actually matter

Measured in `docs/scene-scaling.md`; internalize these:

- **Author, then render *once*.** Every scene edit invalidates the sim, and the
  next `render`/`step` recompiles the whole MuJoCo model. Rendering inside an add
  loop is **O(N²)** — ~17× slower at 400 objects and worsening. Add everything,
  render at the end.
- **The op-log is single-model.** Primitives *replace* the running mesh, so
  `cube`,`cube`,`cube` yields one cube. You can't compose a scene op-by-op; use
  the scene tools for many objects, or merge geometry into one `mesh` op.
- **To path-trace a *whole scene*,** merge it into one op-log `mesh` and run
  `mirage_render` — the Scene/MuJoCo layer can't reach the path tracer. The BVH
  keeps tracing sub-linear in triangles (~40k faces in seconds); the linear knobs
  are `spp`, resolution, `max_bounce`. See `examples/cases/17_city_scene.py`.

## 5. Pointers

- `docs/design.md` — the v0.1 architecture & thesis.
- `docs/scene-scaling.md` — scene scaling, bottlenecks, the whole-scene render.
- `examples/cases/` — 17 runnable scenarios (11–16 model the op-log; 17 = a scene
  at scale; 01–10 = scene/physics/robot/dataset).
- `mirage_render --oplog model.json --out shot.ppm --spp 160` — photoreal still
  of any op-log. `mirage_viewer --oplog model.json` — the live GUI.
