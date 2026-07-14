# AGENTS.md — orientation for coding agents (Claude Code, Codex, …)

Mirage is an **AI-native 3D modeling engine**. Its thesis: **one legible op-log is
the single source of truth**, and a human (the native GUI) and an AI (the MCP
tools) are co-equal operators on it. If you're here to *drive* Mirage (model an
object, compose a scene, render), read **[`skills/mirage/SKILL.md`](skills/mirage/SKILL.md)** —
it's the task-focused guide. This file orients you to the repo itself.

## Layout

| Path | What |
|---|---|
| `src/mirage/` | the Python package: `meshlang.py` (the op-log), `kernel.py` (the B-rep mesh kernel), `session.py`/`scene.py` (the USD scene layer), `mcp_server.py` (the agent control surface), `mujoco_backend.py`, `capture.py` (the making-of recorder). |
| `core/` | the C++20 engine: `mirage_core` (a byte-identical twin of the kernel), `mirage_render` (the offline path tracer), `mirage_viewer` (the native GLFW/ImGui GUI). Built with CMake into `core/build/`. |
| `examples/cases/` | 17 runnable end-to-end scenarios. |
| `docs/` | `design.md` (architecture & roadmap), `scene-scaling.md` (scaling & bottlenecks), `claude-desktop.md`. |
| `skills/mirage/` | the portable agent skill for driving Mirage. |

## Build & test

```bash
uv pip install -e ".[usd,mujoco,mcp,demos]"     # Python surface (or pip install -e ".[...]")
uv run pytest tests -q                            # the suite — keep it green
cmake -S core -B core/build -DMIRAGE_BUILD_VIEWER=ON   # native engine
cmake --build core/build --config Release         # -> core/build/Release/{mirage_render,mirage_viewer}.exe
```

The Python kernel and the C++ core must stay **byte-identical**: `tests/` feeds ~100
shared op-logs through both and asserts the same element counts, the same vertex
POSITIONS, and the same faces — plus, for all but the five `bisect`/`bridge` logs named
in `_NUMBERING_DIFFERS`, the same vertex order, face order and winding. Replaying one
op-log twice in one engine must also give the same mesh, byte for byte. If you touch
either mesh engine, run the differential tests and keep them passing.

Watch for the trap those tests exist to catch: an operator that walks a container keyed
on a pointer (C++) or `id()` (Python) and emits vertices in that order is ordering its
output by heap address. Counts stay right, so it looks fine, but the same op-log then
builds a differently-numbered mesh every run. Walk elements in **id order** instead —
`region_in_id_order()` and the `sorted(..., key=lambda e: e.id)` calls in the kernel are
there for exactly this.

## Driving Mirage (the two surfaces)

The MCP server (`python -m mirage.mcp_server`, auto-registered for Claude Code via
`.mcp.json`) exposes two groups — pick by task:

- **Model one object** → the op-log loop: `new_model` → `apply_mesh_op` (one op at
  a time) → `get_mesh_state` / `lint_mesh_program` → `render_model`. Selection is a
  **query** (`on: {"by": …}`), never a raw index. This is Mirage's sharpest edge.
- **Compose a scene** of many objects + physics → `add_box`/`add_sphere`/… ,
  `place_on`/`stack`, `step`, `render`.
- **Or skip MCP** (you're a local agent in the repo): edit the op-log JSON file
  directly — a list of ops, `place` ops for a scene — then `mirage_render --oplog
  file.json --out shot.ppm` and look. The op-log *is* the model; MCP is for decoupled
  clients (Claude Desktop, remote).

Details, the full op set, and worked examples: **`skills/mirage/SKILL.md`**.

## Performance rules (measured — see `docs/scene-scaling.md`)

- **Author, then render *once*.** A scene edit invalidates the sim; the next render
  recompiles the whole MuJoCo model. Rendering inside an add loop is **O(N²)**.
- **Compose scenes with `place`** — the op-log is natively multi-object. Primitives
  replace the running mesh, but a `place` op *adds* a sub-object at a transform, so a
  whole scene is a legible list of `place` ops (`examples/cases/18_interior_scene.py`).
- **To path-trace a whole scene,** build that `place`-op-log and run `mirage_render`
  (`--cam-*` place the camera, `--threads` caps CPU workers) — the engine composes it,
  no manual merge. The Scene/MuJoCo layer still can't reach the path tracer.

## Conventions

- Apache-2.0. Structured JSON in/out of every tool — data, not prose.
- The op-log JSON is the interchange *and* the model; `mirage_viewer` and the MCP
  tools Load/Save the **same file**.
- Prefer real operators and the op-log over ad-hoc geometry; keep the two engines
  in sync.
