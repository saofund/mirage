# Mirage architecture

Mirage is a small, agent-drivable engine for building 3D worlds, stepping their
physics, and rendering them. It is deliberately a *thin* core: the value is in
the data model and the control surface, not in re-implementing a DCC app.

> **Historical (v0.0.1 scaffold).** This file documents the original pure-Python,
> null-backend scaffold. The engine has since grown a **first-party native C++20
> core** — an op-log modeling kernel, the `mirage_render` path tracer, and the
> `mirage_viewer` GL GUI — so "pure-Python, zero-dependency" and "no path tracer of
> our own" below are **no longer current**. The authoritative design is
> [design.md](design.md); this page is kept for history.

## Goals

- **Agent-first.** Everything an agent needs is a small set of orthogonal
  operations over a fully serializable scene. No hidden, context-dependent state.
- **Deterministic & reproducible.** Same scene + same steps → same result.
  Entities are addressed by stable names, not opaque handles.
- **Light & fast.** Pure-Python core with zero required dependencies.
- **Permissive.** Apache-2.0, with a backend strategy that avoids GPL entanglement.

## Non-goals (for now)

- Being a full DCC / editor UI.
- Shipping our own contact-dynamics solver — we integrate MuJoCo behind a small
  interface. *(This once also disclaimed a path tracer of our own; that's no longer
  true — Mirage ships a first-party path tracer, `mirage_render`. See
  [design.md](design.md).)*

## Core concepts

### Scene (`mirage.scene`)
Plain dataclasses that serialize losslessly to/from JSON (USD later). The scene
is the single source of truth: `Entity`, `Camera`, `Light`, each with a
`Transform`; entities optionally carry `Geometry`, `Material`, and a
`PhysicsBody`. An agent reads it via `to_json()` and rebuilds it via
`from_json()`.

### Backends (`mirage.backends`)
Two tiny interfaces:
- `RenderBackend.render(scene, camera) -> RenderResult`
- `PhysicsBackend.step(scene, dt) -> None`

Reference `NullRenderer` / `NullPhysics` have no heavy deps, so the whole loop
runs and tests pass anywhere. Real backends slot in unchanged.

### Engine (`mirage.engine`)
Owns a `Scene` plus a render and a physics backend, and exposes `step()` /
`render()`.

### MCP server (`mirage.mcp_server`)
The AI-native surface. Wraps an in-memory `Engine` and exposes `reset_scene`,
`add_box`, `add_camera`, `add_light`, `step`, `render`, and `get_scene` as MCP
tools. This is how Claude Code drives Mirage.

## Module layout

```
src/mirage/
  scene.py        # data model + (de)serialization
  backends.py     # RenderBackend / PhysicsBackend + null implementations
  engine.py       # step/render loop
  mcp_server.py   # MCP tools (optional 'mcp' dependency)
examples/         # runnable demos on the null backends
tests/            # serialization + physics-loop tests
```

## Licensing strategy

Mirage is Apache-2.0. To keep the whole stack permissively licensed, real
backends are planned on permissively-licensed components rather than GPL ones:

- **Render:** Cycles (Apache-2.0, usable standalone), Embree (Apache-2.0),
  OpenImageDenoise (Apache-2.0), OpenImageIO (BSD), OpenVDB (MPL).
- **Physics:** a small custom integrator and/or MuJoCo (Apache-2.0).
- **Interchange:** glTF and USD (permissive).

We deliberately avoid linking Blender's GPL application layer (e.g. `bpy`).

## Roadmap

> **Superseded.** The forward-looking design and roadmap now live in
> [docs/design.md](design.md) (v0.1, post-pivot: **OpenUSD as the scene source of
> truth**, Python orchestration over native engines — MuJoCo/Newton physics,
> Hydra/Cycles rendering). This file documents the *current* v0.0.1 scaffold.

Original indicative roadmap (kept for history):

1. ✅ Scene model, null backends, engine loop, MCP server, tests.
2. Scene I/O: glTF import, USD round-trip.
3. Real-time raster preview backend.
4. Offline path-traced backend (Cycles/Embree) with depth/segmentation/normals.
5. Real physics backend (collisions, joints) for robotics.
6. Sensor models (depth, segmentation, LiDAR) + domain randomization for
   synthetic-data generation.
