# Mirage — Design & Architecture

> **Status:** v0.1 design, post-pivot (2026-05). This document is the authoritative
> forward-looking design and **supersedes** the v0.0.1 scaffold's
> "zero-dependency, pure-Python" framing. `docs/architecture.md` still describes
> the *current* scaffold code until P0 lands.

## 1. Vision

Mirage is an **AI-native, USD-centric control layer for 3D + physics** — a thin,
scriptable "cockpit" that lets an agent (Claude Code) *or* a human author scenes,
drive best-in-class permissively-licensed engines (physics: MuJoCo / Newton;
rendering: Hydra / Cycles / Embree), and emit renders and synthetic datasets —
deterministically and reproducibly.

Think **"lightweight Blender + Isaac Sim"**, achieved *not* by re-implementing
either, but by providing the agent-drivable layer that ties chosen engines
together over one common, serializable data model.

### Mirage IS
- A serializable, diffable **scene data model** — OpenUSD as the single source of truth (SoT).
- An **AI-native control surface** — MCP + a matching Python API exposing the full
  *author → simulate → render → inspect* loop, with structured I/O and image returns.
- **Thin adapters** that make permissive engines swappable behind small interfaces.
- A **synthetic-data pipeline** (domain randomization + auto-annotation) for robotics / embodied AI.

### Mirage is NOT (non-goals)
- A new physics solver or path tracer — we integrate MuJoCo / Newton and Cycles / Embree.
- A full DCC — no sculpting / retopo / NLA-editor ambitions.
- Linked against Blender's GPL app layer (`bpy`). Standalone Cycles (Apache-2.0) is fine.

## 2. Locked decisions (2026-05)

| Axis | Choice | Rationale |
|---|---|---|
| **Core stack** | Python orchestration over native engines | Heavy lifting in USD / MuJoCo / Hydra; Python conducts. Matches Isaac (C++ plugins + Python), Genesis (Python + Taichi), Newton (Python + Warp). Rust / Warp reserved for *measured* hotspots. |
| **Wedge** | Robotics + synthetic-data **and** general 3D authoring, in parallel | Shared USD + Hydra + control-surface foundation serves both; tracks branch after P1. |
| **Scene SoT** | OpenUSD | Industrial format compat (USD ⇄ URDF / MJCF / glTF), composition / layers / variants, Hydra rendering pipeline for free. Cost: `usd-core` dependency — no longer zero-dep. |
| **Viewport** | Both web **and** native, prototyped in parallel | One render/stream backend, two frontends: web (three.js / WebGPU, AI-screenshottable) + native (wgpu). |

## 3. Architecture

```
            Claude / agent  ──MCP──┐         human ── Viewport (web | native)
            Python API ───────────┤              │  same actions = API calls
                                   ▼              ▼
                    ┌─────────────────────────────────────┐
                    │  Control surface                     │  ← Mirage's soul
                    │  one command vocabulary, two faces;  │
                    │  structured I/O · image return ·     │
                    │  replayable command log              │
                    └─────────────────────────────────────┘
                                   │
                    ┌─────────────────────────────────────┐
                    │  Runtime / Engine: step/render loop  │
                    └─────────────────────────────────────┘
                       │                          │
              Physics adapter             Render adapter (Hydra)
       MuJoCo → Newton/Warp → PhysX   HdStorm | HdEmbree | Cycles
                       │                  (AOVs: depth/seg/normal/id)
                       └────────────┬─────────────┘
                          OpenUSD Stage  (single source of truth)
                  USD ⇄ URDF/MJCF/glTF · thin agent facade · JSON diff view
                                   │
                  Synthetic-data pipeline (Replicator-style:
                  randomize → batch render → auto-annotate → dataset)
```

### Layers
1. **Scene / Data** — a USD stage is the SoT. A thin, ergonomic facade (`mirage.scene`)
   wraps `pxr` so agents aren't exposed to raw USD verbosity. JSON remains a *derived*
   diff/view format for cheap agent inspection.
2. **Engine adapters** — `PhysicsBackend` (MuJoCo → Newton/MuJoCo-Warp → PhysX) and
   `RenderBackend` over Hydra render delegates. The scaffold's "tiny swappable backend"
   instinct is kept; the backends are now real engines and the interfaces are
   USD/Hydra-shaped rather than invented.
3. **Runtime / Engine** — owns the stage + chosen delegates; drives the step/render
   loop; manages time, sensors, randomization.
4. **Control surface** — one command vocabulary; the MCP server and the Python API are
   two faces of it (MCP tools are thin wrappers over the Python API). Structured,
   round-trippable I/O; `render()` returns a PNG the multimodal agent can *see*; every
   mutating op is recorded to a replayable command log.
5. **Viewport** — a `Viewport` abstraction with web + native adapters off the same
   render/stream backend.
6. **Synthetic-data pipeline** — Replicator-style: scene + randomization spec
   (poses / materials / lighting / camera) → batch render → auto-annotations
   (RGB / depth / segmentation / normals / 2D-3D bbox) → dataset (COCO / KITTI / USD).

## 4. Tech stack & dependencies

Packaged as **extras** so the core stays light and importable with graceful guards:
`mirage[usd]`, `mirage[mujoco]`, `mirage[render]`, `mirage[viewer]`, `mirage[mcp]`, `mirage[all]`.

- **Core:** Python ≥ 3.10; `usd-core` (BSD-style) as SoT.
- **Physics:** `mujoco` (Apache-2.0) first; path to **Newton** (Warp + USD, GPU,
  differentiable) and **PhysX 5** (now fully BSD-3 incl. GPU kernels).
- **Rendering:** USD **Hydra** delegates — HdStorm (realtime), HdEmbree / **Cycles**
  (Apache-2.0 standalone) / Embree for offline; AOVs for sensor modalities.
  *Windows caveat:* pip `usd-core` is core-only (no Hydra imaging); the **early** render
  path uses **MuJoCo's built-in renderer** + the **web viewer**, with Hydra/Cycles as a
  later/optional offline path.
- **Viewport:** web (three.js / WebGPU) + native (wgpu); both consume the render/stream backend.
- **Agent surface:** `mcp` (FastMCP) + the Python API.
- **Optional accel:** NVIDIA **Warp** / **Newton**; **Madrona** batch GPU renderer for vision-RL / synth-data.

## 5. AI-native control surface (the soul)

- **One vocabulary, two faces** — MCP tools wrap the Python API; identical semantics.
- **Structured, round-trippable I/O** — ops return the affected object (JSON), not prose; clear typed errors.
- **The agent can SEE** — `render(camera) → PNG` returned to the multimodal agent; AOVs available.
- **read / diff / edit / reproduce** — full prim CRUD; `get_scene` / `set_scene` (USD
  export/import); `diff`; deterministic command log + replay. (The scaffold could only
  read + append; this closes the loop the README always promised.)
- **Orthogonal tool groups** — scene authoring · asset import (URDF/MJCF/glTF/USD) ·
  sim control · rendering/sensors · introspection (query / raycast / select) ·
  synthetic-data (randomize / batch / annotate).

## 6. Determinism & reproducibility
- A session is `(base USD stage + ordered command log)`; replay reproduces it.
- Stable USD prim paths instead of opaque handles; fixed seeds for randomization; pinned
  engine versions. Same stage + log + seed → same result.

## 7. Roadmap (both wedges in parallel)

> **Status (2026-05): P0–P5 delivered.** USD scene SoT + Session/MCP control surface
> (structured I/O, command log/replay, render-as-PNG); MuJoCo physics (collision, hinge
> joints, articulation, actuators); multimodal AOVs (RGB/depth/segmentation); URDF/MJCF +
> real MuJoCo-Menagerie robots (Franka Panda); OBJ/STL mesh import; web + native viewports;
> and a synthetic-data API — with 7 demo cases and 30 passing tests. Deferred (env-blocked):
> a Hydra/Cycles path-traced backend and native glTF I/O (see §8 and the follow-up tasks).

**Shared foundation**
- **P0 — Foundation.** USD as SoT + thin facade; unified MCP/Python command vocabulary;
  structured I/O. (no engine yet)
- **P1 — "See it".** Render a camera → PNG returned over MCP (MuJoCo renderer / web
  viewer first); prototype web + native viewports behind the `Viewport` abstraction.

**Then two interleaved tracks**
- **Robotics track**
  - **P2** — MuJoCo physics behind the adapter; import URDF/MJCF; collisions / resting
    contact (the falling box actually lands); robot-arm demo.
- **Authoring track**
  - **P3** — full prim CRUD; import glTF/USD assets; materials / lighting; command log + replay.
  - **P4** — USD ⇄ URDF/MJCF/glTF round-trip; Cycles/Embree offline path-traced render + AOVs.

**Converge**
- **P5 — Synthetic data.** Replicator-style domain randomization + batch render +
  auto-annotation → datasets. Optional Warp/Newton + Madrona acceleration. (Serves both wedges.)

**MVP wedge = P0 + P1 + (P2 or P3):** a USD-backed, agent-drivable, *visible* mini-engine —
"a lightweight Isaac/Blender you can drive with Claude."

## 8. Risks / open questions
- **USD/Hydra/Cycles on Windows** — `usd-core` wheel is core-only; full Hydra/usdview or
  Cycles may need a heavier build. *Mitigation:* MuJoCo renderer + web viewer as early
  render paths; Hydra/Cycles treated as later/optional.
- **Dependency weight vs "lightweight" promise** — addressed via extras-based install;
  core importable with graceful guards.
- **Newton maturity** — new (GTC 2025, Linux Foundation); keep MuJoCo as the stable default.

## References
- Isaac Sim / Omniverse (OpenUSD + PhysX + RTX): <https://docs.isaacsim.omniverse.nvidia.com/>
- OpenUSD + Hydra: <https://openusd.org/> · <https://docs.nvidia.com/learn-openusd/>
- MuJoCo (Apache-2.0) + MJX: <https://mujoco.org/> · <https://github.com/google-deepmind/mujoco>
- Newton (NVIDIA + Google DeepMind + Disney, Linux Foundation): <https://github.com/newton-physics/newton>
- NVIDIA Warp: <https://github.com/NVIDIA/warp> · Genesis (Taichi): <https://github.com/Genesis-Embodied-AI/Genesis>
- PhysX 5 (BSD-3): <https://github.com/NVIDIA-Omniverse/PhysX>
- Cycles permissive license: <https://code.blender.org/2013/08/cycles-render-engine-released-with-permissive-license/>
- Robot formats (URDF/MJCF/USD): <https://source-robotics.com/blogs/blog/robot-simulation-files-urdf-vs-mjcf-vs-usd>
