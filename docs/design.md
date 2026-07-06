# Mirage — Design & Architecture

> **Status:** v0.1 design, post-pivot (2026-05), **updated 2026-07** as the native
> core landed. This document is the authoritative forward-looking design and
> **supersedes** the v0.0.1 scaffold's "zero-dependency, pure-Python" framing.
> Since 2026-05 Mirage grew a **first-party native C++20 core** — its own op-log
> modeling kernel, path tracer (`mirage_render`), and GL viewport (`mirage_viewer`) —
> so the old "we never re-implement an engine" stance is now **split**: we still
> integrate physics (MuJoCo), but the modeling kernel and the offline renderer are
> ours. Passages touched by that shift are updated inline. `docs/architecture.md`
> documents the original v0.0.1 scaffold, kept for history.

## 1. Vision

Mirage is an **AI-native 3D modeling engine** — a native application an agent
(Claude Code) *or* a human can drive to author models and scenes, simulate, and
render, deterministically and reproducibly. Its spine is **one legible op-log as
the single source of truth**, on which the GUI and the AI are co-equal operators.

Where it pays off, Mirage **builds the real thing** rather than wrapping a DCC: a
first-party native C++20 mesh kernel, an offline path tracer (`mirage_render`), and
a native GL viewport (`mirage_viewer`) are all in-repo. Where an engine is already
best-in-class and hard to beat, it **integrates** instead — MuJoCo for contact
physics, OpenUSD for the multi-object scene / interchange layer. So "lightweight
Blender + Isaac Sim" is now *literal* on the authoring & render side (our own kernel
and renderer) and *integrative* on the physics & scene side.

### Mirage IS
- A **legible op-log modeling kernel** — a model is an ordered program of mesh
  operators (meshlang); that op-log is the single source of truth the GUI and the AI
  both edit. Implemented **byte-identically** in the Python kernel and the C++ core.
- A **first-party offline path tracer** (`mirage_render`) and a **native GL viewport**
  (`mirage_viewer`) — the ground-truth render and the realtime preview of that same
  op-log.
- A serializable, diffable **scene layer** — OpenUSD for composing many objects and
  interchange (USD ⇄ URDF / MJCF / glTF), with MuJoCo behind it for physics.
- An **AI-native control surface** — MCP + a matching Python API exposing the full
  *author → simulate → render → inspect* loop, with structured I/O and image returns.
- A **synthetic-data pipeline** (domain randomization + auto-annotation) for robotics / embodied AI.

### Mirage is NOT (non-goals)
- **A new physics / contact solver** — we integrate MuJoCo (→ Newton / Warp later)
  rather than write our own rigid-body dynamics. *(The renderer used to sit on this
  line too — it no longer does: `mirage_render` is a first-party path tracer. Physics
  stays integrated; rendering is ours.)*
- A full DCC — no sculpting / retopo / NLA-editor ambitions.
- Linked against any GPL app layer (Blender's `bpy`, …). The core is Apache-2.0 and
  self-contained; its only heavy third-party deps are permissive (MuJoCo, `usd-core`).

## 2. Locked decisions (2026-05)

| Axis | Choice | Rationale |
|---|---|---|
| **Core stack** | First-party native **C++20 core** + Python orchestration | The modeling kernel, path tracer, and viewport are native C++ (CMake); Python conducts and hosts the AI / scene / synthetic-data layers. Physics still goes to MuJoCo. Like Isaac (C++ + Python) — but with the *authoring & render* kernel **owned**, not wrapped. |
| **Wedge** | Robotics + synthetic-data **and** general 3D authoring, in parallel | Shared USD + Hydra + control-surface foundation serves both; tracks branch after P1. |
| **Scene SoT** | OpenUSD | Industrial format compat (USD ⇄ URDF / MJCF / glTF), composition / layers / variants, Hydra rendering pipeline for free. Cost: `usd-core` dependency — no longer zero-dep. |
| **Viewport** | Both web **and** native | Native `mirage_viewer` is C++ **GLFW + OpenGL 3.3 + Dear ImGui**, driven by the op-log and headless-screenshottable; the web viewport (three.js) plays scenes back in a browser. |

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

> **The native authoring / render pillar (2026-07).** Alongside the USD/engine layers
> above, Mirage now carries a self-contained native path from op-log to pixels that
> does *not* route through USD or MuJoCo: the **meshlang op-log** → the **C++ / Python
> mesh kernel** (one model, byte-identical twins) → **`mirage_render`** (path-traced
> stills) and **`mirage_viewer`** (realtime GL preview). Single-object modeling and
> ground-truth rendering live here; the USD/MuJoCo layers own multi-object scenes and
> physics. How the two pillars connect — and where they don't yet — is measured in
> [scene-scaling.md](scene-scaling.md).

## 4. Tech stack & dependencies

Packaged as **extras** so the core stays light and importable with graceful guards:
`mirage[usd]`, `mirage[mujoco]`, `mirage[render]`, `mirage[viewer]`, `mirage[mcp]`, `mirage[all]`.

- **Core:** Python ≥ 3.10; `usd-core` (BSD-style) as SoT.
- **Physics:** `mujoco` (Apache-2.0) first; path to **Newton** (Warp + USD, GPU,
  differentiable) and **PhysX 5** (now fully BSD-3 incl. GPU kernels).
- **Rendering (offline, ground truth):** a **first-party path tracer**, `mirage_render`
  — a from-scratch physically-based Monte-Carlo tracer over the kernel mesh:
  Cook-Torrance / GGX surfaces, next-event estimation for a sky + sun environment, a
  median-split **BVH** (sub-linear in triangles), multi-threaded over scanlines,
  Russian-roulette termination, ACES tonemap; deterministic per (mesh, camera,
  settings). This **replaced** the earlier "Hydra / Cycles later" plan — the render
  pillar is ours now, with no external DCC.
- **Rendering (scene / physics preview):** **MuJoCo's built-in rasterizer** renders the
  USD/MuJoCo scene layer and its AOVs (RGB / depth / segmentation); the web viewport
  (three.js) plays scenes back in a browser.
- **Viewport:** native `mirage_viewer` (C++ GLFW + OpenGL 3.3 + Dear ImGui), driven by the op-log; plus a web viewport (three.js) for scene playback.
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
> and a synthetic-data API — with 7 demo cases and 30 passing tests.
>
> **Update (2026-07): the native core landed** — and it *replaced* the deferred
> Hydra/Cycles plan rather than waiting on it. Delivered since: a first-party **op-log
> modeling kernel** (meshlang) with ~30 operators and selection-as-query, mirrored
> **byte-identically in a C++20 core**; the **`mirage_render` path tracer** (BVH,
> GGX + NEE, multi-threaded); the native **`mirage_viewer`** GL GUI that Loads/Saves the
> same op-log; glTF import/export; a native **`place`** op that composes multi-object
> scenes in the op-log itself; and 18 demo cases with 312 passing tests. Op-log scenes
> now path-trace **directly** (no manual merge — a scene is a legible list of `place`
> ops); what remains is *connecting the USD layer* — lowering a whole USD/MuJoCo
> `Session` into the op-log (see [scene-scaling.md](scene-scaling.md)).

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
  Cycles need a heavy build. *Resolved:* the offline render pillar is now the first-party
  `mirage_render` (no Hydra/Cycles dependency at all); MuJoCo's rasterizer + the web
  viewport cover scene/physics preview. Hydra/Cycles are no longer on the path.
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
