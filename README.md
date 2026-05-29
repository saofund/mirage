# Mirage

**An AI-native 3D renderer + lightweight physics simulator** — built to be driven by coding agents (e.g. Claude Code via [MCP](https://modelcontextprotocol.io)), aimed at robotics and synthetic-data use cases.

> **Status:** 🌱 early scaffold (`v0.0.1`). The architecture and the AI-native control surface are in place with dependency-free *null* backends; real rendering/physics backends are on the roadmap.

## Why

Powerful DCC tools (Blender, …) have large, stateful automation surfaces that are awkward for programmatic/agent control. Full robotics simulators are excellent but heavy. Mirage takes the opposite bet:

- **Scene = plain data.** The whole world is one serializable object (JSON today, USD later). An agent can read it, diff it, edit it, and reproduce it deterministically.
- **Tiny, swappable backends.** A backend just consumes a `Scene`: `render(scene, camera)` or `step(scene, dt)`. Start with zero-dependency null backends; plug in a real renderer (Cycles/Embree/OIDN — all permissively licensed) or physics (e.g. MuJoCo) behind the same interface.
- **AI-native control surface.** A first-class MCP server exposes the build/step/render loop as a handful of orthogonal tools, so Claude Code can drive Mirage out of the box.
- **Light, fast, permissive.** Pure-Python core, zero required dependencies, Apache-2.0, no GPL entanglement.

## Quickstart

```bash
git clone https://github.com/saofund/mirage
cd mirage
pip install -e .
python examples/falling_box.py
```

## Use with Claude Code

This repo ships a **project-scoped** MCP config (`.mcp.json`), so Claude Code
picks Mirage up automatically when you open this folder as the workspace:

```bash
pip install -e ".[mcp]"   # installs the 'mirage' package + the 'mcp' dependency
cd mirage                 # the project root, where .mcp.json lives
claude                    # approve the 'mirage' MCP server when prompted
```

Then `/mcp` shows `mirage` connected and the agent can call `reset_scene`,
`add_box`, `add_camera`, `add_light`, `step`, `render`, and `get_scene`.

Run the server standalone (for any other MCP client):

```bash
python -m mirage.mcp_server
```

## Architecture

See [docs/architecture.md](docs/architecture.md). In one diagram:

```
          agent (Claude Code)
                │  MCP tools
                ▼
          ┌───────────┐    reads / writes    ┌──────────┐
          │  Engine   │◀───────────────────▶ │  Scene   │   (JSON / USD)
          └───────────┘                       └──────────┘
            │       │
     step() │       │ render()
            ▼       ▼
     PhysicsBackend   RenderBackend
     (null → MuJoCo)  (null → Cycles/Embree)
```

## License

[Apache-2.0](LICENSE).
