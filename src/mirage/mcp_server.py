"""Mirage MCP server — the AI-native control surface.

Exposes the scene/sim/render loop as MCP tools so a coding agent (Claude Code,
etc.) can build scenes, step physics, and render — all against the same
serializable ``Scene`` it can also read back as JSON.

Run (stdio transport)::

    pip install "mirage[mcp]"
    python -m mirage.mcp_server
"""
from __future__ import annotations

from .scene import Entity, Transform, Geometry, Material, PhysicsBody, Camera, Light, Scene
from .engine import Engine

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The 'mcp' package is required to run the server: pip install \"mirage[mcp]\""
    ) from exc

mcp = FastMCP("mirage")
_engine = Engine(scene=Scene(name="default"))


@mcp.tool()
def reset_scene(name: str = "default") -> str:
    """Replace the current scene with a fresh, empty one."""
    global _engine
    _engine = Engine(scene=Scene(name=name))
    return f"created empty scene '{name}'"


@mcp.tool()
def add_box(
    name: str,
    position: list[float] | None = None,
    size: list[float] | None = None,
    mass: float = 1.0,
    dynamic: bool = True,
) -> str:
    """Add a box entity. position=[x,y,z], size=[sx,sy,sz]."""
    position = position or [0.0, 0.0, 0.0]
    size = size or [1.0, 1.0, 1.0]
    _engine.scene.add(
        Entity(
            name=name,
            transform=Transform(position=list(position)),
            geometry=Geometry(kind="box", params={"size": list(size)}),
            material=Material(),
            physics=PhysicsBody(kind="dynamic" if dynamic else "static", mass=mass),
        )
    )
    return f"added box '{name}' at {position}"


@mcp.tool()
def add_camera(
    name: str,
    position: list[float] | None = None,
    width: int = 640,
    height: int = 480,
    modalities: list[str] | None = None,
) -> str:
    """Add a camera/sensor. modalities is a subset of rgb|depth|segmentation|normals."""
    position = position or [0.0, -5.0, 2.0]
    _engine.scene.add(
        Camera(
            name=name,
            transform=Transform(position=list(position)),
            width=width,
            height=height,
            modalities=list(modalities or ["rgb"]),
        )
    )
    return f"added camera '{name}'"


@mcp.tool()
def add_light(
    name: str,
    position: list[float] | None = None,
    kind: str = "point",
    intensity: float = 1.0,
) -> str:
    """Add a light (point|sun|area)."""
    position = position or [0.0, 0.0, 5.0]
    _engine.scene.add(
        Light(
            name=name,
            kind=kind,
            transform=Transform(position=list(position)),
            intensity=intensity,
        )
    )
    return f"added {kind} light '{name}'"


@mcp.tool()
def step(dt: float = 1.0 / 60.0, steps: int = 1) -> str:
    """Advance physics by ``steps`` steps of ``dt`` seconds."""
    t = _engine.step(dt=dt, steps=steps)
    return f"stepped {steps}x dt={dt:.4f}s -> t={t:.4f}s"


@mcp.tool()
def render(camera: str) -> str:
    """Render the current scene through a named camera."""
    return _engine.render(camera).summary


@mcp.tool()
def get_scene() -> str:
    """Return the full current scene as JSON (the source of truth)."""
    return _engine.scene.to_json()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
