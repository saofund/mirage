"""Mirage MCP server — the AI-native control surface.

Exposes the full *author → simulate → render → inspect → reproduce* loop as MCP
tools over one ``Session``. Every tool returns **structured JSON** (not prose);
``render`` returns a **PNG image** so a multimodal agent can see the scene and
iterate. Mutations are recorded to a replayable command log.

Run (stdio transport)::

    uv pip install -e ".[usd,mujoco,mcp,demos]"
    python -m mirage.mcp_server
"""
from __future__ import annotations

import io
from typing import Optional

from .session import Session

try:
    from mcp.server.fastmcp import FastMCP, Image
except ImportError:  # pragma: no cover
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.server.fastmcp.utilities.types import Image
    except ImportError as exc:
        raise SystemExit(
            'The "mcp" package is required to run the server: pip install "mirage[mcp]"'
        ) from exc

mcp = FastMCP("mirage")
_session = Session()


# --------------------------------------------------------------------------- #
# Scene lifecycle
# --------------------------------------------------------------------------- #
@mcp.tool()
def reset_scene(name: str = "default") -> dict:
    """Replace the current scene with a fresh, empty one (clears the command log)."""
    global _session
    _session = Session(name=name)
    return _session.reset(name)


# --------------------------------------------------------------------------- #
# Authoring — primitives, camera, light
# --------------------------------------------------------------------------- #
@mcp.tool()
def add_box(name: str, position: Optional[list[float]] = None, size: Optional[list[float]] = None,
            color: Optional[list[float]] = None, mass: float = 1.0, dynamic: bool = True) -> dict:
    """Add a box. position=[x,y,z], size=[sx,sy,sz] (full extents), color=[r,g,b(,a)]."""
    return _session.add_box(name, position=position, size=size, color=color, mass=mass, dynamic=dynamic)


@mcp.tool()
def add_sphere(name: str, position: Optional[list[float]] = None, radius: float = 0.5,
               color: Optional[list[float]] = None, mass: float = 1.0, dynamic: bool = True) -> dict:
    """Add a sphere of the given radius."""
    return _session.add_sphere(name, position=position, radius=radius, color=color, mass=mass, dynamic=dynamic)


@mcp.tool()
def add_cylinder(name: str, position: Optional[list[float]] = None, radius: float = 0.5, height: float = 1.0,
                 color: Optional[list[float]] = None, mass: float = 1.0, dynamic: bool = True) -> dict:
    """Add a cylinder of the given radius and height."""
    return _session.add_cylinder(name, position=position, radius=radius, height=height, color=color, mass=mass, dynamic=dynamic)


@mcp.tool()
def add_mesh(name: str, path: str, position: Optional[list[float]] = None, scale: Optional[list[float]] = None,
             color: Optional[list[float]] = None, mass: float = 1.0, dynamic: bool = True) -> dict:
    """Add a mesh entity from an OBJ/STL file on disk."""
    return _session.add_mesh(name, path, position=position, scale=scale, color=color, mass=mass, dynamic=dynamic)


@mcp.tool()
def add_plane(name: str = "ground", position: Optional[list[float]] = None,
              size: Optional[list[float]] = None, color: Optional[list[float]] = None) -> dict:
    """Add a static ground plane. size=[half_x, half_y]."""
    return _session.add_plane(name, position=position, size=size, color=color)


@mcp.tool()
def add_camera(name: str, position: Optional[list[float]] = None, width: int = 640, height: int = 480,
               modalities: Optional[list[str]] = None) -> dict:
    """Add a camera/sensor. modalities is a subset of rgb|depth|segmentation|normals."""
    return _session.add_camera(name, position=position, width=width, height=height, modalities=modalities)


@mcp.tool()
def add_light(name: str, position: Optional[list[float]] = None, kind: str = "point",
              color: Optional[list[float]] = None, intensity: float = 1.0) -> dict:
    """Add a light (point|sun|area)."""
    return _session.add_light(name, position=position, kind=kind, color=color, intensity=intensity)


# --------------------------------------------------------------------------- #
# Edits
# --------------------------------------------------------------------------- #
@mcp.tool()
def move(name: str, position: list[float]) -> dict:
    """Move an entity to a new position."""
    return _session.move(name, position)


@mcp.tool()
def set_transform(name: str, position: Optional[list[float]] = None,
                  rotation: Optional[list[float]] = None, scale: Optional[list[float]] = None) -> dict:
    """Set an entity's transform. rotation is a quaternion [w,x,y,z]."""
    return _session.set_transform(name, position=position, rotation=rotation, scale=scale)


@mcp.tool()
def set_material(name: str, color: Optional[list[float]] = None,
                 metallic: Optional[float] = None, roughness: Optional[float] = None) -> dict:
    """Update an entity's material (base color / metallic / roughness)."""
    return _session.set_material(name, color=color, metallic=metallic, roughness=roughness)


@mcp.tool()
def set_velocity(name: str, linear: Optional[list[float]] = None, angular: Optional[list[float]] = None) -> dict:
    """Set a dynamic entity's linear and/or angular velocity."""
    return _session.set_velocity(name, linear=linear, angular=angular)


@mcp.tool()
def remove(name: str) -> dict:
    """Remove an object by name."""
    return _session.remove(name)


@mcp.tool()
def rename(old: str, new: str) -> dict:
    """Rename an object."""
    return _session.rename(old, new)


# --------------------------------------------------------------------------- #
# Introspection & reproduce
# --------------------------------------------------------------------------- #
@mcp.tool()
def get(name: str) -> dict:
    """Get a single object (entity/camera/light) as structured data."""
    return _session.get(name)


@mcp.tool()
def list_objects() -> dict:
    """List object names grouped by kind."""
    return _session.list()


@mcp.tool()
def get_scene() -> dict:
    """Return the full scene as structured data (the JSON view of the USD source of truth)."""
    return _session.get_scene()


@mcp.tool()
def set_scene(scene: dict) -> dict:
    """Replace the scene from a structured scene dict (inverse of get_scene)."""
    return _session.set_scene(scene)


@mcp.tool()
def diff_scene(other: dict) -> dict:
    """Structured diff of the current scene against another scene dict ({} means equal)."""
    return _session.diff(other)


@mcp.tool()
def save_scene(path: str) -> dict:
    """Export the scene to USD on disk (.usda text or .usdc binary)."""
    return _session.save(path)


@mcp.tool()
def load_scene(path: str) -> dict:
    """Load a Mirage-authored USD file from disk."""
    return _session.load(path)


# --------------------------------------------------------------------------- #
# Simulate & render
# --------------------------------------------------------------------------- #
@mcp.tool()
def step(dt: float = 1.0 / 60.0, steps: int = 1) -> dict:
    """Advance physics by ``steps`` steps of ``dt`` seconds (MuJoCo when available)."""
    return _session.step(dt=dt, steps=steps)


def _render_png(width: int, height: int, **framing) -> Optional[bytes]:
    out = _session.render(width=width, height=height, modalities=("rgb",), **framing)
    if "rgb" not in out.get("data", {}):
        return None
    import numpy as np
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray(np.asarray(out["data"]["rgb"]).astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


@mcp.tool()
def render(width: int = 640, height: int = 480, azimuth: Optional[float] = None,
           elevation: Optional[float] = None, distance: Optional[float] = None,
           lookat: Optional[list[float]] = None):
    """Render the current scene to a PNG the agent can see. Framing is a free
    orbit camera: azimuth/elevation in degrees, distance in meters, lookat=[x,y,z]."""
    png = _render_png(width, height, azimuth=azimuth, elevation=elevation, distance=distance, lookat=lookat)
    if png is None:
        return {"summary": 'no pixels — install mirage[mujoco] to render', "entities": _session.list()}
    return Image(data=png, format="png")


# --------------------------------------------------------------------------- #
# Spatial layer — relations, grounding, intent-level placement
# --------------------------------------------------------------------------- #
def _view(lookat, azimuth, elevation, distance) -> dict:
    return {"lookat": lookat or [0, 0, 0.2], "azimuth": azimuth if azimuth is not None else 90,
            "elevation": elevation if elevation is not None else -20, "distance": distance or 3.0}


@mcp.tool()
def relations(lookat: Optional[list[float]] = None, azimuth: Optional[float] = None,
              elevation: Optional[float] = None, distance: Optional[float] = None) -> dict:
    """Spatial scene graph: what is on/in/next-to/left-of/aligned-with what, in the
    viewer frame matching `ground`. This is the representation to reason over."""
    return _session.relations(view=_view(lookat, azimuth, elevation, distance))


@mcp.tool()
def place_on(a: str, b: str) -> dict:
    """Rest object a on top of object b (intent -> precise pose)."""
    return _session.place_on(a, b)


@mcp.tool()
def place_beside(a: str, b: str, side: str = "left", gap: float = 0.04) -> dict:
    """Place a beside b. side = left | right | front | back."""
    return _session.place_beside(a, b, side=side, gap=gap)


@mcp.tool()
def place_inside(a: str, b: str) -> dict:
    """Place a inside b."""
    return _session.place_inside(a, b)


@mcp.tool()
def align_tops(names: list[str]) -> dict:
    """Lift the named objects so their tops are level."""
    return _session.align_tops(names)


@mcp.tool()
def stack(names: list[str], base: str) -> dict:
    """Stack the named objects in order on top of base."""
    return _session.stack(names, base)


@mcp.tool()
def ground(width: int = 720, height: int = 540, azimuth: Optional[float] = None,
           elevation: Optional[float] = None, distance: Optional[float] = None,
           lookat: Optional[list[float]] = None):
    """Set-of-Mark grounding render: the scene with each object tagged by its id, so
    the agent can link what it sees to the relation graph. Returns a PNG."""
    try:
        img, _ = _session.set_of_mark(view=_view(lookat, azimuth, elevation, distance), width=width, height=height)
    except Exception as exc:  # mujoco/demos extra missing
        return {"error": str(exc)}
    import numpy as np
    from PIL import Image as PILImage
    buf = io.BytesIO()
    PILImage.fromarray(np.asarray(img).astype("uint8")).save(buf, format="PNG")
    return Image(data=buf.getvalue(), format="png")


# --------------------------------------------------------------------------- #
# Command log (reproducibility)
# --------------------------------------------------------------------------- #
@mcp.tool()
def get_log() -> list:
    """Return the replayable command log of all mutating operations this session."""
    return _session.get_log()


@mcp.tool()
def replay_log() -> dict:
    """Deterministically rebuild the scene+simulation by replaying the command log."""
    global _session
    _session = _session.replay()
    return _session.list()


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
