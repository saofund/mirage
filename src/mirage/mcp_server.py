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
# Mesh modeling (meshlang) — an LLM models by emitting op-log commands; the model
# is built on Mirage's own topological kernel (no trimesh), never by index.
# --------------------------------------------------------------------------- #
from .meshlang import MeshProgram  # noqa: E402

_model = MeshProgram()


@mcp.tool()
def new_model(primitive: Optional[dict] = None) -> dict:
    """Start a fresh mesh program. Optionally seed a primitive, e.g.
    {"op":"cube","size":1.0} or {"op":"cylinder","sides":24,"radius":0.5,"height":1.0}."""
    global _model
    _model = MeshProgram()
    if primitive:
        return apply_mesh_op(primitive)
    return {"ok": True, "program": []}


@mcp.tool()
def apply_mesh_op(command: dict, auto_repair: bool = True) -> dict:
    """Append ONE meshlang op and rebuild. command = {op, on:<selector>, ...params, mark?}.
    Ops: cube/cylinder (primitive); extrude{on,distance}; inset{on,thickness};
    subdivide{levels}; tag{on,name}; scale/translate{on,by}; assert{closed_manifold,euler}.
    Selectors (the `on` value): {"by":"normal","axis":"z","sign":1} | {"by":"tag","name":..} |
    {"by":"extreme","axis":"z","which":"max"} | {"by":"last_created"} | {"by":"all"} |
    {"and":[..]} / {"or":[..]} / {"not":..}.

    On success returns the new state (with `warnings` from the lint pass). On failure,
    if auto_repair is on a high-confidence, intent-preserving fix (a tag typo, a
    too-tight tol, a scalar scale, a numeric string, ...) is applied automatically and
    reported in `repaired`/`applied`; otherwise the op is rolled back and a structured
    `diagnostic` + ranked `suggestions` (intent-changing fixes for YOU to choose) are
    returned."""
    from .repair import repair_program, lint_program
    _model.ops.append(command)
    try:
        return {"ok": True, "repaired": False, "warnings": lint_program(_model), **_model.get_state()}
    except Exception as exc:
        if auto_repair:
            res = repair_program(_model.ops)
            if res.repaired:
                _model.ops[:] = res.program
                return {"ok": True, "repaired": True, "applied": res.applied, "original": command,
                        "warnings": lint_program(_model), **_model.get_state()}
            _model.ops.pop()
            return {"ok": False, "repaired": False, "error": str(exc),
                    "diagnostic": res.diagnostic, "suggestions": res.suggestions}
        _model.ops.pop()
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def get_mesh_state() -> dict:
    """The AI-legible model state: op program + invariants/size/bbox + normal groups +
    tags, plus `warnings` (lint of silent traps that build cleanly but lose intent)."""
    from .repair import lint_program
    try:
        return {"ok": True, "warnings": lint_program(_model), **_model.get_state()}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "program": _model.ops}


@mcp.tool()
def diagnose_mesh_op(command: dict) -> dict:
    """Dry-run an op WITHOUT changing the model: would it build? would auto-repair fix
    it (and how)? what are the ranked suggestions? Use before apply_mesh_op when unsure."""
    from .repair import repair_program
    res = repair_program([*_model.ops, command])
    return {"ok": res.ok, "would_auto_repair": res.repaired, "applied": res.applied,
            "diagnostic": res.diagnostic, "suggestions": res.suggestions, "attempts": res.attempts}


@mcp.tool()
def lint_mesh_program() -> dict:
    """Static check for silent traps (zero-distance extrude, clamped inset thickness,
    which!='max' silently meaning min, last_created==whole-surface) that build cleanly
    but discard intent — build() cannot catch these."""
    from .repair import lint_program
    return {"warnings": lint_program(_model)}


@mcp.tool()
def repair_mesh_geometry(eps: float = 1e-6) -> dict:
    """Geometry-level cleanup of the current built mesh: weld coincident verts, drop
    zero-area faces + orphan verts, detect (NOT delete) non-manifold edges. Returns the
    report. (Op-log programs are always clean; this is for imported/freeform meshes.)"""
    from .repair import repair_mesh
    try:
        mesh = _model.build()
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    _cleaned, report = repair_mesh(mesh, eps=eps)
    return {"ok": True, "report": report}


@mcp.tool()
def get_mesh_program() -> list:
    """The current op-log program (the canonical model)."""
    return _model.ops


@mcp.tool()
def undo_mesh_op() -> dict:
    """Drop the last op."""
    if _model.ops:
        _model.ops.pop()
    return get_mesh_state()


def _render_mesh_png(marked: bool, width: int, height: int, **view) -> Optional[bytes]:
    import numpy as np
    from PIL import Image as PILImage
    mesh = _model.build()
    if marked:
        from .grounding import set_of_mark_mesh
        rgb, _ = set_of_mark_mesh(mesh, view=view, width=width, height=height)
    else:
        import os
        import tempfile
        from .session import Session
        from .mujoco_backend import MujocoSim
        cache = os.path.join(tempfile.gettempdir(), "mirage_grounding")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, "_model.obj").replace("\\", "/")
        mesh.export_obj(path)
        minz = min(v.co[2] for v in mesh.verts)
        s = Session(name="model")
        s.add_plane("ground", size=[8, 8])
        s.add_mesh("model", path, position=[0, 0, -minz + 0.001], color=[0.82, 0.8, 0.74], dynamic=False)
        s.set_material("model", roughness=0.6)
        rgb = MujocoSim.from_scene(s.scene, quality="studio").render(width, height, **view)["rgb"]
    buf = io.BytesIO()
    PILImage.fromarray(np.asarray(rgb).astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


@mcp.tool()
def render_model(azimuth: float = 128, elevation: float = -15, distance: float = 4.0,
                 lookat: Optional[list[float]] = None, width: int = 720, height: int = 560):
    """Studio render of the current model (returns a PNG to look at)."""
    try:
        png = _render_mesh_png(False, width, height, lookat=lookat or [0, 0, 0.4],
                               azimuth=azimuth, elevation=elevation, distance=distance)
    except Exception as exc:
        return {"error": str(exc)}
    return Image(data=png, format="png")


@mcp.tool()
def render_mesh_marked(azimuth: float = 128, elevation: float = -15, distance: float = 3.2,
                       lookat: Optional[list[float]] = None, width: int = 800, height: int = 620):
    """Set-of-Mark render: the model with each visible face tagged F{id} (returns a PNG)."""
    try:
        png = _render_mesh_png(True, width, height, lookat=lookat or [0, 0, 0.4],
                               azimuth=azimuth, elevation=elevation, distance=distance)
    except Exception as exc:
        return {"error": str(exc)}
    return Image(data=png, format="png")


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
