"""Set-of-Mark grounding — render the scene with object-ID overlays.

Gives a multimodal agent a render where every object is tagged with its stable id
(à la Set-of-Mark prompting), so it can link what it *sees* to the symbolic
relation graph from ``mirage.relations``. Reasoning over pixels + symbols together
is far more reliable than either alone.

Needs the `mujoco` + `demos` extras.
"""
from __future__ import annotations

import numpy as np

from .scene import Scene

DEFAULT_VIEW = {"lookat": [0, 0, 0.2], "distance": 3.0, "azimuth": 90, "elevation": -20}


def _geom_to_entity(model, gid: int):
    if gid < 0 or gid >= model.ngeom:
        return None
    name = model.geom(int(gid)).name or ""
    if not name:
        return None
    return name[:-2] if name.endswith("_g") else name


def set_of_mark(scene: Scene, view: dict | None = None, width: int = 720, height: int = 540, quality: str = "basic"):
    """Render the scene and overlay each object's 2D bbox + id. Returns
    ``(image, boxes)`` where boxes maps id -> (x0, y0, x1, y1). Ground planes are
    skipped (they're the floor, not objects)."""
    from .mujoco_backend import MujocoSim
    from .imaging import seg_ids
    view = view or DEFAULT_VIEW
    skip = {n for n in scene.entity_names()
            if scene.get_entity(n).geometry and scene.get_entity(n).geometry.kind == "plane"}

    sim = MujocoSim.from_scene(scene, quality=quality)
    imgs = sim.render(width, height, modalities=("rgb", "segmentation"), **view)
    rgb, ids = imgs["rgb"], seg_ids(imgs["segmentation"])

    boxes = {}
    for gid in np.unique(ids):
        name = _geom_to_entity(sim.model, int(gid))
        if name is None or name in skip:
            continue
        ys, xs = np.where(ids == gid)
        if xs.size < 10:
            continue
        boxes[name] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return _draw(rgb, boxes), boxes


def _draw(rgb, boxes: dict):
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(rgb).astype("uint8")).convert("RGB")
    d = ImageDraw.Draw(img)
    for name, (x0, y0, x1, y1) in boxes.items():
        d.rectangle([x0, y0, x1, y1], outline=(255, 90, 90), width=2)
        ty = max(0, y0 - 13)
        d.rectangle([x0, ty, x0 + 7 * len(name) + 4, ty + 12], fill=(255, 90, 90))
        d.text((x0 + 2, ty), name, fill=(255, 255, 255))
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# Element-level Set-of-Mark for the mesh KERNEL (per-face ids, not whole objects)
# --------------------------------------------------------------------------- #
def _project(points, view, width, height, fovy=45.0):
    """Project world points to pixel coords matching MuJoCo's free camera."""
    from .relations import camera_basis
    right, fwd, up = (np.array(v) for v in camera_basis(
        view["lookat"], view.get("distance", 3.0), view.get("azimuth", 90), view.get("elevation", -20)))
    cam = np.array(view["lookat"], float) - view.get("distance", 3.0) * fwd
    tan_y = np.tan(np.radians(fovy) / 2.0)
    tan_x = tan_y * (width / height)
    out = []
    for p in np.array(points, float):
        d = p - cam
        zc = float(d @ fwd)
        if zc <= 1e-6:
            out.append(None); continue
        u = ((d @ right) / zc / tan_x * 0.5 + 0.5) * width
        v = (1 - ((d @ up) / zc / tan_y * 0.5 + 0.5)) * height
        out.append((float(u), float(v), zc))
    return out, cam


def _draw_face_marks(rgb, marks):
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(rgb).astype("uint8")).convert("RGB")
    d = ImageDraw.Draw(img)
    for label, u, v in marks:
        w = 7 * len(label) + 4
        d.rectangle([u - 1, v - 7, u - 1 + w, v + 7], fill=(35, 115, 215))
        d.text((u + 1, v - 7), label, fill=(255, 255, 255))
        d.ellipse([u - 2, v - 2, u + 2, v + 2], fill=(255, 230, 0))
    return np.asarray(img)


def set_of_mark_mesh(mesh, view: dict | None = None, width: int = 900, height: int = 700,
                     color=(0.72, 0.74, 0.8), max_marks: int = 40):
    """Render a kernel Mesh (studio backdrop) with each visible face tagged ``F{id}``,
    so an agent can point at faces by id; returns (image, snapshot) where snapshot
    maps face id -> {centroid, screen[u,v], tags}. Faces are world == mesh coords
    (the mesh is rendered at the origin)."""
    import os
    import tempfile
    from .mujoco_backend import MujocoSim
    from .session import Session
    from .kernel import face_normal
    view = view or {"lookat": [0, 0, 0.3], "distance": 3.0, "azimuth": 125, "elevation": -15}

    cache = os.path.join(tempfile.gettempdir(), "mirage_grounding")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "_mesh.obj").replace("\\", "/")
    mesh.export_obj(path)
    s = Session(name="ground_mesh")
    s.add_mesh("m", path, position=[0, 0, 0], color=list(color), dynamic=False)
    img = MujocoSim.from_scene(s.scene, quality="studio").render(width, height, **view)["rgb"]

    cents = [[sum(v.co[k] for v in mesh.face_verts(f)) / len(mesh.face_verts(f)) for k in range(3)]
             for f in mesh.faces]
    proj, cam = _project(cents, view, width, height)
    items = []
    for i, f in enumerate(mesh.faces):
        pr = proj[i]
        if pr is None:
            continue
        u, v, zc = pr
        if not (0 <= u < width and 0 <= v < height):
            continue
        n = np.array(face_normal(mesh, f))
        if n @ (cam - np.array(cents[i])) <= 0:   # back-facing cull (ok for convex-ish)
            continue
        items.append((zc, i, u, v, f))
    items.sort()
    items = items[:max_marks]

    overlay = _draw_face_marks(img, [(f"F{i}", u, v) for _, i, u, v, _ in items])
    snapshot = {i: {"centroid": [round(x, 3) for x in cents[i]], "screen": [round(u), round(v)],
                    "tags": [t for t in f.attrs.get("tags", []) if not t.startswith("__")]}
                for _, i, u, v, f in items}
    return overlay, snapshot
