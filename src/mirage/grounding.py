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


def set_of_mark(scene: Scene, view: dict | None = None, width: int = 720, height: int = 540):
    """Render the scene and overlay each object's 2D bbox + id. Returns
    ``(image, boxes)`` where boxes maps id -> (x0, y0, x1, y1). Ground planes are
    skipped (they're the floor, not objects)."""
    from .mujoco_backend import MujocoSim
    from .imaging import seg_ids
    view = view or DEFAULT_VIEW
    skip = {n for n in scene.entity_names()
            if scene.get_entity(n).geometry and scene.get_entity(n).geometry.kind == "plane"}

    sim = MujocoSim.from_scene(scene)
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
