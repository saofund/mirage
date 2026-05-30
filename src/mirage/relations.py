"""Spatial relation layer (L1) — the AI-native scene representation.

Lifts a geometric USD scene into an explicit, queryable spatial scene graph:
*what is ON / IN / NEXT-TO / LEFT-OF / ALIGNED-WITH what* — in a stated reference
frame (by default the viewer/camera frame, so "left" matches the grounding render).
Also provides constraint-style placement ops (`place_on`, `place_beside`,
`place_inside`, `align_tops`, `stack`) so an agent issues intent — "put the lamp on
the shelf" — and the system resolves it to a precise pose. This is the layer an LLM
reasons over, instead of vertex soup + 4x4 matrices.

Needs numpy (the `demos` extra). Not imported by `mirage` core.
"""
from __future__ import annotations

import numpy as np

from .scene import Scene

UP = np.array([0.0, 0.0, 1.0])

_BASIC_COLORS = {
    "red": (0.8, 0.15, 0.15), "green": (0.2, 0.7, 0.3), "blue": (0.2, 0.45, 0.85),
    "yellow": (0.9, 0.8, 0.2), "orange": (0.95, 0.55, 0.2), "purple": (0.55, 0.35, 0.75),
    "pink": (0.9, 0.45, 0.7), "cyan": (0.25, 0.7, 0.7), "white": (0.92, 0.92, 0.92),
    "gray": (0.5, 0.5, 0.52), "black": (0.08, 0.08, 0.1), "brown": (0.45, 0.3, 0.2),
}


def color_name(rgb) -> str:
    if rgb is None:
        return ""
    c = np.array(rgb[:3], float)
    return min(_BASIC_COLORS, key=lambda k: np.linalg.norm(c - np.array(_BASIC_COLORS[k])))


# --------------------------------------------------------------------------- #
# Geometry: world-space bounding boxes
# --------------------------------------------------------------------------- #
def _quat_matrix(q):
    w, x, y, z = [float(v) for v in q]
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _local_half(geom) -> np.ndarray:
    kind = geom.kind if geom else "box"
    p = (geom.params if geom else {}) or {}
    if kind == "box":
        s = p.get("size", [1, 1, 1]); return np.array([s[0] / 2, s[1] / 2, s[2] / 2])
    if kind == "sphere":
        r = p.get("radius", 0.5); return np.array([r, r, r])
    if kind == "cylinder":
        return np.array([p.get("radius", 0.5), p.get("radius", 0.5), p.get("height", 1.0) / 2])
    if kind == "plane":
        s = p.get("size", [10, 10]); return np.array([s[0], s[1], 0.01])
    return np.array([0.25, 0.25, 0.25])  # mesh / unknown fallback


def _local_bounds(geom):
    p = (geom.params if geom else {}) or {}
    if geom and geom.kind == "mesh" and "bbox_lo" in p:  # real mesh bounds baked by add_part
        return np.array(p["bbox_lo"], float), np.array(p["bbox_hi"], float)
    half = _local_half(geom)
    return -half, half


def world_aabb(entity):
    """World-space axis-aligned bounding box (lo, hi) of an entity."""
    lo_l, hi_l = _local_bounds(entity.geometry)
    R = _quat_matrix(entity.transform.rotation)
    s = np.array([float(v) for v in entity.transform.scale])
    p = np.array([float(v) for v in entity.transform.position])
    corners = np.array([
        p + R @ (np.array([cx, cy, cz]) * s)
        for cx in (lo_l[0], hi_l[0]) for cy in (lo_l[1], hi_l[1]) for cz in (lo_l[2], hi_l[2])
    ])
    return corners.min(0), corners.max(0)


def describe(scene: Scene, name: str) -> dict:
    e = scene.get_entity(name)
    lo, hi = world_aabb(e)
    color = e.material.base_color[:3] if e.material else None
    label = e.geometry.kind if e.geometry else "object"
    size = hi - lo
    cn = color_name(color)
    return {
        "id": name, "label": label, "color": cn,
        "center": (lo + hi) / 2, "size": size, "lo": lo, "hi": hi,
        "caption": f"a {cn + ' ' if cn else ''}{label} ({size[0]:.2f}x{size[1]:.2f}x{size[2]:.2f} m)",
        "physics": e.physics.kind if e.physics else None,
    }


# --------------------------------------------------------------------------- #
# Reference frame (matches the grounding render's free camera)
# --------------------------------------------------------------------------- #
def camera_basis(lookat, distance, azimuth_deg, elevation_deg):
    """Right / forward / up unit vectors of the MuJoCo free camera, so directional
    relations agree with what the grounding render shows."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])  # cam -> lookat
    view = view / (np.linalg.norm(view) or 1.0)
    right = np.cross(view, UP); right = right / (np.linalg.norm(right) or 1.0)
    up = np.cross(right, view)
    return right, view, up


def _xy_overlap(A, B):
    return A["lo"][0] < B["hi"][0] and A["hi"][0] > B["lo"][0] and \
           A["lo"][1] < B["hi"][1] and A["hi"][1] > B["lo"][1]


def _contains(B, A, tol):
    center_in = bool(np.all(A["center"] >= B["lo"]) and np.all(A["center"] <= B["hi"]))
    bbox_in = bool(np.all(A["lo"] >= B["lo"] - tol) and np.all(A["hi"] <= B["hi"] + tol))
    return center_in and bbox_in and bool(np.any(B["size"] > A["size"]))


def _gap(A, B):
    d = np.maximum.reduce([A["lo"] - B["hi"], B["lo"] - A["hi"], np.zeros(3)])
    return float(np.linalg.norm(d))


def _hgap(A, B):
    """Horizontal (xy) gap — 'next to' is about side-by-side, not stacked."""
    d = np.maximum.reduce([A["lo"][:2] - B["hi"][:2], B["lo"][:2] - A["hi"][:2], np.zeros(2)])
    return float(np.linalg.norm(d))


def _zoverlap(A, B):
    return A["lo"][2] < B["hi"][2] and A["hi"][2] > B["lo"][2]


def _direction(A, B, basis):
    """A relative to B in the viewer frame -> one dominant relation label."""
    right, view, up = basis
    d = A["center"] - B["center"]
    proj = {"right_of": d @ right, "in_front_of": -(d @ view), "above": d @ up}
    key = max(proj, key=lambda k: abs(proj[k]))
    val = proj[key]
    return {"right_of": "right_of" if val > 0 else "left_of",
            "in_front_of": "in_front_of" if val > 0 else "behind",
            "above": "above" if val > 0 else "below"}[key]


def extract_relations(scene: Scene, view: dict | None = None, near: float = 0.08, tol: float = 0.04) -> dict:
    """Build the spatial scene graph. ``view`` is the grounding render's framing
    (lookat/distance/azimuth/elevation); directionals are expressed in that frame."""
    view = view or {"lookat": [0, 0, 0.2], "distance": 3.0, "azimuth": 90, "elevation": -20}
    basis = camera_basis(view["lookat"], view.get("distance", 3.0),
                         view.get("azimuth", 90), view.get("elevation", -20))
    names = scene.entity_names()
    objs = {n: describe(scene, n) for n in names}

    rels, supports = [], {}
    # 1) containment first; a contained object is excluded from support reasoning
    contained = set()
    for a in names:
        for b in names:
            if a != b and _contains(objs[b], objs[a], tol):
                rels.append({"type": "inside", "a": a, "b": b})
                contained.add(a)
    # 2) support / on (skip pairs touching a contained object)
    for a in names:
        for b in names:
            if a == b or a in contained or b in contained:
                continue
            A, B = objs[a], objs[b]
            if _xy_overlap(A, B) and abs(A["lo"][2] - B["hi"][2]) < tol and A["center"][2] > B["center"][2]:
                rels.append({"type": "on", "a": a, "b": b})
                rels.append({"type": "supports", "a": b, "b": a})
                supports.setdefault(b, []).append(a)

    skip = {(r["a"], r["b"]) for r in rels if r["type"] in ("on", "inside")}
    skip |= {(b, a) for a, b in skip}
    # 3) near = truly adjacent pairs that aren't already on/inside
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if (a, b) in skip or a in contained or b in contained:
                continue
            if _zoverlap(objs[a], objs[b]):
                g = _hgap(objs[a], objs[b])
                if 0 < g < near:
                    rels.append({"type": "near", "a": a, "b": b, "gap": round(g, 3)})
    # 4) directionals + aligned-tops among siblings sharing a supporter (viewer frame)
    for sup, kids in supports.items():
        for i in range(len(kids)):
            for j in range(i + 1, len(kids)):
                a, b = kids[i], kids[j]
                rels.append({"type": _direction(objs[a], objs[b], basis), "a": a, "b": b, "frame": "viewer"})
                if abs(objs[a]["hi"][2] - objs[b]["hi"][2]) < tol:
                    rels.append({"type": "aligned_top", "a": a, "b": b})

    return {
        "frame": "viewer (matches the grounding render)",
        "up_axis": "z",
        "objects": {n: {
            "id": n, "label": o["label"], "color": o["color"], "caption": o["caption"],
            "center": [round(float(v), 3) for v in o["center"]],
            "size": [round(float(v), 3) for v in o["size"]],
        } for n, o in objs.items()},
        "relations": rels,
    }


def relation_sentences(graph: dict) -> list:
    """Human-readable one-liners from the relation graph."""
    out = []
    for r in graph["relations"]:
        t = r["type"]
        if t == "supports":
            out.append(f"{r['b']} rests on {r['a']}")
        elif t == "on":
            continue
        elif t == "inside":
            out.append(f"{r['a']} is inside {r['b']}")
        elif t == "near":
            out.append(f"{r['a']} is next to {r['b']} (gap {r['gap']} m)")
        elif t == "aligned_top":
            out.append(f"{r['a']} and {r['b']} have level tops")
        else:
            out.append(f"{r['a']} is {t.replace('_', ' ')} {r['b']}")
    return out


# --------------------------------------------------------------------------- #
# Constraint-style placement ops (intent -> precise pose)
# --------------------------------------------------------------------------- #
def _pos(scene, name):
    return np.array(scene.get_entity(name).transform.position, float)


def place_on(scene: Scene, a: str, b: str) -> None:
    """Rest A on top of B, centered on B's footprint."""
    A, B = describe(scene, a), describe(scene, b)
    p = _pos(scene, a)
    p[0] += B["center"][0] - A["center"][0]
    p[1] += B["center"][1] - A["center"][1]
    p[2] += B["hi"][2] - A["lo"][2]
    scene.set_transform(a, position=p.tolist())


def place_beside(scene: Scene, a: str, b: str, side: str = "left", gap: float = 0.04) -> None:
    """Place A beside B (left/right/front/back in world axes), bottoms aligned."""
    A, B = describe(scene, a), describe(scene, b)
    p = _pos(scene, a)
    if side == "left":
        p[0] += (B["lo"][0] - gap) - A["hi"][0]; p[1] += B["center"][1] - A["center"][1]
    elif side == "right":
        p[0] += (B["hi"][0] + gap) - A["lo"][0]; p[1] += B["center"][1] - A["center"][1]
    elif side == "front":
        p[1] += (B["lo"][1] - gap) - A["hi"][1]; p[0] += B["center"][0] - A["center"][0]
    elif side == "back":
        p[1] += (B["hi"][1] + gap) - A["lo"][1]; p[0] += B["center"][0] - A["center"][0]
    else:
        raise ValueError(f"side must be left/right/front/back, got {side!r}")
    p[2] += B["lo"][2] - A["lo"][2]
    scene.set_transform(a, position=p.tolist())


def place_inside(scene: Scene, a: str, b: str) -> None:
    """Center A inside B (resting on B's floor)."""
    A, B = describe(scene, a), describe(scene, b)
    p = _pos(scene, a)
    p[0] += B["center"][0] - A["center"][0]
    p[1] += B["center"][1] - A["center"][1]
    p[2] += (B["lo"][2] + 0.01) - A["lo"][2]
    scene.set_transform(a, position=p.tolist())


def align_tops(scene: Scene, names) -> None:
    """Lift each named object so all their tops match the highest top."""
    target = max(describe(scene, n)["hi"][2] for n in names)
    for n in names:
        d = describe(scene, n)
        p = _pos(scene, n)
        p[2] += target - d["hi"][2]
        scene.set_transform(n, position=p.tolist())


def stack(scene: Scene, names, base: str) -> None:
    """Stack names[0] on base, names[1] on names[0], ..."""
    under = base
    for n in names:
        place_on(scene, n, under)
        under = n
