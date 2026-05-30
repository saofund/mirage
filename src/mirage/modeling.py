"""Parametric hard-surface modeling kernel — geometry as an editable op-graph.

A ``Part`` is a *program*: an ordered list of features (primitive + boolean op +
placement / pattern), not a baked mesh. This is the AI-legible representation
argued for in docs/design.md — an agent edits *features and parameters*
(coplanar, concentric, arrayed holes), and the kernel evaluates them to a precise
mesh via robust CSG (trimesh + manifold3d). The program is serializable, so edits
round-trip and replay deterministically.

Needs the `model` extra (trimesh + manifold3d).
"""
from __future__ import annotations

import numpy as np

_ENGINE = "manifold"


def _require():
    try:
        import trimesh  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('modeling needs the model extra: pip install "mirage[model]"') from exc


def _prim_mesh(kind: str, params: dict):
    import trimesh
    if kind == "box":
        return trimesh.creation.box(extents=params["size"])
    if kind == "cylinder":
        return trimesh.creation.cylinder(radius=params["radius"], height=params["height"],
                                         sections=params.get("sections", 48))
    if kind == "sphere":
        return trimesh.creation.icosphere(radius=params["radius"], subdivisions=params.get("subdiv", 3))
    if kind == "cone":
        return trimesh.creation.cone(radius=params["radius"], height=params["height"],
                                     sections=params.get("sections", 48))
    raise ValueError(f"unknown primitive {kind!r}")


def _placed(kind, params, pos=(0, 0, 0), rot=None):
    import trimesh.transformations as tf
    m = _prim_mesh(kind, params)
    T = tf.euler_matrix(*np.radians(rot)) if rot else np.eye(4)
    T[:3, 3] = np.array(pos, float)
    m.apply_transform(T)
    return m


def _instances(f):
    kind, params = f["prim"]["kind"], f["prim"]["params"]
    pos, rot, pat = f.get("pos", [0, 0, 0]), f.get("rot"), f.get("pattern")
    if not pat or pat["type"] == "none":
        return [_placed(kind, params, pos, rot)]
    if pat["type"] == "explicit":
        return [_placed(kind, params, p, rot) for p in pat["positions"]]
    if pat["type"] == "linear":
        sp = np.array(pat["spacing"], float)
        return [_placed(kind, params, np.array(pos, float) + i * sp, rot) for i in range(pat["count"])]
    raise ValueError(f"unknown pattern {pat['type']!r}")


class Part:
    """An editable feature-graph solid. Chain features; call ``build()`` to mesh it."""

    def __init__(self, name: str = "part"):
        _require()
        self.name = name
        self.features: list[dict] = []

    # -- feature accumulation (fluent) --------------------------------------- #
    def _feat(self, op, kind, params, pos, rot, pattern=None) -> "Part":
        self.features.append({"op": op, "prim": {"kind": kind, "params": params},
                              "pos": list(pos), "rot": list(rot) if rot else None, "pattern": pattern})
        return self

    def box(self, size, at=(0, 0, 0), rot=None, op="add"):
        return self._feat(op, "box", {"size": [float(v) for v in size]}, at, rot)

    def cylinder(self, radius, height, at=(0, 0, 0), rot=None, op="add", sections=48):
        return self._feat(op, "cylinder", {"radius": float(radius), "height": float(height), "sections": sections}, at, rot)

    def sphere(self, radius, at=(0, 0, 0), op="add"):
        return self._feat(op, "sphere", {"radius": float(radius)}, at, None)

    def cone(self, radius, height, at=(0, 0, 0), rot=None, op="add"):
        return self._feat(op, "cone", {"radius": float(radius), "height": float(height)}, at, rot)

    # -- composite features -------------------------------------------------- #
    def bolt_circle(self, hole_r, depth, circle_r, count, at=(0, 0, 0), op="cut", sections=32):
        cx, cy, cz = at
        pos = [[cx + circle_r * np.cos(2 * np.pi * i / count),
                cy + circle_r * np.sin(2 * np.pi * i / count), cz] for i in range(count)]
        self.features.append({"op": op, "prim": {"kind": "cylinder", "params": {"radius": float(hole_r), "height": float(depth), "sections": sections}},
                              "pos": [0, 0, 0], "rot": None, "pattern": {"type": "explicit", "positions": pos}})
        return self

    def hole_grid(self, hole_r, depth, nx, ny, spacing, at=(0, 0, 0), op="cut", sections=24):
        cx, cy, cz = at
        sx, sy = spacing
        pos = [[cx + (i - (nx - 1) / 2) * sx, cy + (j - (ny - 1) / 2) * sy, cz]
               for i in range(nx) for j in range(ny)]
        self.features.append({"op": op, "prim": {"kind": "cylinder", "params": {"radius": float(hole_r), "height": float(depth), "sections": sections}},
                              "pos": [0, 0, 0], "rot": None, "pattern": {"type": "explicit", "positions": pos}})
        return self

    # -- evaluation ---------------------------------------------------------- #
    def build(self):
        """Evaluate the feature graph into a watertight trimesh via CSG."""
        _require()
        mesh = None
        for f in self.features:
            insts = _instances(f)
            tool = insts[0]
            for m in insts[1:]:
                tool = tool.union(m, engine=_ENGINE)
            op = f["op"]
            if mesh is None:
                mesh = tool
            elif op == "add":
                mesh = mesh.union(tool, engine=_ENGINE)
            elif op == "cut":
                mesh = mesh.difference(tool, engine=_ENGINE)
            elif op == "intersect":
                mesh = mesh.intersection(tool, engine=_ENGINE)
            else:
                raise ValueError(f"unknown op {op!r}")
        if mesh is None:
            raise ValueError("part has no features")
        return mesh

    def export_obj(self, path) -> str:
        self.build().export(str(path))
        return str(path)

    def stats(self) -> dict:
        m = self.build()
        lo, hi = m.bounds
        return {"watertight": bool(m.is_watertight), "volume": round(float(m.volume), 5),
                "vertices": int(len(m.vertices)), "faces": int(len(m.faces)),
                "size": [round(float(v), 4) for v in (hi - lo)], "features": len(self.features)}

    # -- serialization (the editable program) -------------------------------- #
    def to_dict(self) -> dict:
        return {"name": self.name, "features": self.features}

    @classmethod
    def from_features(cls, features, name: str = "part") -> "Part":
        p = cls(name)
        p.features = list(features)
        return p


# --------------------------------------------------------------------------- #
# A small library of parametric hard-surface parts (each is just a Part program)
# --------------------------------------------------------------------------- #
def l_bracket(arm=0.5, thickness=0.08, width=0.3, hole_r=0.03, holes=2) -> Part:
    """An L-shaped mounting bracket with a row of holes in each arm."""
    p = Part("l_bracket")
    p.box([thickness, width, arm], at=[0, 0, arm / 2])                 # vertical arm
    p.box([arm, width, thickness], at=[arm / 2, 0, thickness / 2])     # horizontal arm
    sp = arm / (holes + 1)
    p.features.append({"op": "cut", "prim": {"kind": "cylinder", "params": {"radius": hole_r, "height": width * 2, "sections": 24}},
                       "pos": [0, 0, 0], "rot": [90, 0, 0],
                       "pattern": {"type": "explicit", "positions": [[thickness / 2, 0, (i + 1) * sp] for i in range(holes)]}})
    p.features.append({"op": "cut", "prim": {"kind": "cylinder", "params": {"radius": hole_r, "height": thickness * 4, "sections": 24}},
                       "pos": [0, 0, 0], "rot": None,
                       "pattern": {"type": "explicit", "positions": [[(i + 1) * sp, 0, thickness / 2] for i in range(holes)]}})
    return p


def flanged_pipe(length=0.6, outer_r=0.12, wall=0.03, flange_r=0.22, bolts=6, bolt_r=0.022) -> Part:
    """A pipe section with a bolt-circle flange at one end."""
    p = Part("flanged_pipe")
    p.cylinder(outer_r, length, at=[0, 0, length / 2])                  # outer body
    p.cylinder(flange_r, 0.04, at=[0, 0, 0.02])                         # flange disk
    p.cylinder(outer_r - wall, length * 1.2, at=[0, 0, length / 2], op="cut")  # bore
    p.bolt_circle(bolt_r, 0.1, (flange_r + outer_r) / 2, bolts, at=[0, 0, 0.02])
    return p


def perforated_plate(w=0.6, d=0.4, t=0.05, nx=5, ny=3, hole_r=0.03) -> Part:
    """A flat plate with a grid of holes."""
    p = Part("perforated_plate")
    p.box([w, d, t], at=[0, 0, t / 2])
    p.hole_grid(hole_r, t * 3, nx, ny, [w / (nx + 1), d / (ny + 1)], at=[0, 0, t / 2])
    return p
