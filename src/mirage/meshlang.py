"""meshlang — the AI-facing modeling language over the mesh kernel.

The model an LLM reads/edits is an **op-log program**: an ordered list of
``{op, on:<selector>, ...params, mark}`` commands. ``.build()`` replays it
deterministically into a ``kernel.Mesh`` (like ``Session.replay`` / ``Part.features``).
The LLM NEVER touches a vertex/face index — because kernel operators rebuild the
mesh and renumber every element (the Topological Naming Problem), a stored index
dies after one op. Instead:

* **Selection is a re-evaluable query** resolved against the *live* mesh:
  ``{by: normal|tag|extreme|side|all|last_created}`` composable with ``and/or/not``.
  The grammar has no field for a raw index, so fragile indexing is unrepresentable.
* **Durable handles are tags** (``mark``), carried in ``Face.attrs`` across rebuilds.
* **Every step is validated**, so a bad command localises to its op instead of
  silently corrupting the mesh.

This is the layer an LLM (or the MCP server) drives.
"""
from __future__ import annotations

import json

from .kernel import (
    Mesh, make_cube, make_cylinder_ngon, make_plane, make_uv_sphere, make_cone,
    make_torus, make_grid, face_normal, faces_by_normal,
    extrude_faces, inset_faces, bevel_faces, loop_cut, edge_bevel,
    delete_faces, bridge_faces, fill_holes, catmull_clark,
)


class SelectorEmpty(Exception):
    """A selector matched zero faces — carries diagnostics so the agent self-corrects."""
    def __init__(self, sel, diagnostics):
        super().__init__(f"selector matched 0 faces: {sel} | available: {diagnostics}")
        self.sel = sel
        self.diagnostics = diagnostics


class MeshLangError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Selectors (declarative, re-runnable against the live mesh)
# --------------------------------------------------------------------------- #
def _centroid(mesh, f):
    vs = mesh.face_verts(f)
    n = len(vs)
    return [sum(v.co[k] for v in vs) / n for k in range(3)]


def _bbox(mesh):
    cs = [v.co for v in mesh.verts]
    return [min(c[k] for c in cs) for k in range(3)], [max(c[k] for c in cs) for k in range(3)]


def _tags(f):
    return f.attrs.get("tags", [])


def _diagnostics(mesh):
    lo, hi = _bbox(mesh)
    tags = sorted({t for f in mesh.faces for t in _tags(f) if not t.startswith("__")})
    hist = {}
    for f in mesh.faces:
        n = face_normal(mesh, f)
        for k, ax in enumerate("xyz"):
            if n[k] > 0.5:
                hist[f"+{ax}"] = hist.get(f"+{ax}", 0) + 1
            elif n[k] < -0.5:
                hist[f"-{ax}"] = hist.get(f"-{ax}", 0) + 1
    return {"faces": len(mesh.faces), "bbox": [lo, hi], "tags": tags, "normal_histogram": hist}


def resolve(mesh: Mesh, sel: dict, last_tag=None) -> list:
    """Resolve a selector to a list of Faces. Raises SelectorEmpty on no match."""
    faces = _resolve(mesh, sel, last_tag)
    seen, out = set(), []
    for f in faces:
        if id(f) not in seen:
            seen.add(id(f)); out.append(f)
    if not out:
        raise SelectorEmpty(sel, _diagnostics(mesh))
    return out


def _resolve(mesh, sel, last_tag):
    if not isinstance(sel, dict):
        raise MeshLangError(f"selector must be a dict, got {sel!r}")
    if "and" in sel:
        sets = [set(map(id, _resolve(mesh, s, last_tag))) for s in sel["and"]]
        keep = set.intersection(*sets) if sets else set()
        return [f for f in mesh.faces if id(f) in keep]
    if "or" in sel:
        keep = set()
        for s in sel["or"]:
            keep |= set(map(id, _resolve(mesh, s, last_tag)))
        return [f for f in mesh.faces if id(f) in keep]
    if "not" in sel:
        excl = set(map(id, _resolve(mesh, sel["not"], last_tag)))
        return [f for f in mesh.faces if id(f) not in excl]
    by = sel.get("by")
    if by == "all":
        return list(mesh.faces)
    if by == "normal":
        if "dir" in sel:
            d = sel["dir"]
            m = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5 or 1.0
            d = [c / m for c in d]
            tol = sel.get("tol", 0.5)
            return [f for f in mesh.faces if sum(face_normal(mesh, f)[k] * d[k] for k in range(3)) > 1 - tol]
        return faces_by_normal(mesh, sel.get("axis", "z"), sel.get("sign", 1.0), sel.get("tol", 0.5))
    if by == "tag":
        name = sel["name"]
        return [f for f in mesh.faces if name in _tags(f)]
    if by == "last_created":
        return [] if last_tag is None else [f for f in mesh.faces if last_tag in _tags(f)]
    if by == "extreme":
        ax = "xyz".index(sel.get("axis", "z"))
        lo, hi = _bbox(mesh)
        tol = sel.get("tol", 0.02) * ((hi[ax] - lo[ax]) or 1.0)
        cents = [(_centroid(mesh, f)[ax], f) for f in mesh.faces]
        target = max(c for c, _ in cents) if sel.get("which", "max") == "max" else min(c for c, _ in cents)
        return [f for c, f in cents if abs(c - target) <= tol]
    if by == "side":
        ax = "xyz".index(sel.get("axis", "x"))
        lo, hi = _bbox(mesh)
        mid = (lo[ax] + hi[ax]) / 2
        return [f for f in mesh.faces if (_centroid(mesh, f)[ax] - mid) * sel.get("sign", 1.0) > 0]
    raise MeshLangError(f"unknown selector {sel!r}")


# --------------------------------------------------------------------------- #
# Edge selection (the same re-evaluable query idea, for edges) — the foundation
# for edge_bevel. Edges are NEVER addressed by index; a query resolves against
# the live mesh (sharp creases, an axis direction, the rim of a face region, ...).
# --------------------------------------------------------------------------- #
def _edge_dir(e):
    a, b = e.v1.co, e.v2.co
    d = [b[k] - a[k] for k in range(3)]
    m = (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5 or 1.0
    return [x / m for x in d]


def _dihedral_deg(mesh, e):
    """Angle between the two faces meeting at e (0 = flat/coplanar, 90 = a cube
    edge). Boundary/non-manifold edges count as fully sharp."""
    import math
    fs = mesh.edge_faces(e)
    if len(fs) != 2:
        return 180.0
    n1, n2 = face_normal(mesh, fs[0]), face_normal(mesh, fs[1])
    d = max(-1.0, min(1.0, sum(n1[k] * n2[k] for k in range(3))))
    return math.degrees(math.acos(d))


def resolve_edges(mesh: Mesh, sel: dict, last_tag=None) -> list:
    """Resolve an edge selector to a deduped list of Edges. Raises SelectorEmpty
    on no match. Grammar: {by: all|sharp|axis|boundary|on_face} + and/or/not."""
    edges = _resolve_edges(mesh, sel, last_tag)
    seen, out = set(), []
    for e in edges:
        if id(e) not in seen:
            seen.add(id(e)); out.append(e)
    if not out:
        raise SelectorEmpty(sel, _diagnostics(mesh))
    return out


def _resolve_edges(mesh, sel, last_tag):
    if not isinstance(sel, dict):
        raise MeshLangError(f"edge selector must be a dict, got {sel!r}")
    if "and" in sel:
        sets = [set(map(id, _resolve_edges(mesh, s, last_tag))) for s in sel["and"]]
        keep = set.intersection(*sets) if sets else set()
        return [e for e in mesh.edges if id(e) in keep]
    if "or" in sel:
        keep = set()
        for s in sel["or"]:
            keep |= set(map(id, _resolve_edges(mesh, s, last_tag)))
        return [e for e in mesh.edges if id(e) in keep]
    if "not" in sel:
        excl = set(map(id, _resolve_edges(mesh, sel["not"], last_tag)))
        return [e for e in mesh.edges if id(e) not in excl]
    by = sel.get("by")
    if by == "all":
        return list(mesh.edges)
    if by == "sharp":
        ang = sel.get("angle", 30.0)
        return [e for e in mesh.edges if _dihedral_deg(mesh, e) >= ang]
    if by == "axis":
        ax = "xyz".index(sel.get("axis", "z"))
        tol = sel.get("tol", 0.1)
        return [e for e in mesh.edges if abs(_edge_dir(e)[ax]) > 1 - tol]
    if by == "boundary":
        return [e for e in mesh.edges if len(mesh.edge_faces(e)) != 2]
    if by == "on_face":
        fset = set(map(id, resolve(mesh, sel["face"], last_tag)))
        return [e for e in mesh.edges if any(id(f) in fset for f in mesh.edge_faces(e))]
    raise MeshLangError(f"unknown edge selector {sel!r}")


class Sel:
    """Python sugar for selector dicts (an LLM emits the dicts directly)."""
    normal = staticmethod(lambda axis="z", sign=1.0, tol=0.5: {"by": "normal", "axis": axis, "sign": sign, "tol": tol})
    dir = staticmethod(lambda d, tol=0.5: {"by": "normal", "dir": list(d), "tol": tol})
    tag = staticmethod(lambda name: {"by": "tag", "name": name})
    extreme = staticmethod(lambda axis="z", which="max": {"by": "extreme", "axis": axis, "which": which})
    side = staticmethod(lambda axis="x", sign=1.0: {"by": "side", "axis": axis, "sign": sign})
    all = staticmethod(lambda: {"by": "all"})
    last = staticmethod(lambda: {"by": "last_created"})
    AND = staticmethod(lambda *s: {"and": list(s)})
    OR = staticmethod(lambda *s: {"or": list(s)})
    NOT = staticmethod(lambda s: {"not": s})


class ESel:
    """Python sugar for EDGE selector dicts (used by edge_bevel)."""
    all = staticmethod(lambda: {"by": "all"})
    sharp = staticmethod(lambda angle=30.0: {"by": "sharp", "angle": angle})
    axis = staticmethod(lambda axis="z", tol=0.1: {"by": "axis", "axis": axis, "tol": tol})
    boundary = staticmethod(lambda: {"by": "boundary"})
    on_face = staticmethod(lambda face: {"by": "on_face", "face": face})
    AND = staticmethod(lambda *s: {"and": list(s)})
    OR = staticmethod(lambda *s: {"or": list(s)})
    NOT = staticmethod(lambda s: {"not": s})


def describe(mesh: Mesh) -> dict:
    """A compact, AI-legible state summary (no vertex soup): invariants, size, the
    named tag groups, and how many faces point along each axis (so the agent knows
    what it can select next)."""
    lo, hi = _bbox(mesh)
    groups = {}
    for k, ax in enumerate("xyz"):
        for sign, lbl in ((1.0, "+"), (-1.0, "-")):
            cnt = sum(1 for f in mesh.faces if face_normal(mesh, f)[k] * sign > 0.5)
            if cnt:
                groups[f"{lbl}{ax}"] = cnt
    tags = {}
    for f in mesh.faces:
        for t in _tags(f):
            if not t.startswith("__"):
                tags[t] = tags.get(t, 0) + 1
    return {
        "stats": mesh.stats(),
        "size": [round(hi[k] - lo[k], 3) for k in range(3)],
        "bbox": [[round(x, 3) for x in lo], [round(x, 3) for x in hi]],
        "normal_groups": groups,
        "tags": tags,
    }


# --------------------------------------------------------------------------- #
# Region transforms (edit the selected verts in place; no rebuild)
# --------------------------------------------------------------------------- #
def _transform(mesh, faces, op, by):
    verts = list({v.id: v for f in faces for v in mesh.face_verts(f)}.values())
    if op == "translate":
        for v in verts:
            v.co = (v.co[0] + by[0], v.co[1] + by[1], v.co[2] + by[2])
    elif op == "scale":
        c = [sum(v.co[k] for v in verts) / len(verts) for k in range(3)]
        for v in verts:
            v.co = tuple(c[k] + (v.co[k] - c[k]) * by[k] for k in range(3))


def _check_assert(mesh, cmd):
    if cmd.get("closed_manifold") and not mesh.is_closed_manifold():
        raise MeshLangError("assert closed_manifold failed")
    if "euler" in cmd and mesh.euler() != cmd["euler"]:
        raise MeshLangError(f"assert euler={cmd['euler']} failed (got {mesh.euler()})")


def _cmd(op, on=None, mark=None, **params):
    c = {"op": op}
    if on is not None:
        c["on"] = on
    c.update(params)
    if mark is not None:
        c["mark"] = mark
    return c


# --------------------------------------------------------------------------- #
# The program (op-log) — the canonical, replayable model
# --------------------------------------------------------------------------- #
class MeshProgram:
    def __init__(self, ops=None):
        self.ops = [dict(o) for o in (ops or [])]

    # -- log editing --------------------------------------------------------- #
    def add(self, **cmd): self.ops.append(cmd); return self
    def insert(self, i, cmd): self.ops.insert(i, dict(cmd)); return self
    def replace(self, i, cmd): self.ops[i] = dict(cmd); return self
    def delete(self, i): del self.ops[i]; return self
    def to_json(self, indent=2): return json.dumps(self.ops, indent=indent)

    @classmethod
    def from_json(cls, s): return cls(json.loads(s))

    # -- fluent builders (sugar; an LLM just emits the op dicts) -------------- #
    def cube(self, size=1.0, mark=None): return self.add(**_cmd("cube", mark=mark, size=size))
    def cylinder(self, sides=24, radius=0.5, height=1.0, mark=None):
        return self.add(**_cmd("cylinder", mark=mark, sides=sides, radius=radius, height=height))
    def plane(self, size_x=1.0, size_y=None, mark=None):
        return self.add(**_cmd("plane", mark=mark, size_x=size_x, size_y=size_y if size_y is not None else size_x))
    def uv_sphere(self, segments=24, rings=16, radius=0.5, mark=None):
        return self.add(**_cmd("uv_sphere", mark=mark, segments=segments, rings=rings, radius=radius))
    def cone(self, sides=24, radius=0.5, height=1.0, mark=None):
        return self.add(**_cmd("cone", mark=mark, sides=sides, radius=radius, height=height))
    def torus(self, major_segments=24, minor_segments=12, major_radius=0.5, minor_radius=0.2, mark=None):
        return self.add(**_cmd("torus", mark=mark, major_segments=major_segments, minor_segments=minor_segments,
                               major_radius=major_radius, minor_radius=minor_radius))
    def grid(self, size_x=1.0, size_y=None, x_div=10, y_div=None, mark=None):
        return self.add(**_cmd("grid", mark=mark, size_x=size_x, size_y=size_y if size_y is not None else size_x,
                               x_div=x_div, y_div=y_div if y_div is not None else x_div))
    def mesh(self, verts, faces, face_materials=None, mark=None):
        cmd = _cmd("mesh", mark=mark, verts=[list(v) for v in verts], faces=[list(f) for f in faces])
        if face_materials is not None:
            cmd["face_materials"] = face_materials
        return self.add(**cmd)
    def delete(self, on): return self.add(**_cmd("delete", on=on))
    def bridge(self, on, mark=None): return self.add(**_cmd("bridge", on=on, mark=mark))
    def fill(self, mark=None): return self.add(**_cmd("fill", mark=mark))
    def extrude(self, on, distance=0.5, mark=None): return self.add(**_cmd("extrude", on=on, mark=mark, distance=distance))
    def inset(self, on, thickness=0.3, mark=None): return self.add(**_cmd("inset", on=on, mark=mark, thickness=thickness))
    def bevel(self, on, width=0.2, depth=0.1, mark=None): return self.add(**_cmd("bevel", on=on, mark=mark, width=width, depth=depth))
    def loop_cut(self, on, axis="z", mark=None): return self.add(**_cmd("loop_cut", on=on, mark=mark, axis=axis))
    def edge_bevel(self, on, width=0.15, mark=None): return self.add(**_cmd("edge_bevel", on=on, mark=mark, width=width))
    def subdivide(self, levels=1): return self.add(**_cmd("subdivide", levels=levels))
    def tag(self, on, name): return self.add(**_cmd("tag", on=on, name=name))
    def material(self, on, color=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5):
        return self.add(**_cmd("material", on=on, color=list(color), metallic=metallic, roughness=roughness))
    def translate(self, on, by): return self.add(**_cmd("translate", on=on, by=list(by)))
    def scale(self, on, by): return self.add(**_cmd("scale", on=on, by=list(by)))
    def assert_(self, **kw): return self.add(**_cmd("assert", **kw))

    # -- replay -------------------------------------------------------------- #
    def build(self) -> Mesh:
        """Replay the program into a fresh, validated mesh."""
        mesh, last_tag = None, None
        for i, cmd in enumerate(self.ops):
            op = cmd.get("op")
            out_tag = f"__out{i}"
            try:
                if op == "cube":
                    mesh = make_cube(cmd.get("size", 1.0)); outs = list(mesh.faces)
                elif op == "cylinder":
                    mesh = make_cylinder_ngon(cmd.get("sides", 24), cmd.get("radius", 0.5), cmd.get("height", 1.0))
                    outs = list(mesh.faces)
                elif op == "plane":
                    mesh = make_plane(cmd.get("size_x", 1.0), cmd.get("size_y"))
                    outs = list(mesh.faces)
                elif op == "uv_sphere":
                    mesh = make_uv_sphere(cmd.get("segments", 24), cmd.get("rings", 16), cmd.get("radius", 0.5))
                    outs = list(mesh.faces)
                elif op == "cone":
                    mesh = make_cone(cmd.get("sides", 24), cmd.get("radius", 0.5), cmd.get("height", 1.0))
                    outs = list(mesh.faces)
                elif op == "torus":
                    mesh = make_torus(cmd.get("major_segments", 24), cmd.get("minor_segments", 12),
                                      cmd.get("major_radius", 0.5), cmd.get("minor_radius", 0.2))
                    outs = list(mesh.faces)
                elif op == "grid":
                    mesh = make_grid(cmd.get("size_x", 1.0), cmd.get("size_y"),
                                     cmd.get("x_div", 10), cmd.get("y_div"))
                    outs = list(mesh.faces)
                elif op == "mesh":
                    # inline geometry (the import seam): raw verts+faces, replayed by
                    # both engines via from_pydata. face_materials[i] (or null) bakes
                    # the per-face PBR. This is what glTF import lowers to.
                    verts = [tuple(float(c) for c in v) for v in cmd.get("verts", [])]
                    faces = [list(f) for f in cmd.get("faces", [])]
                    mesh = Mesh.from_pydata(verts, faces)
                    fmats = cmd.get("face_materials")
                    if fmats:
                        for f, fm in zip(mesh.faces, fmats):
                            if fm:
                                f.attrs["material"] = {"color": list(fm.get("color", [0.8, 0.8, 0.8])),
                                                       "metallic": fm.get("metallic", 0.0),
                                                       "roughness": fm.get("roughness", 0.5)}
                    outs = list(mesh.faces)
                elif mesh is None:
                    raise MeshLangError(f"op '{op}' before any primitive")
                elif op == "extrude":
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    mesh = extrude_faces(mesh, sel, cmd.get("distance", 0.5), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "inset":
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    mesh = inset_faces(mesh, sel, cmd.get("thickness", 0.3), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "bevel":
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    mesh = bevel_faces(mesh, sel, cmd.get("width", 0.2), cmd.get("depth", 0.1), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "loop_cut":
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    mesh = loop_cut(mesh, sel, cmd.get("axis", "z"), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "edge_bevel":
                    esel = resolve_edges(mesh, cmd.get("on", {"by": "all"}), last_tag)
                    mesh = edge_bevel(mesh, esel, cmd.get("width", 0.15), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "delete":
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    mesh = delete_faces(mesh, sel)
                    outs = []  # faces removed -> last_created undefined
                elif op == "bridge":
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    mesh = bridge_faces(mesh, sel, mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "fill":
                    mesh = fill_holes(mesh, mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "subdivide":
                    for _ in range(cmd.get("levels", 1)):
                        mesh = catmull_clark(mesh)
                    outs = []  # global op — last_created is undefined afterward
                elif op == "tag":
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    for f in sel:
                        f.attrs.setdefault("tags", []).append(cmd["name"])
                    outs = sel
                elif op == "material":
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    mat = {"color": list(cmd.get("color", [0.8, 0.8, 0.8])),
                           "metallic": cmd.get("metallic", 0.0), "roughness": cmd.get("roughness", 0.5)}
                    for f in sel:
                        f.attrs["material"] = mat
                    outs = sel
                elif op in ("translate", "scale"):
                    sel = resolve(mesh, cmd.get("on", Sel.all()), last_tag)
                    _transform(mesh, sel, op, cmd.get("by", [1, 1, 1] if op == "scale" else [0, 0, 0]))
                    outs = sel
                elif op == "assert":
                    _check_assert(mesh, cmd); outs = []
                else:
                    raise MeshLangError(f"unknown op '{op}'")
            except (SelectorEmpty, MeshLangError):
                raise
            except Exception as e:  # localise any kernel error to its op
                raise MeshLangError(f"op #{i} '{op}': {type(e).__name__}: {e}") from e

            for f in outs:   # stamp the step's out tag so `last_created` resolves after ANY op
                tags = f.attrs.setdefault("tags", [])
                if out_tag not in tags:          # extrude/inset already stamped it via the kernel
                    tags.append(out_tag)
            mark = cmd.get("mark")
            if mark and outs:
                for f in outs:
                    f.attrs.setdefault("tags", []).append(mark)
            if outs:
                last_tag = out_tag
            try:
                mesh.validate()  # guardrail: every step keeps the mesh valid
            except AssertionError as e:
                raise MeshLangError(f"op #{i} '{op}' produced an invalid mesh: {e}") from e
        if mesh is None:
            raise MeshLangError("empty program")
        return mesh

    def stats(self) -> dict:
        return self.build().stats()

    def get_state(self) -> dict:
        """The AI-legible state: the op program + a semantic summary of the result."""
        return {"program": self.ops, **describe(self.build())}
