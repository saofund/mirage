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
import math

from .kernel import (
    Mesh, make_cube, make_cylinder_ngon, make_plane, make_uv_sphere, make_cone,
    make_torus, make_grid, make_profile, face_normal, faces_by_normal,
    extrude_faces, inset_faces, bevel_faces, loop_cut, edge_bevel,
    delete_faces, bridge_faces, fill_holes, catmull_clark,
    solidify, mirror, array, bisect, spin, screw, boolean,
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


def _place_xform(V, t, rot, s):
    """The `place` op's transform: scale -> rotate (Rz@Ry@Rx, degrees) -> translate.
    Byte-mirrored in the C++ core (program.cpp)."""
    rx, ry, rz = (math.radians(a) for a in rot)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    out = []
    for x, y, z in V:
        x *= s[0]; y *= s[1]; z *= s[2]
        y, z = y * cx - z * sx, y * sx + z * cx
        x, z = x * cy + z * sy, -x * sy + z * cy
        x, y = x * cz - y * sz, x * sz + y * cz
        out.append([x + t[0], y + t[1], z + t[2]])
    return out


def _place_material(pm):
    """Normalise a place op's material (a dict) into the stored {color,metallic,roughness}."""
    if pm is None:
        return None
    return {"color": list(pm.get("color", [0.8, 0.8, 0.8])),
            "metallic": pm.get("metallic", 0.0), "roughness": pm.get("roughness", 0.5)}


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
    if by == "material":
        col, tol = sel.get("color"), sel.get("tol", 0.02)
        out = []
        for f in mesh.faces:
            mat = f.attrs.get("material")
            if not mat:
                continue
            if col is None or all(abs(mat["color"][k] - col[k]) <= tol for k in range(3)):
                out.append(f)
        return out
    if by == "connected":
        comps = _connected_components(mesh)
        if not comps:
            return []
        if "seed" in sel:
            seed = set(map(id, _resolve(mesh, sel["seed"], last_tag)))
            return [f for comp in comps if any(id(g) in seed for g in comp) for f in comp]
        which = sel.get("which", "largest")
        return list(max(comps, key=len) if which == "largest" else min(comps, key=len))
    if by == "box":
        lo = sel.get("min", [-1e30, -1e30, -1e30])
        hi = sel.get("max", [1e30, 1e30, 1e30])
        out = []
        for f in mesh.faces:
            c = _centroid(mesh, f)
            if all(lo[k] <= c[k] <= hi[k] for k in range(3)):
                out.append(f)
        return out
    if by == "area":
        from .kernel import face_area
        if "which" in sel:
            best, bv = None, None       # single extreme face; first-wins on ties (mesh order)
            for f in mesh.faces:
                a = face_area(mesh, f)
                if best is None or (a > bv if sel["which"] == "largest" else a < bv):
                    best, bv = f, a
            return [] if best is None else [best]
        amin, amax = sel.get("min", 0.0), sel.get("max", 1e30)
        return [f for f in mesh.faces if amin <= face_area(mesh, f) <= amax]
    if by == "curvature":
        cmin, cmax = sel.get("min", 0.0), sel.get("max", 180.0)
        return [f for f in mesh.faces if cmin <= _face_curvature(mesh, f) <= cmax]
    raise MeshLangError(f"unknown selector {sel!r}")


def _face_curvature(mesh, f):
    """Local curvature proxy: the mean dihedral angle over the face's boundary edges
    (0 = flat neighbourhood, large = a bent/creased region)."""
    angs = [_dihedral_deg(mesh, lp.edge) for lp in mesh.face_loops(f)]
    return sum(angs) / len(angs) if angs else 0.0


def _connected_components(mesh):
    """Partition faces into edge-connected components (union-find over shared edges),
    returned sorted by each component's lowest face id (deterministic in both engines)."""
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    for f in mesh.faces:
        parent[f.id] = f.id
    for e in mesh.edges:
        fs = mesh.edge_faces(e)
        for i in range(1, len(fs)):
            union(fs[0].id, fs[i].id)
    groups = {}
    for f in mesh.faces:
        groups.setdefault(find(f.id), []).append(f)
    return sorted(groups.values(), key=lambda g: min(x.id for x in g))


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
    material = staticmethod(lambda color=None, tol=0.02: ({"by": "material"} if color is None
                            else {"by": "material", "color": list(color), "tol": tol}))
    connected = staticmethod(lambda which="largest": {"by": "connected", "which": which})
    component_of = staticmethod(lambda seed: {"by": "connected", "seed": seed})
    box = staticmethod(lambda lo, hi: {"by": "box", "min": list(lo), "max": list(hi)})
    area = staticmethod(lambda which="largest": {"by": "area", "which": which})
    area_range = staticmethod(lambda min=0.0, max=1e30: {"by": "area", "min": min, "max": max})
    curvature = staticmethod(lambda min=0.0, max=180.0: {"by": "curvature", "min": min, "max": max})
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
    # material groups (so the agent can re-select by paint) and connected components
    # (so it knows how many separate parts there are before selecting one)
    materials = {}
    for f in mesh.faces:
        mat = f.attrs.get("material")
        if mat:
            key = "rgb(%.2f,%.2f,%.2f)" % tuple(mat["color"])
            materials[key] = materials.get(key, 0) + 1
    comps = _connected_components(mesh)
    return {
        "stats": mesh.stats(),
        "size": [round(hi[k] - lo[k], 3) for k in range(3)],
        "bbox": [[round(x, 3) for x in lo], [round(x, 3) for x in hi]],
        "normal_groups": groups,
        "tags": tags,
        "materials": materials,
        "components": [len(c) for c in comps],   # face count of each separate part
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
# Parametric op-log: a `params` block, arithmetic EXPRESSIONS in numeric fields,
# and a `repeat` loop — resolved to a plain op-log before build(). This is what
# turns the model into a re-runnable *generator*: change one parameter and the
# whole form rebuilds. (A `place` inside a `repeat` whose transform is an
# expression of the loop index `i` grows a spiral, a colonnade, a tower...)
# --------------------------------------------------------------------------- #
_EXPR_FUNCS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "sqrt": math.sqrt,
    "abs": abs, "floor": math.floor, "ceil": math.ceil, "round": round,
    "exp": math.exp, "log": math.log, "min": min, "max": max, "pow": pow,
    "atan2": math.atan2, "hypot": math.hypot, "sign": lambda x: float((x > 0) - (x < 0)),
    "lerp": lambda a, b, t: a + (b - a) * t, "clamp": lambda x, a, b: max(a, min(b, x)),
}
_EXPR_CONSTS = {"pi": math.pi, "tau": math.tau, "e": math.e}


def _tokenize(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
        elif c in "+-*/%^(),":
            out.append(c); i += 1
        elif c.isdigit() or c == ".":
            j = i
            while j < n and (s[j].isdigit() or s[j] in ".eE" or (s[j] in "+-" and s[j - 1] in "eE")):
                j += 1
            out.append(("num", float(s[i:j]))); i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            out.append(("name", s[i:j])); i = j
        else:
            raise ValueError(f"bad character {c!r} in expression {s!r}")
    return out


def _eval_expr(s, env):
    """Evaluate an arithmetic expression string over ``env`` (the params). Recursive
    descent: + -  <  * / %  <  unary +/-  <  ^ (right-assoc)  <  atom (number | const |
    param | fn(args...)). Deterministic and side-effect-free."""
    if not isinstance(s, str):
        return s
    toks = _tokenize(s)
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = toks[pos[0]]; pos[0] += 1; return t

    def atom():
        t = take()
        if t == "(":
            v = expr()
            if take() != ")":
                raise ValueError(f"missing ) in {s!r}")
            return v
        if isinstance(t, tuple) and t[0] == "num":
            return t[1]
        if isinstance(t, tuple) and t[0] == "name":
            name = t[1]
            if peek() == "(":                      # function call
                take(); args = []
                if peek() != ")":
                    args.append(expr())
                    while peek() == ",":
                        take(); args.append(expr())
                if take() != ")":
                    raise ValueError(f"missing ) after {name}( in {s!r}")
                return float(_EXPR_FUNCS[name](*args))
            if name in _EXPR_CONSTS:
                return _EXPR_CONSTS[name]
            if name in env:
                return float(env[name])
            raise ValueError(f"unknown name {name!r} in expression {s!r}")
        raise ValueError(f"unexpected token {t!r} in {s!r}")

    def powf():
        b = atom()
        if peek() == "^":
            take(); return b ** powf()             # right-associative
        return b

    def unary():
        if peek() == "-":
            take(); return -unary()
        if peek() == "+":
            take(); return unary()
        return powf()

    def term():
        v = unary()
        while peek() in ("*", "/", "%"):
            o = take(); r = unary()
            v = v * r if o == "*" else (v / r if o == "/" else math.fmod(v, r))
        return v

    def expr():
        v = term()
        while peek() in ("+", "-"):
            o = take(); r = term()
            v = v + r if o == "+" else v - r
        return v

    v = expr()
    if pos[0] != len(toks):
        raise ValueError(f"trailing tokens in expression {s!r}")
    return v


# The numeric fields of each op (values that may be a number OR an expression string,
# possibly nested in a list). Structural fields (op / axis / plane / on / mode / name)
# are never resolved. Counts are rounded to ints so range()/tessellation stays exact.
_NUM_FIELDS = {
    "cube": ("size",), "cylinder": ("sides", "radius", "height"),
    "plane": ("size_x", "size_y"), "uv_sphere": ("segments", "rings", "radius"),
    "cone": ("sides", "radius", "height"),
    "torus": ("major_segments", "minor_segments", "major_radius", "minor_radius"),
    "grid": ("size_x", "size_y", "x_div", "y_div"), "profile": ("points",),
    "extrude": ("distance",), "inset": ("thickness",), "bevel": ("width", "depth"),
    "edge_bevel": ("width",), "solidify": ("thickness",),
    "array": ("count", "offset"), "bisect": ("point", "normal"),
    "spin": ("steps", "angle"), "screw": ("steps", "turns", "height", "angle"),
    "subdivide": ("levels",), "crease": ("weight",),
    "material": ("color", "metallic", "roughness"),
    "translate": ("by",), "scale": ("by",),
    "place": ("translate", "rotate", "scale"),
}
_INT_FIELDS = {"sides", "steps", "segments", "rings", "levels", "count",
               "major_segments", "minor_segments", "x_div", "y_div", "turns"}


def _resolve_value(v, env):
    if isinstance(v, str):
        return _eval_expr(v, env)
    if isinstance(v, list):
        return [_resolve_value(x, env) for x in v]
    return v


def _resolve_program(ops, env=None):
    """Resolve a parametric op-log to a plain one: apply ``params``, expand ``repeat``
    loops (binding the loop index), and evaluate every expression in a numeric field.
    Idempotent on a plain op-log (numbers pass straight through), so it's safe to run
    unconditionally before build — existing op-logs are unaffected."""
    env = dict(env or {})
    out = []
    for cmd in ops:
        op = cmd.get("op")
        if op == "params":
            for k, val in (cmd.get("set") or {}).items():
                env[k] = _resolve_value(val, env)
            continue
        if op == "repeat":
            count = cmd.get("count", 0)
            n = int(round(_eval_expr(count, env) if isinstance(count, str) else count))
            idx = cmd.get("index", "i")
            for k in range(max(0, n)):
                e2 = dict(env); e2[idx] = float(k)
                out.extend(_resolve_program(cmd.get("body", []), e2))
            continue
        c = dict(cmd)
        for f in _NUM_FIELDS.get(op, ()):
            if f in c:
                val = _resolve_value(c[f], env)
                if f in _INT_FIELDS and isinstance(val, float):
                    val = int(round(val))
                c[f] = val
        if op == "place":
            if isinstance(c.get("material"), dict):
                m = dict(c["material"])
                for mk in ("color", "metallic", "roughness"):
                    if mk in m:
                        m[mk] = _resolve_value(m[mk], env)
                c["material"] = m
            if "program" in c:
                c["program"] = _resolve_program(c["program"], env)
        out.append(c)
    return out


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
    def profile(self, points, plane="xz", closed=False, mark=None):
        return self.add(**_cmd("profile", mark=mark, points=[list(p) for p in points],
                               plane=plane, closed=closed))
    def delete(self, on): return self.add(**_cmd("delete", on=on))
    def bridge(self, on, mark=None): return self.add(**_cmd("bridge", on=on, mark=mark))
    def fill(self, mark=None): return self.add(**_cmd("fill", mark=mark))
    def extrude(self, on, distance=0.5, mark=None): return self.add(**_cmd("extrude", on=on, mark=mark, distance=distance))
    def inset(self, on, thickness=0.3, mark=None): return self.add(**_cmd("inset", on=on, mark=mark, thickness=thickness))
    def bevel(self, on, width=0.2, depth=0.1, mark=None): return self.add(**_cmd("bevel", on=on, mark=mark, width=width, depth=depth))
    def loop_cut(self, on, axis="z", mark=None): return self.add(**_cmd("loop_cut", on=on, mark=mark, axis=axis))
    def edge_bevel(self, on, width=0.15, mark=None): return self.add(**_cmd("edge_bevel", on=on, mark=mark, width=width))
    def solidify(self, thickness=0.1, mark=None): return self.add(**_cmd("solidify", mark=mark, thickness=thickness))
    def mirror(self, axis="x", mark=None): return self.add(**_cmd("mirror", mark=mark, axis=axis))
    def array(self, count=3, offset=(1.1, 0.0, 0.0), mark=None):
        return self.add(**_cmd("array", mark=mark, count=count, offset=list(offset)))
    def bisect(self, point=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), fill=False, mark=None):
        return self.add(**_cmd("bisect", mark=mark, point=list(point), normal=list(normal), fill=fill))
    def spin(self, axis="z", steps=24, angle=360.0, mark=None):
        return self.add(**_cmd("spin", mark=mark, axis=axis, steps=steps, angle=angle))
    def screw(self, axis="z", steps=24, turns=1, height=1.0, angle=360.0, mark=None):
        return self.add(**_cmd("screw", mark=mark, axis=axis, steps=steps, turns=turns,
                               height=height, angle=angle))
    def boolean(self, mode, cutter, mark=None):
        """Real mesh-mesh boolean. `cutter` (operand B) is a built Mesh or a
        (verts, faces) pair; the current mesh is operand A. mode: union/difference/
        intersection. The cutter geometry is baked into the op (world space)."""
        if isinstance(cutter, Mesh):
            verts = [list(v.co) for v in cutter.verts]
            faces = [[lp.vert.id for lp in cutter.face_loops(f)] for f in cutter.faces]
        else:
            verts, faces = cutter
            verts = [list(v) for v in verts]; faces = [list(f) for f in faces]
        return self.add(**_cmd("boolean", mark=mark, mode=mode, verts=verts, faces=faces))
    def subdivide(self, levels=1): return self.add(**_cmd("subdivide", levels=levels))
    def crease(self, on, weight=1.0):
        """Hold the selected EDGES sharp through a following subdivide. weight is in
        subdivision levels, so weight >= levels stays hard all the way down."""
        return self.add(**_cmd("crease", on=on, weight=weight))
    def tag(self, on, name): return self.add(**_cmd("tag", on=on, name=name))
    def material(self, on, color=(0.8, 0.8, 0.8), metallic=0.0, roughness=0.5):
        return self.add(**_cmd("material", on=on, color=list(color), metallic=metallic, roughness=roughness))
    def translate(self, on, by): return self.add(**_cmd("translate", on=on, by=list(by)))
    def scale(self, on, by): return self.add(**_cmd("scale", on=on, by=list(by)))

    def place(self, obj=None, at=(0.0, 0.0, 0.0), rotate=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0),
              material=None, verts=None, faces=None, mark=None):
        """Compose a sub-object into the running model at a transform — the SCENE op.

        ``obj`` is a sub-program (a ``MeshProgram`` or an op list) built into its own
        mesh, or pass inline ``verts``/``faces``. It is transformed (scale -> rotate
        [degrees, XYZ] -> ``at``) then **disjoint-unioned** onto the current mesh —
        which it *starts* if this is the first op. ``material`` (a dict) paints the
        placed object; without it the sub-object keeps its own materials.
        ``last_created`` resolves to the placed faces, so you can edit what you just
        placed. This is what makes the op-log natively multi-object: a scene is a
        legible list of ``place`` ops, each carrying its object's operators."""
        params = {"translate": list(at), "rotate": list(rotate), "scale": list(scale)}
        if obj is not None:
            params["program"] = obj.ops if isinstance(obj, MeshProgram) else [dict(o) for o in obj]
        else:
            params["verts"] = [list(v) for v in verts]
            params["faces"] = [list(f) for f in faces]
        if material is not None:
            params["material"] = material if isinstance(material, dict) else {
                "color": list(material[0]),
                "metallic": material[1] if len(material) > 1 else 0.0,
                "roughness": material[2] if len(material) > 2 else 0.5}
        return self.add(**_cmd("place", mark=mark, **params))

    def assert_(self, **kw): return self.add(**_cmd("assert", **kw))

    # -- parametric: the op-log as a re-runnable generator ------------------- #
    def params(self, **values):
        """Set named parameters. Later ops can reference them in expression strings
        (e.g. ``size="w*2"``); change a parameter and the whole model rebuilds."""
        return self.add(**_cmd("params", set=dict(values)))

    def repeat(self, count, body, index="i"):
        """Instantiate ``body`` (a MeshProgram / op list) ``count`` times, binding the
        loop index (``i``) so each instance's expressions can vary — a colonnade, a
        stack of floors, a spiral. ``count`` may itself be a parameter/expression."""
        body_ops = body.ops if isinstance(body, MeshProgram) else [dict(o) for o in body]
        return self.add(**_cmd("repeat", count=count, index=index, body=body_ops))

    def resolved(self):
        """The plain op-log this parametric program resolves to (params applied,
        repeats expanded, expressions evaluated) — what the kernels actually build."""
        return _resolve_program(self.ops)

    def resolved_json(self, indent=None):
        return json.dumps(_resolve_program(self.ops), indent=indent)

    # -- replay -------------------------------------------------------------- #
    def build(self) -> Mesh:
        """Replay the program into a fresh, validated mesh. The op-log is resolved first
        (``params`` / ``repeat`` / expressions -> a plain op-log), so a parametric program
        and its resolved form build identically."""
        mesh, last_tag = None, None
        for i, cmd in enumerate(_resolve_program(self.ops)):
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
                elif op == "profile":
                    # a first-class 2D generatrix (a wire polyline) — the real input to
                    # the lathe. spin/screw revolve it into a single-walled surface.
                    mesh = make_profile(cmd.get("points", []), cmd.get("plane", "xz"),
                                        cmd.get("closed", False))
                    outs = list(mesh.faces)   # a wire has no faces; last_created is undefined
                elif op == "place":
                    # compose a sub-object at a transform (the scene op). It can START
                    # the model (mesh is None) or append to it — a disjoint union, so
                    # the op-log is natively multi-object. Byte-mirrored in C++.
                    sub_ops = cmd.get("program")
                    if sub_ops is not None:
                        sub = MeshProgram(sub_ops).build()
                    else:
                        sub = Mesh.from_pydata([tuple(float(c) for c in v) for v in cmd.get("verts", [])],
                                               [list(f) for f in cmd.get("faces", [])])
                    subV = _place_xform([list(v.co) for v in sub.verts], cmd.get("translate", [0, 0, 0]),
                                        cmd.get("rotate", [0, 0, 0]), cmd.get("scale", [1, 1, 1]))
                    sidx = {v.id: k for k, v in enumerate(sub.verts)}
                    subF = [[sidx[lp.vert.id] for lp in sub.face_loops(f)] for f in sub.faces]
                    subTags = [list(_tags(f)) for f in sub.faces]
                    subMats = [f.attrs.get("material") for f in sub.faces]
                    if mesh is None:                       # place starts the model
                        aV, aF, aTags, aMats = [], [], [], []
                    else:
                        aidx = {v.id: k for k, v in enumerate(mesh.verts)}
                        aV = [list(v.co) for v in mesh.verts]
                        aF = [[aidx[lp.vert.id] for lp in mesh.face_loops(f)] for f in mesh.faces]
                        aTags = [list(_tags(f)) for f in mesh.faces]
                        aMats = [f.attrs.get("material") for f in mesh.faces]
                    nA, base = len(aF), len(aV)
                    V = aV + subV
                    F = aF + [[base + i for i in f] for f in subF]
                    Tags = aTags + subTags
                    mesh = Mesh.from_pydata(V, F, [({"tags": list(t)} if t else {}) for t in Tags])
                    place_mat = _place_material(cmd.get("material"))
                    all_mats = aMats + ([place_mat] * len(subF) if place_mat is not None else subMats)
                    for f, m2 in zip(mesh.faces, all_mats):
                        if m2:
                            f.attrs["material"] = {"color": list(m2["color"]),
                                                   "metallic": m2.get("metallic", 0.0),
                                                   "roughness": m2.get("roughness", 0.5)}
                    outs = list(mesh.faces[nA:])           # last_created = the placed object
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
                elif op == "solidify":
                    mesh = solidify(mesh, cmd.get("thickness", 0.1), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "mirror":
                    mesh = mirror(mesh, cmd.get("axis", "x"), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "array":
                    mesh = array(mesh, cmd.get("count", 3), cmd.get("offset", [1.1, 0.0, 0.0]), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "bisect":
                    mesh = bisect(mesh, cmd.get("point", [0.0, 0.0, 0.0]), cmd.get("normal", [0.0, 0.0, 1.0]),
                                  cmd.get("fill", False), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "spin":
                    mesh = spin(mesh, cmd.get("axis", "z"), cmd.get("steps", 24), cmd.get("angle", 360.0),
                                mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "screw":
                    mesh = screw(mesh, cmd.get("axis", "z"), cmd.get("steps", 24), cmd.get("turns", 1),
                                 cmd.get("height", 1.0), cmd.get("angle", 360.0), mark=out_tag)
                    outs = [f for f in mesh.faces if out_tag in _tags(f)]
                elif op == "boolean":
                    # current mesh = operand A; inline verts+faces = operand B (the tool/cutter)
                    bverts = [tuple(float(c) for c in v) for v in cmd.get("verts", [])]
                    bfaces = [list(f) for f in cmd.get("faces", [])]
                    mesh = boolean(mesh, Mesh.from_pydata(bverts, bfaces), cmd.get("mode", "difference"))
                    outs = []   # a fresh welded mesh — last_created is undefined
                elif op == "crease":
                    esel = resolve_edges(mesh, cmd.get("on", {"by": "all"}), last_tag)
                    w = cmd.get("weight", 1.0)
                    for e in esel:
                        e.crease = w
                    # Stamps sharpness for the next subdivide; no geometry changes and nothing
                    # is created, so outs stays empty (last_created still means the prior op).
                    outs = []
                elif op == "subdivide":
                    levels = cmd.get("levels", 1)
                    if levels > 0:
                        mesh = catmull_clark(mesh, levels)  # creases decay across levels
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
