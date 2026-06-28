"""A from-scratch topological mesh kernel — the actual engine of a modeling tool.

This is NOT a wrapper around a geometry library; it owns the connectivity, the way
Blender's BMesh does. A mesh is **Verts, Edges, Loops, Faces** joined by circular
linked lists:

* **loop cycle** — the ordered loops around a face (its boundary). A *Loop* is a
  per-face-corner element (it is to a polygon what a half-edge is to an edge):
  it knows its vert, its edge, its face, and ``next``/``prev`` around the face.
* **radial cycle** — the loops that share one edge (``radial_next``/``radial_prev``).
  This is what lets an edge carry 1 face (boundary), 2 (manifold) or N
  (non-manifold) — the thing a plain half-edge can't do, and why Blender uses a
  radial-edge structure (after Weiler).

Modeling operators (extrude / loop-cut / inset / bevel / subdivide) are built on
this owned topology, editing only local connectivity and keeping it valid. The
academic baseline is the half-edge / DCEL structure (2-manifold); BMesh = that
idea plus the radial cycle.

Pure Python and dependency-free on purpose — clarity first; optimization later.
"""
from __future__ import annotations

from typing import Iterator, Optional


def _copy_attrs(attrs, add_tag=None):
    """Copy a face's attrs (the 'tags' list is copied, not shared); optionally append
    a tag — the durable handle that survives an operator's mesh rebuild. ``material``
    is intentionally NOT propagated: it is a final-mesh assignment (matching the C++
    engine), so assign materials after the geometry ops."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in attrs.items() if k != "material"}
    if add_tag is not None:
        out.setdefault("tags", []).append(add_tag)
    return out


class Vert:
    __slots__ = ("id", "co", "loop")

    def __init__(self, vid: int, co):
        self.id = vid
        self.co = tuple(float(c) for c in co)
        self.loop: Optional["Loop"] = None  # one incident loop (entry for traversal)


class Edge:
    __slots__ = ("id", "v1", "v2", "loop")

    def __init__(self, eid: int, v1: Vert, v2: Vert):
        self.id = eid
        self.v1, self.v2 = v1, v2
        self.loop: Optional["Loop"] = None  # one loop in this edge's radial cycle

    def other(self, v: Vert) -> Vert:
        return self.v2 if v is self.v1 else self.v1


class Loop:
    """A face corner: the BMesh/half-edge atom."""
    __slots__ = ("id", "vert", "edge", "face", "next", "prev", "radial_next", "radial_prev")

    def __init__(self, lid: int):
        self.id = lid
        self.vert: Optional[Vert] = None
        self.edge: Optional[Edge] = None
        self.face: Optional["Face"] = None
        self.next: Optional["Loop"] = None
        self.prev: Optional["Loop"] = None
        self.radial_next: "Loop" = self
        self.radial_prev: "Loop" = self


class Face:
    __slots__ = ("id", "loop", "attrs")

    def __init__(self, fid: int):
        self.id = fid
        self.loop: Optional[Loop] = None  # entry into the loop cycle
        self.attrs: dict = {}


class Mesh:
    """A boundary-representation mesh with explicit, editable topology."""

    def __init__(self):
        self.verts: list[Vert] = []
        self.edges: list[Edge] = []
        self.faces: list[Face] = []
        self._edge_map: dict[tuple[int, int], Edge] = {}
        self._lid = 0

    # -- construction -------------------------------------------------------- #
    def add_vert(self, co) -> Vert:
        v = Vert(len(self.verts), co)
        self.verts.append(v)
        return v

    def _edge(self, a: Vert, b: Vert) -> Edge:
        key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
        e = self._edge_map.get(key)
        if e is None:
            e = Edge(len(self.edges), a, b)
            self.edges.append(e)
            self._edge_map[key] = e
        return e

    @staticmethod
    def _radial_insert(e: Edge, loop: Loop) -> None:
        if e.loop is None:
            e.loop = loop
            loop.radial_next = loop.radial_prev = loop
        else:
            tail = e.loop.radial_prev
            tail.radial_next = loop
            loop.radial_prev = tail
            loop.radial_next = e.loop
            e.loop.radial_prev = loop

    def add_face(self, verts: list[Vert], attrs: dict | None = None) -> Face:
        """Create a face from an ordered list of verts (a closed loop)."""
        n = len(verts)
        if n < 3:
            raise ValueError("a face needs >= 3 verts")
        if len({id(v) for v in verts}) != n:
            raise ValueError("face has repeated vertices")
        face = Face(len(self.faces))
        if attrs:
            face.attrs = dict(attrs)
        loops = []
        for v in verts:
            lp = Loop(self._lid); self._lid += 1
            lp.vert = v; lp.face = face
            if v.loop is None:
                v.loop = lp
            loops.append(lp)
        for i, lp in enumerate(loops):
            lp.next = loops[(i + 1) % n]
            lp.prev = loops[(i - 1) % n]
            e = self._edge(verts[i], verts[(i + 1) % n])  # outgoing edge of this corner
            lp.edge = e
            self._radial_insert(e, lp)
        face.loop = loops[0]
        self.faces.append(face)
        return face

    @classmethod
    def from_pydata(cls, positions, faces, face_attrs=None) -> "Mesh":
        m = cls()
        vs = [m.add_vert(p) for p in positions]
        for i, f in enumerate(faces):
            m.add_face([vs[j] for j in f], attrs=face_attrs[i] if face_attrs else None)
        return m

    def copy(self) -> "Mesh":
        faces = [[lp.vert.id for lp in self.face_loops(f)] for f in self.faces]
        attrs = [_copy_attrs(f.attrs) for f in self.faces]
        return Mesh.from_pydata([list(v.co) for v in self.verts], faces, attrs)

    # -- traversal ----------------------------------------------------------- #
    @staticmethod
    def face_loops(f: Face) -> Iterator[Loop]:
        start = f.loop
        lp = start
        while True:
            yield lp
            lp = lp.next
            if lp is start:
                break

    def face_verts(self, f: Face) -> list[Vert]:
        return [lp.vert for lp in self.face_loops(f)]

    @staticmethod
    def edge_loops(e: Edge) -> list[Loop]:
        if e.loop is None:
            return []
        out, lp = [], e.loop
        while True:
            out.append(lp)
            lp = lp.radial_next
            if lp is e.loop:
                break
        return out

    def edge_faces(self, e: Edge) -> list[Face]:
        return [lp.face for lp in self.edge_loops(e)]

    # -- invariants ---------------------------------------------------------- #
    def euler(self) -> int:
        """V - E + F (2 for a closed genus-0 surface like a cube)."""
        return len(self.verts) - len(self.edges) + len(self.faces)

    def is_closed_manifold(self) -> bool:
        return all(len(self.edge_loops(e)) == 2 for e in self.edges)

    def validate(self) -> None:
        for f in self.faces:
            loops = list(self.face_loops(f))
            assert len(loops) >= 3, "degenerate face"
            for lp in loops:
                assert lp.face is f, "loop/face mismatch"
                assert lp.next.prev is lp, "loop cycle broken"
                assert lp.vert in (lp.edge.v1, lp.edge.v2), "loop/edge mismatch"
        for e in self.edges:
            for lp in self.edge_loops(e):
                assert lp.edge is e, "radial/edge mismatch"
                assert lp.radial_next.radial_prev is lp, "radial cycle broken"
        referenced = {lp.vert.id for f in self.faces for lp in self.face_loops(f)}
        for v in self.verts:
            assert v.id in referenced, f"orphan vertex {v.id} (no incident face)"
        for f in self.faces:  # no degenerate (zero-area) faces
            vs = self.face_verts(f)
            nx = ny = nz = 0.0
            for i in range(len(vs)):
                a, b = vs[i].co, vs[(i + 1) % len(vs)].co
                nx += (a[1] - b[1]) * (a[2] + b[2])
                ny += (a[2] - b[2]) * (a[0] + b[0])
                nz += (a[0] - b[0]) * (a[1] + b[1])
            assert nx * nx + ny * ny + nz * nz > 1e-16, f"degenerate face {f.id}"

    # -- export (triangulate only to *view* the result elsewhere) ------------ #
    def triangulate(self):
        positions = [list(v.co) for v in self.verts]
        tris = []
        for f in self.faces:
            vids = [lp.vert.id for lp in self.face_loops(f)]
            for i in range(1, len(vids) - 1):  # fan
                tris.append([vids[0], vids[i], vids[i + 1]])
        return positions, tris

    def export_obj(self, path) -> str:
        pos, tris = self.triangulate()
        with open(path, "w", encoding="utf-8") as fh:
            for p in pos:
                fh.write(f"v {p[0]} {p[1]} {p[2]}\n")
            for t in tris:
                fh.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
        return str(path)

    def stats(self) -> dict:
        return {"verts": len(self.verts), "edges": len(self.edges), "faces": len(self.faces),
                "euler": self.euler(), "closed_manifold": self.is_closed_manifold()}


# --------------------------------------------------------------------------- #
# Operators (built on the owned topology). More to come: extrude / loop-cut / ...
# --------------------------------------------------------------------------- #
def catmull_clark(mesh: Mesh) -> Mesh:
    """One level of Catmull-Clark subdivision — the classic test that a half-edge/
    loop kernel actually works: it needs face points, edge points and re-weighted
    vertex points, all found by walking the topology, then rebuilds quad faces."""
    import numpy as np
    if not mesh.faces:
        return Mesh()

    def arr(v):
        return np.array(v.co, float)

    face_list = list(mesh.faces)
    fp = {f: np.mean([arr(lp.vert) for lp in mesh.face_loops(f)], axis=0) for f in face_list}

    edge_mid, edge_pt, edge_adj = {}, {}, {}
    for e in mesh.edges:
        faces = mesh.edge_faces(e)
        edge_adj[e] = faces
        mid = (arr(e.v1) + arr(e.v2)) / 2.0
        edge_mid[e] = mid
        edge_pt[e] = (arr(e.v1) + arr(e.v2) + fp[faces[0]] + fp[faces[1]]) / 4.0 if len(faces) == 2 else mid

    v_edges: dict[Vert, list[Edge]] = {v: [] for v in mesh.verts}
    v_faces: dict[Vert, list[Face]] = {v: [] for v in mesh.verts}
    for e in mesh.edges:
        v_edges[e.v1].append(e); v_edges[e.v2].append(e)
    for f in face_list:
        for lp in mesh.face_loops(f):
            v_faces[lp.vert].append(f)

    new_v = {}
    for v in mesh.verts:
        inc_e, inc_f, P = v_edges[v], v_faces[v], arr(v)
        boundary = [e for e in inc_e if len(edge_adj[e]) < 2]
        if boundary:  # standard cubic B-spline boundary rule: (6P + sum neighbors)/(6+k)
            new_v[v] = (6 * P + sum(arr(e.other(v)) for e in boundary)) / (6 + len(boundary))
        else:
            n = len(inc_e)
            F = np.mean([fp[f] for f in inc_f], axis=0)
            R = np.mean([edge_mid[e] for e in inc_e], axis=0)
            new_v[v] = (F + 2 * R + (n - 3) * P) / n

    out_pos, iv, ie, jf = [], {}, {}, {}

    def add(p):
        out_pos.append([float(p[0]), float(p[1]), float(p[2])])
        return len(out_pos) - 1

    for v in mesh.verts:
        iv[v] = add(new_v[v])
    for e in mesh.edges:
        ie[e] = add(edge_pt[e])
    for f in face_list:
        jf[f] = add(fp[f])

    new_faces, new_attrs = [], []
    for f in face_list:
        for lp in mesh.face_loops(f):  # one quad per corner; child inherits parent face's tags
            new_faces.append([iv[lp.vert], ie[lp.edge], jf[f], ie[lp.prev.edge]])
            new_attrs.append(_copy_attrs(f.attrs))
    return Mesh.from_pydata(out_pos, new_faces, new_attrs)


def face_normal(mesh: Mesh, f: Face):
    """Unit face normal via Newell's method (robust for non-planar polygons)."""
    vs = mesh.face_verts(f)
    nx = ny = nz = 0.0
    for i in range(len(vs)):
        a, b = vs[i].co, vs[(i + 1) % len(vs)].co
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    m = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
    return (nx / m, ny / m, nz / m)


def faces_by_normal(mesh: Mesh, axis: str = "z", sign: float = 1.0, tol: float = 0.5) -> list:
    """Select faces whose normal points mostly along +/- an axis (e.g. the top)."""
    k = "xyz".index(axis)
    return [f for f in mesh.faces if face_normal(mesh, f)[k] * sign > tol]


def _compact(positions, faces):
    """Drop positions not referenced by any face and remap indices (kills orphans)."""
    used = sorted({i for f in faces for i in f})
    remap = {old: new for new, old in enumerate(used)}
    return [positions[i] for i in used], [[remap[i] for i in f] for f in faces]


def extrude_faces(mesh: Mesh, faces, distance: float = 0.5, mark: str | None = None) -> Mesh:
    """Extrude a region of faces. Each region vertex moves along the average of its
    incident region-face normals (so opposite/symmetric selections don't cancel),
    side walls bridge the boundary (edges with one region face), and orphaned
    interior verts are compacted away. Returns a fresh, valid mesh."""
    import numpy as np
    region = set(faces)
    if not region or abs(distance) < 1e-9:
        return mesh.copy()
    pos = [list(v.co) for v in mesh.verts]
    fn = {f: np.array(face_normal(mesh, f)) for f in region}
    vacc: dict = {}
    for f in region:
        for v in mesh.face_verts(f):
            vacc[v.id] = vacc.get(v.id, np.zeros(3)) + fn[f]

    new_pos, newid = list(pos), {}
    for vid in sorted(vacc):
        n = vacc[vid]
        nl = float(np.linalg.norm(n))
        d = (n / nl) * distance if nl > 1e-9 else np.zeros(3)
        newid[vid] = len(new_pos)
        new_pos.append([pos[vid][0] + d[0], pos[vid][1] + d[1], pos[vid][2] + d[2]])

    new_faces, new_attrs = [], []
    for f in mesh.faces:  # untouched faces
        if f not in region:
            new_faces.append([lp.vert.id for lp in mesh.face_loops(f)]); new_attrs.append(_copy_attrs(f.attrs))
    for e in mesh.edges:  # side walls
        adj = [f for f in mesh.edge_faces(e) if f in region]
        if len(adj) == 1:
            lp = next(l for l in mesh.face_loops(adj[0]) if l.edge is e)
            a, b = lp.vert.id, lp.next.vert.id
            new_faces.append([a, b, newid[b], newid[a]]); new_attrs.append(_copy_attrs(adj[0].attrs))
    for f in sorted(region, key=lambda f: f.id):  # caps in deterministic id order (reproducible replay)
        new_faces.append([newid[lp.vert.id] for lp in mesh.face_loops(f)]); new_attrs.append(_copy_attrs(f.attrs, add_tag=mark))

    new_pos, new_faces = _compact(new_pos, new_faces)
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def inset_faces(mesh: Mesh, faces, thickness: float = 0.3, mark: str | None = None) -> Mesh:
    """Inset each face: a smaller copy inside, ringed by border quads. ``thickness``
    is a centroid-proportional inset, clamped to the open interval (0, 1) to avoid
    the degenerate (t<=0/t>=1) and self-intersecting (bowtie) cases. The inner face
    of the last inset face is mesh.faces[-1] (handy for inset-then-extrude)."""
    import numpy as np
    region = set(faces)
    if not region:
        return mesh.copy()
    thickness = min(max(float(thickness), 1e-3), 0.999)
    new_pos = [list(v.co) for v in mesh.verts]
    new_faces, new_attrs = [], []
    for f in mesh.faces:
        if f not in region:
            new_faces.append([lp.vert.id for lp in mesh.face_loops(f)]); new_attrs.append(_copy_attrs(f.attrs))
    for f in sorted(region, key=lambda f: f.id):  # deterministic id order (reproducible replay)
        vids = [lp.vert.id for lp in mesh.face_loops(f)]
        centroid = np.mean([new_pos[i] for i in vids], axis=0)
        inner = []
        for i in vids:
            p = np.array(new_pos[i], float)
            ip = p + (centroid - p) * thickness
            inner.append(len(new_pos))
            new_pos.append([float(ip[0]), float(ip[1]), float(ip[2])])
        n = len(vids)
        for k in range(n):
            new_faces.append([vids[k], vids[(k + 1) % n], inner[(k + 1) % n], inner[k]]); new_attrs.append(_copy_attrs(f.attrs))
        new_faces.append(inner); new_attrs.append(_copy_attrs(f.attrs, add_tag=mark))  # inner face tagged
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def bevel_faces(mesh: Mesh, faces, width: float = 0.2, depth: float = 0.1, mark: str | None = None) -> Mesh:
    """Bevel (chamfer) the rim of each face: an inset ring of ``width`` whose inner
    face is offset ``depth`` along the face normal, so the border quads slant into a
    chamfer. This is the face-region analogue of an edge bevel (and exactly
    Blender's Inset Faces = Thickness + Depth). depth>0 raises a chamfered boss,
    depth<0 sinks a chamfered recess; depth=0 is a plain inset. Same topology as
    inset, so euler/manifold are preserved. The inner face is mesh.faces[-1]."""
    import numpy as np
    region = set(faces)
    if not region:
        return mesh.copy()
    width = min(max(float(width), 1e-3), 0.999)
    new_pos = [list(v.co) for v in mesh.verts]
    new_faces, new_attrs = [], []
    for f in mesh.faces:
        if f not in region:
            new_faces.append([lp.vert.id for lp in mesh.face_loops(f)]); new_attrs.append(_copy_attrs(f.attrs))
    for f in sorted(region, key=lambda f: f.id):  # deterministic id order (reproducible replay)
        normal = np.array(face_normal(mesh, f))
        vids = [lp.vert.id for lp in mesh.face_loops(f)]
        centroid = np.mean([new_pos[i] for i in vids], axis=0)
        inner = []
        for i in vids:
            p = np.array(new_pos[i], float)
            ip = p + (centroid - p) * width + normal * depth  # inset toward centroid, then lift along normal
            inner.append(len(new_pos))
            new_pos.append([float(ip[0]), float(ip[1]), float(ip[2])])
        n = len(vids)
        for k in range(n):
            new_faces.append([vids[k], vids[(k + 1) % n], inner[(k + 1) % n], inner[k]]); new_attrs.append(_copy_attrs(f.attrs))
        new_faces.append(inner); new_attrs.append(_copy_attrs(f.attrs, add_tag=mark))  # inner face tagged
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def loop_cut(mesh: Mesh, seed_faces, axis: str = "z", mark: str | None = None) -> Mesh:
    """Insert an edge loop. From a seed quad, walk the ring of quads whose shared
    edges run along ``axis`` and bisect each one, threading a continuous loop of
    new midpoint vertices (so the cut is watertight). Only quad strips are cut —
    an n-gon (e.g. a cylinder cap) stops the walk. Returns a fresh, valid mesh.

    This is the classic hard-surface loop cut: e.g. a loop around a cylinder's
    waist, or a horizontal band around a cube. The cut faces are tagged ``mark``."""
    import numpy as np
    seed = next(iter(seed_faces), None)
    if seed is None or len(mesh.face_verts(seed)) != 4:
        return mesh.copy()  # only a quad can seed a loop cut
    ax = "xyz".index(axis)

    def quad_edges(f):
        return [lp.edge for lp in mesh.face_loops(f)]

    def opposite_edge(f, e):
        es = quad_edges(f)
        return es[(es.index(e) + 2) % 4]

    def other_face(e, f):
        fs = [x for x in mesh.edge_faces(e) if x is not f]
        return fs[0] if fs else None

    # The seed's crossed edge-pair: the opposite pair most aligned with `axis`
    # (so the new bisecting edge runs perpendicular — the loop encircles `axis`).
    loops = list(mesh.face_loops(seed))
    es = [lp.edge for lp in loops]
    dirs = [np.array(lp.next.vert.co) - np.array(lp.vert.co) for lp in loops]

    def pair_align(i):
        a = abs(dirs[i][ax]) / (np.linalg.norm(dirs[i]) or 1.0)
        b = abs(dirs[(i + 2) % 4][ax]) / (np.linalg.norm(dirs[(i + 2) % 4]) or 1.0)
        return a + b

    i0 = 0 if pair_align(0) >= pair_align(1) else 1

    # Walk the connected ring of quads linked through the crossed edges.
    cross = {id(seed): (es[i0], es[(i0 + 2) % 4])}
    seen, ring, crossed = set(), [], set()
    stack = [seed]
    while stack:
        f = stack.pop()
        if id(f) in seen or len(mesh.face_verts(f)) != 4:
            continue
        seen.add(id(f))
        ea, ec = cross[id(f)]
        crossed.add(ea); crossed.add(ec)
        ring.append(f)
        for e in (ea, ec):
            nf = other_face(e, f)
            if nf is not None and id(nf) not in seen and len(mesh.face_verts(nf)) == 4 and id(nf) not in cross:
                cross[id(nf)] = (e, opposite_edge(nf, e))
                stack.append(nf)

    new_pos = [list(v.co) for v in mesh.verts]
    mid = {}  # crossed edge -> new midpoint vertex id
    for e in crossed:
        p = (np.array(e.v1.co) + np.array(e.v2.co)) * 0.5
        mid[e] = len(new_pos); new_pos.append([float(p[0]), float(p[1]), float(p[2])])

    ring_ids = {id(f) for f in ring}
    new_faces, new_attrs = [], []
    for f in mesh.faces:
        if id(f) not in ring_ids:
            new_faces.append([lp.vert.id for lp in mesh.face_loops(f)]); new_attrs.append(_copy_attrs(f.attrs))
            continue
        vids = [lp.vert.id for lp in mesh.face_loops(f)]
        fedges = quad_edges(f)
        ea, ec = cross[id(f)]
        i = next(k for k in range(4) if fedges[k] in (ea, ec))  # crossed pair at i and i+2
        mi, mj = mid[fedges[i]], mid[fedges[(i + 2) % 4]]
        quad1 = [vids[i], mi, mj, vids[(i + 3) % 4]]
        quad2 = [mi, vids[(i + 1) % 4], vids[(i + 2) % 4], mj]
        for q in (quad1, quad2):
            new_faces.append(q); new_attrs.append(_copy_attrs(f.attrs, add_tag=mark))
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


# --------------------------------------------------------------------------- #
# Open meshes — boundary edges (1 incident loop) are first-class. These operators
# cut, join and close them, which is what bridge / hole-filling need.
# --------------------------------------------------------------------------- #
def delete_faces(mesh: Mesh, faces, mark: str | None = None) -> Mesh:
    """Remove the selected faces (opening the mesh: their edges become boundary).
    Orphaned vertices are compacted away. The result may be an open mesh."""
    rem = set(id(f) for f in faces)
    new_pos = [list(v.co) for v in mesh.verts]
    new_faces, new_attrs = [], []
    for f in mesh.faces:
        if id(f) not in rem:
            new_faces.append([lp.vert.id for lp in mesh.face_loops(f)]); new_attrs.append(_copy_attrs(f.attrs))
    if not new_faces:
        return Mesh()  # everything deleted
    new_pos, new_faces = _compact(new_pos, new_faces)
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def _boundary_loops(mesh: Mesh) -> list:
    """Chain the boundary edges (1 incident face) into ordered vertex cycles, each
    wound for an OUTWARD fill face (opposite to the existing face along each edge)."""
    nxt = {}  # vert id -> next vert id around the hole
    for e in mesh.edges:
        loops = mesh.edge_loops(e)
        if len(loops) == 1:
            lp = loops[0]
            a, b = lp.vert.id, lp.next.vert.id   # the face traverses a->b
            nxt[b] = a                            # so the hole/fill goes b->a
    out, seen = [], set()
    for start in list(nxt):
        if start in seen:
            continue
        loop, cur = [], start
        while cur is not None and cur not in seen:
            seen.add(cur); loop.append(cur); cur = nxt.get(cur)
            if cur == start:
                break
        if len(loop) >= 3:
            out.append(loop)
    return out


def fill_holes(mesh: Mesh, mark: str | None = None) -> Mesh:
    """Close every boundary loop with a single n-gon face (cap the holes)."""
    new_pos = [list(v.co) for v in mesh.verts]
    new_faces = [[lp.vert.id for lp in mesh.face_loops(f)] for f in mesh.faces]
    new_attrs = [_copy_attrs(f.attrs) for f in mesh.faces]
    for loop in _boundary_loops(mesh):
        # the loop is already wound opposite to the wall face along each edge -> an
        # outward-facing cap that stays manifold.
        new_faces.append(loop)
        new_attrs.append({"tags": [mark]} if mark else {})
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def bridge_faces(mesh: Mesh, faces, mark: str | None = None) -> Mesh:
    """Bridge two selected faces into a tunnel: delete both and connect their
    boundary rims with a ring of quads. The two faces must have equal vertex count
    and share no vertices or edges (separate openings — e.g. two disjoint quads, or
    a cube's top+bottom after the sides are deleted). Vertices are paired by nearest
    correspondence with the second rim reversed; wound to stay manifold."""
    import numpy as np
    sel = list(faces)
    if len(sel) < 2:
        return mesh.copy()
    fa, fb = sel[0], sel[1]
    la = [lp.vert.id for lp in mesh.face_loops(fa)]
    lb = [lp.vert.id for lp in mesh.face_loops(fb)]
    n = len(la)
    if fa is fb or n != len(lb) or set(la) & set(lb):
        return mesh.copy()                       # need two distinct, vertex-disjoint faces
    # no existing edge may join the two rims (else a bridge quad would be the 3rd face on it)
    sa, sb = set(la), set(lb)
    for e in mesh.edges:
        if (e.v1.id in sa and e.v2.id in sb) or (e.v1.id in sb and e.v2.id in sa):
            return mesh.copy()

    pa = [np.array(mesh.verts[i].co) for i in la]
    lb_rev = lb[::-1]
    pb = [np.array(mesh.verts[i].co) for i in lb_rev]
    best = None
    for off in range(n):                          # nearest rotational correspondence
        rolled = lb_rev[off:] + lb_rev[:off]
        rp = pb[off:] + pb[:off]
        cost = sum(float(np.dot(pa[i] - rp[i], pa[i] - rp[i])) for i in range(n))
        if best is None or cost < best[0]:
            best = (cost, rolled)
    pair = best[1]

    new_pos = [list(v.co) for v in mesh.verts]
    new_faces, new_attrs = [], []
    for f in mesh.faces:
        if f is fa or f is fb:
            continue                              # delete the two bridged faces
        new_faces.append([lp.vert.id for lp in mesh.face_loops(f)]); new_attrs.append(_copy_attrs(f.attrs))
    for i in range(n):                            # wall quads, wound for outward normals
        j = (i + 1) % n
        new_faces.append([la[j], la[i], pair[i], pair[j]]); new_attrs.append(_copy_attrs(fa.attrs, add_tag=mark))
    new_pos, new_faces = _compact(new_pos, new_faces)
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def make_plane(size_x: float = 1.0, size_y: float | None = None) -> Mesh:
    """A single quad in the z=0 plane (an open mesh: 4 boundary edges)."""
    sy = size_x if size_y is None else size_y
    hx, hy = size_x / 2.0, sy / 2.0
    p = [(-hx, -hy, 0.0), (hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0)]
    return Mesh.from_pydata(p, [[0, 1, 2, 3]])


def _faces_around_vertex(mesh: Mesh, vid: int) -> list:
    """The faces incident to vertex ``vid``, in rotational (umbrella) order — walk
    from face to face across the edges that meet at the vertex."""
    info, e2f = {}, {}
    for f in mesh.faces:
        for lp in mesh.face_loops(f):
            if lp.vert.id == vid:
                ein, eout = lp.prev.edge, lp.edge
                info[id(f)] = (f, ein, eout)
                e2f.setdefault(id(ein), []).append(f)
                e2f.setdefault(id(eout), []).append(f)
                break
    if not info:
        return []
    start = next(iter(info.values()))[0]
    ordered, seen = [start], {id(start)}
    bridge = info[id(start)][2]
    while len(ordered) < len(info):
        nxt = next((g for g in e2f.get(id(bridge), []) if id(g) not in seen), None)
        if nxt is None:
            break
        seen.add(id(nxt)); ordered.append(nxt)
        _, ein, eout = info[id(nxt)]
        bridge = eout if id(ein) == id(bridge) else ein
    return ordered


def _vertex_face_edges(mesh: Mesh, f, vid: int) -> set:
    """The two edges of face ``f`` incident to vertex ``vid`` (as a set of ids)."""
    for lp in mesh.face_loops(f):
        if lp.vert.id == vid:
            return {id(lp.prev.edge), id(lp.edge)}
    return set()


def edge_bevel(mesh: Mesh, edges, width: float = 0.15, mark: str | None = None) -> Mesh:
    """Bevel (round/chamfer) any set of selected edges — general mixed valence.

    At each touched vertex the edge-star is split into SECTORS by the selected
    edges: a run of faces joined by *un*-selected edges shares one inward-moved
    corner (so those edges stay manifold), while each selected edge is a sector
    boundary that becomes a chamfer quad linking the two sectors' corners. A vertex
    whose entire star is selected closes into a corner face; a partially-selected
    vertex's sectors are stitched by the chamfers alone (no corner face).

    A selected edge is only bevelled where BOTH endpoints carry >=2 selected edges
    (a lone cut can't separate its two faces) — a fixpoint prunes the rest. So this
    rounds closed edge loops and whole-face / region edges (`edge_bevel(sharp)`
    rounds every hard edge; `edge_bevel(top_four)` rounds just that loop), and
    safely no-ops on an open edge path or a single edge. Always watertight."""
    import numpy as np
    sel = set(id(e) for e in edges)
    if not sel:
        return mesh.copy()
    t = min(max(float(width), 1e-3), 0.49)

    # a vertex is interior only if its WHOLE star is 2-manifold; boundary vertices
    # (on an open mesh) have an open umbrella the sector walk can't handle, so we
    # never bevel them (their edges are pruned -> safe, never a crash).
    vinterior = {}
    for e in mesh.edges:
        man = len(mesh.edge_faces(e)) == 2
        for vid in (e.v1.id, e.v2.id):
            vinterior[vid] = vinterior.get(vid, True) and man

    while True:  # prune edges whose endpoint is a lone cut (<2) or a boundary vertex
        deg = {}
        for e in mesh.edges:
            if id(e) in sel:
                deg[e.v1.id] = deg.get(e.v1.id, 0) + 1
                deg[e.v2.id] = deg.get(e.v2.id, 0) + 1
        new_sel = {id(e) for e in mesh.edges
                   if id(e) in sel and vinterior.get(e.v1.id) and vinterior.get(e.v2.id)
                   and deg.get(e.v1.id, 0) >= 2 and deg.get(e.v2.id, 0) >= 2}
        if new_sel == sel:
            break
        sel = new_sel
    if not sel:
        return mesh.copy()

    bevel_vert = {e.v1.id for e in mesh.edges if id(e) in sel} | {e.v2.id for e in mesh.edges if id(e) in sel}

    new_pos = [list(v.co) for v in mesh.verts]
    sector_corner = {}   # (vid, sector_index) -> new vertex id
    face_sector = {}     # (vid, face_id) -> sector_index
    vert_ns = {}         # vid -> number of sectors
    vert_repface = {}    # vid -> a representative incident face (for the corner tag)

    for vid in sorted(bevel_vert):
        ring = _faces_around_vertex(mesh, vid)
        k = len(ring)
        if k == 0:
            continue
        # cut[i]: the edge shared by ring[i] and ring[i+1] (at vid) is selected
        cut = [any(eid in sel for eid in
                   _vertex_face_edges(mesh, ring[i], vid) & _vertex_face_edges(mesh, ring[(i + 1) % k], vid))
               for i in range(k)]
        start = next((i for i in range(k) if cut[(i - 1) % k]), 0)  # begin a sector after a cut
        sectors, cur, idx = [], [], start
        for _ in range(k):
            cur.append(ring[idx])
            if cut[idx]:
                sectors.append(cur); cur = []
            idx = (idx + 1) % k
        if cur:
            sectors.append(cur)
        vert_ns[vid] = len(sectors)
        vert_repface[vid] = ring[0]
        vco = np.array(mesh.verts[vid].co)
        for si, sec in enumerate(sectors):
            avg = np.mean([np.mean([w.co for w in mesh.face_verts(f)], axis=0) for f in sec], axis=0)
            p = vco + (avg - vco) * t
            sector_corner[(vid, si)] = len(new_pos); new_pos.append([float(p[0]), float(p[1]), float(p[2])])
            for f in sec:
                face_sector[(vid, id(f))] = si

    def corner_of(vid, f):  # the vertex face f uses in place of vid (its sector corner)
        return sector_corner[(vid, face_sector[(vid, id(f))])] if vid in bevel_vert else vid

    new_faces, new_attrs = [], []
    for f in mesh.faces:  # the shrunk original faces (corners moved per sector)
        new_faces.append([corner_of(lp.vert.id, f) for lp in mesh.face_loops(f)])
        new_attrs.append(_copy_attrs(f.attrs))
    for e in mesh.edges:  # a chamfer quad per selected edge (linking the two sectors)
        if id(e) not in sel:
            continue
        fs = mesh.edge_faces(e)
        if len(fs) != 2:
            continue
        f1, f2 = fs
        lp1 = next(lp for lp in mesh.face_loops(f1) if lp.edge is e)
        u, w = lp1.vert.id, lp1.next.vert.id      # f1 traverses u->w along the edge
        u1, w1 = corner_of(u, f1), corner_of(w, f1)
        u2, w2 = corner_of(u, f2), corner_of(w, f2)
        new_faces.append([w1, u1, u2, w2])        # opposite to f1's u1->w1 -> manifold
        new_attrs.append(_copy_attrs(f1.attrs, add_tag=mark))
    for vid in sorted(bevel_vert):  # corner face fills the sector-corner cycle (>=3 sectors)
        ns = vert_ns.get(vid, 0)
        if ns < 3:  # 2 sectors -> the two chamfers already share the lone cross-edge
            continue
        new_faces.append([sector_corner[(vid, si)] for si in reversed(range(ns))])  # reversed -> outward
        new_attrs.append(_copy_attrs(vert_repface[vid].attrs, add_tag=mark))
    new_pos, new_faces = _compact(new_pos, new_faces)  # drop the now-orphaned original verts
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def solidify(mesh: Mesh, thickness: float = 0.1, mark: str | None = None) -> Mesh:
    """Give a surface thickness — a shell. Builds an inner copy offset along the
    inverted vertex normals (reversed winding) and bridges every boundary edge with
    a wall quad, so an OPEN surface (a plane, a grid, an open box) becomes a watertight
    solid. Vertex order: all outer verts, then the matching inner verts."""
    import numpy as np
    n = len(mesh.verts)
    if n == 0 or abs(thickness) < 1e-9:
        return mesh.copy()
    vacc = {v.id: np.zeros(3) for v in mesh.verts}
    for f in mesh.faces:
        fn = np.array(face_normal(mesh, f))
        for v in mesh.face_verts(f):
            vacc[v.id] += fn
    pos = [list(v.co) for v in mesh.verts]
    new_pos = list(pos)
    for vid in range(n):                       # inner verts: pos - normal*thickness
        a = vacc[vid]; l = float(np.linalg.norm(a))
        d = (a / l) * thickness if l > 1e-9 else np.zeros(3)
        new_pos.append([pos[vid][0] - d[0], pos[vid][1] - d[1], pos[vid][2] - d[2]])

    def inner(i): return i + n
    new_faces, new_attrs = [], []
    for f in mesh.faces:                       # outer shell (original)
        new_faces.append([lp.vert.id for lp in mesh.face_loops(f)])
        new_attrs.append(_copy_attrs(f.attrs))
    for f in mesh.faces:                       # inner shell (reversed winding)
        ids = [lp.vert.id for lp in mesh.face_loops(f)]
        new_faces.append([inner(i) for i in reversed(ids)])
        new_attrs.append(_copy_attrs(f.attrs, add_tag=mark))
    for e in mesh.edges:                       # walls bridge the boundary (1-face edges)
        fs = mesh.edge_faces(e)
        if len(fs) == 1:
            lp = next(l for l in mesh.face_loops(fs[0]) if l.edge is e)
            a, b = lp.vert.id, lp.next.vert.id
            new_faces.append([b, a, inner(a), inner(b)])  # opposite to the outer edge -> manifold
            new_attrs.append(_copy_attrs(fs[0].attrs, add_tag=mark))
    new_pos, new_faces = _compact(new_pos, new_faces)
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def mirror(mesh: Mesh, axis: str = "x", mark: str | None = None) -> Mesh:
    """Mirror the mesh across the axis=0 plane and weld the seam. Verts ON the plane
    are shared (the join), off-plane verts get a reflected copy with reversed winding
    (a reflection flips orientation). Model a symmetric half whose open boundary lies
    on the plane, mirror it, and the seam edges merge into a watertight whole."""
    k = "xyz".index(axis)
    tol = 1e-6
    pos = [list(v.co) for v in mesh.verts]
    new_pos = list(pos)
    mir = {}                                   # original vid -> its mirrored vid
    for vid in range(len(pos)):
        if abs(pos[vid][k]) < tol:
            mir[vid] = vid                     # on the plane -> shared seam vertex
        else:
            mir[vid] = len(new_pos)
            p = list(pos[vid]); p[k] = -p[k]
            new_pos.append(p)
    new_faces, new_attrs = [], []
    for f in mesh.faces:                       # the original half
        new_faces.append([lp.vert.id for lp in mesh.face_loops(f)])
        new_attrs.append(_copy_attrs(f.attrs))
    for f in mesh.faces:                       # the reflected half (reversed winding)
        ids = [lp.vert.id for lp in mesh.face_loops(f)]
        new_faces.append([mir[i] for i in reversed(ids)])
        new_attrs.append(_copy_attrs(f.attrs, add_tag=mark))
    new_pos, new_faces = _compact(new_pos, new_faces)
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def array(mesh: Mesh, count: int = 3, offset=(1.1, 0.0, 0.0), mark: str | None = None) -> Mesh:
    """Linear array: ``count`` copies of the mesh, copy c shifted by offset*c. Copies
    are disjoint (no weld); the last copy's faces carry ``mark``."""
    count = max(int(count), 1)
    n = len(mesh.verts)
    pos = [list(v.co) for v in mesh.verts]
    faces0 = [[lp.vert.id for lp in mesh.face_loops(f)] for f in mesh.faces]
    attrs0 = [f.attrs for f in mesh.faces]
    new_pos, new_faces, new_attrs = [], [], []
    for c in range(count):
        ox, oy, oz = offset[0] * c, offset[1] * c, offset[2] * c
        for p in pos:
            new_pos.append([p[0] + ox, p[1] + oy, p[2] + oz])
        for fi, f in enumerate(faces0):
            new_faces.append([i + c * n for i in f])
            new_attrs.append(_copy_attrs(attrs0[fi], add_tag=(mark if c == count - 1 else None)))
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def make_cube(size: float = 1.0) -> Mesh:
    s = size / 2.0
    p = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
         (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
         [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]  # outward-facing quads
    return Mesh.from_pydata(p, f)


def make_cylinder_ngon(sides: int = 24, radius: float = 0.5, height: float = 1.0) -> Mesh:
    """An n-gon prism (bottom cap, top cap, side quads) — a closed manifold."""
    import math
    half = height / 2.0
    ring = [(radius * math.cos(2 * math.pi * i / sides), radius * math.sin(2 * math.pi * i / sides)) for i in range(sides)]
    m = Mesh()
    vb = [m.add_vert((x, y, -half)) for x, y in ring]
    vt = [m.add_vert((x, y, half)) for x, y in ring]
    m.add_face(list(reversed(vb)))           # bottom cap (normal -z)
    m.add_face(vt)                           # top cap (normal +z)
    for i in range(sides):
        j = (i + 1) % sides
        m.add_face([vb[i], vb[j], vt[j], vt[i]])  # side quad
    return m


def make_uv_sphere(segments: int = 24, rings: int = 16, radius: float = 0.5) -> Mesh:
    """A UV sphere: ``segments`` longitudinal slices, ``rings`` latitudinal bands.
    Triangle fans at the poles, quad strips between. A closed 2-manifold (euler 2).

    Vertex order (so the C++ engine matches exactly): north pole, then ``rings-1``
    interior latitude circles of ``segments`` verts each, then south pole."""
    import math
    S, R = max(int(segments), 3), max(int(rings), 2)
    m = Mesh()
    north = m.add_vert((0.0, 0.0, radius))
    circ = []                                 # circ[i] = verts of interior ring i (i in 0..R-2)
    for i in range(1, R):
        theta = math.pi * i / R               # polar angle from +z
        z, rr = radius * math.cos(theta), radius * math.sin(theta)
        circ.append([m.add_vert((rr * math.cos(2 * math.pi * j / S),
                                 rr * math.sin(2 * math.pi * j / S), z)) for j in range(S)])
    south = m.add_vert((0.0, 0.0, -radius))
    for j in range(S):                        # north cap fan
        jn = (j + 1) % S
        m.add_face([circ[0][j], circ[0][jn], north])
    for i in range(R - 2):                     # quad bands (upper ring i, lower ring i+1)
        up, lo = circ[i], circ[i + 1]
        for j in range(S):
            jn = (j + 1) % S
            m.add_face([lo[j], lo[jn], up[jn], up[j]])
    for j in range(S):                        # south cap fan
        jn = (j + 1) % S
        m.add_face([south, circ[R - 2][jn], circ[R - 2][j]])
    return m


def make_cone(sides: int = 24, radius: float = 0.5, height: float = 1.0) -> Mesh:
    """A cone: an ``sides``-gon base capped by triangles meeting at the apex.
    Base ring at z=-h/2, apex at z=+h/2 (centered, like the cylinder)."""
    import math
    half = height / 2.0
    m = Mesh()
    vb = [m.add_vert((radius * math.cos(2 * math.pi * i / sides),
                      radius * math.sin(2 * math.pi * i / sides), -half)) for i in range(sides)]
    apex = m.add_vert((0.0, 0.0, half))
    m.add_face(list(reversed(vb)))             # base cap (normal -z)
    for i in range(sides):
        j = (i + 1) % sides
        m.add_face([vb[i], vb[j], apex])       # side triangle
    return m


def make_torus(major_segments: int = 24, minor_segments: int = 12,
               major_radius: float = 0.5, minor_radius: float = 0.2) -> Mesh:
    """A torus (genus-1 closed manifold, euler 0) — a grid of quads wrapped in both
    directions. ``major_segments`` around the ring, ``minor_segments`` around the tube.
    Vertex index = i*minor + j, so the C++ engine matches exactly."""
    import math
    M, N = max(int(major_segments), 3), max(int(minor_segments), 3)
    m = Mesh()
    grid = []
    for i in range(M):
        u = 2 * math.pi * i / M
        row = []
        for j in range(N):
            v = 2 * math.pi * j / N
            cr = major_radius + minor_radius * math.cos(v)
            row.append(m.add_vert((cr * math.cos(u), cr * math.sin(u), minor_radius * math.sin(v))))
        grid.append(row)
    for i in range(M):
        ii = (i + 1) % M
        for j in range(N):
            jn = (j + 1) % N
            m.add_face([grid[i][j], grid[i][jn], grid[ii][jn], grid[ii][j]])
    return m


def make_grid(size_x: float = 1.0, size_y: float | None = None,
              x_div: int = 10, y_div: int | None = None) -> Mesh:
    """A subdivided quad in z=0 (an OPEN mesh: a boundary loop of edges). ``x_div`` by
    ``y_div`` cells; vertex index = iy*(x_div+1) + ix."""
    sy = size_x if size_y is None else size_y
    nx, ny = max(int(x_div), 1), max(int(x_div if y_div is None else y_div), 1)
    hx, hy = size_x / 2.0, sy / 2.0
    pos = []
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            pos.append((-hx + size_x * ix / nx, -hy + sy * iy / ny, 0.0))
    faces = []
    for iy in range(ny):
        for ix in range(nx):
            a = iy * (nx + 1) + ix
            faces.append([a, a + 1, a + nx + 2, a + nx + 1])
    return Mesh.from_pydata(pos, faces)
