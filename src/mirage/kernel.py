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
        attrs = [dict(f.attrs) for f in self.faces]
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

    new_faces = []
    for f in face_list:
        for lp in mesh.face_loops(f):  # one quad per corner
            new_faces.append([iv[lp.vert], ie[lp.edge], jf[f], ie[lp.prev.edge]])
    return Mesh.from_pydata(out_pos, new_faces)


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


def extrude_faces(mesh: Mesh, faces, distance: float = 0.5) -> Mesh:
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
            new_faces.append([lp.vert.id for lp in mesh.face_loops(f)]); new_attrs.append(dict(f.attrs))
    for e in mesh.edges:  # side walls
        adj = [f for f in mesh.edge_faces(e) if f in region]
        if len(adj) == 1:
            lp = next(l for l in mesh.face_loops(adj[0]) if l.edge is e)
            a, b = lp.vert.id, lp.next.vert.id
            new_faces.append([a, b, newid[b], newid[a]]); new_attrs.append(dict(adj[0].attrs))
    for f in region:  # lifted caps
        new_faces.append([newid[lp.vert.id] for lp in mesh.face_loops(f)]); new_attrs.append(dict(f.attrs))

    new_pos, new_faces = _compact(new_pos, new_faces)
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def inset_faces(mesh: Mesh, faces, thickness: float = 0.3) -> Mesh:
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
            new_faces.append([lp.vert.id for lp in mesh.face_loops(f)]); new_attrs.append(dict(f.attrs))
    for f in region:
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
            new_faces.append([vids[k], vids[(k + 1) % n], inner[(k + 1) % n], inner[k]]); new_attrs.append(dict(f.attrs))
        new_faces.append(inner); new_attrs.append(dict(f.attrs))
    return Mesh.from_pydata(new_pos, new_faces, new_attrs)


def make_cube(size: float = 1.0) -> Mesh:
    s = size / 2.0
    p = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s),
         (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
         [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]  # outward-facing quads
    return Mesh.from_pydata(p, f)
