"""Ask the picture where things are in space.

A renderer that only returns pixels makes the model guess. Every question that actually
comes up while building -- what is under this pixel, which faces am I looking at, where in
the world is that highlight, is this part edge-on -- has an exact answer that the tracer
already computed and then threw away. This module hands those answers back.

Two halves:

* **camera** -- the projection, matching `core/src/raytrace.cpp` exactly:

      fwd = normalize(target - eye);  right = normalize(cross(fwd, up));  up2 = cross(right, fwd)
      th  = tan(fov / 2);  aspect = w / h;  image y DOWNWARD

  It lives here rather than in a script because there were already two copies of it and a
  third would have followed. A tool re-deriving the engine's own camera is a tool that
  will disagree with it on some frame nobody renders twice.

* **probe** -- queries over the AOVs (`--depth`, `--normal`, `--face-ids`):

      raycast(x, y)     one pixel  -> world position, normal, face, depth
      pick(region)      a screen rectangle or mask -> the faces under it, by pixel count
      face_screen_pos   every face's centroid in pixels, and whether it is visible

`pick` is the piece that closes the loop opened by the marked render: the model can see
face labels in a picture, and now a region of that picture converts back into a selection
it can hand to an op. Selecting geometry stops meaning "write a bounding box in world
coordinates and hope", which is how this repo has selected a 60 x 5 mm sliver instead of a
52 x 60 mm face, embossed onto zero faces, and stippled a region it meant to inset.

The unprojection is exact, not approximate: `--depth` is metric distance along the VIEW
AXIS (not along the ray), so a pixel's world point is eye + dir * (depth / dot(dir, fwd)).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Camera -- one definition, shared by the tracer, the tools and the probes
# --------------------------------------------------------------------------- #
def basis(pose):
    """(eye, fwd, right, up2) for a pose dict with eye / target / up."""
    eye = np.asarray(pose["eye"], float)
    fwd = np.asarray(pose["target"], float) - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.asarray(pose["up"], float))
    right = right / np.linalg.norm(right)
    return eye, fwd, right, np.cross(right, fwd)


def project(points, pose):
    """World points -> (u, v, depth). u right, v DOWN, both tangents of the off-axis angle.

    No focal length enters, so ratios and relative positions need no agreement about the
    fov convention. `depth` is along the view axis, matching the depth AOV.
    """
    eye, fwd, right, up2 = basis(pose)
    p = np.asarray(points, float).reshape(-1, 3) - eye
    d = p @ fwd
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    return np.stack([(p @ right) / d, -(p @ up2) / d, d], axis=1)


def to_pixels(points, pose, w, h):
    """World points -> (px, py) in a frame of that size."""
    q = project(points, pose)
    th = math.tan(pose["fov"] * 0.5)
    aspect = float(w) / float(h)
    return np.stack([(q[:, 0] / (th * aspect) + 1.0) * 0.5 * w,
                     (q[:, 1] / th + 1.0) * 0.5 * h], axis=1)


def ray(pose, x, y, w, h):
    """The unit direction the tracer would fire through pixel centre (x, y)."""
    eye, fwd, right, up2 = basis(pose)
    th = math.tan(pose["fov"] * 0.5)
    aspect = float(w) / float(h)
    u = ((x + 0.5) / w * 2.0 - 1.0) * th * aspect
    v = ((y + 0.5) / h * 2.0 - 1.0) * th
    d = fwd + right * u - up2 * v
    return eye, d / np.linalg.norm(d)


def unproject(pose, x, y, depth, w, h):
    """Pixel + its depth-AOV value -> the world point the ray hit.

    The division by dot(dir, fwd) is not a nicety. The AOV stores distance along the VIEW
    AXIS, so treating it as distance along the ray puts every point off-centre by up to
    the cosine of the half-fov -- 5 per cent at the corners of a 0.7 rad frame, which is
    tens of millimetres on a part measured to two.
    """
    eye, d = ray(pose, x, y, w, h)
    _, fwd, _, _ = basis(pose)
    return eye + d * (float(depth) / float(np.dot(d, fwd)))


# --------------------------------------------------------------------------- #
# Probes over the AOVs
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    x: int
    y: int
    hit: bool
    depth: float = 0.0
    face: int = -1
    world: tuple = (0.0, 0.0, 0.0)
    normal: tuple = (0.0, 0.0, 0.0)

    def __str__(self):
        if not self.hit:
            return f"({self.x}, {self.y}): background"
        return (f"({self.x}, {self.y}): face {self.face} at "
                f"({self.world[0]:+.4f}, {self.world[1]:+.4f}, {self.world[2]:+.4f}) "
                f"depth {self.depth:.4f} normal "
                f"({self.normal[0]:+.3f}, {self.normal[1]:+.3f}, {self.normal[2]:+.3f})")


class Probe:
    """AOVs from one render, queryable in space.

    `depth`, `normal` and `face_ids` are all optional; whatever is present is what can be
    asked about. Shapes are checked against each other on construction, because AOVs from
    two different renders line up pixel for pixel and mean nothing together -- a mistake
    that produces plausible numbers rather than an error.
    """

    def __init__(self, pose, depth=None, normal=None, face_ids=None, shape=None):
        self.pose = pose
        self.depth = None if depth is None else np.asarray(depth, float)
        self.normal = None if normal is None else np.asarray(normal, float)
        self.face_ids = None if face_ids is None else np.asarray(face_ids).astype(np.int64)
        shapes = [a.shape[:2] for a in (self.depth, self.normal, self.face_ids)
                  if a is not None]
        if shape is not None:
            shapes.append(tuple(shape))
        if not shapes:
            raise ValueError("Probe needs at least one AOV")
        if len({s for s in shapes}) != 1:
            raise ValueError(f"AOVs disagree about the frame: {shapes}")
        self.h, self.w = shapes[0]

    def raycast(self, x, y):
        """What is under one pixel."""
        x, y = int(x), int(y)
        if not (0 <= x < self.w and 0 <= y < self.h):
            raise IndexError(f"({x}, {y}) is outside the {self.w}x{self.h} frame")
        d = float(self.depth[y, x]) if self.depth is not None else 0.0
        f = int(self.face_ids[y, x]) if self.face_ids is not None else -1
        hit = (d > 0.0) if self.depth is not None else (f >= 0)
        s = Sample(x=x, y=y, hit=hit, depth=d, face=f)
        if hit and self.depth is not None:
            s.world = tuple(float(v) for v in
                            unproject(self.pose, x, y, d, self.w, self.h))
        if hit and self.normal is not None:
            s.normal = tuple(float(v) for v in self.normal[y, x])
        return s

    def pick(self, region):
        """A screen region -> {face id: pixels}, most-covered first.

        `region` is either a (x0, y0, x1, y1) rectangle or a boolean mask. This is how a
        selection gets made by looking instead of by guessing a bounding box in world
        coordinates.
        """
        if self.face_ids is None:
            raise ValueError("pick needs the face-id AOV (--face-ids)")
        if isinstance(region, (tuple, list)) and len(region) == 4:
            x0, y0, x1, y1 = (int(v) for v in region)
            m = np.zeros((self.h, self.w), bool)
            m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        else:
            m = np.asarray(region, bool)
            if m.shape != (self.h, self.w):
                raise ValueError(f"mask is {m.shape}, frame is {(self.h, self.w)}")
        ids = self.face_ids[m & (self.face_ids >= 0)]
        if ids.size == 0:
            return {}
        u, c = np.unique(ids, return_counts=True)
        order = np.argsort(-c)
        return {int(u[i]): int(c[i]) for i in order}

    def coverage(self, face):
        """How many pixels one face occupies."""
        if self.face_ids is None:
            raise ValueError("coverage needs the face-id AOV (--face-ids)")
        return int((self.face_ids == int(face)).sum())


def face_screen_pos(mesh, pose, w, h, face_ids=None):
    """Every face's centroid, in world AND in pixels, with its visible pixel count.

    The marked render already computes this to place its labels and then keeps only the
    picture. Handing the numbers back is free, and it is what lets "the label F17 is over
    there" become "face 17 is at this world point, facing this way, and 340 pixels of it
    are visible".
    """
    faces = sorted(mesh.faces, key=lambda f: f.id)
    cents = np.array([[sum(v.co[k] for v in mesh.face_verts(f)) / len(mesh.face_verts(f))
                       for k in range(3)] for f in faces], float)
    px = to_pixels(cents, pose, w, h)
    q = project(cents, pose)
    seen = {}
    if face_ids is not None:
        ids = np.asarray(face_ids).astype(np.int64).ravel()
        ids = ids[ids >= 0]
        if ids.size:
            u, c = np.unique(ids, return_counts=True)
            seen = dict(zip(u.tolist(), c.tolist()))
    return [{"face": f.id,
             "world": tuple(float(v) for v in cents[i]),
             "pixel": (float(px[i, 0]), float(px[i, 1])),
             "depth": float(q[i, 2]),
             "visible_px": int(seen.get(f.id, 0))}
            for i, f in enumerate(faces)]
