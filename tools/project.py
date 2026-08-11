"""Project a built mesh through the renderer's own camera and report where it lands.

Why this exists
---------------
Three times in this repo a part has been placed by computing the view direction **at the
panel** and then reasoning about a part that stands 130 mm proud of it.  A fuel door swung
back past ninety degrees has its centroid a long way outside the body, and the camera can
easily be on the other side of it: at one point the door was 4.9 degrees from edge-on by
that reasoning and 23 degrees from edge-on in fact, which is the difference between a
5 mm bright line and a 106 mm blue lens.  Parallax is not a correction here, it is the
whole effect.

So: no more deriving where a part will appear.  Build it, push it through the same basis
the tracer uses, and read the answer off.

The basis is `core/src/raytrace.cpp`:

    fwd   = normalize(target - eye)
    right = normalize(cross(fwd, up))
    up2   = cross(right, fwd)

with image y increasing DOWNWARD, which is why `up2` enters the projection negated.

Self-calibration
----------------
Absolute pixels need the fov convention to be right, and getting that wrong silently
rescales every answer.  `relative_to` avoids the question: project a reference ring (the
pocket aperture, whose radius is known in mm) in the same pass and report the part in
units of that radius.  That is the same quantity measured off the photograph -- the door
edge sits at 1.40 aperture radii -- so the two are directly comparable without either
side having to be calibrated.
"""

from __future__ import annotations

import math

import numpy as np


def basis(pose):
    eye = np.asarray(pose["eye"], float)
    target = np.asarray(pose["target"], float)
    up = np.asarray(pose["up"], float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    return eye, fwd, right, up2


def project(points, pose):
    """Camera-space (u, v, depth) for world points; v is DOWNWARD, u is RIGHT.

    u and v are tangents of angles off the axis -- multiply by any focal length to get
    pixels.  Ratios and positions relative to another projected feature need no focal
    length at all, which is the point.
    """
    eye, fwd, right, up2 = basis(pose)
    p = np.asarray(points, float) - eye
    d = p @ fwd
    d = np.where(np.abs(d) < 1e-9, 1e-9, d)
    return np.stack([(p @ right) / d, -(p @ up2) / d, d], axis=1)


def verts(prog, offset=(0.0, 0.0, 0.0)):
    m = prog.build()
    return np.array([v.co for v in m.verts], float) + np.asarray(offset, float)


def silhouette(points, pose):
    """Axis-aligned extent of a projected point set, in tangent units."""
    q = project(points, pose)
    return {
        "u0": float(q[:, 0].min()), "u1": float(q[:, 0].max()),
        "v0": float(q[:, 1].min()), "v1": float(q[:, 1].max()),
        "w": float(q[:, 0].max() - q[:, 0].min()),
        "h": float(q[:, 1].max() - q[:, 1].min()),
        "depth": float(q[:, 2].mean()),
    }


def ring(radius, centre=(0.0, 0.0, 0.0), n=180):
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    return np.stack([radius * np.cos(a) + centre[0],
                     radius * np.sin(a) + centre[1],
                     np.full(n, centre[2], float)], axis=1)


def relative_to(points, pose, ref_radius, ref_centre=(0.0, 0.0, 0.0)):
    """Report a part's silhouette in units of a reference ring's projected radius.

    Returns width/height/edge positions as multiples of that radius, measured from the
    ring's projected centre -- exactly the numbers a photograph gives up when the same
    ring is visible in it.
    """
    r = project(ring(ref_radius, ref_centre), pose)
    cu, cv = r[:, 0].mean(), r[:, 1].mean()
    scale = 0.5 * (r[:, 0].max() - r[:, 0].min())
    q = project(points, pose)
    return {
        "u0": float((q[:, 0].min() - cu) / scale), "u1": float((q[:, 0].max() - cu) / scale),
        "v0": float((q[:, 1].min() - cv) / scale), "v1": float((q[:, 1].max() - cv) / scale),
        "w": float((q[:, 0].max() - q[:, 0].min()) / scale),
        "h": float((q[:, 1].max() - q[:, 1].min()) / scale),
        "aspect": float((q[:, 0].max() - q[:, 0].min()) /
                        max(1e-9, q[:, 1].max() - q[:, 1].min())),
    }


def plane_normal(points):
    c = np.asarray(points, float).mean(0)
    _, _, vt = np.linalg.svd(np.asarray(points, float) - c)
    return c, vt[2]


def off_edge_on(points, pose):
    """Degrees the part's own plane is away from containing the view ray THROUGH IT.

    0 means edge-on.  Evaluated at the part's centroid, never at the origin -- that is
    the whole reason this module exists.
    """
    c, n = plane_normal(points)
    eye = np.asarray(pose["eye"], float)
    v = eye - c
    v /= np.linalg.norm(v)
    return 90.0 - math.degrees(math.acos(min(1.0, abs(float(n @ v)))))
