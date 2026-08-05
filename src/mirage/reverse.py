"""Measured geometry in, op-log out — modelling a real object from its own data.

Mirage can already go from an op-log to a picture and from a photograph to a camera
(`mirage.solve`, `mirage.photoscene`). This is the missing direction: from something the
world was measured with — a depth camera, a scanner, or one RGB frame put through a
monocular depth network — to the numbers an operator actually takes.

The point is narrow and worth stating exactly. A `spin` op consumes a **section**: a list
of (radius, height). Any depth map of a round-ish part *contains* that list. Every time it
is written by hand instead, the model comes out with plausible features and wrong surfaces
between them — which is a specific, repeatable failure, not a matter of skill. Case 27 hand-
built a fuel-filler pocket whose cap diameter, rib size and aperture were all within a few
per cent of measured, while the trench between them was 33 mm deep in reality and 6 mm in
the model, because nothing in the workflow ever asked what the shape *between* the features
was. `section_from_cloud` asks.

    from mirage.reverse import cloud_from_depth, section_from_cloud, superellipse_plan
    cloud = cloud_from_depth(depth_m, K)                   # HxW metres + 3x3 intrinsics
    sec   = section_from_cloud(cloud, origin=c, axis=n)    # [(r, z), ...] in metres
    prog  = MeshProgram().profile(sec, plane="xz").spin(steps=56,
                                                        plan=superellipse_plan(3.6, 56))

Nothing here fits a depth network — that is somebody else's model and a moving target.
It takes a depth map, which is what those networks emit.

Needs numpy.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["cloud_from_depth", "principal_axis", "section_from_cloud", "section_to_profile",
           "superellipse_plan", "plan_from_cloud"]


def cloud_from_depth(depth, K, mask=None, stride=1):
    """Unproject a metric depth map into an (N,3) cloud in camera coordinates.

    `depth` is HxW in metres with 0 (or NaN) meaning "no answer", `K` the 3x3 intrinsic
    matrix in pixels. The frame is OpenCV's — x right, y **down**, z forward — because that
    is what every depth sensor and every monocular network reports, and quietly using a
    y-up convention here would flip every model built through this module."""
    d = np.asarray(depth, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("depth must be HxW")
    K = np.asarray(K, dtype=np.float64)
    h, w = d.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = d[::stride, ::stride]
    ok = np.isfinite(z) & (z > 0)
    if mask is not None:
        ok &= np.asarray(mask, bool)[::stride, ::stride]
    if not ok.any():
        return np.zeros((0, 3))
    x = (xs[ok] + 0.5 - K[0, 2]) / K[0, 0] * z[ok]
    y = (ys[ok] + 0.5 - K[1, 2]) / K[1, 1] * z[ok]
    return np.stack([x, y, z[ok]], 1)


def principal_axis(points, toward=None):
    """The axis of a roughly-round patch: the normal of its best-fit plane.

    Returned pointing at the viewer by default, since a cloud is measured from somewhere
    and "which way is out" is otherwise a coin flip that silently mirrors the section."""
    P = np.asarray(points, dtype=np.float64)
    c = P.mean(0)
    _, _, V = np.linalg.svd(P - c, full_matrices=False)
    n = V[2]
    ref = np.asarray(toward, float) if toward is not None else np.array([0.0, 0.0, -1.0])
    if n @ ref < 0:
        n = -n
    return c, n


def _frame(origin, axis):
    n = np.asarray(axis, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    t = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(t, n)
    u /= np.linalg.norm(u)
    return np.asarray(origin, float), u, np.cross(n, u), n


def section_from_cloud(points, origin=None, axis=None, r_max=None, step=0.002,
                       min_pts=6, reduce="median"):
    """The (radius, height) SECTION of a cloud about an axis — a lathe's input, measured.

    Heights are relative to `origin` along `axis`, radii are distance from the axis. Each
    radial bin reduces to a median by default, which is what makes this usable on real
    sensor data: a mean chases flying pixels at every depth discontinuity, and those sit on
    exactly the edges a section is trying to locate.

    Returns an (M,2) array of (r, z) in the cloud's own units, gaps dropped. Feed it to
    `section_to_profile` before handing it to `MeshProgram.profile`."""
    P = np.asarray(points, dtype=np.float64)
    if len(P) < min_pts:
        return np.zeros((0, 2))
    if origin is None or axis is None:
        c, n = principal_axis(P)
        origin = origin if origin is not None else c
        axis = axis if axis is not None else n
    o, u, v, n = _frame(origin, axis)
    L = np.stack([(P - o) @ u, (P - o) @ v, (P - o) @ n], 1)
    r = np.hypot(L[:, 0], L[:, 1])
    if r_max is None:
        r_max = float(np.percentile(r, 99.5))
    keep = r <= r_max
    if keep.sum() < min_pts:
        return np.zeros((0, 2))
    r, hz = r[keep], L[keep, 2]
    idx = (r / step).astype(int)
    out = []
    red = np.median if reduce == "median" else np.mean
    for k in range(idx.max() + 1):
        m = idx == k
        if m.sum() >= min_pts:
            out.append(((k + 0.5) * step, float(red(hz[m]))))
    return np.array(out) if out else np.zeros((0, 2))


def section_to_profile(section, close_axis=True, floor=None, outer=None):
    """Turn a measured section into a profile polyline a `spin` can close into a solid.

    A measured section starts wherever the data starts, which is never the axis — a cap
    covers the middle of a recess, a scanner sees nothing down a bore. `spin` needs a
    polyline that begins and ends ON the axis or it produces a tube with two open rims,
    silently, and the failure only surfaces much later as a mesh that will not build.

    So this walks: axis -> under the section -> out to `outer` -> back IN along the measured
    surface -> down to `floor` -> home. `floor`/`outer` default to a little beyond the data."""
    S = np.asarray(section, dtype=np.float64)
    if len(S) < 2:
        raise ValueError("section needs at least two samples")
    r0, z0 = S[0]
    r1 = float(outer if outer is not None else S[-1, 0] * 1.04)
    zf = float(floor if floor is not None else S[:, 1].min() - 0.25 * (S[:, 1].max() - S[:, 1].min() + 1e-9))
    pts = [(0.0, zf)]
    if close_axis:
        pts.append((r1, zf))
    pts.append((r1, float(S[-1, 1])))
    pts += [(float(r), float(z)) for r, z in S[::-1]]
    pts.append((float(r0), zf + 0.02 * abs(zf - z0)))
    pts.append((0.0, zf + 0.02 * abs(zf - z0)))
    return pts


def superellipse_plan(n, steps, normalise=True, by="area"):
    """A `spin` plan that presses a round section into a rounded rectangle.

    `n` = 2 is an ellipse, 3-5 the rounded rectangles most pressed panels actually are,
    large n a rectangle. `normalise` divides by the mean so squaring the plan does not also
    INFLATE it — the raw form is 1 on the axes and 1.17 on the diagonals at n = 3.6, so an
    un-normalised "square" ring is also an 8% bigger ring, which moves every wall outward
    and is a size error wearing a shape error's clothes."""
    if n <= 1.0:
        return [1.0] * int(steps)
    ks = []
    for j in range(int(steps)):
        a = 2.0 * math.pi * j / int(steps)
        c, s = math.cos(a), math.sin(a)
        ks.append((abs(c) ** n + abs(s) ** n) ** (-1.0 / n))
    if normalise:
        # Normalise by AREA (root-mean-square) rather than by arithmetic mean. A ring's
        # contribution to anything measured over a surface goes as r^2, so a plan whose
        # arithmetic mean is 1 still enlarges the swept area — and, worse, a measurement
        # binned by physical radius then samples the section further IN wherever the plan
        # pushed outward, which reads as the whole outer surface sitting a few millimetres
        # low. `by="mean"` keeps the old behaviour for anything that wants the perimeter
        # preserved instead.
        if by == "area":
            m = (sum(k * k for k in ks) / len(ks)) ** 0.5
        else:
            m = sum(ks) / len(ks)
        ks = [k / m for k in ks]
    return ks


def plan_from_cloud(points, origin=None, axis=None, steps=48, r_band=None, normalise=True):
    """The measured PLAN: how the outline's radius varies with direction.

    The companion to `section_from_cloud`. Together they describe a part that is turned in
    section and pressed in plan without either being assumed — take the section from one,
    the plan from the other, and `spin(plan=...)` puts them back together."""
    P = np.asarray(points, dtype=np.float64)
    if origin is None or axis is None:
        c, n = principal_axis(P)
        origin = origin if origin is not None else c
        axis = axis if axis is not None else n
    o, u, v, n = _frame(origin, axis)
    L = np.stack([(P - o) @ u, (P - o) @ v], 1)
    r = np.hypot(L[:, 0], L[:, 1])
    if r_band is not None:
        keep = (r >= r_band[0]) & (r <= r_band[1])
        L, r = L[keep], r[keep]
    if len(r) < steps * 2:
        return [1.0] * int(steps)
    # Bin to the NEAREST direction index, not the one below it. `superellipse_plan` samples
    # entry k at angle k/steps*2pi, so binning by floor would offset a recovered plan by
    # half a bin against a generated one — and half a bin on a 4-lobed outline is enough to
    # make the two look uncorrelated. The two functions have to share a phase convention or
    # they cannot be fed to each other, which is the whole point of having both.
    a = np.mod(np.arctan2(L[:, 1], L[:, 0]), 2 * math.pi)
    b = np.rint(a / (2 * math.pi) * steps).astype(int) % int(steps)
    out = []
    for k in range(int(steps)):
        m = b == k
        out.append(float(np.percentile(r[m], 95)) if m.sum() >= 3 else np.nan)
    out = np.array(out)
    if np.isnan(out).all():
        return [1.0] * int(steps)
    good = ~np.isnan(out)
    out = np.interp(np.arange(len(out)), np.flatnonzero(good), out[good])
    if normalise:
        out = out / out.mean()
    return [float(x) for x in out]
