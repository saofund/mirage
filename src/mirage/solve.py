"""Measurement — recovering a camera and a ground plane from a photograph.

Reproducing a real scene fails the same way every time: you eyeball the render against the
photo, nudge a number, re-render, and repeat until you run out of patience. Nothing in that
loop is a measurement, so nothing in it converges. This module is the missing half — the
part that turns "it looks about right" into a number in pixels.

Three things, in dependency order:

  ``solve_camera``  image<->world correspondences -> eye / target / fov / lens, AND the RMS
                    reprojection residual. The residual is the point: a camera you have not
                    reprojected is a camera you are guessing at.
  ``homography``    four coplanar correspondences -> the exact image<->ground mapping. Paint
                    on a forecourt, tiles on a floor, a road's markings: all coplanar, all
                    solvable. Guessing their layout by hand is choosing not to measure.
  ``rectify``       the photo's ground region -> a top-down orthophoto. Which is a texture:
                    every line, stain and puddle exactly where the camera saw it.

Everything is numpy — the DLT, the Gauss-Newton, the distortion inverse. No scipy, no cv2,
matching the rest of the codebase.

CRITICAL: ``project`` must agree with ``raytrace.cpp``'s ray generation to the last term. A
solver whose forward model disagrees with the renderer will fit a camera that is confidently,
precisely wrong, and every residual it reports will be a lie. ``tests/test_solve.py`` pins
the two together by rendering a known scene and re-solving it.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["homography", "apply_h", "rectify", "project", "solve_camera",
           "distort", "undistort", "Camera"]


def _n(v):
    v = np.asarray(v, float)
    l = np.linalg.norm(v)
    return v / l if l > 1e-12 else v


class Camera:
    """The renderer's camera, exactly: eye/target/up/fov_y plus a radial lens.

    k1/k2 are the same convention as ``RenderSettings::lens_k1/k2`` — applied to the
    NORMALISED image coordinates (a, b), where b spans [-1, 1] top to bottom and a spans
    [-aspect, aspect], so r = 1 on the top and bottom edges. Positive k1 = barrel.
    """

    def __init__(self, eye, target, up=(0, 0, 1), fov_y=0.7, k1=0.0, k2=0.0):
        self.eye = np.asarray(eye, float)
        self.target = np.asarray(target, float)
        self.up = np.asarray(up, float)
        self.fov_y = float(fov_y)
        self.k1 = float(k1)
        self.k2 = float(k2)

    def basis(self):
        """fwd / right / up — derived exactly as raytrace.cpp does, from eye/target/up."""
        fwd = _n(self.target - self.eye)
        right = _n(np.cross(fwd, self.up))
        up2 = np.cross(right, fwd)
        return fwd, right, up2

    def render_flags(self):
        """The mirage_render CLI flags for this camera."""
        f = ["--cam-eye", *(f"{v:.5f}" for v in self.eye),
             "--cam-target", *(f"{v:.5f}" for v in self.target),
             "--cam-fov", f"{self.fov_y:.5f}"]
        if self.k1 or self.k2:
            f += ["--lens-k1", f"{self.k1:.6f}", "--lens-k2", f"{self.k2:.6f}"]
        return f

    def __repr__(self):
        return (f"Camera(eye={np.round(self.eye, 3).tolist()}, "
                f"target={np.round(self.target, 3).tolist()}, "
                f"fov_y={self.fov_y:.4f}, k1={self.k1:.5f}, k2={self.k2:.5f})")


# --------------------------------------------------------------------------- #
# lens
# --------------------------------------------------------------------------- #
def distort(ab, k1, k2):
    """Ideal normalised coords -> what the lens actually puts on the sensor.

    This is the direction the RENDERER runs: it takes a pixel, forms (a, b), scales them
    radially, and shoots the ray. So the renderer needs no iteration; the inverse lives
    here in Python, where it is free.
    """
    ab = np.atleast_2d(np.asarray(ab, float))
    r2 = (ab ** 2).sum(1)
    s = 1.0 + k1 * r2 + k2 * r2 * r2
    return ab * s[:, None]


def undistort(ab, k1, k2, iters=24):
    """The inverse of `distort`: given what the lens produced, recover the ideal coords.

    Fixed-point — ab_ideal <- ab_distorted / s(|ab_ideal|) — not Newton, and the difference
    is not cosmetic. r*s(r) is NOT monotonic once k2 opposes k1: at k1=0.25, k2=-0.05 it
    turns over at r=2, so the equation r*s(r)=R has a second root on the far side of the
    fold. Newton started at r=R happily climbs to that one and returns a point reflected
    somewhere across the frame, silently. The fixed-point iteration contracts toward the
    small root — the physical one — from the same start. (This is what OpenCV's
    undistortPoints does, for the same reason.)

    Exact identity when k1 = k2 = 0.
    """
    ab = np.atleast_2d(np.asarray(ab, float))
    if k1 == 0.0 and k2 == 0.0:
        return ab
    out = ab.copy()
    for _ in range(iters):
        r2 = (out ** 2).sum(1)
        s = 1.0 + k1 * r2 + k2 * r2 * r2
        out = ab / np.where(np.abs(s) < 1e-9, 1e-9, s)[:, None]
    return out


# --------------------------------------------------------------------------- #
# projection — the forward model, matched to raytrace.cpp
# --------------------------------------------------------------------------- #
def project(cam: Camera, world, w: int, h: int):
    """World points -> pixel coordinates, mirroring the tracer's ray generation.

    The tracer builds a primary ray as::

        a  = (2*(x+0.5)/w - 1) * aspect          # normalised, a in [-aspect, aspect]
        b  = (1 - 2*(y+0.5)/h)                   #             b in [-1, 1]
        a, b = distort(a, b)                     # the lens
        rd = norm(fwd + right*(a*th) + up*(b*th))

    so inverting it means: perspective-divide onto (a*th, b*th), undo the lens, and undo the
    pixel mapping. Points behind the eye come back as NaN rather than a mirrored ghost.
    """
    world = np.atleast_2d(np.asarray(world, float))
    fwd, right, up2 = cam.basis()
    th = math.tan(cam.fov_y * 0.5)
    aspect = w / h

    d = world - cam.eye
    z = d @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        A = (d @ right) / (z * th)      # == a*s(r): the lens-distorted coords
        B = (d @ up2) / (z * th)
    ab = undistort(np.stack([A, B], 1), cam.k1, cam.k2)
    x = (ab[:, 0] / aspect + 1.0) * w * 0.5
    y = (1.0 - ab[:, 1]) * h * 0.5
    out = np.stack([x, y], 1)
    out[z <= 1e-9] = np.nan          # behind the camera
    return out


# --------------------------------------------------------------------------- #
# homography — the ground plane
# --------------------------------------------------------------------------- #
def homography(src, dst):
    """The 3x3 H with dst ~ H @ src, from >= 4 correspondences (DLT, Hartley-normalised).

    Normalisation is not optional: raw pixel coordinates are ~1e3 while world metres are
    ~1e0, and the un-normalised DLT's design matrix is then so badly conditioned that the
    SVD returns confident nonsense. Centre both clouds and scale to mean radius sqrt(2).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError(f"need >= 4 matched points, got {len(src)} and {len(dst)}")

    def norm_t(p):
        c = p.mean(0)
        d = np.linalg.norm(p - c, axis=1).mean()
        s = math.sqrt(2) / d if d > 1e-12 else 1.0
        T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
        q = (T @ np.hstack([p, np.ones((len(p), 1))]).T).T
        return q[:, :2], T

    sn, Ts = norm_t(src)
    dn, Td = norm_t(dst)
    rows = []
    for (x, y), (u, v) in zip(sn, dn):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_h(H, pts):
    """Map points through a homography (returns Nx2)."""
    pts = np.atleast_2d(np.asarray(pts, float))
    q = (H @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
    return q[:, :2] / q[:, 2:3]


def rectify(img, H_img_to_world, extent, px_per_m=100.0):
    """The photo's ground region, straightened into a top-down orthophoto.

    `extent` is (x0, x1, y0, y1) in world metres. The result is a texture: every painted
    line and every stain lands where the camera actually saw it, which no amount of
    hand-placed rectangles will match. Nearest-neighbour is deliberate — this is usually
    upsampling a distant, oblique part of the frame, and bilinear only adds a false smoothness.
    """
    arr = np.asarray(img)
    ih, iw = arr.shape[:2]
    x0, x1, y0, y1 = extent
    W = max(1, int((x1 - x0) * px_per_m))
    Hh = max(1, int((y1 - y0) * px_per_m))
    # world coords of every output texel (+y is up in the texture, so flip rows)
    xs = np.linspace(x0, x1, W)
    ys = np.linspace(y1, y0, Hh)
    gx, gy = np.meshgrid(xs, ys)
    world = np.stack([gx.ravel(), gy.ravel()], 1)
    src = apply_h(np.linalg.inv(H_img_to_world), world)
    u = np.rint(src[:, 0]).astype(int)
    v = np.rint(src[:, 1]).astype(int)
    ok = (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
    out = np.zeros((Hh, W, arr.shape[2] if arr.ndim == 3 else 1), arr.dtype)
    flat = out.reshape(-1, out.shape[2])
    flat[ok] = arr[v[ok], u[ok]] if arr.ndim == 3 else arr[v[ok], u[ok]][:, None]
    return out.reshape(Hh, W, -1)


# --------------------------------------------------------------------------- #
# camera resection
# --------------------------------------------------------------------------- #
def _pack(eye, yaw, pitch, roll, fov, k1):
    return np.array([*eye, yaw, pitch, roll, fov, k1], float)


def _unpack(p):
    eye = p[:3]
    yaw, pitch, roll, fov, k1 = p[3:8]
    fwd = np.array([math.cos(pitch) * math.cos(yaw),
                    math.cos(pitch) * math.sin(yaw),
                    math.sin(pitch)])
    r0 = _n(np.cross(fwd, [0, 0, 1.0]))
    u0 = np.cross(r0, fwd)
    up = -r0 * math.sin(roll) + u0 * math.cos(roll)
    return Camera(eye, eye + fwd, up, fov, k1)


def solve_camera(img_pts, world_pts, w, h, *, guess=None, fit_lens=True,
                 fit_fov=True, iters=140):
    """Fit a camera to image<->world correspondences. Returns (Camera, info).

    Nine unknowns (eye, yaw, pitch, roll, fov, k1) against 2N equations, by damped
    Gauss-Newton with a numerical Jacobian — Levenberg damping because the pose and the lens
    trade off against each other near the optimum and plain Gauss-Newton oscillates.

    ``info`` carries what actually matters: ``rms_px`` and ``max_px``, the reprojection
    residuals. A solve that does not report those is a guess wearing a lab coat. Rules of
    thumb on a 2560-wide frame: < 3 px is a good fit, > 10 px means the correspondences are
    wrong, the lens model is too weak, or the points are near-degenerate (all coplanar and
    all in one small patch of the frame).
    """
    img_pts = np.atleast_2d(np.asarray(img_pts, float))
    world_pts = np.atleast_2d(np.asarray(world_pts, float))
    n = len(img_pts)
    if n < 5 or n != len(world_pts):
        raise ValueError(f"need >= 5 matched points, got {n} image and {len(world_pts)} world")

    if guess is None:
        c = world_pts.mean(0)
        guess = Camera(c + np.array([0.0, -np.ptp(world_pts[:, 1]) - 4.0, 5.0]), c)
    g_fwd = _n(guess.target - guess.eye)
    p = _pack(guess.eye,
              math.atan2(g_fwd[1], g_fwd[0]),
              math.asin(np.clip(g_fwd[2], -1, 1)),
              0.0, guess.fov_y, guess.k1)

    free = np.ones(8, bool)
    if not fit_fov:
        free[7 - 1] = False      # fov
    if not fit_lens:
        free[7] = False          # k1

    def resid(par):
        r = project(_unpack(par), world_pts, w, h) - img_pts
        return np.nan_to_num(r, nan=1e4).ravel()

    lam = 1e-3
    r = resid(p)
    cost = float(r @ r)
    for _ in range(iters):
        # numerical Jacobian: 8 params, and an analytic one buys nothing at this size
        J = np.zeros((len(r), 8))
        for j in range(8):
            if not free[j]:
                continue
            step = 1e-6 if j < 3 else 1e-7
            q = p.copy()
            q[j] += step
            J[:, j] = (resid(q) - r) / step
        JtJ = J.T @ J
        Jtr = J.T @ r
        for _ in range(30):
            try:
                dp = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), -Jtr)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            q = p + dp * free
            rq = resid(q)
            cq = float(rq @ rq)
            if cq < cost:                       # accept, and trust the model more
                p, r, cost, lam = q, rq, cq, max(lam * 0.3, 1e-9)
                break
            lam *= 10                           # reject, and step more like gradient descent
        else:
            break
        if np.linalg.norm(dp) < 1e-10:
            break

    cam = _unpack(p)
    proj = project(cam, world_pts, w, h)
    err = np.linalg.norm(proj - img_pts, axis=1)
    info = {"rms_px": float(np.sqrt((err ** 2).mean())),
            "max_px": float(err.max()),
            "per_point_px": err.round(2).tolist(),
            "n_points": n}
    return cam, info
