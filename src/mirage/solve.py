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
           "distort", "undistort", "Camera",
           "place_from_footprint", "solve_sun", "estimate_sun_env_ratio",
           "paint_mask", "fit_line", "trace_line", "line_intersection", "vanishing_point",
           "fit_quad"]


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
# --------------------------------------------------------------------------- #
# reading a line off a photograph, instead of squinting at it
# --------------------------------------------------------------------------- #
def paint_mask(img, min_value=0.52, max_sat=0.16):
    """Road/floor MARKINGS, as a boolean mask: bright and unsaturated.

    Brightness alone does not work on a wet surface — standing water is bright too, because
    it is mirroring the sky. But it is bright and BLUE, while paint is bright and colourless,
    so saturation is what separates a line from a puddle. On the forecourt this is the whole
    difference between a clean mask and a green smear.
    """
    a = np.asarray(img, float)
    if a.max() > 1.5:
        a = a / 255.0
    mx = a.max(2)
    mn = a.min(2)
    sat = (mx - mn) / (mx + 1e-6)
    return (mx > min_value) & (sat < max_sat)


def fit_line(mask, seed_p0, seed_p1, halfwidth=14, iters=5):
    """Fit a painted line precisely, given a rough two-point seed through it.

    This is the right division of labour. A human is good at "there is a line roughly here"
    and hopeless at its sub-pixel position — reading a bay corner off a photo by eye is what
    produced a quad that missed the paint entirely. The machine is the exact opposite. So
    seed it by hand and let it measure: collect the mask pixels within `halfwidth` of the
    current estimate, refit by total least squares (PCA, so it handles vertical lines that
    y = mx + c cannot), and iterate — the band re-centres onto the true line each pass.

    Returns {'line': (a,b,c) normalised so a^2+b^2 = 1, 'p0', 'p1', 'n', 'rms', 'width_px'}.

    Read `rms` correctly: it is the inliers' spread about the fit, which for a clean painted
    stroke is NOT the fit error — it is the stroke's own half-width. A uniform band of width
    w has standard deviation w/sqrt(12), so a crisp 9 px line reports rms 2.6 and is perfect.
    `width_px` reports that back as a thickness (rms * sqrt(12)) so the number means
    something. What a bad fit looks like is rms far larger than any plausible line: fitting a
    smear of wet tarmac gives tens of pixels. Compare it to the width you expect, not to zero.
    """
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(float)
    p0 = np.asarray(seed_p0, float)
    p1 = np.asarray(seed_p1, float)
    d = _n(p1 - p0)
    c = (p0 + p1) / 2

    for _ in range(iters):
        nrm = np.array([-d[1], d[0]])
        off = (pts - c) @ nrm
        along = (pts - c) @ d
        L = np.linalg.norm(p1 - p0) / 2
        keep = (np.abs(off) < halfwidth) & (np.abs(along) < L * 1.15)
        sel = pts[keep]
        if len(sel) < 12:
            raise ValueError(f"only {len(sel)} paint pixels near the seed — wrong place, or "
                             f"halfwidth too small")
        c = sel.mean(0)
        u, s, vt = np.linalg.svd(sel - c)
        d = vt[0]                       # principal direction = the line
    nrm = np.array([-d[1], d[0]])
    rms = float(np.sqrt((((sel - c) @ nrm) ** 2).mean()))
    a, b = nrm
    return {"line": (float(a), float(b), float(-(a * c[0] + b * c[1]))),
            "p0": (c - d * L).tolist(), "p1": (c + d * L).tolist(),
            "n": int(len(sel)), "rms": rms,
            "width_px": float(rms * math.sqrt(12.0))}   # a uniform band of width w has sd w/sqrt(12)


def trace_line(mask, seed_p0, seed_p1, samples=30, search=70, min_width=6):
    """Follow a painted line by scanning ACROSS it and keeping the WIDEST run per scanline.

    Use this instead of `fit_line` whenever anything bright sits near the line. `fit_line`
    takes every mask pixel inside its band, and a band wide enough to tolerate a rough seed
    is also wide enough to swallow a kerb, a concrete apron or a scuff — which then drag the
    fit silently. A road marking is the WIDEST bright thing on its scanline; clutter is not.
    Taking only the widest run throws the clutter away rather than averaging it in.

    It caught a real one. On the forecourt, `fit_line` reported the left bay edge at 45 px
    wide against the right edge's 31 for the same paint, and its thickness WOBBLED down the
    frame (48,44,46,49,47,36,32) where a constant-width line in perspective must grow
    smoothly. The right edge (nothing beside it) was already clean. Tracing fixes the left.

    Do not seed all the way into a corner. Where two edges meet, the scan runs into the OTHER
    edge and, if that one is wider, the widest-run rule hands back the wrong line entirely. The
    forecourt's near bay divider is 16 px wide and its long edges are 28 and 43, so the end
    scanlines reported widths of [63, 18, 17, ..., 16, 50] — the 63 and the 50 are junctions,
    not the line. Two of them out of sixteen survived the 2.5-sigma cut (they inflate sigma
    enough to shelter each other) and dragged the fit hard enough to move a downstream focal
    length by 36%. Trimming the seed to the middle 80% took that line's rms from 4.67 to 0.60.
    A `widths` list that spikes at both ends is the signature.

    Returns the fit plus `widths` and `pts`, so the physics check — does the image thickness
    grow toward the camera? — is available to the caller. That check is worth more than any
    residual, because the fit has never seen it.
    """
    mask = np.asarray(mask)
    H, W = mask.shape[:2]
    p0 = np.asarray(seed_p0, float)
    p1 = np.asarray(seed_p1, float)
    d = _n(p1 - p0)
    nrm = np.array([-d[1], d[0]])

    ts = np.arange(-search, search + 1.0)
    pts, widths = [], []
    for s in np.linspace(0, 1, samples):
        c = p0 + (p1 - p0) * s
        q = c[None, :] + ts[:, None] * nrm[None, :]
        xi = np.rint(q[:, 0]).astype(int)
        yi = np.rint(q[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        v = np.zeros(len(ts), bool)
        v[ok] = mask[yi[ok], xi[ok]]
        # widest run of True along the scan
        best, bw, i = None, 0, 0
        while i < len(v):
            if v[i]:
                j = i
                while j < len(v) and v[j]:
                    j += 1
                if j - i > bw:
                    bw, best = j - i, (i + j - 1) / 2.0
                i = j
            else:
                i += 1
        if best is None or bw < min_width:
            continue
        pts.append(c + ts[int(round(best))] * nrm)
        widths.append(bw)

    if len(pts) < 5:
        raise ValueError(f"only {len(pts)} scanlines found a run >= {min_width} px — "
                         f"wrong seed, or min_width too high")
    P = np.asarray(pts)
    Wd = np.asarray(widths, float)

    # Robust refit. Even taking the widest run, a scanline can pick the WRONG thing where
    # the line is scuffed or something crosses it, and that centre is then metres out in
    # world terms while looking like a normal point. Fit, drop whatever sits > 2.5 sigma off,
    # refit. Without this, two bad scanlines out of fourteen quietly tilt the whole line.
    keep = np.ones(len(P), bool)
    for _ in range(3):
        ctr = P[keep].mean(0)
        _, _, vt = np.linalg.svd(P[keep] - ctr)
        dd = vt[0]
        n2 = np.array([-dd[1], dd[0]])
        off = np.abs((P - ctr) @ n2)
        sd = float(np.sqrt((off[keep] ** 2).mean()))
        nk = off < max(2.5 * sd, 1.5)
        if nk.sum() < 5 or (nk == keep).all():
            break
        keep = nk

    ctr = P[keep].mean(0)
    _, _, vt = np.linalg.svd(P[keep] - ctr)
    dd = vt[0]
    n2 = np.array([-dd[1], dd[0]])
    rms = float(np.sqrt((((P[keep] - ctr) @ n2) ** 2).mean()))
    a, b = n2
    return {"line": (float(a), float(b), float(-(a * ctr[0] + b * ctr[1]))),
            "p0": (ctr - dd * np.linalg.norm(p1 - p0) / 2).tolist(),
            "p1": (ctr + dd * np.linalg.norm(p1 - p0) / 2).tolist(),
            "n": int(keep.sum()), "rejected": int((~keep).sum()),
            "rms": rms, "pts": P[keep].tolist(),
            "widths": [int(x) for x in Wd[keep]],
            "width_px": float(np.median(Wd[keep]))}


def line_intersection(l1, l2):
    """Where two image lines cross — a corner, to sub-pixel.

    For a CORNER this is well posed: the lines were traced either side of the crossing, so the
    answer sits in the middle of the evidence. For a VANISHING POINT it is not — the answer is
    an extrapolation far outside the frame. Use `vanishing_point` there; it computes the same
    number and then tells you whether to believe it.
    """
    a1, b1, c1 = l1["line"] if isinstance(l1, dict) else l1
    a2, b2, c2 = l2["line"] if isinstance(l2, dict) else l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-12:
        raise ValueError("those lines are parallel — they have no corner")
    return np.array([(b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det])


def vanishing_point(l1, l2, min_angle_deg=5.0):
    """Where two lines PARALLEL IN THE WORLD meet — and whether the answer means anything.

    The forecourt taught this one the expensive way. Its bay strip gives two pairs of parallel
    world lines, all four traced clean (rms 0.5–1.9 px, every one landing on its paint). The
    pair along the strip meets 17.7 deg apart, 1.7 kpx away: solid, and it reproduced to ~20 px
    under a bootstrap. The pair across the strip meets 2.5 deg apart, 16 kpx away — six frame
    widths outside the image — and it is worthless. Both came back as a tidy pair of floats.

    What separates them is not residual. Both pairs fit beautifully; the cross pair fits
    beautifully and is still worthless, because the camera looks nearly ALONG the strip, so
    those two lines are nearly parallel in the image and a 0.1 deg wobble in either sweeps the
    intersection by kilopixels. `angle_deg` and `reach` see that. An rms cannot.

    Returns `p`, plus:
      angle_deg  the angle between the two image lines. Below ~5 deg, stop.
      reach      |p - centroid of the evidence| / (extent of the evidence). How many times the
                 length of the lines you actually traced you are extrapolating. reach 1 is
                 interpolation; the forecourt's bad VP had reach 22.
      sigma_px   first-order propagation of each line's own rms into the position of `p`.

    sigma_px IS A LOWER BOUND and it will lie to you if you let it. It only knows the scatter
    of the traced centres about their own fit, so it sees random error and is blind to any bias
    that moves the whole line together — lens curvature, a trace that grabbed a kerb, a scan
    that ran into a corner. On the forecourt it reported the bad VP's focal length as
    3714 +- 270 px, while trimming the seeds by 10% moved that same number from 2722 to 3714.
    A shift several times sigma under a change that should not have mattered IS the detection
    of a systematic. So do not read sigma and stop: perturb the seed and re-run. If the answer
    moves by more than sigma, sigma is decoration and the real error is whatever you just saw.
    """
    a1, b1, c1 = l1["line"] if isinstance(l1, dict) else l1
    a2, b2, c2 = l2["line"] if isinstance(l2, dict) else l2
    d1 = _n([-b1, a1])
    d2 = _n([-b2, a2])
    ang = float(np.degrees(np.arccos(np.clip(abs(d1 @ d2), -1.0, 1.0))))
    if ang < min_angle_deg:
        raise ValueError(
            f"those lines are {ang:.2f} deg apart — too near parallel for a vanishing point. "
            f"Their intersection is a number, but a {ang:.2f} deg wedge places it wherever "
            f"the last tenth of a degree says, so it carries no information about the scene. "
            f"Find a pair that converges harder, or accept that this direction is not "
            f"measurable from this photograph.")
    p = line_intersection(l1, l2)

    ev, sig = [], 0.0
    for l in (l1, l2):
        if not isinstance(l, dict) or "pts" not in l:
            continue
        P = np.asarray(l["pts"], float)
        ev.append(P)
        span = float(np.linalg.norm(P[-1] - P[0]))
        if span > 1e-9 and len(P) > 2:
            # sd of the fitted direction: rms scatter / (lever arm) / sqrt(n)
            sig += (l["rms"] * np.sqrt(12.0) / (span * np.sqrt(len(P)))) ** 2
    if ev:
        A = np.vstack(ev)
        ctr = A.mean(0)
        extent = float(np.linalg.norm(A.max(0) - A.min(0)))
        reach = float(np.linalg.norm(p - ctr) / extent) if extent > 1e-9 else float("inf")
        sigma = (float(np.linalg.norm(p - ctr)) * np.sqrt(sig)
                 / max(np.sin(np.radians(ang)), 1e-9)) if sig > 0 else float("nan")
    else:
        reach, sigma = float("nan"), float("nan")
    return {"p": p, "angle_deg": ang, "reach": reach, "sigma_px": sigma}


def fit_quad(mask, seeds, halfwidth=14):
    """Four seeded edges -> the quad's four corners, measured.

    `seeds` is four (p0, p1) pairs in order, each roughly along one edge. Corners come from
    intersecting consecutive fitted edges, so they are recovered even where the paint is
    scuffed away or hidden — which is usually exactly where the corner is.
    """
    lines = [fit_line(mask, a, b, halfwidth=halfwidth) for a, b in seeds]
    corners = [line_intersection(lines[i - 1], lines[i]) for i in range(len(lines))]
    return {"corners": np.array(corners), "lines": lines,
            "rms": [round(l["rms"], 3) for l in lines]}


# --------------------------------------------------------------------------- #
# placing things, and lighting them, by measurement
# --------------------------------------------------------------------------- #
def place_from_footprint(img_quad, H, height=None):
    """An object's ground footprint, traced in the photo -> where it actually stands.

    Returns {'at', 'rotate', 'size'} ready for a ``place`` op. `img_quad` is its four
    footprint corners in image pixels, wound consistently; the first edge (p0->p1) defines
    the object's local +x, so the yaw comes out of the photo rather than out of a guess.

    This replaces the single biggest waste of time in matching a scene by hand: nudging an
    object's x / y / rotate and re-rendering to see if it landed. The ground plane already
    knows where it is. Trace it once and read the answer.
    """
    w = apply_h(H, img_quad)
    if len(w) != 4:
        raise ValueError(f"a footprint is four corners, got {len(w)}")
    e0 = w[1] - w[0]
    e1 = w[3] - w[0]
    centre = w.mean(0)
    yaw = math.degrees(math.atan2(e0[1], e0[0]))
    out = {"at": [float(centre[0]), float(centre[1]), 0.0],
           "rotate": [0.0, 0.0, float(yaw)],
           "size": [float(np.linalg.norm(e0)), float(np.linalg.norm(e1))],
           # how far from a true rectangle the traced quad is — a sanity check on the trace,
           # not on the maths: opposite edges of a rectangle stay equal through a homography
           "squareness": float(abs(np.linalg.norm(w[2] - w[3]) - np.linalg.norm(e0))
                               / max(np.linalg.norm(e0), 1e-6))}
    if height is not None:
        out["size"].append(float(height))
        out["at"][2] = float(height) / 2.0
    return out


def solve_sun(base_px, shadow_tip_px, height, H):
    """The sun direction, from one shadow. Two clicks and a height.

    A vertical of height `h` standing at ground point B casts its tip's shadow at ground
    point S. The top of the object is (B.x, B.y, h), so the sun lies along
    (top - S), normalised — and that is the whole solve. Both B and S are ON THE GROUND, so
    the homography already knows where they are; nothing here needs the camera pose.

    Note the split: the AZIMUTH comes from the ground vector S->B alone and needs no height
    at all. Only the ELEVATION needs `h`, through tan(elev) = h / |S-B|. So a rough height
    tilts the sun but cannot swing it around the compass — which is the good failure mode.

    Returns a dict with the direction (pointing TOWARD the sun, matching
    RenderSettings::sun_dir) plus the azimuth/elevation it implies, in degrees.
    """
    b = apply_h(H, np.atleast_2d(base_px))[0]
    s = apply_h(H, np.atleast_2d(shadow_tip_px))[0]
    ground = b - s
    L = float(np.linalg.norm(ground))
    if L < 1e-9:
        raise ValueError("the base and the shadow tip are the same point — no shadow to solve")
    d = _n(np.array([ground[0], ground[1], float(height)]))
    return {"sun_dir": [float(v) for v in d],
            "azimuth_deg": float(math.degrees(math.atan2(ground[1], ground[0]))),
            "elevation_deg": float(math.degrees(math.atan2(height, L))),
            "shadow_len_m": L}


# The tracer's sun is SUN_E = (6.5, 6.0, 5.0) scaled by sun_intensity, and its sky averages
# ~0.45 over the hemisphere scaled by env_intensity. Kept here so the estimate below speaks
# the renderer's units rather than some abstract ratio.
_SUN_E_LUM = 0.2126 * 6.5 + 0.7152 * 6.0 + 0.0722 * 5.0     # 6.034
_SKY_MEAN = 0.45


def estimate_sun_env_ratio(lit_luma, shadow_luma, n_dot_l):
    """sun_intensity / env_intensity, from one lit patch and one shadowed patch.

    Same material, both flat: the lit one gets sun + sky, the shadowed one only sky, so

        lit / shadow = 1 + (SUN_E/pi * sun * NoL) / (SKY_MEAN * env)

    and the contrast between them IS the ratio. This is what makes a scene's key readable
    off the photograph rather than dialled in by feel.

    ESTIMATE, and honestly so: it assumes the shadow sees the whole sky (it does not — the
    thing casting the shadow occludes part of it, which biases the ratio DOWN), ignores
    bounce from nearby surfaces, and takes the tracer's sky as uniform. Treat it as the
    starting point for the exposure loop, not as an answer. It gets the order of magnitude
    and the character (hard key vs overcast) right, which is what actually matters.
    """
    if shadow_luma <= 1e-6 or n_dot_l <= 1e-6:
        raise ValueError("need a lit patch, a shadowed patch, and a positive N.L")
    ratio = lit_luma / shadow_luma
    if ratio <= 1.0:
        return {"sun_over_env": 0.0, "contrast": float(ratio),
                "note": "no measurable shadow contrast — overcast; the key is the sky"}
    k = (_SUN_E_LUM / math.pi) / _SKY_MEAN
    return {"sun_over_env": float((ratio - 1.0) / (k * n_dot_l)),
            "contrast": float(ratio),
            "note": "estimate: assumes the shadow sees the full sky (biases low)"}


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
