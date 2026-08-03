"""Placing objects by MEASUREMENT — project them back onto the photograph and fit.

The camera for case 26 was solved to 4.2 px and then everything in the scene was put where
it looked about right. That asymmetry is the whole problem: one number for the camera, and
guesses for the twenty objects the camera is looking at. A layout built by eye cannot be
audited, cannot be improved without re-eyeballing everything, and quietly absorbs errors
that belong somewhere else — an object that is two metres too far away gets "fixed" by
making it bigger, and now two things are wrong.

Three tools:

  :func:`overlay` — project the parts' footprints and boxes onto the photograph and draw
  them. No render, no lighting, milliseconds. This is the gate to run after every layout
  change: a wireframe sitting beside the thing it is meant to be is unmissable, where the
  same error inside a finished render reads as "hmm, something's off".

  :func:`fit_ground` — search (x, y, yaw) for one object so its projected SILHOUETTE lands
  on real edges in the photograph. Scored with photomatch's asymmetric chamfer, so the
  photo's own clutter costs nothing and only the outline the object draws is paid for. It
  rasterises in numpy rather than path-tracing, so a trial is a millisecond and a hundred
  placements per object are free. This is the same loss `chamfer_per_object` reports; here
  it is being minimised instead of read.

  :func:`fit_contacts` — the one to reach for when the object is a VEHICLE against a wall,
  or anything else sitting in clutter. It ignores the edge map entirely and fits two things
  the photograph states without ambiguity: where the object's ground contacts are, and which
  columns it spans. See its docstring for why that is worth a whole second fitter.

The camera is held FIXED. It was solved from four measured corners and a rectangle
constraint, and letting an object-placement search push it around would trade a known
quantity for a hundred unknown ones.

    from mirage.layout import overlay, fit_ground

Needs numpy + Pillow.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .photomatch import edge_map, edt
from .solve import Camera, project

__all__ = ["Camera", "project_mesh", "silhouette", "outline", "fit_ground", "fit_contacts",
           "ground_line", "overlay", "bbox_wire", "footprint"]


def project_mesh(cam: Camera, verts, w: int, h: int):
    """World vertices -> pixel coordinates, through the same camera the renderer uses."""
    return project(cam, np.asarray(verts, float), w, h)


def _tri_fill(mask, a, b, c):
    """Rasterise one triangle into a boolean mask (barycentric, over its own bbox)."""
    x0 = max(int(np.floor(min(a[0], b[0], c[0]))), 0)
    x1 = min(int(np.ceil(max(a[0], b[0], c[0]))) + 1, mask.shape[1])
    y0 = max(int(np.floor(min(a[1], b[1], c[1]))), 0)
    y1 = min(int(np.ceil(max(a[1], b[1], c[1]))) + 1, mask.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(d) < 1e-12:
        return
    l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / d
    l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / d
    mask[y0:y1, x0:x1] |= (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)


def silhouette(verts, faces, cam: Camera, w: int, h: int):
    """The object's filled screen footprint — every face rasterised, no depth, no shading.

    Depth is deliberately absent: what a placement fit needs is WHERE the object covers, and
    a solid stamp of its whole projection is exactly that. Adding a z-buffer would only let
    the object hide from its own outline."""
    px = project_mesh(cam, verts, w, h)
    mask = np.zeros((h, w), bool)
    behind = ~np.isfinite(px).all(1)
    for f in faces:
        for i in range(1, len(f) - 1):
            k = (f[0], f[i], f[i + 1])
            if behind[list(k)].any():
                continue
            _tri_fill(mask, px[k[0]], px[k[1]], px[k[2]])
    return mask


def outline(mask):
    """The silhouette's boundary — the pixels a photograph could plausibly support."""
    m = np.asarray(mask, bool)
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] != m[:-1, :]
    e[:, 1:] |= m[:, 1:] != m[:, :-1]
    return e & m


def reference_field(reference, radius=48, keep=0.05):
    """Distance-to-nearest-strong-photo-edge. Computed ONCE; every trial is then a lookup."""
    ref = np.asarray(reference, float)
    ref = ref / 255.0 if ref.max() > 1.5 else ref
    ef = edge_map(ref, blur=0, normalize=False)
    return edt(ef >= max(float(np.quantile(ef, 1.0 - keep)), 1e-9), radius=radius)


def score(field, mask):
    """Mean distance from this silhouette's outline to real photo structure (lower better).

    An object that projects to nothing scores perfectly, which is the same trap `chamfer`
    documents one level up — so callers must reject empty and near-empty silhouettes rather
    than celebrate them. `fit_ground` does."""
    o = outline(mask)
    n = int(o.sum())
    return (float(field.max()), 0) if n == 0 else (float(field[o].mean()), n)


def fit_ground(make, cam: Camera, field, w: int, h: int, *, dx=(-1.5, 1.5), dy=(-1.5, 1.5),
               yaw=(-12.0, 12.0), steps=5, rounds=3, min_px=400, log=None):
    """Search (x, y, yaw) around a starting placement so the object lands on real edges.

    `make(dx, dy, dyaw)` returns the object's world geometry as (verts, faces) at that
    offset from wherever the layout currently puts it. The search is a coarse-to-fine grid —
    not a gradient, because the chamfer field is piecewise flat wherever the outline sits in
    empty photograph and a gradient method stalls there immediately.

    Returns (best_offset, best_score, start_score) so a caller can refuse a "win" that is
    not one. That matters more than it sounds: this loss is minimised by an object that
    covers less, so a search left unattended will happily walk something out of frame.
    `min_px` is the floor that stops it.
    """
    best = (0.0, 0.0, 0.0)
    v, f = make(*best)
    s0, n0 = score(field, silhouette(v, f, cam, w, h))
    best_s = s0
    span = [dx, dy, yaw]
    for r in range(rounds):
        grid = [np.linspace(lo, hi, steps) + c
                for (lo, hi), c in zip(span, best)]
        for a in grid[0]:
            for b in grid[1]:
                for g in grid[2]:
                    v, f = make(a, b, g)
                    sc, n = score(field, silhouette(v, f, cam, w, h))
                    if n < min_px:          # shrinking out of frame is not an improvement
                        continue
                    if sc < best_s:
                        best_s, best = sc, (float(a), float(b), float(g))
        span = [((lo - hi) / (2 * steps), (hi - lo) / (2 * steps)) for lo, hi in span]
        if log:
            log(f"  round {r + 1}: {best_s:.2f} px at dx={best[0]:+.2f} dy={best[1]:+.2f} "
                f"yaw={best[2]:+.1f}")
    return best, best_s, s0


def ground_line(rows, cols=None):
    """Least-squares (k, c) for the contact line `row = k * col + c`.

    `rows` is a mapping {column: row} — the bottom of the dark band under a vehicle, read off
    the photograph column by column. Pass the ones that are clean; drop any column with a
    cone, a puddle edge or another object's shadow in it, because one bad column tilts the
    line and the line is what sets the object's DISTANCE."""
    if cols is None:
        cols, rows = list(rows.keys()), list(rows.values())
    A = np.stack([np.asarray(cols, float), np.ones(len(cols))], 1)
    k, c = np.linalg.lstsq(A, np.asarray(rows, float), rcond=None)[0]
    return float(k), float(c)


def fit_contacts(verts, cam: Camera, w: int, h: int, *, columns, line, start=(0, 0, 0, 1),
                 near_flank=True, z_eps=0.06, top_row=None, scale_range=(0.85, 1.18),
                 rounds=200, log=None):
    """Solve an object's (x, y, yaw, scale) from its GROUND CONTACTS and its column span.

    `fit_ground` minimises chamfer against the photograph's edges, and its own docstring
    warns that an object in clutter is near something wherever you put it. Case 26's van is
    the worst case of that: it stands against a roller-shuttered wall whose every slat is an
    edge, and it sat three and a half metres too close to the camera for weeks while the
    scorecard reported its chamfer as 5.3 px — the best of any vehicle in the scene — and its
    tone, colour and cast all inside tolerance. Nothing that reads the render could see it.

    This reads the photograph instead, and only where the photograph is unambiguous:

    * `line` (k, c) — the contact line, `row = k*col + c`, traced from the bottom of the dark
      band under the object. Tyres touch z=0, which is the one height in a photograph that
      needs no assumption, so this fixes the DISTANCE outright. Unprojecting a sill or a
      waistline instead requires assuming its height, and an elevated point read as ground
      always lands beyond itself.
    * `columns` (x0, x1) — the first and last column the object covers. With the distance
      pinned, this is a LENGTH.
    * `top_row` — an optional row the object's top must not fall below. Use it when the
      photograph crops the object (case 26's van has its roof cut off by the frame): that is
      real information about height, and without it the fit is free to shrink.

    Only the NEAR flank's contacts are fitted. Both flanks touch the ground, but the far one
    is a vehicle's width further away and so projects several pixels higher; asking one line
    to fit both leaves a residual no placement can remove, and the fit pays for it in yaw —
    measured on the van, 9.0 px of residual and 5 degrees of error that vanish when the far
    flank is dropped.

    Returns (x, y, yaw_deg, scale, rms_px).
    """
    V = np.asarray(verts, float)
    x0, x1 = columns
    k, c = line
    # ABSOLUTE z, not `min() + z_eps`. Relative would silently accept an object whose lowest
    # point is not on the ground — a hanging sign, a body handed over without its wheels —
    # and then fit its distance as though it were, which is precisely the error this function
    # exists to prevent. Parts here are authored with their base at z=0; anything else should
    # say so by failing.
    ground = V[:, 2] < z_eps
    if near_flank:
        ground = ground & (V[:, 1] < V[:, 1].mean())
    if ground.sum() < 4:
        raise ValueError(f"only {int(ground.sum())} ground contacts below z={z_eps} — pass the "
                         "object in its own frame with its base on z=0, or raise z_eps")
    G = V[ground]

    def put(p, pts):
        x, y, yaw, s = p
        a = np.radians(yaw)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
        return (pts * s) @ R.T + np.array([x, y, 0.0])

    def cost(p):
        if not (scale_range[0] <= p[3] <= scale_range[1]):
            return np.inf
        q = project(cam, put(p, V), w, h)
        q = q[np.isfinite(q).all(1)]
        if len(q) < 32:
            return np.inf
        g = project(cam, put(p, G), w, h)
        g = g[np.isfinite(g).all(1)]
        if len(g) < 4:
            return np.inf
        d = g[:, 1] - (k * g[:, 0] + c)
        e = (q[:, 0].min() - x0) ** 2 + (q[:, 0].max() - x1) ** 2 + 4.0 * float((d ** 2).mean())
        n = 6.0
        if top_row is not None:
            e += 2.0 * max(0.0, q[:, 1].min() - top_row) ** 2
            n += 2.0
        return e / n

    best = np.asarray(start, float)
    step = np.array([1.2, 1.6, 8.0, 0.04])
    fb = cost(best)
    for _ in range(rounds):
        moved = False
        for i in range(4):
            for sgn in (+1, -1):
                cand = best.copy(); cand[i] += sgn * step[i]
                fc = cost(cand)
                if fc < fb - 1e-9:
                    best, fb, moved = cand, fc, True
        if not moved:
            step *= 0.55
            if step.max() < 1e-4:
                break
        if log:
            log(f"  {best[0]:+.2f} {best[1]:+.2f} yaw {best[2]:+.1f} x{best[3]:.3f}"
                f"  rms {np.sqrt(fb):.2f}px")
    return float(best[0]), float(best[1]), float(best[2]), float(best[3]), float(np.sqrt(fb))


def footprint(verts, z_eps=0.06):
    """The object's ground contact patch — the convex hull of its lowest vertices."""
    V = np.asarray(verts, float)
    low = V[V[:, 2] <= V[:, 2].min() + z_eps]
    if len(low) < 3:
        low = V
    c = low[:, :2].mean(0)
    order = np.argsort(np.arctan2(low[:, 1] - c[1], low[:, 0] - c[0]))
    return low[order][:, :2]


def bbox_wire(verts):
    """The twelve edges of the world-axis bounding box, as vertex pairs."""
    V = np.asarray(verts, float)
    lo, hi = V.min(0), V.max(0)
    c = [[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]], [lo[0], hi[1], lo[2]],
         [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]], [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.array(c), e


COLOURS = [(255, 90, 60), (90, 220, 120), (90, 170, 255), (255, 210, 60), (230, 110, 230),
           (60, 230, 230), (255, 150, 90), (170, 255, 90)]


def overlay(cam: Camera, items, reference, out, w=None, h=None, alpha=0.62):
    """Draw each object's footprint and bounding box on the photograph, labelled.

    `items` is [(name, verts)]. This is the layout gate: run it after moving anything. A
    wireframe next to the object it is supposed to be is a fact; the same error seen inside
    a finished render is a feeling.
    """
    img = Image.open(reference).convert("RGB") if isinstance(reference, (str, bytes)) \
        else Image.fromarray((np.clip(np.asarray(reference, float), 0, 1) * 255).astype(np.uint8))
    if w and h and img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    W, H = img.size
    base = Image.blend(img, Image.new("RGB", img.size, (10, 10, 12)), 1.0 - alpha)
    d = ImageDraw.Draw(base)
    for i, (name, verts) in enumerate(items):
        col = COLOURS[i % len(COLOURS)]
        fp = footprint(verts)
        p = project_mesh(cam, np.column_stack([fp, np.zeros(len(fp))]), W, H)
        good = np.isfinite(p).all(1)
        if good.sum() >= 3:
            d.polygon([tuple(q) for q in p[good]], outline=col)
        c, edges = bbox_wire(verts)
        q = project_mesh(cam, c, W, H)
        for a, b in edges:
            if np.isfinite(q[a]).all() and np.isfinite(q[b]).all():
                d.line([tuple(q[a]), tuple(q[b])], fill=col, width=1)
        lab = q[np.isfinite(q).all(1)]
        if len(lab):
            d.text((float(lab[:, 0].min()) + 2, float(lab[:, 1].min()) - 11), name, fill=col)
    base.save(out)
    return out
