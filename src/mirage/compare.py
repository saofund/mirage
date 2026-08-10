"""Score a render against the photograph it is copying, instead of looking at it.

Every judgement in the fuel-filler work so far was made by eye, and the eye is exactly the
wrong instrument for it: it cannot tell 2.0 cap-diameters from 2.9 (that error survived
rounds of looking), it reads "too bright" off a picture whose tone already matches to one
part in 150, and it cannot see at all that a surface is facing the wrong way. It is also
easily satisfied by a render that reuses the reference's own pixels.

Four numbers, each answering a question the eye answers badly:

    silhouette   IoU and a symmetric contour distance in pixels -- is it the same SHAPE
    keypoints    per-feature displacement in millimetres -- is each part in the right PLACE
    tone         per-region luma, so "too bright" is a number and not an impression
    structure    where the local difference is, as a map, so the answer is a location

None of these need a calibrated camera or a ground-truth mesh. They need the render and
the photograph framed the same way, which is what the case's compare sheet already does.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _gray(a):
    import numpy as np
    a = np.asarray(a)
    if a.ndim == 3:
        return (0.114 * a[..., 0] + 0.587 * a[..., 1] + 0.299 * a[..., 2]).astype(np.float64)
    return a.astype(np.float64)


def _erode(m):
    """4-neighbour binary erosion, numpy only."""
    import numpy as np
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    return e


def _edt(mask):
    """Distance to the nearest True pixel.

    `scipy.ndimage.distance_transform_edt` when it is importable, otherwise a two-pass
    chamfer with the (3, 4)/3 weights, which is within a couple of per cent of Euclidean
    and needs nothing but numpy. This module sits in `mirage` core, so it does not get to
    require OpenCV the way a case script does.
    """
    import numpy as np
    try:
        from scipy.ndimage import distance_transform_edt
        return distance_transform_edt(~mask)
    except Exception:
        pass
    INF = 1e9
    d = np.where(mask, 0.0, INF)
    a, b = 1.0, 1.41421356
    for _ in range(2):                       # forward then backward
        for j in range(d.shape[0]):
            for i in range(d.shape[1]):
                v = d[j, i]
                if j > 0:
                    v = min(v, d[j - 1, i] + a)
                    if i > 0:
                        v = min(v, d[j - 1, i - 1] + b)
                    if i + 1 < d.shape[1]:
                        v = min(v, d[j - 1, i + 1] + b)
                if i > 0:
                    v = min(v, d[j, i - 1] + a)
                d[j, i] = v
        d = d[::-1, ::-1].copy()
    return d


def silhouette(mask_a, mask_b):
    """IoU plus a symmetric contour distance, both on boolean masks of the same size.

    IoU alone is a poor guide near 1: two shapes can share 90% of their area and still have
    an edge several pixels out everywhere it matters. The contour distance says how far the
    outlines actually are apart, which is the number that corresponds to what somebody sees.
    """
    import numpy as np
    a = np.asarray(mask_a, bool)
    b = np.asarray(mask_b, bool)
    if a.shape != b.shape:
        raise ValueError(f"masks differ in size: {a.shape} vs {b.shape}")
    inter = float((a & b).sum())
    union = float((a | b).sum())
    iou = inter / union if union else 1.0
    ea, eb = a & ~_erode(a), b & ~_erode(b)
    da, db = _edt(a), _edt(b)
    d_ab = float(db[ea].mean()) if ea.any() else 0.0
    d_ba = float(da[eb].mean()) if eb.any() else 0.0
    both = np.r_[db[ea], da[eb]] if (ea.any() and eb.any()) else np.zeros(1)
    return {"iou": iou, "contour_px": 0.5 * (d_ab + d_ba),
            "contour_p95_px": float(np.percentile(both, 95)),
            "area_ratio": float(b.sum()) / float(a.sum()) if a.sum() else float("inf")}


def tone(img_a, img_b, regions=None):
    """Per-region luma, so "too bright" stops being an impression.

    `regions` is a dict of name -> boolean mask. With none given it splits on the
    photograph's own luma into the three bands that matter for a part in a cavity: the lit
    body, the mid tones, and the trap.
    """
    import numpy as np
    ga, gb = _gray(img_a), _gray(img_b)
    if regions is None:
        hi, lo = np.percentile(ga, 88), np.percentile(ga, 20)
        regions = {"lit": ga >= hi, "mid": (ga < hi) & (ga > lo), "cavity": ga <= lo}
    out = {}
    for name, m in regions.items():
        m = np.asarray(m, bool)
        if not m.any():
            continue
        out[name] = {"ref": float(ga[m].mean()), "render": float(gb[m].mean()),
                     "delta": float(gb[m].mean() - ga[m].mean())}
    out["_frame"] = {"ref": float(ga.mean()), "render": float(gb.mean()),
                     "delta": float(gb.mean() - ga.mean())}
    return out


def keypoints(pairs, px_per_mm):
    """Displacement per named feature, in millimetres.

    `pairs` is {name: ((ax, ay), (bx, by))} in pixels. The eye is good at seeing that
    something is off and bad at saying by how much or in which direction; this says both,
    and in the unit the model is authored in.
    """
    import math
    out = {}
    for name, ((ax, ay), (bx, by)) in pairs.items():
        dx, dy = (bx - ax) / px_per_mm, (by - ay) / px_per_mm
        out[name] = {"dx_mm": dx, "dy_mm": dy, "dist_mm": math.hypot(dx, dy)}
    if out:
        out["_worst"] = max((k for k in out), key=lambda k: out[k]["dist_mm"])
    return out


def structure(img_a, img_b, win=24):
    """WHERE the two differ, as a coarse grid of local mean-absolute differences.

    A single scalar over the whole frame is nearly useless — it goes down when the render
    gets blurrier. A grid says "the top-left of the pocket is what is wrong", which is a
    place to go and look.
    """
    import numpy as np
    ga, gb = _gray(img_a), _gray(img_b)
    h, w = ga.shape
    ny, nx = max(1, h // win), max(1, w // win)
    grid = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            sa = ga[j * win:(j + 1) * win, i * win:(i + 1) * win]
            sb = gb[j * win:(j + 1) * win, i * win:(i + 1) * win]
            grid[j, i] = float(np.abs(sa - sb).mean())
    j, i = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return {"grid": grid, "mean": float(grid.mean()), "max": float(grid.max()),
            "worst_cell": (int(i), int(j)),
            "worst_px": (int(i * win + win // 2), int(j * win + win // 2))}


@dataclass
class Report:
    silhouette: dict = field(default_factory=dict)
    tone: dict = field(default_factory=dict)
    keypoints: dict = field(default_factory=dict)
    structure: dict = field(default_factory=dict)

    def lines(self):
        out = []
        if self.silhouette:
            s = self.silhouette
            out.append(f"silhouette  IoU {s['iou']:.3f}   contour {s['contour_px']:.1f} px "
                       f"(p95 {s['contour_p95_px']:.1f})   area x{s['area_ratio']:.3f}")
        if self.tone:
            f = self.tone.get("_frame", {})
            out.append(f"tone        frame {f.get('ref', 0):.1f} -> {f.get('render', 0):.1f} "
                       f"({f.get('delta', 0):+.1f})")
            for k, v in self.tone.items():
                if k != "_frame":
                    out.append(f"              {k:8s} {v['ref']:6.1f} -> {v['render']:6.1f} "
                               f"({v['delta']:+6.1f})")
        if self.keypoints:
            w = self.keypoints.get("_worst")
            for k, v in self.keypoints.items():
                if k != "_worst":
                    out.append(f"keypoint    {k:14s} {v['dist_mm']:5.1f} mm "
                               f"({v['dx_mm']:+.1f}, {v['dy_mm']:+.1f})"
                               + ("   <-- worst" if k == w else ""))
        if self.structure:
            st = self.structure
            out.append(f"structure   mean |diff| {st['mean']:.1f}  worst {st['max']:.1f} "
                       f"at px {st['worst_px']}")
        return out

    def __str__(self):
        return "\n".join(self.lines())


def compare(ref_rgb, render_rgb, ref_mask=None, render_mask=None, px_per_mm=None,
            key_pairs=None, win=24):
    """The whole report. Masks and keypoints are optional; everything else is free."""
    r = Report()
    if ref_mask is not None and render_mask is not None:
        r.silhouette = silhouette(ref_mask, render_mask)
    r.tone = tone(ref_rgb, render_rgb)
    if key_pairs and px_per_mm:
        r.keypoints = keypoints(key_pairs, px_per_mm)
    r.structure = structure(ref_rgb, render_rgb, win=win)
    return r
