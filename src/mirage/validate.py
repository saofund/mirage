"""Move the camera. Anything that was only true from one viewpoint stops being true.

A reproduction matched to a single photograph is matched to a single *projection* of that
photograph, and several very different mistakes are invisible from the matching viewpoint:

* artwork projected from the reference camera looks perfect there and smears everywhere
  else — the whole reason this exists;
* geometry that is flat but painted to look modelled has no parallax;
* a part placed by eye at the right pixel is usually at the wrong depth, which shows as
  soon as the baseline changes;
* a surface that only faces the right way in one direction.

None of these are found by looking harder at the matched view. All of them are found by
rendering the same scene from a few degrees away, which costs one extra render each.

`orbit()` returns validation poses around a matched one; `parallax()` says how much the
picture actually changed between two of them, which is the number that separates "the model
has depth" from "the model is a photograph on a card".
"""
from __future__ import annotations

import math


def _norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 1e-12 else list(v)


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def orbit(eye, target, up, degrees=(-6.0, -3.0, 3.0, 6.0), axis="both"):
    """Poses that keep the target and the distance and move the camera a few degrees.

    Small on purpose. The point is not a turntable — a large move changes what is occluded
    and makes the comparison meaningless. It is to break the coincidence that a single
    viewpoint grants, and three to six degrees does that while leaving every feature still
    in frame and still lit the same way.

    `axis`: "yaw" swings horizontally in the camera's own frame, "pitch" vertically,
    "both" interleaves them, which is what catches a fault that happens to lie along one.
    """
    e = [float(x) for x in eye]
    t = [float(x) for x in target]
    u = _norm(up)
    d = [t[k] - e[k] for k in range(3)]
    dist = math.sqrt(sum(c * c for c in d))
    f = _norm(d)
    right = _norm(_cross(f, u))
    true_up = _cross(right, f)

    out = []
    for deg in degrees:
        for kind in (("yaw", "pitch") if axis == "both" else (axis,)):
            a = math.radians(deg)
            ax = true_up if kind == "yaw" else right
            # rotate the view direction about `ax` (Rodrigues), keep the distance
            c, s = math.cos(a), math.sin(a)
            dot = sum(f[k] * ax[k] for k in range(3))
            cr = _cross(ax, f)
            nf = [f[k] * c + cr[k] * s + ax[k] * dot * (1.0 - c) for k in range(3)]
            ne = [t[k] - nf[k] * dist for k in range(3)]
            out.append({"eye": tuple(ne), "target": tuple(t), "up": tuple(u),
                        "label": f"{kind}{deg:+.1f}", "degrees": deg, "axis": kind})
    return out


def parallax(img_ref, img_moved, mask=None):
    """How much did the picture change when the camera moved?

    A modelled scene shifts and re-occludes; a picture painted on a card mostly does not.
    Returns the mean absolute luma change and the fraction of pixels that moved by more
    than a few levels, both restricted to `mask` if one is given.

    Read it as a floor, not a target: a *low* number after a real camera move is the
    warning. It means whatever is being looked at has no depth in it.
    """
    import numpy as np
    a = np.asarray(img_ref, float)
    b = np.asarray(img_moved, float)
    if a.shape != b.shape:
        raise ValueError(f"frames differ in size: {a.shape} vs {b.shape}")
    if a.ndim == 3:
        a = a.mean(-1)
        b = b.mean(-1)
    d = np.abs(a - b)
    if mask is not None:
        m = np.asarray(mask, bool)
        d = d[m]
    return {"mean_abs": float(d.mean()), "p95": float(np.percentile(d, 95)),
            "moved_frac": float((d > 6.0).mean())}


def flatness_warning(views, threshold=2.0):
    """Given [(label, parallax_dict), ...], name the views that barely changed.

    A reproduction whose parallax is near zero a few degrees off the matched camera is
    either flat or projected. This does not decide which — it says where to look.
    """
    weak = [(lab, p) for lab, p in views if p["mean_abs"] < threshold]
    return {"suspect": [lab for lab, _ in weak],
            "worst": min(views, key=lambda t: t[1]["mean_abs"])[0] if views else None,
            "note": ("parallax below %.1f luma levels: the geometry under these views is "
                     "either flat or the shading is not coming from it" % threshold)
            if weak else "all views moved"}
