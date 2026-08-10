"""The part's own profile, measured from the real captures and from the model, compared.

This is the loop that should have been built first, and everything else this case has done
by eye or by rendering is downstream of it.

**The frame is the part's, not the camera's.** Height above the cap's own plane against
radius from the cap's axis. Every capture then contributes to the same curve no matter
where it was shot from, no pose has to be fitted, and the model can be sampled by exactly
the same rule — so the two numbers in a row are directly comparable and a disagreement is
a millimetre figure at a named radius rather than an impression about a picture.

**Aggregate.** One capture is a sensor reading: flying pixels at every depth discontinuity,
dropouts on black plastic, a hole wherever the cap shadows the projector. The median over a
hundred of them is the part. The inter-quartile spread is reported next to it, because a
disagreement inside the spread is not a disagreement.

What it found the first time it was run, against a model that had had a day of careful
work on it: the trench that is the whole shape of a filler pocket -- 28 mm deep at r = 42 --
was **entirely absent**, and the body panel beyond r = 100 was 33 mm too high. Both errors
are an order of magnitude larger than anything that had been argued about all day.

    python -m fuelcap.profile              # the real profile vs hero.py
    python -m fuelcap.profile --sectors 8  # per-angle, for parts that are not round
"""
from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np

from .fit import REF

BINS = np.arange(0.0, 138.0, 6.0)


def _cap_frame(xyz, lab):
    """The cap's centre and outward normal, from the labelled points themselves."""
    cap = xyz[lab == 1]
    if len(cap) < 200:
        return None
    c = cap.mean(0)
    n = np.linalg.svd(cap - c, full_matrices=False)[2][-1]
    if n[2] > 0:                      # point it back toward the camera
        n = -n
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        return None
    e1 /= np.linalg.norm(e1)
    return c, n, e1, np.cross(n, e1)


def capture_profile(path, sectors=1):
    d = np.load(path)
    fr = _cap_frame(d["xyz"], d["label"])
    if fr is None:
        return None
    c, n, e1, e2 = fr
    p = d["xyz"] - c
    x, y = p @ e1, p @ e2
    r = np.hypot(x, y) * 1e3
    h = (p @ n) * 1e3
    th = (np.degrees(np.arctan2(y, x)) % 360.0)
    out = np.full((sectors, len(BINS) - 1), np.nan)
    for s in range(sectors):
        lo, hi = s * 360.0 / sectors, (s + 1) * 360.0 / sectors
        sel = (th >= lo) & (th < hi) if sectors > 1 else np.ones(len(r), bool)
        for i in range(len(BINS) - 1):
            m = sel & (r >= BINS[i]) & (r < BINS[i + 1])
            if m.sum() >= 12:
                out[s, i] = np.median(h[m])
    return out


def model_profile(mesh, cap_z=0.0, sectors=1, exclude=("door", "tether")):
    """The same measurement on a built mesh.

    The 85th percentile of height in a bin, not the median: a depth sensor sees the SURFACE
    NEAREST it, and a mesh bin contains the back faces too. Taking the median would compare
    the camera's view of the real part against the inside of the model's.

    `exclude` drops parts the reference does not contain. The captures are of the pocket
    with the door out of frame, so leaving the open door in the model's profile put +37 mm
    at r = 120 where the body is +6 -- a 33 mm "error" that was entirely the comparison
    measuring two different things. Any statistic over "the whole model" is wrong when the
    reference is a crop.
    """
    keep = []
    for f in mesh.faces:
        tags = f.attrs.get("tags", []) if hasattr(f, "attrs") else []
        if any(t in exclude for t in tags):
            continue
        pts = [lp.vert.co for lp in mesh.face_loops(f)]
        keep.append(np.mean(pts, axis=0))
    co = np.array(keep, float) if keep else np.zeros((0, 3))
    if not len(co):
        return np.full((sectors, len(BINS) - 1), np.nan)
    r = np.hypot(co[:, 0], co[:, 1]) * 1e3
    h = (co[:, 2] - cap_z) * 1e3
    th = (np.degrees(np.arctan2(co[:, 1], co[:, 0])) % 360.0)
    out = np.full((sectors, len(BINS) - 1), np.nan)
    for s in range(sectors):
        lo, hi = s * 360.0 / sectors, (s + 1) * 360.0 / sectors
        sel = (th >= lo) & (th < hi) if sectors > 1 else np.ones(len(r), bool)
        for i in range(len(BINS) - 1):
            m = sel & (r >= BINS[i]) & (r < BINS[i + 1])
            if m.sum() >= 3:
                out[s, i] = np.percentile(h[m], 85)
    return out


def real_profile(n=140, sectors=1, cloud_dir="orbbec_clouds"):
    fs = sorted(glob.glob(os.path.join(REF, cloud_dir, "[!_]*.npz")))[:n]
    got = [p for p in (capture_profile(f, sectors) for f in fs) if p is not None]
    if not got:
        raise SystemExit(f"no usable captures in {cloud_dir}")
    a = np.array(got)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(a, 0)
        iqr = np.nanpercentile(a, 75, axis=0) - np.nanpercentile(a, 25, axis=0)
    return med, iqr, len(got)


def report(model, real, iqr, sectors=1):
    lines = []
    for s in range(sectors):
        if sectors > 1:
            lines.append(f"--- sector {s * 360 // sectors}-{(s + 1) * 360 // sectors} deg ---")
        lines.append("  r(mm)     REAL     IQR    MODEL     diff")
        err = []
        for i in range(len(BINS) - 1):
            a, b, c = real[s, i], iqr[s, i], model[s, i]
            if np.isnan(a) and np.isnan(c):
                continue
            f = lambda v, w=8, p=1: (f"{v:+{w}.{p}f}" if not np.isnan(v) else " " * (w - 1) + "-")
            d = c - a
            flag = ""
            if not np.isnan(d) and abs(d) > max(6.0, (b if not np.isnan(b) else 0) * 0.75):
                flag = "  <-- outside the spread"
                err.append((abs(d), BINS[i]))
            lines.append(f"  {BINS[i]:5.0f} {f(a)} {f(b, 7)} {f(c)} {f(d)}{flag}")
        ok = ~(np.isnan(real[s]) | np.isnan(model[s]))
        if ok.any():
            e = np.abs(model[s][ok] - real[s][ok])
            lines.append(f"  mean |error| {e.mean():.1f} mm   worst {e.max():.1f} mm "
                         f"at r={BINS[:-1][ok][int(np.argmax(e))]:.0f} mm")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sectors", type=int, default=1,
                    help="angular sectors; >1 for a part that is not a solid of revolution")
    ap.add_argument("-n", type=int, default=140, help="captures to aggregate")
    a = ap.parse_args(argv)
    from . import hero as H
    real, iqr, used = real_profile(a.n, a.sectors)
    model = model_profile(H.build().build(), cap_z=-H.RECESS, sectors=a.sectors)
    print(f"{used} captures aggregated\n")
    print(report(model, real, iqr, a.sectors))


if __name__ == "__main__":
    main()
