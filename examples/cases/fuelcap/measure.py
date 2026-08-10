"""Measure ANY car's filler region from its own photograph, with no hand work.

Everything this case has learned about reading a photograph, in one pass that takes a file
and returns numbers. It exists so that reproducing the eleventh car costs what reproducing
the second one cost, instead of a day.

The method, and why each step is the way it is:

1. **The cap is a circle.** Under weak perspective a circle in the body panel images as an
   ellipse whose major axis is unforeshortened and whose minor is that same diameter times
   cos(tilt). Fitting that ellipse to the cap's own annotation polygon gives the panel's
   tilt, the tilt axis, and the pixel/mm scale -- with no camera calibration and no
   assumption beyond the cap being round.

2. **Measure along the tilt axis.** Distances there are unforeshortened, so an extent read
   along it is directly comparable to the cap's major axis. Measuring one direction against
   another is the mistake that produced a 2.46 that should have been 2.18.

3. **Ratios, not millimetres.** Nothing in a photograph fixes absolute size. What the eye
   reads -- and what was wrong in every early render -- is the RATIO of the opening to the
   cap, so that is what this returns.

4. **Report what it could not do.** A car whose aperture will not segment returns a reason,
   not a guess. Ten quiet guesses are worse than five measurements and five skips.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np

from .fit import REF


def _imread(path):
    import cv2
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _polys(txt, w, h):
    out = {}
    if not os.path.exists(txt):
        return out
    for line in open(txt, encoding="utf-8"):
        p = line.split()
        if len(p) >= 7:
            out[int(p[0])] = (np.array([float(x) for x in p[1:]], np.float32)
                              .reshape(-1, 2) * [w, h])
    return out


def measure(png, cap_mm=57.0):
    """Everything derivable from one annotated photograph. Returns a dict, or {'skip': why}."""
    import cv2
    im = _imread(png)
    if im is None:
        return {"skip": "unreadable"}
    h, w = im.shape[:2]
    pol = _polys(png[:-4] + ".txt", w, h)
    if 1 not in pol or len(pol[1]) < 5:
        return {"skip": "no cap polygon"}

    (cx, cy), (aw, ah), ang = cv2.fitEllipse(pol[1].astype(np.float32))
    D, minor = max(aw, ah), min(aw, ah)
    if D < 40:
        return {"skip": f"cap only {D:.0f} px"}
    cosT = min(1.0, minor / D)
    tilt = math.degrees(math.acos(cosT))
    axis = (ang + 90.0) if ah >= aw else ang

    g = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    # paint level from a ring well outside the aperture
    ring = []
    for a in np.arange(0, 360, 3.0):
        t = math.radians(a)
        x, y = cx + 2.1 * D * math.cos(t), cy + 2.1 * D * math.sin(t)
        if 0 <= x < w and 0 <= y < h:
            ring.append(g[int(y), int(x)])
    if len(ring) < 40:
        return {"skip": "aperture too near the frame edge"}
    paint = float(np.median(ring))
    if paint < 45:
        return {"skip": f"body too dark to segment (paint {paint:.0f})"}

    # walk the TILT AXIS both ways: unforeshortened, so directly comparable to D
    spans = []
    for sgn in (+1, -1):
        t = math.radians(axis)
        dx, dy = sgn * math.cos(t), sgn * math.sin(t)
        hit = None
        for r in np.arange(D * 0.55, D * 2.6, 1.0):
            x, y = cx + dx * r, cy + dy * r
            if not (2 <= x < w - 2 and 2 <= y < h - 2):
                break
            # A SUSTAINED return to paint level, not merely a bright pixel. A liner has
            # highlights on it -- a rolled lip, a wet patch, a screw head -- and at a 0.62
            # threshold held for 10 px the search stops on them: half the cars came back
            # with an "aperture" 1.10 times the cap, which is the cap's own rim. Paint
            # continues; a highlight does not.
            if float(cv2.getRectSubPix(g, (5, 5), (x, y)).mean()) > 0.80 * paint:
                run = [float(cv2.getRectSubPix(g, (5, 5),
                                               (cx + dx * rr, cy + dy * rr)).mean())
                       for rr in np.arange(r, min(r + 30.0, D * 2.6), 3.0)
                       if 2 <= cx + dx * rr < w - 2 and 2 <= cy + dy * rr < h - 2]
                if len(run) >= 6 and min(run) > 0.72 * paint:
                    hit = r
                    break
        spans.append(hit)
    if None in spans:
        return {"skip": "aperture edge not found along the tilt axis"}

    open_px = spans[0] + spans[1]
    # A result AT the search floor is not a measurement, it is the loop giving up wearing a
    # number: both rays stopped at their first sample, which happens on a dark car where the
    # liner is as bright as the paint. 1.10 came back for four cars this way and would have
    # gone into a template as a fact. The docstring says report what it could not do; this
    # is where that has to be enforced rather than intended.
    if open_px <= D * 1.16:
        return {"skip": "aperture ratio at the search floor (dark body?)"}
    return {
        "cap_px": D, "tilt_deg": tilt, "tilt_axis_deg": axis % 180,
        "px_per_mm": D / cap_mm,
        "open_over_cap": open_px / D,
        "open_mm": open_px / (D / cap_mm),
        "cap_offset_px": abs(spans[0] - spans[1]) / 2.0,
        "recess_mm": (abs(spans[0] - spans[1]) / 2.0) / (D / cap_mm) / max(math.tan(math.radians(tilt)), 1e-3),
    }


def survey(limit=None, cap_mm=57.0, one_per_car=True):
    rows, skips = [], {}
    seen = set()
    for png in sorted(glob.glob(os.path.join(REF, "bycar", "*", "*.png"))):
        car = os.path.basename(os.path.dirname(png))
        if one_per_car and car in seen:
            continue
        m = measure(png, cap_mm)
        if "skip" in m:
            skips[m["skip"]] = skips.get(m["skip"], 0) + 1
            continue
        seen.add(car)
        m["car"] = car
        m["png"] = png
        rows.append(m)
        if limit and len(rows) >= limit:
            break
    return rows, skips


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-n", type=int, default=None)
    ap.add_argument("--cap-mm", type=float, default=57.0)
    a = ap.parse_args(argv)
    rows, skips = survey(a.n, a.cap_mm)
    print(f"{'car':16s} {'cap px':>7s} {'tilt':>6s} {'axis':>6s} {'open/cap':>9s} "
          f"{'open mm':>8s} {'recess mm':>10s}")
    for r in rows:
        print(f"{r['car'][:16]:16s} {r['cap_px']:7.0f} {r['tilt_deg']:6.1f} "
              f"{r['tilt_axis_deg']:6.1f} {r['open_over_cap']:9.2f} {r['open_mm']:8.0f} "
              f"{r['recess_mm']:10.1f}")
    if rows:
        v = np.array([r["open_over_cap"] for r in rows])
        print(f"\nopen/cap over {len(rows)} cars: p10 {np.percentile(v,10):.2f}  "
              f"median {np.median(v):.2f}  p90 {np.percentile(v,90):.2f}")
    if skips:
        print("\nskipped:")
        for k, n in sorted(skips.items(), key=lambda t: -t[1]):
            print(f"  {n:4d}  {k}")


if __name__ == "__main__":
    main()
