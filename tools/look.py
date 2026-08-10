"""Look at a reproduction the way somebody judging it looks at it: magnified, region by
region, with the numbers underneath.

This exists because of a specific, repeated failure. The compare sheets this repo generates
are ~900 px wide with the subject about 90 px across, and *everything looks fine at 90 px*:
a cap with no surface texture, a latch that reads as cardboard, a pocket interior two stops
too dark and a bar twice the height it should be all survive that magnification untouched.
Days of work went past on the strength of pictures too small to show what was wrong.

So: pick regions once, then after EVERY change emit the same regions at 3-4x with the
photograph beside the render, and the per-region tone and structure numbers beside those.
Looking early and looking often is the whole point; a tool that makes it one command is the
only way that actually happens.

    python tools/look.py compare.png --regions pocket=330,120,330,330 cap=430,240,190,190
    python tools/look.py compare.png --auto        # split the frame into a 3x3

The input is a side-by-side (photograph left, render right) of equal halves — the form
every case in this repo already produces.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _put(img, text, xy, scale=0.6, col=(60, 255, 255)):
    import cv2
    cv2.putText(img, text, xy, 0, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, xy, 0, scale, col, 1, cv2.LINE_AA)


def sheet(path, regions, out=None, zoom=380, cols=1):
    """Magnified photograph|render pairs, one row per region, with numbers on each."""
    import cv2
    from mirage.compare import tone, structure

    im = cv2.imread(str(path))
    if im is None:
        raise SystemExit(f"cannot read {path}")
    h, w = im.shape[:2]
    half = w // 2
    left, right = im[:, :half], im[:, half:]

    rows, report = [], []
    for name, (x, y, rw, rh) in regions.items():
        a = left[y:y + rh, x:x + rw]
        b = right[y:y + rh, x:x + rw]
        if a.size == 0 or b.size == 0 or a.shape != b.shape:
            print(f"  skipping {name}: region outside the frame")
            continue
        k = zoom / max(rw, rh)
        rs = lambda t: cv2.resize(t, (int(rw * k), int(rh * k)), interpolation=cv2.INTER_LANCZOS4)
        A, B = rs(a), rs(b)
        t = tone(a[:, :, ::-1], b[:, :, ::-1])
        st = structure(a[:, :, ::-1], b[:, :, ::-1], win=max(8, min(rw, rh) // 8))
        d = t["_frame"]["delta"]
        pair = np.hstack([A, np.full((A.shape[0], 6, 3), 40, np.uint8), B])
        _put(pair, f"{name}   luma {t['_frame']['ref']:.0f} -> {t['_frame']['render']:.0f} "
                   f"({d:+.0f})   |diff| {st['mean']:.0f}", (8, 24))
        # a red dot where the local difference is worst -- the place to go and look
        wx, wy = st["worst_px"]
        cv2.circle(pair, (int(wx * k), int(wy * k)), 9, (60, 60, 255), 2)
        rows.append(pair)
        report.append((name, t["_frame"]["ref"], t["_frame"]["render"], d, st["mean"]))

    if not rows:
        raise SystemExit("no usable regions")
    mw = max(r.shape[1] for r in rows)
    canvas = np.full((sum(r.shape[0] + 8 for r in rows), mw, 3), 18, np.uint8)
    yy = 0
    for r in rows:
        canvas[yy:yy + r.shape[0], :r.shape[1]] = r
        yy += r.shape[0] + 8
    out = out or (os.path.splitext(str(path))[0] + "_look.png")
    cv2.imwrite(str(out), canvas)

    print(f"{'region':14s} {'photo':>7s} {'render':>7s} {'delta':>7s} {'|diff|':>7s}")
    for name, a, b, d, s in sorted(report, key=lambda r: -r[4]):
        print(f"{name:14s} {a:7.1f} {b:7.1f} {d:+7.1f} {s:7.1f}"
              + ("   <-- worst" if (name, a, b, d, s) == max(report, key=lambda r: r[4]) else ""))
    print(f"\nwrote {out}")
    return out, report


def auto_regions(path, n=3):
    import cv2
    im = cv2.imread(str(path))
    h, w = im.shape[:2]
    half = w // 2
    rw, rh = half // n, h // n
    return {f"r{j}c{i}": (i * rw, j * rh, rw, rh) for j in range(n) for i in range(n)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("compare", help="a side-by-side png: photograph left, render right")
    ap.add_argument("--regions", nargs="*", default=[],
                    metavar="NAME=X,Y,W,H", help="regions in LEFT-half pixels")
    ap.add_argument("--auto", type=int, nargs="?", const=3, default=None,
                    help="instead, split the frame into an NxN grid")
    ap.add_argument("--zoom", type=int, default=380)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.auto:
        regions = auto_regions(a.compare, a.auto)
    else:
        regions = {}
        for spec in a.regions:
            name, _, box = spec.partition("=")
            regions[name] = tuple(int(v) for v in box.split(","))
        if not regions:
            raise SystemExit("give --regions NAME=X,Y,W,H ... or --auto")
    sheet(a.compare, regions, a.out, a.zoom)


if __name__ == "__main__":
    sys.exit(main())
