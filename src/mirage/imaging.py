"""Image helpers: PNG/GIF I/O, depth/segmentation colorization, 2D bbox extraction.

Used by the synthetic-data pipeline and the demo cases. Needs the ``demos`` extra
(numpy/imageio/matplotlib/pillow); not imported by ``mirage`` core.
"""
from __future__ import annotations

import numpy as np


def save_png(arr, path) -> None:
    import imageio.v3 as iio
    iio.imwrite(str(path), np.asarray(arr).astype(np.uint8))


def save_gif(frames, path, fps: int = 30) -> None:
    import imageio
    imageio.mimsave(str(path), [np.asarray(f).astype(np.uint8) for f in frames], fps=fps, loop=0)


def colorize_depth(depth, lo: float | None = None, hi: float | None = None):
    """Depth (meters) -> RGB heatmap; far/background clamped, near = warm."""
    import matplotlib.cm as cm
    d = np.asarray(depth, dtype=float)
    finite = np.isfinite(d) & (d > 0)
    if not finite.any():
        return np.zeros((*d.shape, 3), np.uint8)
    lo = float(np.percentile(d[finite], 2)) if lo is None else lo
    hi = float(np.percentile(d[finite], 98)) if hi is None else hi
    t = np.clip((d - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = cm.turbo(1.0 - t)[..., :3]
    rgb[~finite] = 0.08
    return (rgb * 255).astype(np.uint8)


def _id_color(i: int):
    if i < 0:
        return np.array([25, 28, 32], np.uint8)
    rng = np.random.default_rng(i * 9781 + 12345)
    return (rng.random(3) * 200 + 40).astype(np.uint8)


def seg_ids(seg):
    """Per-pixel object ids from a MuJoCo segmentation buffer."""
    s = np.asarray(seg)
    return s[..., 0].astype(int) if s.ndim == 3 else s.astype(int)


def colorize_seg(seg):
    ids = seg_ids(seg)
    out = np.zeros((*ids.shape, 3), np.uint8)
    for u in np.unique(ids):
        out[ids == u] = _id_color(int(u))
    return out


def bboxes_from_seg(seg, ignore=(-1,), min_pixels: int = 12) -> dict:
    """Axis-aligned 2D bboxes per object id: {id: (x0, y0, x1, y1)}."""
    ids = seg_ids(seg)
    boxes = {}
    for u in np.unique(ids):
        if u in ignore:
            continue
        ys, xs = np.where(ids == u)
        if xs.size < min_pixels:
            continue
        boxes[int(u)] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return boxes


def overlay_boxes(rgb, boxes: dict):
    """Draw labeled rectangles on a copy of an RGB image."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(rgb).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for oid, (x0, y0, x1, y1) in boxes.items():
        c = tuple(int(v) for v in _id_color(oid))
        draw.rectangle([x0, y0, x1, y1], outline=c, width=3)
        draw.text((x0 + 2, max(0, y0 - 11)), f"id{oid}", fill=c)
    return np.asarray(img)
