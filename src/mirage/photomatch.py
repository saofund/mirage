"""A loss — how far a render is from the photograph it is trying to be.

The missing number. Without it, matching a reference is: look, nudge, re-render, squint,
repeat — a loop with no gradient, which is why it never converges and why you cannot tell a
real improvement from a change of mood. Every metric here answers a question you would
otherwise be guessing at:

  ``exposure``  am I lit right?     Median linear luma, render vs reference. A render whose
                                    concrete is near-white against a reference at 0.48 sRGB
                                    is WRONG, however pleasant it looks. This catches that
                                    in one number instead of an hour.
  ``colour``    is my palette right? Per-region mean error in LINEAR space, because sRGB
                                    distances lie about dark values — exactly where the
                                    tarmac, the leather and the shadows live.
  ``edges``     is my geometry right? Soft-edge correlation. Blur both edge maps into a
                                    field and correlate: high when structure lands in the
                                    same place, and it degrades smoothly with misalignment
                                    rather than falling off a cliff like a hard IoU.

KNOW WHAT `edges` IS NOT. Between two renders it measures alignment honestly. Between a
PHOTOGRAPH and an untextured render it mostly measures the appearance gap: a real frame
carries wet tarmac, cracks, stains, lettering and a timestamp overlay, so its edge map has
orders of magnitude more content than a render of flat painted boxes, and the correlation
sits near zero however good the camera is. Measured on the forecourt: -0.011, with a diff
plate that is almost solid magenta ("the photo has edges here and you don't") — true, and
useless as a geometry signal.

For geometry against a photograph, the honest instrument is `mirage.solve.solve_camera`'s
**rms_px**: reprojection residual on real correspondences, in pixels, with no appearance
term in it at all. Use `edges` to compare two renders, or once the render carries projected
photo texture. Reach for residuals otherwise.

`compare` also writes a diff plate — reference, render, and where the edges disagree — since
a number tells you THAT you are wrong and the picture tells you WHERE.

numpy + PIL only, matching the rest of the codebase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["srgb_to_linear", "linear_to_srgb", "compare", "edge_map", "diff_plate"]


def srgb_to_linear(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def _load(p, size=None):
    im = Image.open(p).convert("RGB")
    if size and im.size != size:
        im = im.resize(size, Image.LANCZOS)
    return np.asarray(im, float) / 255.0


def _luma(lin):
    return lin @ np.array([0.2126, 0.7152, 0.0722])


def edge_map(rgb, blur=5):
    """A soft edge field: Sobel magnitude, then blurred into a distance-like ramp.

    Blurring is the point. Two hard edge maps a few pixels apart score ZERO overlap and the
    metric can't tell "nearly aligned" from "completely wrong" — no gradient to follow. Blur
    them and near-misses score high, so the number improves as the geometry approaches.
    """
    g = _luma(srgb_to_linear(rgb))
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    m = np.hypot(gx, gy)
    m = m / (m.max() + 1e-9)
    if blur > 1:
        im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
        from PIL import ImageFilter
        im = im.filter(ImageFilter.GaussianBlur(radius=blur))
        m = np.asarray(im, float) / 255.0
    return m


def compare(render, reference, regions=None, plate=None):
    """Score a render against a reference photograph. Returns a dict of metrics.

    `regions` is an optional {name: (x0, y0, x1, y1)} in FRACTIONS of the frame, so the same
    spec survives a resolution change. Use it to ask focused questions — "is the tarmac the
    right grey", "is the bay the right orange" — rather than one global number that averages
    a good sky against a bad floor and reports mediocre.
    """
    ref_im = Image.open(reference).convert("RGB")
    ren = _load(render, size=ref_im.size)
    ref = np.asarray(ref_im, float) / 255.0

    lin_r, lin_f = srgb_to_linear(ren), srgb_to_linear(ref)
    out = {
        "size": ref_im.size,
        "exposure": {
            "render_median_luma": round(float(np.median(_luma(lin_r))), 4),
            "ref_median_luma": round(float(np.median(_luma(lin_f))), 4),
        },
        "colour_rmse_linear": round(float(np.sqrt(((lin_r - lin_f) ** 2).mean())), 4),
    }
    ratio = (np.median(_luma(lin_r)) + 1e-6) / (np.median(_luma(lin_f)) + 1e-6)
    out["exposure"]["ratio"] = round(float(ratio), 3)
    out["exposure"]["stops_off"] = round(float(np.log2(max(ratio, 1e-6))), 2)

    er, ef = edge_map(ren), edge_map(ref)
    a, b = er - er.mean(), ef - ef.mean()
    out["edge_correlation"] = round(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), 4)

    if regions:
        h, w = ref.shape[:2]
        out["regions"] = {}
        for name, (x0, y0, x1, y1) in regions.items():
            sl = (slice(int(y0 * h), max(int(y1 * h), int(y0 * h) + 1)),
                  slice(int(x0 * w), max(int(x1 * w), int(x0 * w) + 1)))
            mr, mf = lin_r[sl].reshape(-1, 3).mean(0), lin_f[sl].reshape(-1, 3).mean(0)
            out["regions"][name] = {
                "render_linear": [round(float(v), 3) for v in mr],
                "ref_linear": [round(float(v), 3) for v in mf],
                "err": round(float(np.abs(mr - mf).mean()), 4),
            }
    if plate:
        diff_plate(ren, ref, er, ef, plate)
        out["plate"] = str(plate)
    return out


def diff_plate(ren, ref, er, ef, path):
    """reference | render | edge disagreement — the number says you are wrong, this says where.

    Magenta = an edge the photo has and the render does not (something missing or misplaced);
    green = an edge the render invents. Grey where they agree.
    """
    h, w = ref.shape[:2]
    rgb = np.zeros((h, w, 3))
    rgb[..., 0] = np.clip(ef - er, 0, 1)          # missing -> magenta
    rgb[..., 2] = np.clip(ef - er, 0, 1)
    rgb[..., 1] = np.clip(er - ef, 0, 1)          # invented -> green
    agree = np.minimum(er, ef)[..., None] * 0.55
    rgb = np.clip(rgb * 2.2 + agree, 0, 1)

    tile = [(ref * 255).astype(np.uint8), (ren * 255).astype(np.uint8),
            (rgb * 255).astype(np.uint8)]
    sw = 900
    ims = [Image.fromarray(t).resize((sw, int(h * sw / w)), Image.LANCZOS) for t in tile]
    ih = ims[0].height
    out = Image.new("RGB", (sw, ih * 3 + 8), (16, 16, 18))
    from PIL import ImageDraw
    for i, (im, label) in enumerate(zip(ims, ["reference", "render", "edges: magenta = missing, green = invented"])):
        out.paste(im, (0, i * (ih + 4)))
        d = ImageDraw.Draw(out)
        d.rectangle([0, i * (ih + 4), 330, i * (ih + 4) + 16], fill=(12, 12, 14))
        d.text((5, i * (ih + 4) + 3), label, fill=(240, 240, 240))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    return path
