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

  ``chamfer``   is my geometry right? For each RENDER edge, the distance in pixels to the
                                    nearest strong REFERENCE edge. Never the reverse — and
                                    that asymmetry is the whole thing. See below.

KNOW WHAT `edges` IS NOT. Between two renders it measures alignment honestly. Between a
PHOTOGRAPH and an untextured render it mostly measures the appearance gap: a real frame
carries wet tarmac, cracks, stains, lettering and a timestamp overlay, so its edge map has
orders of magnitude more content than a render of flat painted boxes, and the correlation
sits near zero however good the camera is. Measured on the forecourt: -0.011, with a diff
plate that is almost solid magenta ("the photo has edges here and you don't") — true, and
useless as a geometry signal.

This file used to answer that with "reach for residuals" — go get correspondences from
`solve.solve_camera` instead. That sentence cost a working week. Residuals need
correspondences, correspondences on a photograph need lines, lines need paint, and paint
needs a road: the forecourt got solved by hand-seeding traces off zoomed crops, which works
on exactly one photograph of exactly one petrol station and generalises to nothing. A
living room has no road markings. The loss failing is not a reason to go surveying; it is a
reason to fix the loss.

`chamfer` is the fix, and the fix is asymmetry. Correlation asks "do these two edge maps
look alike", so the photo's cracks and stains — which the render can never have and should
never be asked to have — dominate it. Chamfer asks only "is every edge I DREW supported by
something real nearby". Structure the photo has and the render lacks is free; structure the
render invents is paid for. That question is answerable against any photograph of anything,
which is what `edges` never was, and it needs no paint, no rectangles and no hand-seeding.

`compare` also writes a diff plate — reference, render, and where the edges disagree — since
a number tells you THAT you are wrong and the picture tells you WHERE.

numpy + PIL only, matching the rest of the codebase — including the distance transform.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

__all__ = ["srgb_to_linear", "linear_to_srgb", "compare", "edge_map", "diff_plate",
           "chamfer_per_object", "read_ids",
           "chamfer", "edt"]


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


def edge_map(rgb, blur=5, normalize=True):
    """A soft edge field: Sobel magnitude, then blurred into a distance-like ramp.

    Blurring is the point. Two hard edge maps a few pixels apart score ZERO overlap and the
    metric can't tell "nearly aligned" from "completely wrong" — no gradient to follow. Blur
    them and near-misses score high, so the number improves as the geometry approaches.

    `normalize` divides by the map's own peak, which makes two maps comparable and makes any
    ABSOLUTE reading of the result meaningless: a frame holding one faint edge and a frame
    holding a hundred crisp ones both come out peaking at 1.0. Pass False when you need to
    know how much edge is actually there — see `chamfer`'s `edge_mass`, which is exactly that
    question and which silently answered a constant for as long as it was reading a
    self-normalised map through a quantile.
    """
    g = _luma(srgb_to_linear(rgb))
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    m = np.hypot(gx, gy)
    if normalize:
        m = m / (m.max() + 1e-9)
    if blur > 1:
        im = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8))
        from PIL import ImageFilter
        im = im.filter(ImageFilter.GaussianBlur(radius=blur))
        m = np.asarray(im, float) / 255.0
    return m


def edt(mask, radius=48):
    """Distance from every pixel to the nearest True in `mask`, exact out to `radius`.

    Squared Euclidean distance is separable — d2(y,x) = min_x' [(x-x')^2 + min_y' ((y-y')^2 +
    f(y',x'))] — so two passes of shifted minima give the exact answer, with no scipy and no
    per-row Python loop. Cost is O(radius) vectorised passes, not O(radius^2).

    Truncating at `radius` is a feature, not a shortcut. Beyond it the distance saturates,
    which makes anything downstream a ROBUST loss: a render edge with no support at all —
    something the proxy invented, or a real object the model does not have yet — contributes a
    bounded penalty instead of dragging the whole fit toward itself. An untruncated chamfer is
    a least-squares fit to its own worst outlier.
    """
    mask = np.asarray(mask, bool)
    r = int(radius)
    big = float(r * r + 1)
    f = np.where(mask, 0.0, big)
    g = f.copy()
    for dx in range(1, r + 1):                      # min over x', exactly
        c = float(dx * dx)
        g[:, :-dx] = np.minimum(g[:, :-dx], f[:, dx:] + c)
        g[:, dx:] = np.minimum(g[:, dx:], f[:, :-dx] + c)
    d = g.copy()
    for dy in range(1, r + 1):                      # then min over y'
        c = float(dy * dy)
        d[:-dy, :] = np.minimum(d[:-dy, :], g[dy:, :] + c)
        d[dy:, :] = np.minimum(d[dy:, :], g[:-dy, :] + c)
    return np.sqrt(np.minimum(d, big))


def chamfer(render, reference, radius=48, ref_keep=0.05, size=None):
    """How far every edge the render DREW sits from the nearest strong edge in the photograph.

    The one metric here built to survive a real photograph against a rough untextured proxy,
    and the only one that does not care what the scene is. It asks a single question — "is
    what I drew supported by something real nearby" — and never the reverse, so a photo full
    of cracks, stains, lettering and a burnt-in timestamp costs nothing. Those are edges the
    render lacks; lacking them is correct.

    `ref_keep` is the fraction of reference pixels kept as edges, so the threshold is a
    QUANTILE, not a level. A quantile transfers between photographs; an absolute threshold is
    tuned to one image's contrast and is another thing that only works on the picture you
    tuned it on. The default keeps the strongest 5% — on a forecourt that is the paint, the
    kerbs and the hardware, not the tarmac's grain.

    Returns `chamfer_px` (lower is better, saturating at `radius`) plus `edge_mass`.

    MIND THE EDGE MASS. The score is a weighted mean over the render's OWN edges, so a render
    with no edges scores a perfect 0. Point the camera at the sky and the loss is delighted.
    An optimiser will find that before it finds the answer, so `edge_mass` — the mean raw
    gradient of the render — is here to catch a step that buys its improvement by drawing
    less, and it must be read as ABSOLUTE.

    It was born broken and that is worth keeping. The first version thresholded the RENDER at
    its own 95th percentile and reported the fraction of pixels kept, which is 5% of pixels by
    the definition of a percentile: it read exactly 0.0500 for all eleven cameras of a real
    sweep and could not have read anything else. A guard computed from a self-normalising
    quantile guards nothing. The render is now weighted by raw gradient STRENGTH with no
    threshold at all — strong edges dominate on their merits — and only the reference keeps a
    quantile, where adapting to the photograph's own contrast is the point.
    """
    if isinstance(render, (str, Path)):
        ref_im = Image.open(reference).convert("RGB")
        ren = _load(render, size=size or ref_im.size)
        ref = np.asarray(ref_im.resize(size, Image.LANCZOS) if size else ref_im, float) / 255.0
    else:
        ren, ref = np.asarray(render, float), np.asarray(reference, float)
        if ren.max() > 1.5:
            ren = ren / 255.0
        if ref.max() > 1.5:
            ref = ref / 255.0

    er = edge_map(ren, blur=0, normalize=False)
    ef = edge_map(ref, blur=0, normalize=False)
    d = edt(ef >= max(float(np.quantile(ef, 1.0 - ref_keep)), 1e-9), radius=radius)

    mass = float(er.mean())
    tot = float(er.sum())
    if tot < 1e-9:
        return {"chamfer_px": float(radius), "edge_mass": 0.0,
                "note": "the render has no edges at all — scoring it as maximally wrong "
                        "rather than perfectly right"}
    return {"chamfer_px": round(float((er * d).sum() / tot), 3),
            "edge_mass": round(mass, 6)}


def read_ids(path):
    """Read mirage_render's --ids AOV: a 16-bit binary PGM of object ids, 0 = nothing."""
    with open(path, "rb") as f:
        def tok():
            out = b""
            while True:
                c = f.read(1)
                if not c or c.isspace():
                    if out:
                        return out
                    if not c:
                        raise ValueError("truncated PGM header")
                    continue
                if c == b"#":
                    f.readline()
                    continue
                out += c
        magic = tok()
        if magic != b"P5":
            raise ValueError(f"{path}: expected a binary PGM (P5), got {magic!r}")
        w, h, mx = int(tok()), int(tok()), int(tok())
        if mx != 65535:
            raise ValueError(f"{path}: expected 16-bit ids, got maxval {mx}")
        buf = f.read(w * h * 2)
    return np.frombuffer(buf, dtype=">u2").reshape(h, w).astype(int)


def chamfer_per_object(render, ids, reference, names=None, radius=48, ref_keep=0.05, grow=2):
    """`chamfer`, but one number per PLACED OBJECT instead of one for the frame.

    The whole-frame score answers "is the picture right", which on a real reproduction is a
    question with no useful answer. Sweeping eleven cameras over the forecourt, the frame
    score sat between 14.1 and 15.0 px for everything from fov 0.9 to 1.6 — flat, because
    most of what it was averaging was a proxy yard and a box van that will never match, and
    those drown whatever the camera is doing. A number that says "the frame is 14 px wrong"
    tells you nothing about what to fix.

    Split by object and the average stops hiding things. `ids` comes from the renderer's
    --ids AOV, so the split is the op-log's own `place(mark=...)` list — the scene's real
    decomposition, not regions somebody drew on the image. Each object is scored on its own
    pixels plus a `grow`-px collar (its silhouette lives on the boundary, so masking to the
    interior alone would clip exactly the edge that matters).

    Returns {name: {chamfer_px, px, edge_mass}}. Sorting on chamfer_px is a MEASURED work
    order — worst object first — which is what photoscene's `worst_first` was guessing at.

    `px` is the object's screen area and it is the guard: chamfer is a mean over an object's
    own edges, so an object that shrinks toward nothing scores beautifully. Read them together
    or the loss will happily optimise an object out of the frame. Same trap as `edge_mass`,
    one level down.

    KNOW WHAT THIS SCORES. "Supported", not "correct". The distance field does not know which
    object a photo edge belongs to, so an object sitting in clutter is near SOMETHING no matter
    where you put it, and it is flattered for it. On case 26 the fire box scores 3.93 px while
    the forecourt scores 18.02 — the fire box is 8 kpx of hardware buried in a thicket of hoses
    and kerb, and it would score well nailed to the wrong spot. Trust the ranking where an
    object is large and its surroundings are clean, and read a good score in clutter as "not
    yet contradicted" rather than "right".
    """
    ren = np.asarray(render, float)
    ref = np.asarray(reference, float)
    ren = ren / 255.0 if ren.max() > 1.5 else ren
    ref = ref / 255.0 if ref.max() > 1.5 else ref
    ids = np.asarray(ids)
    if ids.shape != ren.shape[:2]:
        raise ValueError(f"ids {ids.shape} do not match the render {ren.shape[:2]} — the AOV "
                         f"must be rendered at the same size, and never resampled")

    er = edge_map(ren, blur=0, normalize=False)
    ef = edge_map(ref, blur=0, normalize=False)
    d = edt(ef >= max(float(np.quantile(ef, 1.0 - ref_keep)), 1e-9), radius=radius)

    out = {}
    for k in sorted(int(v) for v in np.unique(ids) if v > 0):
        m = edt(ids == k, radius=max(grow, 1)) <= grow      # the object, plus its silhouette
        w = np.where(m, er, 0.0)
        tot = float(w.sum())
        name = names[k - 1] if names and k - 1 < len(names) else f"id{k}"
        out[name] = {"chamfer_px": float(radius) if tot < 1e-9
                     else round(float((w * d).sum() / tot), 3),
                     "px": int((ids == k).sum()),
                     "edge_mass": round(float(tot / max(int(m.sum()), 1)), 6)}
    return out


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
