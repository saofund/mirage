"""A scorecard — which object in this render is the worst, and in what way.

`photomatch` answers "is what I drew supported by something real". That question is
deliberately one-directional, and it has a blind spot big enough to lose a scene in: a flat
grey box scores BEAUTIFULLY. It draws almost no edges, so almost nothing it draws is
unsupported. Every object in case 26 was a box, and the chamfer loss was content.

This module asks the other three questions, per object, and ranks the answers:

  ``tone``    am I the right BRIGHTNESS here?  Median luma of the object's pixels, render
              against photograph. The forecourt apron sat at 0.30 linear — twice the
              photo's — through two rounds of texture work, because nothing was measuring
              it and a picture looks fine until you put the real one beside it.

  ``detail``  am I DENSE enough here?  Relative contrast — mean |grad L| over mean L — in
              the render against the photograph, over the same pixels. Scale-free, so it
              compares a dark van to a bright one. This is the box detector: a chamfered,
              labelled, hardware-covered part lands around 0.4-0.8 of the reference; a
              painted box lands near 0.05, and says so in a number instead of waiting for
              somebody to say 都是方盒子.

  ``cast``    am I the right COLOUR here?  Red-minus-blue in chromaticity. The renderer's
              sky is a cool gradient, so an untuned scene drifts blue everywhere at once —
              a global error that is invisible object by object and unmistakable as a
              column of numbers all carrying the same sign.

WHAT `detail` IS NOT: a target. A photograph carries sensor noise, dirt, dents, reflections
of things outside the frame and infinite micro-texture, and chasing a ratio of 1.0 would
mean faking all of it. It is a FLOOR. Under ~0.25 an object is reading flat and wants
geometry or a map; between 0.4 and 1.0 it is doing its job; far ABOVE 1.0 means the render
is noisier than the photograph, which on a path-traced image usually means under-sampled
rather than well-detailed. Read it as a flag, never as a loss to minimise.

The ranking is what makes it useful without a human in the loop: severity is area-weighted,
so the table's top row is the largest thing that is most wrong — the next piece of work,
chosen by measurement instead of by whatever caught the eye.

    from mirage.critique import scorecard, report
    rows = scorecard(render, reference, ids, names)
    print(report(rows))

Needs numpy + Pillow. The reference photograph never leaves this process: the optional
plate tints the RENDER only, so a scorecard is safe to publish even when its reference
is not.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .photomatch import _load, _luma, chamfer_per_object, linear_to_srgb, srgb_to_linear

__all__ = ["scorecard", "report", "plate"]

# Thresholds. They are flags, not losses — see the module docstring on `detail`.
TONE_TOL = 0.06      # sRGB luma difference that starts to read as "wrong exposure"
DETAIL_FLOOR = 0.25  # below this an object is reading flat
CAST_TOL = 0.020     # chromaticity (r-b) difference that starts to read as a colour cast


def _detail(srgb, mask, blur=1.2):
    """Mean |grad| of DISPLAY luma over a mask — how much visible structure is here.

    Measured on the sRGB-encoded image on purpose, which is the one place in this codebase
    that is right. The first version divided a linear gradient by the region's mean
    luminance to be "scale-free", and that blows up on anything dark: the hoses — thin
    near-black tubes against a bright floor — scored 4.06 and got flagged as *noisy*, and
    the tiny sliver of visible island scored 3.57. A ratio whose denominator goes to zero
    tells you about its denominator. Display luma is bounded, is what the eye is looking
    at, and makes a flat panel read low whether it is lit or shadowed.

    Both images are lightly BLURRED first, and the reason matters more than it sounds. A
    CCTV frame carries sensor noise and compression mush over every square inch; a path
    traced image is clean. Unblurred, a big smooth region scores the photograph's NOISE as
    detail the render is missing, so the apron's ratio sat at 0.18 and would not move for
    twice the staining, a tile half the size, or turning the denoiser down — none of which
    were the reason. A 1.2 px blur removes what no render should imitate and leaves the
    stains, cracks, kerbs and tyre marks, which are the structure actually being asked
    about."""
    g = _luma(srgb)
    if blur:
        from PIL import ImageFilter
        im = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
        g = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=blur)), float) / 255.0
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return float(np.hypot(gx, gy)[mask].mean())


def _cast(srgb, mask):
    """Red-minus-blue in CHROMATICITY — bounded in [-1, 1], so a dark region cannot post a
    cast of -2.5 the way a luma-normalised difference did."""
    s = np.clip(srgb, 0, None)
    tot = s.sum(axis=-1) + 1e-6
    return float(((s[..., 0] - s[..., 2]) / tot)[mask].mean())


def scorecard(render, reference, ids, names=None, chamfer=True):
    """Score every object in the ids AOV against the photograph. Returns rows, worst first.

    `ids` is the renderer's --ids output (see photomatch.read_ids) and `names` the --id-tags
    list in the same order, so the split is the op-log's own `place(mark=...)` decomposition
    — the scene's real parts, not regions drawn on the image afterwards.
    """
    ren = np.asarray(render, float)
    ren = ren / 255.0 if ren.max() > 1.5 else ren
    ref = np.asarray(reference, float)
    ref = ref / 255.0 if ref.max() > 1.5 else ref
    ids = np.asarray(ids)
    if ids.shape != ren.shape[:2]:
        raise ValueError(f"ids {ids.shape} do not match the render {ren.shape[:2]} — the AOV "
                         f"must be rendered at the same size, and never resampled")
    lin_ren, lin_ref = srgb_to_linear(ren), srgb_to_linear(ref)
    total = float(ids.size)

    cham = {}
    if chamfer:
        cham = chamfer_per_object(ren, ids, ref, names=names)

    rows = []
    for k in sorted(int(v) for v in np.unique(ids) if v > 0):
        mask = ids == k
        px = int(mask.sum())
        if px < 800:
            # Too few pixels for a median or a mean gradient to mean anything. At 295 px the
            # visible sliver of "island" posted detail 3.20 and got flagged noisy every round
            # — a row that could only ever be read and dismissed, which is a row that trains
            # you to skim the table.
            continue
        name = names[k - 1] if names and k - 1 < len(names) else f"id{k}"
        t_ren = float(linear_to_srgb(np.median(_luma(lin_ren)[mask])))
        t_ref = float(linear_to_srgb(np.median(_luma(lin_ref)[mask])))
        d_ren, d_ref = _detail(ren, mask), _detail(ref, mask)
        ratio = d_ren / max(d_ref, 1e-6)
        c_ren, c_ref = _cast(ren, mask), _cast(ref, mask)
        # severity: how wrong, weighted by how much of the frame it is. sqrt so a huge
        # background does not drown a foreground object that is completely wrong.
        area_w = (px / total) ** 0.5
        sev = area_w * (max(abs(t_ren - t_ref) - TONE_TOL, 0.0) * 6.0
                        + max(DETAIL_FLOOR - ratio, 0.0) * 4.0
                        + max(abs(c_ren - c_ref) - CAST_TOL, 0.0) * 20.0)
        flags = []
        if t_ren - t_ref > TONE_TOL:
            flags.append("too light")
        elif t_ref - t_ren > TONE_TOL:
            flags.append("too dark")
        if ratio < DETAIL_FLOOR:
            flags.append("reads flat")
        elif ratio > 2.5:
            flags.append("noisy (raise spp?)")
        if abs(c_ren - c_ref) > CAST_TOL:
            flags.append("cool cast" if c_ren < c_ref else "warm cast")
        rows.append({"name": name, "px": px, "tone": round(t_ren, 3), "tone_ref": round(t_ref, 3),
                     "detail": round(ratio, 3), "cast": round(c_ren - c_ref, 3),
                     "chamfer_px": cham.get(name, {}).get("chamfer_px"),
                     "severity": round(sev, 4), "flags": flags})
    rows.sort(key=lambda r: -r["severity"])
    return rows


def report(rows, width=96):
    """The scorecard as a table — worst first, so the top line is the next piece of work."""
    out = [f"{'object':<14}{'px':>8}{'tone':>7}{'ref':>7}{'detail':>8}{'cast':>7}"
           f"{'cham':>7}{'sev':>7}  flags", "-" * width]
    for r in rows:
        c = "  -  " if r["chamfer_px"] is None else f"{r['chamfer_px']:5.1f}"
        out.append(f"{r['name']:<14}{r['px']:>8,}{r['tone']:>7.3f}{r['tone_ref']:>7.3f}"
                   f"{r['detail']:>8.2f}{r['cast']:>+7.3f}{c:>7}{r['severity']:>7.3f}"
                   f"  {', '.join(r['flags'])}")
    worst = [r for r in rows if r["flags"]]
    out.append("-" * width)
    # ONE number for the whole frame. The per-object rows say WHAT to fix; this says whether
    # the last change helped, which is the question a loop with nobody in it has to answer
    # for itself. Track it across iterations: 7.23 -> 4.97 is progress, and it is not a mood.
    out.append(f"TOTAL SEVERITY {sum(r['severity'] for r in rows):.3f}"
               f"   ({len(worst)} of {len(rows)} objects flagged"
               + (f"; worst: {worst[0]['name']} - {', '.join(worst[0]['flags'])}" if worst else "")
               + ")")
    return "\n".join(out)


def plate(render, ids, rows, path, names=None):
    """Tint each object by severity over the RENDER — green fine, amber marginal, red bad.

    Deliberately the render alone: a scorecard you cannot show anybody is a scorecard you
    stop running, and the photograph behind these numbers is not always ours to publish.
    """
    ren = np.asarray(render, float)
    ren = ren / 255.0 if ren.max() > 1.5 else ren
    ids = np.asarray(ids)
    sev = {r["name"]: r["severity"] for r in rows}
    hi = max([s for s in sev.values()] + [1e-6])
    out = ren * 0.55
    for k in sorted(int(v) for v in np.unique(ids) if v > 0):
        name = names[k - 1] if names and k - 1 < len(names) else f"id{k}"
        if name not in sev:
            continue
        t = min(sev[name] / hi, 1.0)
        tint = np.array([0.30 + 0.70 * t, 0.85 - 0.55 * t, 0.25])
        m = ids == k
        out[m] = np.clip(ren[m] * 0.45 + tint * 0.55 * (0.35 + 0.65 * t), 0, 1)
    Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8)).save(path)
    return Path(path)


def load_pair(render_path, reference_path):
    """Load a render and a reference resized to it (the ids AOV fixes the geometry, so the
    render is the one that must not be resampled)."""
    ren = _load(render_path)
    size = (ren.shape[1], ren.shape[0])
    ref = _load(reference_path, size=size)
    return ren, ref
