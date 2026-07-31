"""The camera that took the picture — grain, chroma response, lens falloff.

A path tracer produces something no camera ever produced: an image with no sensor in front
of it. Measured against a CCTV still of the same scene, a finished render of case 26 was

    luma high-frequency    photo 0.061   render 0.032     half the fine detail
    chroma high-frequency  photo 0.0045  render 0.0125    three times the colour noise

Both differences are the imaging chain, not the scene. The render is missing the sensor's
grain and carrying the path tracer's own colour noise, which no camera has because every
camera resolves chroma far more coarsely than luma (a Bayer array interpolates it, and JPEG
subsamples it outright). Fixing the geometry, the materials and the light cannot close
either gap, and leaving them open is a large part of what "looks rendered" means.

    from mirage.sensor import apply, measure
    out = apply(rgb, grain=0.05, chroma_blur=1.4)
    print(measure(rgb))          # {'luma_hf': ..., 'chroma_hf': ...}

`match` reads the two numbers off a reference and returns the parameters that land on them,
so this is calibrated to a specific camera rather than tuned by eye.

NOT a beauty filter. Everything here models something a real imaging chain does; nothing
here exists to make the picture nicer. If a scene is meant to look like a render — a product
shot, a diagram — do not apply it.

Needs numpy + Pillow.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

__all__ = ["apply", "measure", "match", "noise_floor"]

# Rec.709 luma, the same weights photomatch and critique use.
_W = np.array([0.2126, 0.7152, 0.0722])


def _blur(ch, radius):
    if radius <= 0:
        return ch
    im = Image.fromarray((np.clip(ch, 0, 1) * 255 + 0.5).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=radius)), float) / 255.0


def _highpass(ch, radius=1.6):
    return ch - _blur(ch, radius)


def measure(rgb, radius=1.6, mask=None):
    """The two numbers that separate a render from a photograph of the same thing.

    `luma_hf` is the standard deviation of a high-passed luma channel — fine detail plus
    grain. `chroma_hf` is the same on the colour-only residual, which for a real camera is
    almost nothing and for a path tracer is its own sampling noise."""
    img = np.asarray(rgb, float)
    img = img / 255.0 if img.max() > 1.5 else img
    y = img @ _W
    hp = _highpass(y, radius)
    ch = np.stack([_highpass(img[..., k], radius) for k in range(3)], -1)
    ch = ch - ch.mean(-1, keepdims=True)          # colour-only: luma removed
    if mask is not None:
        hp, ch = hp[mask], ch[mask]
    return {"luma_hf": float(hp.std()), "chroma_hf": float(ch.std())}


def noise_floor(rgb, radius=1.6, mask=None):
    """The NOISE in an image, separated from its detail.

    A standard deviation of high-passed luma cannot tell the two apart, and on anything with
    texture or edges in it the detail wins: measured that way, a CCTV frame's ground reads
    0.070 and its "flat" van panel reads 0.114, neither of which is its noise. Calibrating
    grain against those numbers buries the render in film grain, which is exactly what the
    first attempt did.

    The median absolute deviation does tell them apart. Noise is everywhere and edges are a
    small minority of pixels, so the median is blind to the edges that dominate a mean.
    1.4826 * MAD is the usual consistent estimator of a gaussian sigma."""
    img = np.asarray(rgb, float)
    img = img / 255.0 if img.max() > 1.5 else img
    hp = _highpass(img @ _W, radius)
    if mask is not None:
        hp = hp[mask]
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))))


def apply(rgb, grain=0.0, chroma_blur=0.0, shadow_grain=1.8, sharpen=0.0, vignette=0.0,
          seed=7):
    """Put the image through an imaging chain.

    `chroma_blur` — blur the colour difference channels only, leaving luma untouched. This
    is what a Bayer sensor and a JPEG both do, and it removes a path tracer's colour speckle
    without costing any real detail, because there is almost no real detail in chroma.

    `grain` — monochrome sensor noise, scaled by `shadow_grain` in the darks, because that
    is where a sensor's read noise dominates and where a CCTV's gain is highest. Added to
    LUMA only: coloured grain is what the render already had too much of.

    `sharpen` — a touch of unsharp, standing in for the sharpening every camera pipeline
    applies. `vignette` — corner falloff, in stops at the corner.
    """
    img = np.asarray(rgb, float)
    img = img / 255.0 if img.max() > 1.5 else img
    y = img @ _W
    if chroma_blur > 0:
        c = img - y[..., None]                    # colour difference
        c = np.stack([_blur(c[..., k] + 0.5, chroma_blur) - 0.5 for k in range(3)], -1)
        img = np.clip(y[..., None] + c, 0, 1)
        y = img @ _W
    if sharpen > 0:
        img = np.clip(img + sharpen * (img - np.stack([_blur(img[..., k], 1.1)
                                                       for k in range(3)], -1)), 0, 1)
        y = img @ _W
    if grain > 0:
        rng = np.random.default_rng(seed)
        n = rng.normal(0.0, 1.0, y.shape)
        # more noise in the shadows, and none in the blown highlights
        gain = 1.0 + (shadow_grain - 1.0) * np.clip(1.0 - y / 0.5, 0, 1)
        gain *= np.clip((1.0 - y) / 0.15, 0, 1)
        img = np.clip(img + (grain * n * gain)[..., None], 0, 1)
    if vignette > 0:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2)) / np.sqrt(2.0)
        img = np.clip(img * (2.0 ** (-vignette * r ** 2))[..., None], 0, 1)
    return img


def match(render, reference, mask=None, radius=1.6, max_grain=0.14):
    """Solve `grain` and `chroma_blur` so the render's two statistics land on the photo's.

    Grain adds in quadrature to whatever fine detail the render already has, so the amount
    needed is sqrt(target^2 - have^2) — and if the render already has MORE than the
    reference, none is needed and the answer is zero rather than a negative number that a
    caller would have to remember to clamp.
    """
    # Grain against the NOISE FLOOR, not against luma_hf. luma_hf is detail plus noise, and
    # on a textured photograph the detail is most of it — matching that number with grain
    # buries the render (measured: it wanted 0.064 where the real noise is a fifth of that).
    na, nb = noise_floor(reference, radius, mask), noise_floor(render, radius, mask)
    need = na ** 2 - nb ** 2
    grain = float(min(np.sqrt(need), max_grain)) if need > 0 else 0.0
    a, b = measure(reference, radius, mask), measure(render, radius, mask)
    # chroma: how much blur takes b down to a. A gaussian's residual falls roughly as 1/r,
    # so one solve then one correction lands close enough for an 8-bit image.
    cb = 0.0
    if b["chroma_hf"] > a["chroma_hf"] > 0:
        cb = 1.0
        for _ in range(6):
            got = measure(apply(render, chroma_blur=cb), radius, mask)["chroma_hf"]
            if got <= a["chroma_hf"]:
                break
            cb *= 1.45
    return {"grain": round(grain, 4), "chroma_blur": round(cb, 2)}
