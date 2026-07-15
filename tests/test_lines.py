"""Reading painted lines off a photograph: the mask, the fit, and the corners.

The division of labour these encode: a human is good at "there is a line roughly here" and
hopeless at its sub-pixel position — reading a bay corner by eye produced a quad that missed
the paint entirely. The machine is the exact opposite. So the seed is rough by design, and
the tests check that a rough seed still lands on the true line.
"""
import numpy as np
import pytest

from mirage.solve import (fit_line, fit_quad, homography, line_intersection, paint_mask,
                          apply_h)


def _canvas(h=400, w=600, bg=(0.10, 0.11, 0.13)):
    return np.tile(np.asarray(bg, float), (h, w, 1))


def _stroke(img, p0, p1, width=7, col=(0.92, 0.92, 0.90)):
    """Rasterise a thick line: distance to the SEGMENT, in one vectorised pass."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    d = p1 - p0
    L2 = float(d @ d) or 1.0
    t = np.clip(((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L2, 0, 1)
    dist = np.hypot(xx - (p0[0] + t * d[0]), yy - (p0[1] + t * d[1]))
    img[dist <= width / 2] = col
    return img


# --- the mask --------------------------------------------------------------- #
def test_paint_mask_finds_bright_unsaturated_marks():
    img = _canvas()
    _stroke(img, (50, 350), (550, 60))
    m = paint_mask(img)
    assert 2000 < m.sum() < 12000
    assert m[350 - 2:350 + 3, 48:53].any()


def test_paint_mask_rejects_a_wet_reflection():
    """The reason saturation is in there at all: standing water is BRIGHT — it is mirroring
    the sky — and brightness alone calls a puddle a line. Paint is bright and colourless."""
    img = _canvas()
    img[100:200, 100:400] = (0.30, 0.48, 0.86)     # a bright blue puddle
    _stroke(img, (50, 350), (550, 300))            # real white paint
    m = paint_mask(img)
    assert not m[100:200, 100:400].any(), "a sky reflection was mistaken for paint"
    assert m[345:355, 60:70].any()


def test_paint_mask_accepts_uint8_or_float():
    img = _canvas()
    _stroke(img, (10, 200), (590, 200))
    a = paint_mask(img)
    b = paint_mask((img * 255).astype(np.uint8))
    assert a.sum() == b.sum()


# --- the fit ---------------------------------------------------------------- #
def test_fit_line_beats_a_sloppy_seed():
    """The whole point: seed it badly, get the line anyway."""
    img = _canvas()
    _stroke(img, (60, 340), (560, 80), width=9)
    m = paint_mask(img)
    r = fit_line(m, (40, 300), (600, 120), halfwidth=40)     # deliberately off
    a, b, c = r["line"]
    for p in ((60, 340), (310, 210), (560, 80)):             # true points lie on the fit
        assert abs(a * p[0] + b * p[1] + c) < 3.0


def test_rms_measures_the_stroke_width_not_the_error():
    """`rms` is the inliers' spread about the fit, and for a clean stroke that IS its own
    width: a uniform band of width w has sd w/sqrt(12). So a crisp 9 px line reports ~2.6 and
    is perfect. Reading it as fit error makes you chase a number that is already right."""
    img = _canvas()
    _stroke(img, (60, 340), (560, 80), width=9)
    r = fit_line(paint_mask(img), (40, 300), (600, 120), halfwidth=40)
    assert r["width_px"] == pytest.approx(9, abs=1.6)
    assert r["rms"] == pytest.approx(9 / np.sqrt(12), abs=0.5)

    img2 = _canvas()
    _stroke(img2, (60, 340), (560, 80), width=21)
    r2 = fit_line(paint_mask(img2), (40, 300), (600, 120), halfwidth=40)
    assert r2["width_px"] == pytest.approx(21, abs=3.0)
    assert r2["rms"] > r["rms"] * 1.8            # thicker line, same fit quality


def test_fit_line_handles_a_vertical():
    """PCA, not y = mx + c — a vertical line has infinite slope and a naive fit explodes."""
    img = _canvas()
    _stroke(img, (300, 20), (300, 380), width=9)
    m = paint_mask(img)
    r = fit_line(m, (296, 40), (304, 360), halfwidth=30)
    a, b, c = r["line"]
    assert abs(b) < 0.02                     # the normal is horizontal => the line is vertical
    assert abs(-c / a - 300) < 1.5


def test_fit_line_reports_rms_honestly():
    # a smear, not a line: the fit must SAY it is a bad fit rather than return it silently
    img = _canvas()
    rng = np.random.default_rng(1)
    for _ in range(400):
        p = rng.uniform([100, 100], [500, 300])
        img[int(p[1]):int(p[1]) + 6, int(p[0]):int(p[0]) + 6] = (0.9, 0.9, 0.9)
    m = paint_mask(img)
    r = fit_line(m, (100, 200), (500, 200), halfwidth=110)
    assert r["rms"] > 20, "a random smear should not report a tight fit"


def test_fit_line_complains_when_seeded_at_nothing():
    img = _canvas()
    _stroke(img, (60, 340), (560, 80))
    m = paint_mask(img)
    with pytest.raises(ValueError):
        fit_line(m, (30, 30), (120, 45), halfwidth=6)      # empty corner of the frame


# --- corners ---------------------------------------------------------------- #
def test_line_intersection():
    h = {"line": (0.0, 1.0, -100.0)}     # y = 100
    v = {"line": (1.0, 0.0, -250.0)}     # x = 250
    assert line_intersection(h, v) == pytest.approx([250, 100])


def test_parallel_lines_have_no_corner():
    with pytest.raises(ValueError):
        line_intersection({"line": (0, 1, -10)}, {"line": (0, 1, -50)})


def test_fit_quad_recovers_corners_through_a_gap():
    """Corners come from intersecting EDGES, so they survive the paint being scuffed away —
    which on a real forecourt is exactly where the corner is."""
    img = _canvas()
    quad = np.array([[120, 320], [480, 300], [430, 120], [170, 140.0]])
    for i in range(4):
        _stroke(img, quad[i], quad[(i + 1) % 4], width=7)
    # scrub the paint away around one corner
    img[290:335, 100:150] = (0.10, 0.11, 0.13)
    m = paint_mask(img)
    seeds = [(quad[i] + (quad[(i + 1) % 4] - quad[i]) * 0.25,
              quad[i] + (quad[(i + 1) % 4] - quad[i]) * 0.75) for i in range(4)]
    r = fit_quad(m, seeds, halfwidth=14)
    # fit_quad's corner i is edge[i-1] x edge[i] -> that is quad[i]
    for i in range(4):
        assert np.linalg.norm(r["corners"][i] - quad[i]) < 4.0, \
            f"corner {i}: {r['corners'][i]} vs {quad[i]}"
    assert max(r["rms"]) < 2.5


def test_fit_quad_feeds_a_homography():
    """The actual use: traced quad -> assumed real size -> the ground plane."""
    img = _canvas()
    quad = np.array([[120, 320], [480, 300], [430, 120], [170, 140.0]])
    for i in range(4):
        _stroke(img, quad[i], quad[(i + 1) % 4], width=7)
    m = paint_mask(img)
    seeds = [(quad[i] * 0.75 + quad[(i + 1) % 4] * 0.25,
              quad[i] * 0.25 + quad[(i + 1) % 4] * 0.75) for i in range(4)]
    r = fit_quad(m, seeds)
    world = np.array([[0, 0], [3.4, 0], [3.4, 6.0], [0, 6.0]])
    H = homography(r["corners"], world)
    got = apply_h(H, r["corners"])
    assert np.abs(got - world).max() < 1e-9
