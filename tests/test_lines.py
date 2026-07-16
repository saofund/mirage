"""Reading painted lines off a photograph: the mask, the fit, and the corners.

The division of labour these encode: a human is good at "there is a line roughly here" and
hopeless at its sub-pixel position — reading a bay corner by eye produced a quad that missed
the paint entirely. The machine is the exact opposite. So the seed is rough by design, and
the tests check that a rough seed still lands on the true line.
"""
import math

import numpy as np
import pytest

from mirage.solve import (Camera, apply_h, camera_from_vanishing_points, direction_vp,
                          fit_line, fit_quad, ground_point, homography, line_intersection,
                          paint_mask, project, trace_line, vanishing_point)


def _canvas(h=400, w=600, bg=(0.10, 0.11, 0.13)):
    return np.tile(np.asarray(bg, float), (h, w, 1))


def _n(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


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


# --- tracing: the widest run wins ------------------------------------------- #
def test_trace_line_ignores_clutter_beside_the_line():
    """The reason trace_line exists. `fit_line` takes every mask pixel in its band, and a
    band wide enough for a rough seed also swallows whatever is lying next to the line —
    on the forecourt, the island's bright concrete apron. The marking is the WIDEST bright
    thing on its scanline; the clutter is not."""
    img = _canvas()
    _stroke(img, (200, 40), (200, 360), width=16)             # the real line
    for y in range(50, 350, 14):                              # persistent junk 30 px away
        _stroke(img, (230, y), (238, y + 5), width=5)
    m = paint_mask(img)
    f = fit_line(m, (200, 60), (200, 340), halfwidth=45)
    t = trace_line(m, (200, 60), (200, 340), search=45, min_width=8)
    # the junk drags the band fit off the true x = 200; the trace ignores it
    fx = -f["line"][2] / f["line"][0]
    tx = -t["line"][2] / t["line"][0]
    assert abs(tx - 200) < 1.5, f"trace landed at {tx}"
    assert abs(fx - 200) > abs(tx - 200), "fit_line should be the one that got dragged"


def test_trace_line_reports_the_width_profile():
    """`widths` is what lets a caller check the fit against physics the fit never saw: a
    constant-width line must thicken smoothly toward the camera."""
    img = _canvas(h=400, w=400)
    # a wedge: genuinely wider at the bottom, like a line receding in perspective
    for i, y in enumerate(range(40, 360)):
        half = 4 + i * 0.02
        img[y, int(200 - half):int(200 + half) + 1] = (0.92, 0.92, 0.90)
    t = trace_line(paint_mask(img), (200, 60), (200, 340), search=40, min_width=5)
    assert t["widths"][0] < t["widths"][-1]
    assert t["width_px"] > 8


def test_trace_line_rejects_outlier_scanlines():
    """Even taking the widest run, a scanline can grab the wrong thing where the paint is
    scuffed. Two bad centres out of fourteen quietly tilt the whole line, so drop them."""
    img = _canvas()
    _stroke(img, (100, 60), (500, 300), width=12)
    # a big bright blob squarely across the line -> those scanlines pick it instead
    img[150:200, 250:340] = (0.93, 0.93, 0.91)
    t = trace_line(paint_mask(img), (100, 60), (500, 300), search=60, min_width=6)
    assert t["rejected"] > 0, "the blob's scanlines should have been rejected"
    assert t["rms"] < 3.0
    a, b, c = t["line"]
    for p in ((100, 60), (300, 180), (500, 300)):
        assert abs(a * p[0] + b * p[1] + c) < 4.0


def test_trace_line_complains_when_it_finds_nothing():
    img = _canvas()
    _stroke(img, (100, 60), (500, 300), width=12)
    with pytest.raises(ValueError):
        trace_line(paint_mask(img), (60, 330), (120, 370), search=8, min_width=6)


# --- corners ---------------------------------------------------------------- #
def test_line_intersection():
    h = {"line": (0.0, 1.0, -100.0)}     # y = 100
    v = {"line": (1.0, 0.0, -250.0)}     # x = 250
    assert line_intersection(h, v) == pytest.approx([250, 100])


def test_parallel_lines_have_no_corner():
    with pytest.raises(ValueError):
        line_intersection({"line": (0, 1, -10)}, {"line": (0, 1, -50)})


# --- vanishing points: the answer, and whether it means anything ------------- #
def _traced(p0, p1, n=16, jitter=0.0, seed=0):
    """A trace_line-shaped result, so vanishing_point can propagate its scatter."""
    rng = np.random.default_rng(seed)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0, 1, n)[:, None]
    P = p0 + (p1 - p0) * t
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    P = P + np.array([-d[1], d[0]]) * rng.normal(0, jitter, (n, 1))
    ctr = P.mean(0)
    _, _, vt = np.linalg.svd(P - ctr)
    nn = np.array([-vt[0][1], vt[0][0]])
    return {"line": (nn[0], nn[1], -nn @ ctr), "pts": P.tolist(),
            "rms": float(np.sqrt((((P - ctr) @ nn) ** 2).mean()))}


def test_vanishing_point_agrees_with_the_plain_intersection():
    a, b = _traced((100, 900), (400, 100)), _traced((900, 900), (600, 100))
    r = vanishing_point(a, b)
    assert r["p"] == pytest.approx(line_intersection(a, b))
    assert r["angle_deg"] == pytest.approx(41.1, abs=1.0)


def test_vanishing_point_refuses_a_near_parallel_pair():
    """The forecourt's cross edges: 2.5 deg apart, meeting 16 kpx away. Both lines fit to well
    under a pixel and the intersection is a perfectly ordinary pair of floats. It is worthless,
    and no residual says so — only the angle does. So the tool must be the one to refuse."""
    far = _traced((879, 551), (1353, 499))
    near = _traced((1107, 913), (1742, 828))
    assert abs(np.degrees(np.arctan2(52, 474)) - np.degrees(np.arctan2(85, 635))) < 3.0
    with pytest.raises(ValueError, match="too near parallel"):
        vanishing_point(far, near)
    p = line_intersection(far, near)          # the raw number is still there, and still fine
    assert p[0] > 10000, "the intersection exists; that was never the problem"


def test_vanishing_point_accepts_the_forecourt_pair_that_is_real():
    """The same photo's edges ALONG the strip: 17.7 deg apart, and they hold up."""
    left = _traced((1023, 780), (1326, 1260))
    right = _traced((1584, 700), (2250, 1260))
    r = vanishing_point(left, right)
    assert r["angle_deg"] == pytest.approx(17.7, abs=1.5)
    assert r["p"][1] < 0, "the strip recedes upward — its VP is above the frame"
    assert r["p"] == pytest.approx([272, -412], abs=25)


def test_reach_counts_how_far_past_the_evidence_you_are():
    """rms cannot tell interpolation from extrapolation. reach can: it is the distance to the
    answer in units of the evidence that produced it."""
    near = vanishing_point(_traced((100, 900), (400, 100)), _traced((900, 900), (600, 100)))
    far = vanishing_point(_traced((100, 900), (400, 100)), _traced((900, 900), (1000, 100)))
    assert near["angle_deg"] > far["angle_deg"]        # 41 deg vs 13: a shallower wedge...
    assert near["reach"] < 1.5
    assert far["reach"] > near["reach"] * 2            # ...puts the answer 4x further out


def test_sigma_blows_up_as_the_lines_close_up():
    """Same scatter on the traced centres, shallower wedge -> the answer knows less. This is
    the part rms refuses to report."""
    wide = vanishing_point(_traced((100, 900), (400, 100), jitter=0.8, seed=1),
                           _traced((900, 900), (600, 100), jitter=0.8, seed=2))
    tight = vanishing_point(_traced((100, 900), (400, 100), jitter=0.8, seed=1),
                            _traced((900, 900), (1000, 100), jitter=0.8, seed=2))
    assert tight["sigma_px"] > wide["sigma_px"] * 3
    assert wide["sigma_px"] < 40


def test_sigma_is_a_lower_bound_and_says_so():
    """The trap this whole function exists for. Bias every centre on one line the SAME way —
    a lens bending it, a scan grabbing a kerb. The scatter about the fit is untouched, so rms
    and sigma do not move at all, and the vanishing point walks away. sigma sees random error
    only; a systematic is invisible to it and must be found by perturbing the input."""
    a = _traced((100, 900), (400, 100), jitter=0.8, seed=1)
    b = _traced((900, 900), (600, 100), jitter=0.8, seed=2)
    clean = vanishing_point(a, b)

    P = np.asarray(b["pts"], float)
    P[:, 0] += np.linspace(0, 6, len(P))       # a 6 px systematic tilt, no added scatter
    ctr = P.mean(0)
    _, _, vt = np.linalg.svd(P - ctr)
    nn = np.array([-vt[0][1], vt[0][0]])
    bent = {"line": (nn[0], nn[1], -nn @ ctr), "pts": P.tolist(),
            "rms": float(np.sqrt((((P - ctr) @ nn) ** 2).mean()))}
    r = vanishing_point(a, bent)

    assert bent["rms"] == pytest.approx(b["rms"], abs=0.15), "the bias left the residual alone"
    assert r["sigma_px"] == pytest.approx(clean["sigma_px"], rel=0.35), "...and sigma too"
    moved = np.linalg.norm(r["p"] - clean["p"])
    assert moved > clean["sigma_px"], \
        f"the VP moved {moved:.1f} px, sigma claimed {clean['sigma_px']:.1f} — that gap IS the"


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


# --- putting a pixel on the ground, and testing the camera that does it ------- #
W, H_ = 2560, 1440
CASE26 = Camera([2.0, -4.2, 4.3], [1.53, 2.57, 0.06], fov_y=1.181)   # case 26, as asserted


def test_ground_point_inverts_project():
    rng = np.random.default_rng(7)
    for z in (0.0, 0.35):
        w = np.stack([rng.uniform(-8, 8, 40), rng.uniform(1, 14, 40), np.full(40, z)], 1)
        back = ground_point(CASE26, project(CASE26, w, W, H_), W, H_, z=z)
        assert np.abs(back - w).max() < 1e-9


def test_ground_point_inverts_project_through_a_lens():
    """Looser, and the slack is not a disagreement: `project` undistorts by fixed-point
    iteration and this distorts in closed form, so the gap is that iteration's tolerance."""
    cam = Camera([2.0, -4.2, 4.3], [1.53, 2.57, 0.06], fov_y=1.181, k1=-0.18, k2=0.04)
    rng = np.random.default_rng(11)
    w = np.stack([rng.uniform(-8, 8, 40), rng.uniform(1, 14, 40), np.zeros(40)], 1)
    back = ground_point(cam, project(cam, w, W, H_), W, H_)
    assert np.abs(back - w).max() < 5e-3


def test_ground_point_refuses_a_ray_that_never_lands():
    """A pixel above the horizon has no ground under it. The guard matters: t = (z-eye_z)/d_z
    goes NEGATIVE there, and eye + t*d is then a confident point BEHIND the camera."""
    up = ground_point(CASE26, [[1280, 5]], W, H_)         # near the top edge, above the horizon
    assert np.isnan(up).all()
    down = ground_point(CASE26, [[1280, 1400]], W, H_)    # near the bottom edge: real ground
    assert np.isfinite(down).all() and down[0, 2] == pytest.approx(0.0, abs=1e-9)


def test_direction_vp_is_the_limit_of_a_receding_point():
    """The definition, checked against different algebra: a vanishing point is where a point
    on the line goes as it recedes forever. `direction_vp` drops the translation instead."""
    u = np.array([0.3, 1.0, 0.0]) / np.linalg.norm([0.3, 1.0, 0.0])
    far = project(CASE26, [[0.0, 0.0, 0.0] + u * 1e7], W, H_)[0]
    assert direction_vp(CASE26, u, W, H_) == pytest.approx(far, abs=0.5)


def test_direction_vp_matches_a_vanishing_point_measured_off_the_projection():
    """The two halves meeting: build two REAL parallel world lines, project them, measure
    their VP off the image with `vanishing_point`, and predict it with `direction_vp`. One
    reads pixels, the other reads a camera; they have no code in common past the basis."""
    u = np.array([0.45, 1.0, 0.0]) / np.linalg.norm([0.45, 1.0, 0.0])
    perp = np.array([-u[1], u[0], 0.0])
    lines = []
    for off in (0.0, 3.4):                       # two rails 3.4 m apart, both along u
        a = np.array([0.5, 1.5, 0.0]) + perp * off
        px = project(CASE26, [a + u * t for t in (0.0, 2.0, 4.0, 6.0)], W, H_)
        lines.append(_traced(px[0], px[-1]))
    measured = vanishing_point(lines[0], lines[1])
    assert measured["angle_deg"] > 5.0, "the fixture must give the gate something to pass"
    assert measured["p"] == pytest.approx(direction_vp(CASE26, u, W, H_), abs=1.0)


def test_direction_vp_is_the_same_for_a_direction_and_its_reverse():
    """A line's two ends share one vanishing point; flipping u flips both terms of the ratio."""
    u = np.array([0.3, 1.0, -0.2])
    assert direction_vp(CASE26, u, W, H_) == pytest.approx(direction_vp(CASE26, -u, W, H_))


def test_direction_vp_has_no_finite_answer_across_the_view_axis():
    fwd = CASE26.basis()[0]
    across = np.cross(fwd, [0, 0, 1.0])          # perpendicular to the axis: stays parallel
    assert np.isnan(direction_vp(CASE26, across, W, H_)).all()


def _aimed(yaw_deg, pitch_deg, fov=1.0, eye=(0.0, 0.0, 4.3)):
    """A camera aimed BY yaw and pitch, so a test that varies one varies only one.

    Writing the targets out by hand does not do that. [0.4, 1, 0.2] and [-0.9, 1, 0.2] look
    like a pure yaw change — the z is the same in both — but pitch is z against the HORIZONTAL
    length, and widening x from 0.4 to -0.9 took that from 1.077 to 1.345 and the pitch with
    it, 3.4 deg of it. The literals hid the coupling; a parameterisation cannot.
    """
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    d = np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])
    return Camera(eye, np.asarray(eye, float) + d, fov_y=fov)


def test_every_horizontal_direction_lands_on_one_horizon():
    """What lets dx and dy be read apart. With no roll the horizon is one horizontal image
    line, so yaw slides a VP ALONG it and can never lift it — only pitch and fov do that."""
    cam = _aimed(20, -25)
    ys = [direction_vp(cam, [math.sin(a), math.cos(a), 0.0], W, H_)[1]
          for a in np.radians([-40, -12, 0, 25, 60])]
    assert np.ptp(ys) < 1e-6, f"horizontal directions left the horizon: {ys}"
    for yaw in (-40, 0, 55):                        # same pitch and fov, anywhere in yaw
        assert direction_vp(_aimed(yaw, -25), [0, 1.0, 0], W, H_)[1] == \
            pytest.approx(ys[0], abs=1e-6)
    for pitch in (-24, -26):                        # ...and pitch is what does lift it
        assert abs(direction_vp(_aimed(20, pitch), [0, 1.0, 0], W, H_)[1] - ys[0]) > 20


def test_the_forecourts_asserted_camera_does_not_survive_its_own_photograph():
    """The finding this pair was built to make. Case 26's camera was asserted from framing
    cues and the case recorded that it "checks out" — against those same cues. The bay strip's
    VP is measured (17.7 deg apart, the pair that passed the gate) and the camera never saw it.

    fov is swept because that is the parameter everyone reaches for first, and it is not the
    problem: the miss is mostly dx, and dx is yaw.
    """
    measured = np.array([272.0, -412.0])
    assert np.linalg.norm(direction_vp(CASE26, [0, 1, 0], W, H_) - measured) > 1000
    best = min(np.linalg.norm(direction_vp(Camera(CASE26.eye, CASE26.target, fov_y=f),
                                           [0, 1, 0], W, H_) - measured)
               for f in np.linspace(0.4, 2.4, 41))
    assert best > 1000, f"a plain fov sweep got to {best:.0f} px — then fov WAS the story"


# --- and the inverse: two vanishing points back to a camera ------------------- #
def test_camera_from_vanishing_points_round_trips_direction_vp():
    """The inverse of direction_vp, tested as one. Take a known camera, image two orthogonal
    world directions, hand ONLY those two pixels and the two directions back, and demand the
    orientation and fov return. eye is passed through — VPs cannot carry it — so the recovered
    camera reproduces the SAME VPs from the SAME eye. A rough look_world breaks the four-fold
    flip; it need only have the right signs, so the crude true direction rounded hard is fair."""
    for eye, tgt, fov in [([2.0, -4.2, 4.3], [1.53, 2.57, 0.06], 1.181),
                          ([-3.0, -9.0, 6.5], [1.0, 4.0, 0.4], 0.7),
                          ([0.0, 0.0, 12.0], [2.0, 5.0, 0.0], 0.95)]:
        cam = Camera(eye, tgt, fov_y=fov)
        dirs = np.array([[0, 1.0, 0], [1.0, 0, 0]])          # two orthogonal ground directions
        vps = np.array([direction_vp(cam, d, W, H_) for d in dirs])
        look = np.sign(np.array(tgt) - eye)                  # a crude "that way", signs only
        got, info = camera_from_vanishing_points(vps, dirs, W, H_, eye=eye, look_world=look)
        assert info["fov_y"] == pytest.approx(fov, abs=1e-6)
        assert info["residual_px"] < 1e-6
        assert not info["sense_ambiguous"]
        fwd_true, fwd_got = _n(np.array(tgt) - eye), _n(got.target - got.eye)
        assert fwd_got == pytest.approx(fwd_true, abs=1e-6)


def test_camera_from_vanishing_points_is_blind_to_position():
    """The property that both frees it and limits it: slide the eye anywhere and the VPs do
    not move, so the solver returns the same orientation and cannot object to the wrong eye."""
    cam = Camera([2.0, -4.2, 4.3], [1.53, 2.57, 0.06], fov_y=0.9)
    dirs = np.array([[0, 1.0, 0], [1.0, 0, 0]])
    vps = np.array([direction_vp(cam, d, W, H_) for d in dirs])
    look = [0, 1.0, -1.0]
    a, _ = camera_from_vanishing_points(vps, dirs, W, H_, eye=[2.0, -4.2, 4.3], look_world=look)
    b, _ = camera_from_vanishing_points(vps, dirs, W, H_, eye=[40.0, 15.0, 4.3], look_world=look)
    assert _n(a.target - a.eye) == pytest.approx(_n(b.target - b.eye), abs=1e-9)


def test_camera_from_vanishing_points_is_a_line_not_a_ray_without_a_look_hint():
    """The four-fold flip is real, not a bug: VP(u) == VP(-u), so a camera turned 180 deg
    about a world axis reproduces every vanishing point. Without look_world the solver keeps
    the most upright of the four and says the sense is unpinned; the fov and tilt are still
    right, and a look hint recovers the true aim exactly."""
    cam = Camera([0, 0, 6.0], [1.0, 3.0, -2.0], fov_y=0.8)
    dirs = np.array([[0, 1.0, 0], [1.0, 0, 0]])
    vps = np.array([direction_vp(cam, d, W, H_) for d in dirs])
    fwd_true = _n(cam.target - cam.eye)

    guess, info = camera_from_vanishing_points(vps, dirs, W, H_, eye=[0, 0, 6.0])
    assert info["sense_ambiguous"] and info["residual_px"] < 1e-6   # a valid camera...
    assert info["fov_y"] == pytest.approx(0.8, abs=1e-6)            # ...fov and tilt intact...
    assert guess.basis()[2][2] > 0                                  # ...and upright...
    # ...but the aim may be one of the flips, and a rough look pins the real one.
    aimed, info2 = camera_from_vanishing_points(vps, dirs, W, H_, eye=[0, 0, 6.0],
                                                look_world=[0, 1, -1])
    assert not info2["sense_ambiguous"]
    assert _n(aimed.target - aimed.eye) == pytest.approx(fwd_true, abs=1e-6)


def test_camera_from_vanishing_points_refuses_directions_it_cannot_orthogonalise():
    """Two vanishing points on the same side of centre cannot be images of perpendicular
    directions under any focal length (A_i.A_j + B_i.B_j stays positive). The solver must say
    so, not return an imaginary focal length as a nan or a confident wrong number."""
    with pytest.raises(ValueError, match="orthogonal"):
        camera_from_vanishing_points(np.array([[1300.0, 300.0], [1350.0, 310.0]]),
                                     np.array([[0, 1.0, 0], [1.0, 0, 0]]), W, H_)


def test_camera_from_vanishing_points_overdetermines_fov_with_a_third_axis():
    """Three orthogonal directions give three pairs, so fov is overdetermined and the pairs'
    disagreement is a real bar. On exact synthetic VPs it collapses to zero; that it is zero
    HERE and non-zero on noisy input is the point of reporting it."""
    cam = Camera([1.0, -5.0, 5.0], [1.2, 2.0, 0.3], fov_y=1.05)
    dirs = np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])   # +y, +x, +z: mutually orthogonal
    vps = np.array([direction_vp(cam, d, W, H_) for d in dirs])
    got, info = camera_from_vanishing_points(vps, dirs, W, H_, eye=[1.0, -5.0, 5.0])
    assert info["n_vps"] == 3
    assert info["fov_y"] == pytest.approx(1.05, abs=1e-6)
    assert info["f_spread"] < 1e-6


def test_camera_from_vanishing_points_reports_case_26_fov_as_a_range():
    """The honest ending. Case 26's cross direction is degenerate and its VP is only good to
    13%; the two edge estimates put it at x=9672 and x=11010 on the horizon y=-412. Feed each
    with the clean strip VP and the fov comes back 0.48-0.53 — a range, from the INPUT spread
    that no residual sees. Both solves are internally exact."""
    strip = np.array([272.0, -412.0])
    fovs = []
    for x_cross in (9672.0, 11010.0):
        _, info = camera_from_vanishing_points(
            np.array([strip, [x_cross, -412.0]]),
            np.array([[0, 1.0, 0], [1.0, 0, 0]]), W, H_, eye=[2.0, -4.2, 4.3])
        assert info["residual_px"] < 1e-6            # each is internally perfect...
        fovs.append(info["fov_y"])
    assert min(fovs) == pytest.approx(0.484, abs=0.01)   # ...yet they disagree by 0.04 rad,
    assert max(fovs) == pytest.approx(0.525, abs=0.01)   # which is the real error bar
    assert all(f < 0.6 for f in fovs), "the asserted 1.181 is nowhere near this range"
