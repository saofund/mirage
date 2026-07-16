"""The loss: does it actually rank a better match above a worse one?

A metric that doesn't order things correctly is worse than none — it launders guessing as
measurement. Each test here asks the metric to notice a specific, deliberate error of the
kind that actually happened while matching a real photograph.
"""
import numpy as np
import pytest
from PIL import Image

from mirage.photomatch import (chamfer, chamfer_per_object, compare, edge_map, edt,
                               linear_to_srgb, read_ids, srgb_to_linear)


def _save(tmp_path, name, arr):
    p = tmp_path / name
    Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8)).save(p)
    return p


def _scene(shade=0.5, shift=0, w=240, h=160):
    """A grey field with a darker bar — a stand-in for 'concrete with a painted bay'.

    Laid out in FRACTIONS of the frame so the same scene at another resolution really is the
    same scene; absolute pixel coords would make a rescaled copy a different picture, and the
    resize test would be measuring my own mistake.
    """
    a = np.full((h, w, 3), shade, float)
    a[int(0.25 * h):int(0.69 * h),
      int(0.25 * w) + shift:int(0.625 * w) + shift] = shade * 0.35
    return a


def test_srgb_linear_roundtrip():
    c = np.linspace(0, 1, 64)
    assert np.allclose(linear_to_srgb(srgb_to_linear(c)), c, atol=1e-6)


def test_srgb_is_not_linear_at_the_dark_end():
    # why the metrics work in linear: sRGB compresses exactly where tarmac and leather live,
    # so a "small" sRGB distance down there can be a large physical error
    assert srgb_to_linear(0.5) == pytest.approx(0.214, abs=0.005)


def test_exposure_catches_a_render_that_is_too_bright(tmp_path):
    """The concrete error, in one number. Matching the forecourt, the first render's slab
    came out ~0.78 sRGB against a reference at ~0.48 and I only found out by eye."""
    ref = _save(tmp_path, "ref.png", _scene(0.48))
    bright = _save(tmp_path, "bright.png", _scene(0.78))
    ok = _save(tmp_path, "ok.png", _scene(0.49))

    bad = compare(bright, ref)
    good = compare(ok, ref)
    assert bad["exposure"]["ratio"] > 2.0, bad["exposure"]
    assert bad["exposure"]["stops_off"] > 1.0
    assert abs(good["exposure"]["stops_off"]) < 0.1
    assert good["colour_rmse_linear"] < bad["colour_rmse_linear"]


def test_identical_images_score_perfectly(tmp_path):
    ref = _save(tmp_path, "ref.png", _scene())
    m = compare(ref, ref)
    assert m["colour_rmse_linear"] == pytest.approx(0, abs=1e-6)
    assert m["edge_correlation"] == pytest.approx(1.0, abs=1e-6)
    assert m["exposure"]["ratio"] == pytest.approx(1.0, abs=1e-3)


def test_edge_correlation_degrades_with_misalignment(tmp_path):
    """The geometry signal, and it must be a RAMP, not a cliff: a hard edge IoU scores zero
    for both a 4 px miss and a 40 px miss, which gives nothing to follow."""
    ref = _save(tmp_path, "ref.png", _scene(shift=0))
    scores = []
    for i, s in enumerate((2, 10, 40)):
        p = _save(tmp_path, f"s{i}.png", _scene(shift=s))
        scores.append(compare(p, ref)["edge_correlation"])
    assert scores[0] > scores[1] > scores[2], scores
    assert scores[0] > 0.5, "a 2 px miss should still score well"


def test_edge_map_is_blind_to_a_flat_field():
    flat = np.full((60, 60, 3), 0.5)
    assert edge_map(flat).max() < 1e-6


def test_regions_localise_the_error(tmp_path):
    """One global number averages a good sky against a bad floor and reports 'mediocre'.
    Regions say WHICH part is wrong."""
    ref = np.full((160, 240, 3), 0.5)
    ref[:80] = (0.2, 0.3, 0.8)                 # top: blue
    ref[80:] = (0.7, 0.35, 0.15)               # bottom: orange
    ren = ref.copy()
    ren[80:] = (0.2, 0.6, 0.2)                 # bottom badly wrong; top exact

    rp = _save(tmp_path, "ref.png", ref)
    gp = _save(tmp_path, "ren.png", ren)
    m = compare(gp, rp, regions={"top": (0, 0, 1, 0.5), "bottom": (0, 0.5, 1, 1)})
    assert m["regions"]["top"]["err"] < 1e-3
    assert m["regions"]["bottom"]["err"] > 0.1
    assert m["regions"]["bottom"]["err"] > m["regions"]["top"]["err"] * 50


def test_compare_resizes_a_mismatched_render(tmp_path):
    ref = _save(tmp_path, "ref.png", _scene(w=240, h=160))
    ren = _save(tmp_path, "ren.png", _scene(w=480, h=320))
    m = compare(ren, ref)
    assert m["size"] == (240, 160)
    assert m["edge_correlation"] > 0.9


def test_diff_plate_is_written(tmp_path):
    ref = _save(tmp_path, "ref.png", _scene(shift=0))
    ren = _save(tmp_path, "ren.png", _scene(shift=12))
    out = tmp_path / "plate" / "diff.png"
    m = compare(ren, ref, plate=out)
    assert out.exists() and m["plate"] == str(out)
    im = Image.open(out)
    assert im.height > im.width          # reference / render / edges, stacked


# --- the distance transform ------------------------------------------------- #
def test_edt_is_exact_against_brute_force():
    """Separable shifted-minima, so it had better agree with the definition."""
    rng = np.random.default_rng(3)
    m = rng.random((40, 55)) < 0.03
    m[0, 0] = True
    got = edt(m, radius=30)

    ys, xs = np.nonzero(m)
    yy, xx = np.mgrid[0:40, 0:55]
    want = np.full((40, 55), np.inf)
    for y, x in zip(ys, xs):
        want = np.minimum(want, np.hypot(yy - y, xx - x))
    want = np.minimum(want, np.sqrt(30 * 30 + 1))       # the truncation
    assert np.abs(got - want).max() < 1e-9


def test_edt_saturates_rather_than_running_away():
    """Truncation is what makes chamfer robust: an edge with no support anywhere pays a
    bounded price instead of dragging the fit toward itself."""
    m = np.zeros((80, 80), bool)
    m[40, 40] = True
    d = edt(m, radius=10)
    assert d[40, 40] == 0.0
    assert d[40, 47] == pytest.approx(7.0)
    assert d[0, 0] == pytest.approx(np.sqrt(101))       # far corner: clamped, not 56.6
    assert d.max() == pytest.approx(np.sqrt(101))


# --- chamfer ---------------------------------------------------------------- #
def test_chamfer_is_zero_on_itself_and_grows_with_misalignment():
    base = _scene()
    assert chamfer(base, base)["chamfer_px"] < 0.6
    prev = -1.0
    for shift in (0, 3, 8, 16):
        c = chamfer(_scene(shift=shift), base)["chamfer_px"]
        assert c > prev, f"shift {shift} did not score worse than the one before"
        prev = c


def _stripe(shift=0, w=240, h=160, shade=0.5):
    """A FULL-HEIGHT bar: every edge it owns is vertical, so a sideways shift moves all of
    them perpendicular to themselves and the displacement is fully visible in the pixels."""
    a = np.full((h, w, 3), shade, float)
    a[:, 100 + shift:140 + shift] = shade * 0.35
    return a


def test_chamfer_reads_out_in_pixels():
    """The units are the point — 'the geometry is 8 px out' is actionable, 'the correlation
    is 0.31' is not."""
    for s in (4, 8, 14):
        c = chamfer(_stripe(shift=s), _stripe())["chamfer_px"]
        assert c == pytest.approx(s, abs=1.5), f"a {s} px shift read as {c}"


def test_chamfer_cannot_see_a_slide_along_an_edge():
    """The aperture problem. Every edge-based loss has it and this one must not be read as if
    it does not.

    `_scene`'s bar shifted 8 px sideways scores 3.3, not 8. Its two vertical sides did move
    8 px — but its top and bottom slid ALONG themselves and stayed exactly on the reference's,
    which is not an error the pixels contain. The full-height stripe, whose every edge is
    perpendicular to the motion, reads the full 8.

    So `chamfer_px` is a loss to minimise, not a displacement to trust. It is a mean over the
    render's edges and edges parallel to the error are entitled to say nothing.
    """
    slid = chamfer(_scene(shift=8), _scene())["chamfer_px"]
    perp = chamfer(_stripe(shift=8), _stripe())["chamfer_px"]
    assert perp == pytest.approx(8, abs=1.5)
    assert slid < perp * 0.6, "the top/bottom edges did not move; the metric should say so"


def _grubby(a, seed=0):
    """The photograph problem: a real frame is the scene PLUS cracks, stains, grain and a
    burnt-in timestamp. None of that is structure a proxy render can or should reproduce."""
    rng = np.random.default_rng(seed)
    b = a.copy()
    h, w = b.shape[:2]
    b += rng.normal(0, 0.05, b.shape)                       # grain
    for _ in range(60):                                     # cracks and stains
        y, x = rng.integers(0, h - 6), rng.integers(0, w - 20)
        b[y:y + rng.integers(1, 3), x:x + rng.integers(5, 20)] *= rng.uniform(0.4, 1.6)
    b[4:12, 4:90] = 0.95                                    # the timestamp overlay
    return np.clip(b, 0, 1)


def test_chamfer_survives_the_photograph_where_correlation_does_not(tmp_path):
    """The reason this metric exists, stated as a test.

    Same aligned geometry, but the reference is a PHOTOGRAPH: same scene plus grain, cracks
    and a timestamp. Correlation collapses because it is comparing appearance, and it collapses
    to roughly what a WRONG camera scores — so it cannot rank them, and that is what sent me
    surveying. Chamfer only asks whether the render's own edges are supported, so the grubbiness
    is free and the ranking survives.
    """
    truth, photo = _scene(), _grubby(_scene())
    wrong = _scene(shift=16)

    good_c = chamfer(truth, photo)["chamfer_px"]
    bad_c = chamfer(wrong, photo)["chamfer_px"]

    good_e = compare(_save(tmp_path, "a.png", truth), _save(tmp_path, "p.png", photo))
    bad_e = compare(_save(tmp_path, "b.png", wrong), _save(tmp_path, "p.png", photo))
    ge = good_e["edge_correlation"]
    be = bad_e["edge_correlation"]

    assert bad_c > good_c * 3, f"chamfer must rank them: aligned {good_c}, 16 px out {bad_c}"
    assert good_c < 2.0, f"the grubbiness cost the aligned render {good_c} px — it should be free"
    assert ge - be < bad_c - good_c, \
        f"correlation separated them by {ge - be:.3f}; it is not the instrument here"


def test_chamfer_refuses_to_reward_an_empty_render():
    """The trap in any asymmetric loss: it is a mean over the render's OWN edges, so drawing
    nothing is flawless. Point the camera at the sky and the score is perfect. It has to say so
    instead, or an optimiser will find this before it finds the answer."""
    blank = np.full((160, 240, 3), 0.5)
    r = chamfer(blank, _scene())
    assert r["edge_mass"] == 0.0
    assert r["chamfer_px"] >= 48.0, "an empty render scored well — the optimiser will exploit it"


def test_edge_mass_actually_varies_with_how_much_the_render_drew():
    """The test that was missing, and its absence shipped a constant.

    `edge_mass` guards against an optimiser buying its score by drawing less, so the one thing
    it must do is CHANGE when the render draws less. The first version thresholded the render
    at its own 95th percentile and reported the fraction kept — which is 5% of pixels, always.
    It read 0.0500 for all eleven cameras of a real sweep. Every other test still passed,
    because none of them ever compared the guard between two different renders.
    """
    plain = _scene()
    busy = _scene().copy()
    for x in range(10, 230, 8):                      # the same frame, far more structure
        busy[:, x:x + 2] *= 0.4
    faint = np.full((160, 240, 3), 0.5)
    faint[60:100, 60:180] = 0.497                    # one barely-there edge

    m_plain = chamfer(plain, plain)["edge_mass"]
    m_busy = chamfer(busy, plain)["edge_mass"]
    m_faint = chamfer(faint, plain)["edge_mass"]

    assert m_busy > m_plain * 3, f"a much busier render read {m_busy} vs {m_plain}"
    assert m_faint < m_plain / 10, f"a nearly blank render read {m_faint} vs {m_plain}"
    assert len({round(m_plain, 6), round(m_busy, 6), round(m_faint, 6)}) == 3


def test_edge_mass_is_absolute_not_relative_to_the_frames_own_peak():
    """Why edge_map(normalize=False). Self-normalising makes every frame peak at 1.0, so a
    frame with one faint edge and a frame with a hundred crisp ones look identical to any
    absolute reading — which is how the guard became a constant in the first place."""
    strong = _scene(shade=0.5)
    weak = np.full((160, 240, 3), 0.5)
    weak[40:110, 60:150] = 0.5 * 0.97                # same shape, 20x less contrast
    assert chamfer(weak, strong)["edge_mass"] < chamfer(strong, strong)["edge_mass"] / 5


# --- per object: one number each, instead of one for the frame -------------- #
def _three_objects(shift_b=0, w=240, h=160):
    """A frame with three things in it and the id AOV that names them."""
    img = np.full((h, w, 3), 0.55, float)
    ids = np.zeros((h, w), int)
    boxes = [(20, 60, 30, 80), (20, 60, 100 + shift_b, 150 + shift_b), (90, 140, 40, 190)]
    for k, (y0, y1, x0, x1) in enumerate(boxes, start=1):
        img[y0:y1, x0:x1] = 0.55 * (0.3 + 0.15 * k)
        ids[y0:y1, x0:x1] = k
    return img, ids


def test_per_object_isolates_the_one_that_is_wrong():
    """The reason this exists. On the forecourt the WHOLE-FRAME score sat at 14.1–15.0 px for
    every camera from fov 0.9 to 1.6 — flat, because a proxy yard and a box van that can never
    match drowned everything. Move ONE object and the frame average barely twitches; the
    per-object score names it.
    """
    truth, ids_t = _three_objects()
    ren, ids_r = _three_objects(shift_b=22)          # object 2 is 22 px out; 1 and 3 are exact
    names = ["left", "middle", "wide"]

    per = chamfer_per_object(ren, ids_r, truth, names=names)
    assert per["middle"]["chamfer_px"] > 8.0, per
    assert per["left"]["chamfer_px"] < 1.5, per
    assert per["wide"]["chamfer_px"] < 1.5, per
    assert per["middle"]["chamfer_px"] > per["left"]["chamfer_px"] + 6

    whole = chamfer(ren, truth)["chamfer_px"]
    assert whole < per["middle"]["chamfer_px"] / 2, \
        f"the frame average ({whole}) hid a {per['middle']['chamfer_px']} px error"


def test_per_object_gives_a_measured_work_order():
    """`worst_first` stops being a guess: sort on the number."""
    truth, _ = _three_objects()
    ren, ids_r = _three_objects(shift_b=22)
    per = chamfer_per_object(ren, ids_r, truth, names=["left", "middle", "wide"])
    order = sorted(per, key=lambda n: -per[n]["chamfer_px"])
    assert order[0] == "middle"


def test_per_object_reports_area_so_an_object_cannot_optimise_itself_away():
    """Same trap as edge_mass, one level down: chamfer is a mean over an object's OWN edges,
    so an object shrinking toward nothing scores beautifully. `px` is what catches it."""
    truth, _ = _three_objects()
    ren, ids = _three_objects()
    ids[ids == 2] = 0                       # object 2 all but vanishes from the AOV
    ids[30:33, 120:123] = 2
    per = chamfer_per_object(ren, ids, truth, names=["left", "middle", "wide"])
    assert per["middle"]["px"] < 60
    assert per["middle"]["px"] * 20 < per["left"]["px"]


def test_per_object_refuses_a_mismatched_id_buffer():
    """An id is a label, not a brightness: resampling it averages 1 and 3 into 2 and invents an
    object that was never there. So a size mismatch is an error, never a silent resize."""
    truth, _ = _three_objects()
    ren, ids = _three_objects(w=120, h=80)
    with pytest.raises(ValueError, match="never resampled"):
        chamfer_per_object(truth, ids, truth)


def test_read_ids_round_trips_a_16_bit_pgm(tmp_path):
    _, ids = _three_objects()
    p = tmp_path / "ids.pgm"
    with open(p, "wb") as f:
        f.write(b"P5\n%d %d\n65535\n" % (ids.shape[1], ids.shape[0]))
        f.write(ids.astype(">u2").tobytes())
    assert np.array_equal(read_ids(p), ids)
