"""The loss: does it actually rank a better match above a worse one?

A metric that doesn't order things correctly is worse than none — it launders guessing as
measurement. Each test here asks the metric to notice a specific, deliberate error of the
kind that actually happened while matching a real photograph.
"""
import numpy as np
import pytest
from PIL import Image

from mirage.photomatch import compare, edge_map, linear_to_srgb, srgb_to_linear


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
