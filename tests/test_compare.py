"""The reverse-modelling metrics, checked against cases whose answer is known."""
from __future__ import annotations

import numpy as np
import pytest

from mirage import compare as C


def _disc(n, cx, cy, r):
    y, x = np.mgrid[0:n, 0:n]
    return (x - cx) ** 2 + (y - cy) ** 2 < r * r


def test_silhouette_scores_a_perfect_match_and_a_shifted_one():
    a = _disc(160, 80, 80, 40)
    same = C.silhouette(a, a)
    assert same["iou"] == pytest.approx(1.0)
    assert same["contour_px"] == pytest.approx(0.0, abs=1e-6)
    assert same["area_ratio"] == pytest.approx(1.0)

    shifted = C.silhouette(a, _disc(160, 90, 80, 40))
    assert shifted["iou"] < 0.85
    # a 10 px shift moves the contour by about 10 px on the flanks, less at top and bottom
    assert 3.0 < shifted["contour_px"] < 10.0
    assert shifted["area_ratio"] == pytest.approx(1.0, abs=0.02)


def test_silhouette_separates_same_area_from_same_shape():
    """The failure IoU alone hides: two shapes can share most of their area and still have
    an outline far apart, which is what a viewer sees."""
    a = _disc(200, 100, 100, 60)
    b = _disc(200, 100, 100, 60) | _disc(200, 100, 100, 66) & ~_disc(200, 100, 100, 60)
    grown = C.silhouette(a, b)
    assert grown["iou"] > 0.80          # looks close by area
    assert grown["contour_px"] > 3.0     # and the edge is nowhere near


def test_tone_reports_a_matching_frame_as_matching():
    rng = np.random.default_rng(0)
    ref = (rng.random((120, 120)) * 255).astype(np.uint8)
    t = C.tone(ref, ref)
    assert t["_frame"]["delta"] == pytest.approx(0.0, abs=1e-9)
    for k, v in t.items():
        assert v["delta"] == pytest.approx(0.0, abs=1e-9)


def test_tone_finds_a_cavity_that_is_too_bright_while_the_frame_matches():
    """The exact case the eye got wrong: overall means equal, one region two stops out."""
    ref = np.full((100, 100), 200.0)
    ref[60:, :] = 20.0                      # a dark cavity in a light frame
    ren = ref.copy()
    ren[60:, :] = 90.0                      # cavity far too bright
    ren[:60, :] = 200.0 - (90.0 - 20.0) * 40 / 60   # frame pulled down to compensate
    t = C.tone(ref, ren)
    assert abs(t["_frame"]["delta"]) < 1.0          # the frame says "fine"
    assert t["cavity"]["delta"] > 50.0              # the region says otherwise


def test_keypoints_report_millimetres_and_name_the_worst():
    k = C.keypoints({"cap": ((100, 100), (110, 100)),
                     "latch": ((50, 50), (52, 51))}, px_per_mm=2.0)
    assert k["cap"]["dist_mm"] == pytest.approx(5.0)
    assert k["cap"]["dx_mm"] == pytest.approx(5.0)
    assert k["_worst"] == "cap"


def test_structure_points_at_where_the_difference_is():
    a = np.zeros((96, 96))
    b = a.copy()
    b[10:30, 60:80] = 120.0                 # one bad patch, upper right
    st = C.structure(a, b, win=16)
    assert st["max"] > 20.0
    x, y = st["worst_px"]
    assert 55 < x < 85 and 5 < y < 35


def test_report_prints_every_section_it_was_given():
    a = _disc(120, 60, 60, 30)
    b = _disc(120, 64, 60, 30)
    rep = C.compare(np.where(a, 200, 20).astype(float), np.where(b, 200, 20).astype(float),
                    ref_mask=a, render_mask=b, px_per_mm=3.0,
                    key_pairs={"centre": ((60, 60), (64, 60))})
    text = str(rep)
    for word in ("silhouette", "tone", "keypoint", "structure"):
        assert word in text
