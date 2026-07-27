"""The scorecard has to be right about images whose answer is known.

A critic that is itself wrong is worse than no critic: it sends work at the wrong object
and the work looks justified. Both of this module's metrics shipped broken once — a
luma-normalised contrast that exploded on dark regions and flagged the hoses as *noisy*,
and an unblurred gradient that scored the reference photograph's sensor noise as detail the
render was missing. Neither was caught by looking at a table; both are caught here.
"""
import numpy as np
import pytest

from mirage.critique import DETAIL_FLOOR, plate, report, scorecard


def _ids(h, w, boxes):
    """An id AOV: {id: (y0, y1, x0, x1)} painted onto a zero field."""
    ids = np.zeros((h, w), int)
    for k, (y0, y1, x0, x1) in boxes.items():
        ids[y0:y1, x0:x1] = k
    return ids


def _noise(h, w, seed, amp=0.15, base=0.5):
    rng = np.random.default_rng(seed)
    return np.clip(base + rng.normal(0, amp, (h, w, 3)), 0, 1)


def test_identical_images_score_zero():
    # the degenerate case that any comparison metric must get right
    img = _noise(80, 120, 1)
    ids = _ids(80, 120, {1: (10, 70, 10, 110)})
    rows = scorecard(img, img, ids, names=["thing"], chamfer=False)
    assert len(rows) == 1
    assert rows[0]["flags"] == []
    assert rows[0]["severity"] == pytest.approx(0.0, abs=1e-9)
    assert rows[0]["detail"] == pytest.approx(1.0, abs=1e-6)


def test_a_flat_box_is_flagged_as_flat():
    # THE case this module exists for: a render with no structure where the photo has some.
    # chamfer cannot see this — a flat box draws no edges, so nothing it draws is unsupported.
    ref = _noise(80, 120, 2, amp=0.18)
    ren = np.full_like(ref, 0.5)                       # same mean, no structure at all
    ids = _ids(80, 120, {1: (10, 70, 10, 110)})
    row = scorecard(ren, ref, ids, names=["box"], chamfer=False)[0]
    assert row["detail"] < DETAIL_FLOOR
    assert "reads flat" in row["flags"]
    assert row["severity"] > 0


def test_detail_is_not_fooled_by_reference_noise():
    # a reference that is nothing BUT pixel noise, against a render carrying real structure:
    # the blur must stop the noise counting as detail the render owes. Unblurred this
    # inverts, which is exactly how the apron got blamed for two rounds.
    h, w = 96, 96
    ref = _noise(h, w, 3, amp=0.10, base=0.5)          # noise only, no structure
    ren = np.zeros((h, w, 3))
    ren[:, :w // 2] = 0.30                             # one big honest edge down the middle
    ren[:, w // 2:] = 0.70
    ids = _ids(h, w, {1: (5, 91, 5, 91)})
    row = scorecard(ren, ref, ids, names=["striped"], chamfer=False)[0]
    assert "reads flat" not in row["flags"], "a real edge lost to the reference's grain"


def test_tone_and_cast_are_flagged_with_the_right_sign():
    ref = _noise(80, 120, 4, base=0.45)
    ids = _ids(80, 120, {1: (10, 70, 10, 110)})
    light = scorecard(np.clip(ref + 0.25, 0, 1), ref, ids, names=["a"], chamfer=False)[0]
    dark = scorecard(np.clip(ref - 0.25, 0, 1), ref, ids, names=["a"], chamfer=False)[0]
    assert "too light" in light["flags"] and "too dark" in dark["flags"]
    cool = ref.copy()
    cool[..., 2] = np.clip(cool[..., 2] + 0.20, 0, 1)   # more blue
    warm = ref.copy()
    warm[..., 0] = np.clip(warm[..., 0] + 0.20, 0, 1)   # more red
    assert "cool cast" in scorecard(cool, ref, ids, names=["a"], chamfer=False)[0]["flags"]
    assert "warm cast" in scorecard(warm, ref, ids, names=["a"], chamfer=False)[0]["flags"]


def test_cast_stays_bounded_on_a_near_black_region():
    # the bug that made the hoses read -1.8: dividing a colour difference by a luminance
    # that is going to zero. Chromaticity cannot do that.
    ref = np.full((60, 60, 3), 0.02)
    ren = np.full((60, 60, 3), 0.02)
    ren[..., 2] = 0.05
    ids = _ids(60, 60, {1: (5, 55, 5, 55)})
    row = scorecard(ren, ref, ids, names=["dark"], chamfer=False)[0]
    assert abs(row["cast"]) <= 1.0


def test_ranking_is_area_weighted_and_reported():
    # a small disaster must not outrank a large one: the table's job is to name the next
    # piece of WORK, and the next piece of work is the biggest thing that is most wrong.
    ref = _noise(100, 200, 5, amp=0.16)
    ren = ref.copy()
    ren[5:95, 5:105] = 0.5                              # big flat region
    ren[5:50, 140:180] = 0.5                            # smaller flat region, same fault
    # both regions must clear the 800 px floor, or this stops testing the ranking and
    # starts testing the floor — which is how it broke when the floor was introduced.
    ids = _ids(100, 200, {1: (5, 95, 5, 105), 2: (5, 50, 140, 180)})
    rows = scorecard(ren, ref, ids, names=["big", "small"], chamfer=False)
    assert [r["name"] for r in rows] == ["big", "small"]
    text = report(rows)
    assert "TOTAL SEVERITY" in text and "big" in text


def test_ids_must_match_the_render(tmp_path):
    img = _noise(40, 40, 6)
    with pytest.raises(ValueError, match="do not match"):
        scorecard(img, img, np.zeros((20, 20), int), chamfer=False)


def test_plate_paints_only_the_render(tmp_path):
    # the plate must be publishable even when the reference photograph is not
    ren = _noise(40, 60, 7)
    ids = _ids(40, 60, {1: (5, 35, 5, 55)})
    rows = scorecard(ren, np.clip(ren + 0.3, 0, 1), ids, names=["a"], chamfer=False)
    p = plate(ren, ids, rows, tmp_path / "plate.png", names=["a"])
    assert p.exists()
