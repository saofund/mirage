"""The imaging chain has to be right about images whose answer is known.

Every number in `sensor` is calibrated against a photograph, so an error here does not look
like an error — it looks like a picture with slightly the wrong amount of grain, which is
indistinguishable from an artistic choice. The first version of `match` calibrated grain
against a standard deviation and buried the render in film grain; that is the class of
mistake these tests exist to catch.
"""
import numpy as np

from mirage.sensor import apply, match, measure, noise_floor


def _flat(h=180, w=240, v=0.45):
    return np.full((h, w, 3), v)


def _noisy(h=180, w=240, sigma=0.03, seed=1, v=0.45):
    rng = np.random.default_rng(seed)
    return np.clip(_flat(h, w, v) + rng.normal(0, sigma, (h, w, 1)), 0, 1)


def test_noise_floor_recovers_a_known_sigma():
    for sigma in (0.01, 0.03, 0.06):
        got = noise_floor(_noisy(sigma=sigma))
        # the high-pass keeps most but not all of the noise power, so this is a ratio test
        assert 0.4 * sigma < got < 1.2 * sigma, f"sigma {sigma} -> {got}"


def test_noise_floor_ignores_edges_where_a_std_would_not():
    # THE bug this estimator exists for: detail must not be counted as noise.
    # Edges must be a MINORITY of pixels for a median to ignore them, which in a real image
    # they are. At a 1.6 px high-pass each edge disturbs about five columns, so a stripe
    # every 40 px leaves ~88% of the frame clean and the median sits in it.
    img = _flat()
    img[:, ::40] = 0.85                      # hard edges, no noise at all
    assert noise_floor(img) < 0.01
    assert measure(img)["luma_hf"] > 0.03    # ...which a standard deviation happily reports


def test_grain_lands_on_the_reference_noise_floor():
    ref, ren = _noisy(sigma=0.05, seed=2), _noisy(sigma=0.01, seed=3)
    par = match(ren, ref)
    assert par["grain"] > 0
    out = apply(ren, **par)
    a, b = noise_floor(ref), noise_floor(out)
    assert 0.6 * a < b < 1.6 * a, f"{b} vs {a}"


def test_no_grain_when_the_render_is_already_noisier():
    ref, ren = _noisy(sigma=0.01, seed=4), _noisy(sigma=0.05, seed=5)
    assert match(ren, ref)["grain"] == 0.0    # never a negative amount, never a clamp bug


def test_chroma_blur_removes_colour_noise_and_keeps_luma():
    rng = np.random.default_rng(6)
    base = _flat()
    img = np.clip(base + rng.normal(0, 0.04, base.shape), 0, 1)   # independent per channel
    before = measure(img)
    out = apply(img, chroma_blur=2.0)
    after = measure(out)
    assert after["chroma_hf"] < 0.5 * before["chroma_hf"]
    assert after["luma_hf"] > 0.7 * before["luma_hf"], "luma detail must survive"


def test_apply_is_a_no_op_with_no_parameters():
    img = _noisy()
    assert np.allclose(apply(img), img)


def test_vignette_darkens_corners_not_the_centre():
    img = _flat(120, 200, 0.7)
    out = apply(img, vignette=1.0)
    h, w = img.shape[:2]
    assert out[h // 2, w // 2, 0] > 0.69
    assert out[2, 2, 0] < 0.55


def test_tone_curve_is_monotone_and_bounded():
    from mirage.sensor import tone_curve
    x = np.linspace(0, 1, 256)[None, :, None] * np.ones((1, 1, 3))
    y = tone_curve(x, black=0.12, white=0.80, gamma=1.2)[0, :, 0]
    assert np.all(np.diff(y) >= -1e-12), "a transfer curve must not fold"
    assert y.min() >= 0.0 and y.max() <= 1.0


def test_fit_tone_lands_the_percentiles():
    # a render-like image (compressed range) against a photo-like one (full range)
    rng = np.random.default_rng(11)
    photo = np.clip(rng.normal(0.46, 0.20, (200, 300, 3)), 0, 1)
    render = np.clip(photo * 0.62 + 0.17, 0, 1)          # squashed toward the middle
    from mirage.sensor import fit_tone, tone_curve
    par = fit_tone(render, photo)
    out = tone_curve(render, **par)
    for p in (1, 50, 99):
        a = np.percentile(photo.mean(-1), p)
        b = np.percentile(out.mean(-1), p)
        assert abs(a - b) < 0.06, f"p{p}: {b:.3f} vs {a:.3f}"


def test_fit_tone_does_not_crush_the_middle_to_buy_the_ends():
    # the degenerate fit this guards against solved gamma from the top percentile alone and
    # put the median at 0.264 against a target of 0.459
    rng = np.random.default_rng(12)
    photo = np.clip(rng.normal(0.46, 0.20, (200, 300, 3)), 0, 1)
    render = np.clip(photo * 0.62 + 0.17, 0, 1)
    from mirage.sensor import fit_tone, tone_curve
    out = tone_curve(render, **fit_tone(render, photo))
    assert abs(np.percentile(out.mean(-1), 50) - np.percentile(photo.mean(-1), 50)) < 0.05
