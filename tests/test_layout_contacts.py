"""`layout.fit_contacts` — recovering a known placement from a synthetic photograph.

The point of these is that the fitter is checked against ground truth, not against a picture.
A box is placed at a known (x, y, yaw), projected, and the two observations a real photograph
would give — the columns it spans and the line its contacts trace — are read off that
projection. The fitter then has to find its way back from a deliberately wrong start.
"""
import numpy as np
import pytest

from mirage.layout import fit_contacts, ground_line
from mirage.solve import Camera, project

CAM = Camera(eye=(-3.1, -12.0, 6.1), target=(-2.8, -11.2, 5.7), fov_y=0.505)
W, H = 1600, 900


def box(length=6.5, width=2.0, height=1.95, n=9):
    """A vehicle-shaped box resting on z=0, with its edges sampled.

    Sampled rather than eight corners because `fit_contacts` fits the NEAR flank's contacts
    and eight corners leave it two — enough to define a line, not enough to be a fit."""
    hx, hy = length / 2, width / 2
    xs = np.linspace(-hx, hx, n)
    return np.array([[x, sy * hy, sz * height]
                     for x in xs for sy in (-1, 1) for sz in (0, 1)], float)


def put(v, x, y, yaw, s=1.0):
    a = np.radians(yaw)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    return (np.asarray(v, float) * s) @ R.T + np.array([x, y, 0.0])


def observe(v, x, y, yaw, s=1.0):
    """What a photograph of this placement would tell us: columns, and the contact line."""
    q = project(CAM, put(v, x, y, yaw, s), W, H)
    cols = (float(q[:, 0].min()), float(q[:, 0].max()))
    low = np.asarray(v, float)
    near = low[(low[:, 2] < 0.06) & (low[:, 1] < low[:, 1].mean())]
    g = project(CAM, put(near, x, y, yaw, s), W, H)
    return cols, ground_line({float(p[0]): float(p[1]) for p in g})


def test_recovers_a_known_placement():
    v = box()
    cols, line = observe(v, 9.8, 18.1, -4.0)
    x, y, yaw, sc, rms = fit_contacts(v, CAM, W, H, columns=cols, line=line,
                                      start=(7.9, 14.6, -14.5, 1.0))
    assert rms < 1.0
    assert x == pytest.approx(9.8, abs=0.25)
    assert y == pytest.approx(18.1, abs=0.25)
    assert sc == pytest.approx(1.0, abs=0.03)


def test_recovers_distance_from_far_start():
    """The failure this exists for: an object metres too close to the camera."""
    v = box()
    cols, line = observe(v, 9.5, 18.5, 0.0)
    x, y, yaw, sc, rms = fit_contacts(v, CAM, W, H, columns=cols, line=line,
                                      start=(9.5, 12.0, 0.0, 1.0))
    assert y == pytest.approx(18.5, abs=0.3)
    assert rms < 1.5


def test_recovers_scale():
    """A body built the wrong size shows up as a scale, not as a bad fit."""
    v = box(length=5.4)
    cols, line = observe(box(length=6.5), 9.8, 18.1, 0.0)
    x, y, yaw, sc, rms = fit_contacts(v, CAM, W, H, columns=cols, line=line,
                                      start=(9.8, 18.1, 0.0, 1.0), scale_range=(0.8, 1.4))
    assert sc == pytest.approx(6.5 / 5.4, rel=0.06)


def test_top_row_stops_it_shrinking():
    """Without the crop constraint a shorter box at a nearer distance fits equally well."""
    v = box()
    cols, line = observe(v, 9.8, 18.1, 0.0)
    _, _, _, sc, _ = fit_contacts(v, CAM, W, H, columns=cols, line=line,
                                  start=(9.8, 18.1, 0.0, 1.0), top_row=-5.0,
                                  scale_range=(0.8, 1.3))
    assert sc >= 0.97


def test_near_flank_only_beats_both_flanks():
    """Fitting both flanks to one line leaves a residual no placement can remove."""
    v = box()
    cols, line = observe(v, 9.8, 18.1, -4.0)
    _, _, _, _, rms_near = fit_contacts(v, CAM, W, H, columns=cols, line=line,
                                        start=(7.9, 14.6, -14.5, 1.0))
    _, _, _, _, rms_both = fit_contacts(v, CAM, W, H, columns=cols, line=line,
                                        start=(7.9, 14.6, -14.5, 1.0), near_flank=False)
    assert rms_near < rms_both


def test_ground_line_fits_a_line():
    k, c = ground_line({760: 109.0, 900: 102.0, 1100: 92.0})
    assert k == pytest.approx(-0.05, abs=0.005)
    assert c == pytest.approx(147.0, abs=1.0)


def test_needs_contacts():
    v = box()
    v[:, 2] += 3.0                                   # nothing touching the ground
    with pytest.raises(ValueError):
        fit_contacts(v, CAM, W, H, columns=(700, 1100), line=(-0.05, 148.0), z_eps=0.01)


def observe_rows(v, x, y, yaw, s=1.0):
    q = project(CAM, put(v, x, y, yaw, s), W, H)
    return (float(q[:, 1].min()), float(q[:, 1].max()))


def test_rows_alone_recover_distance():
    """For an object whose contact line the photograph hides — behind a kerb, in shade."""
    v = box()
    cols, _ = observe(v, 9.8, 18.1, 0.0)
    rows = observe_rows(v, 9.8, 18.1, 0.0)
    x, y, yaw, sc, rms = fit_contacts(v, CAM, W, H, columns=cols, rows=rows,
                                      start=(9.8, 13.0, 0.0, 1.0))
    assert y == pytest.approx(18.1, abs=0.4)
    assert rms < 2.0


def test_a_clipped_bound_can_be_left_out():
    """The frame clips the object, so one bound describes the frame and not the object."""
    v = box()
    cols, line = observe(v, 9.8, 18.1, 0.0)
    x, y, yaw, sc, rms = fit_contacts(v, CAM, W, H, columns=(None, cols[1]), line=line,
                                      start=(8.0, 15.0, 0.0, 1.0))
    assert y == pytest.approx(18.1, abs=0.4)


def test_needs_something_to_fit_distance_with():
    with pytest.raises(ValueError):
        fit_contacts(box(), CAM, W, H, columns=(700, 1100))
