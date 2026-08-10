"""Exact geometric checks for the photograph-matched Polo filler region."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "cases"))

from fuelcap2 import polo  # noqa: E402


def test_polo_region_builds_and_replays_identically():
    a = polo.build().build()
    b = polo.build().build()
    a.validate()
    b.validate()
    assert [v.co for v in a.verts] == [v.co for v in b.verts]
    fa = [[loop.vert.id for loop in a.face_loops(face)] for face in a.faces]
    fb = [[loop.vert.id for loop in b.face_loops(face)] for face in b.faces]
    assert fa == fb


def test_polo_scene_retains_the_measured_aperture_axes():
    # The mild oval is measured separately from the camera's remaining foreshortening.
    radii = polo._ellipse_plan(polo.OPEN_RX, polo.OPEN_RY, 96)
    plan = [r / polo.OPEN_R for r in radii]
    steps = 96
    mesh = polo._smooth_liner(plan, polo.OPEN_R, steps=steps).build()
    co = np.asarray([v.co for v in mesh.verts], float)
    # The OUTERMOST RING, not everything above a z plane. The band selector held only while
    # the liner ran straight from the paint into the bowl; once it grew the flange the
    # reference has, the band caught rings that are meant to be narrower and the test failed
    # on correct geometry. `_smooth_liner` emits ring 0 first, and ring 0 is the rim — which
    # is what "retains the measured aperture axes" was always about.
    rim = co[:steps]
    radii = np.hypot(rim[:, 0], rim[:, 1])
    assert abs(radii.min() - polo.OPEN_RX) < 1 * polo.MM
    assert abs(radii.max() - polo.OPEN_RY) < 1 * polo.MM
    assert 0.90 < polo.OPEN_RX / polo.OPEN_RY < 0.92
