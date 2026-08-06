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


def test_polo_scene_retains_the_measured_circular_aperture():
    # The top edge of the liner is authored in the panel plane. Check that the radius is
    # circular before the camera foreshortens it into the photographed vertical ellipse.
    radii = polo._ellipse_plan(polo.OPEN_RX, polo.OPEN_RY, 96)
    plan = [r / polo.OPEN_R for r in radii]
    mesh = polo._smooth_liner(plan, polo.OPEN_R, steps=96).build()
    co = np.asarray([v.co for v in mesh.verts], float)
    rim = co[co[:, 2] > -4.0 * polo.MM]
    radii = np.hypot(rim[:, 0], rim[:, 1])
    assert 18 * polo.MM < radii.max() - radii.min() < 30 * polo.MM
    assert abs(radii.max() - polo.OPEN_R) < 2 * polo.MM
