"""Validation poses and the flatness check that catches a projected reproduction."""
from __future__ import annotations

import math

import numpy as np
import pytest

from mirage import validate as V


def test_orbit_keeps_the_target_and_the_distance():
    eye, target, up = (0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    d0 = math.dist(eye, target)
    for p in V.orbit(eye, target, up, degrees=(-6.0, 6.0)):
        assert p["target"] == target
        assert math.dist(p["eye"], target) == pytest.approx(d0, rel=1e-9)


def test_orbit_moves_by_the_angle_it_was_asked_for():
    eye, target, up = (0.0, 0.0, 0.5), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
    f0 = np.array(target) - np.array(eye)
    f0 = f0 / np.linalg.norm(f0)
    for p in V.orbit(eye, target, up, degrees=(4.0,), axis="yaw"):
        f1 = np.array(p["target"]) - np.array(p["eye"])
        f1 = f1 / np.linalg.norm(f1)
        ang = math.degrees(math.acos(float(np.clip(f0 @ f1, -1, 1))))
        assert ang == pytest.approx(4.0, abs=1e-6)


def test_orbit_both_gives_a_yaw_and_a_pitch_per_angle():
    ps = V.orbit((0, 0, 1), (0, 0, 0), (0, 1, 0), degrees=(-3.0, 3.0), axis="both")
    assert len(ps) == 4
    assert {p["axis"] for p in ps} == {"yaw", "pitch"}
    assert {p["label"] for p in ps} == {"yaw-3.0", "yaw+3.0", "pitch-3.0", "pitch+3.0"}


def test_parallax_is_near_zero_for_a_flat_card_and_large_for_a_moved_scene():
    rng = np.random.default_rng(3)
    card = (rng.random((80, 80)) * 255)
    # a projected/flat reproduction barely changes when the camera moves
    flat = card + rng.normal(0, 0.4, card.shape)
    # a modelled one shifts and re-occludes
    moved = np.roll(card, 5, axis=1)
    p_flat = V.parallax(card, flat)
    p_moved = V.parallax(card, moved)
    assert p_flat["mean_abs"] < 1.0
    assert p_moved["mean_abs"] > 20.0
    assert p_flat["moved_frac"] < 0.02
    assert p_moved["moved_frac"] > 0.5


def test_flatness_warning_names_the_views_that_did_not_move():
    views = [("yaw-3.0", {"mean_abs": 0.4, "p95": 1.0, "moved_frac": 0.0}),
             ("yaw+3.0", {"mean_abs": 18.0, "p95": 60.0, "moved_frac": 0.7})]
    w = V.flatness_warning(views, threshold=2.0)
    assert w["suspect"] == ["yaw-3.0"]
    assert w["worst"] == "yaw-3.0"
    assert "flat" in w["note"]

    ok = V.flatness_warning([("a", {"mean_abs": 9.0, "p95": 1, "moved_frac": 0.4})])
    assert ok["suspect"] == []
    assert ok["note"] == "all views moved"
