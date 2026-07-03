"""GUI regression: the native viewport's orbit control, tested without a human.

The viewport's camera controls live in the GUI shell (core/viewer/viewer.cpp),
which the differential op-log tests never touch — so a flipped sign in the
mouse->camera mapping (exactly the "left-drag is inverted" class of bug) would
sail through every other test. This closes that hole.

`mirage_viewer.exe --drag` synthesises a mouse drag through the REAL input
handlers (on_mouse press -> on_cursor moves -> release) in a hidden GL window and
prints the resulting yaw/pitch delta. We assert the SIGN of that mapping, so the
orbit direction is pinned: drag right must turn the model one specific way, and
any future edit that flips it fails here instead of only being felt at the mouse.

Skipped when the viewer isn't built or a headless GL context can't be created
(e.g. CI without a GPU) — the assertion only runs when the real binary runs.
"""
import os
import re
import subprocess

import pytest

_VIEWER = os.path.join(
    os.path.dirname(__file__), "..", "core", "build", "Release", "mirage_viewer.exe"
)

_DRAG_RE = re.compile(
    r"drag\s+\S+\s+dx=(-?\d+).*?yaw\s+[-\d.]+\s*->\s*[-\d.]+\s*\(d=([+-][\d.]+)\)"
)


def _orbit(dx, dy, tmp_path):
    """Run the viewer headless with a synthetic left-drag; return (yaw_delta, pitch_delta)."""
    if not os.path.isfile(_VIEWER):
        pytest.skip("mirage_viewer.exe not built")
    shot = os.path.join(str(tmp_path), "orbit.ppm")
    try:
        out = subprocess.run(
            [_VIEWER, "--cam", "0.7", "0.5", "4",
             "--drag", "L", str(dx), str(dy), "--screenshot", shot],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"viewer could not run headless: {e}")
    if out.returncode != 0 or "drag" not in out.stdout:
        pytest.skip(f"viewer produced no drag readout (headless GL unavailable?): {out.stderr.strip()[:200]}")
    m = _DRAG_RE.search(out.stdout)
    assert m, f"could not parse drag readout from:\n{out.stdout}"
    # yaw delta is the number after (d=...); pitch we re-extract from the second (d=...)
    yaw_d = float(m.group(2))
    pitch_m = re.findall(r"\(d=([+-][\d.]+)\)", out.stdout)
    pitch_d = float(pitch_m[1]) if len(pitch_m) > 1 else 0.0
    return yaw_d, pitch_d


def test_drag_right_turns_model_negative_yaw(tmp_path):
    # Dragging the cursor to the RIGHT (dx>0) must DECREASE yaw. This is the sign
    # the user confirmed as correct; the old code had it inverted (+=).
    yaw_d, _ = _orbit(150, 0, tmp_path)
    assert yaw_d < 0, f"drag-right should decrease yaw, got d={yaw_d:+.3f}"


def test_drag_left_turns_model_positive_yaw(tmp_path):
    yaw_d, _ = _orbit(-150, 0, tmp_path)
    assert yaw_d > 0, f"drag-left should increase yaw, got d={yaw_d:+.3f}"


def test_horizontal_drag_is_symmetric(tmp_path):
    # Equal-and-opposite horizontal drags must produce equal-and-opposite yaw, and
    # a pure horizontal drag must not tilt (pitch unchanged).
    right_yaw, right_pitch = _orbit(150, 0, tmp_path)
    left_yaw, _ = _orbit(-150, 0, tmp_path)
    assert right_pitch == pytest.approx(0.0, abs=1e-6), "horizontal drag must not change pitch"
    assert right_yaw == pytest.approx(-left_yaw, abs=1e-4), "opposite drags must cancel"


def test_drag_down_changes_pitch(tmp_path):
    # A vertical drag must move pitch (and, being pure-vertical, leave yaw alone).
    yaw_d, pitch_d = _orbit(0, 150, tmp_path)
    assert yaw_d == pytest.approx(0.0, abs=1e-6), "vertical drag must not change yaw"
    assert pitch_d != 0.0, "vertical drag should change pitch"
