import json

import pytest

from mirage import Session
from mirage.viewport import WebViewport


def test_web_viewport_writes_artifacts(tmp_path):
    s = Session(name="vp")
    s.add_plane("g")
    s.add_box("b", position=[0, 0, 1], color=[1, 0, 0])
    index = WebViewport(s.scene, frames=[{"b": [0, 0, 0.5]}]).write(tmp_path)
    assert index.exists()
    html = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "three" in html.lower() and "scene.json" in html
    scn = json.loads((tmp_path / "scene.json").read_text())
    assert "b" in scn["entities"]
    frames = json.loads((tmp_path / "frames.json").read_text())
    assert frames and "b" in frames[0]


def test_trajectory_from_sim(tmp_path):
    pytest.importorskip("mujoco")
    from mirage.mujoco_backend import MujocoSim
    from mirage.viewport import trajectory_from_sim
    s = Session(name="t")
    s.add_plane("g")
    s.add_box("b", position=[0, 0, 1.5])
    frames = trajectory_from_sim(MujocoSim.from_scene(s.scene), s.scene, steps=10, dt=0.02)
    assert len(frames) == 10 and "b" in frames[0]
    assert frames[-1]["b"][2] < 1.5  # fell over the rollout
