import pytest

from mirage import Session


def test_authoring_returns_structured_dicts():
    s = Session(name="t")
    box = s.add_box("b", position=[0, 0, 1], size=[0.4, 0.4, 0.4], color=[1, 0, 0])
    assert box["name"] == "b"
    assert box["transform"]["position"] == [0, 0, 1]
    assert box["geometry"]["params"]["size"] == [0.4, 0.4, 0.4]
    assert box["material"]["base_color"] == [1.0, 0.0, 0.0, 1.0]
    assert s.add_camera("cam")["name"] == "cam"
    assert s.list()["entities"] == ["b"]


def test_command_log_records_only_mutations():
    s = Session()
    s.add_plane("g")
    s.add_box("b", position=[0, 0, 1])
    s.get("b")          # read — not logged
    s.list()            # read — not logged
    assert [e["op"] for e in s.get_log()] == ["add_plane", "add_box"]


def test_replay_reproduces_authoring():
    s = Session()
    s.add_plane("g")
    s.add_box("b", position=[1, 2, 3], color=[0, 1, 0])
    s.set_material("b", roughness=0.2)
    assert s.replay().scene.diff(s.scene) == {}


def test_set_scene_roundtrip_and_diff():
    s = Session(name="r")
    s.add_box("b", position=[0, 0, 2])
    snap = s.get_scene()
    s.reset("r")
    assert s.list()["entities"] == []
    s.set_scene(snap)
    assert s.diff(snap) == {}


def test_render_returns_rgb_array():
    pytest.importorskip("mujoco")
    s = Session()
    s.add_plane("g")
    s.add_box("b", position=[0, 0, 0.3])
    out = s.render(160, 120, modalities=("rgb",))
    assert out["data"]["rgb"].shape == (120, 160, 3)
    assert out["data"]["rgb"].mean() > 1  # not all black


def test_step_reports_engine_and_contacts():
    pytest.importorskip("mujoco")
    s = Session()
    s.add_plane("g")
    s.add_box("b", position=[0, 0, 0.3])
    res = s.step(dt=0.05, steps=20)
    assert res["engine"] == "mujoco"
    assert res["ncontact"] >= 1  # resting on the ground


def test_replay_with_step_is_deterministic():
    pytest.importorskip("mujoco")
    s = Session()
    s.add_plane("g")
    s.add_box("b", position=[0, 0, 1.5])
    s.step(dt=0.05, steps=20)
    z1 = s.scene.get_position("b")[2]
    z2 = s.replay().scene.get_position("b")[2]
    assert z1 < 1.5                       # it fell
    assert z2 == pytest.approx(z1, abs=1e-6)  # replay is bit-reproducible


def test_studio_render():
    pytest.importorskip("mujoco")
    s = Session(name="q")
    s.add_plane("g", size=[4, 4])
    s.add_box("b", position=[0, 0, 0.3], color=[0.8, 0.2, 0.2])
    s.set_material("b", metallic=0.8, roughness=0.2)
    img = s.render_studio(width=160, height=120)
    assert img.shape == (120, 160, 3) and img.mean() > 1
