import pytest

pytest.importorskip("mcp")
import mirage.mcp_server as M


def test_tools_return_structured_json():
    M.reset_scene("t")
    assert M.add_plane("g")["name"] == "g"
    box = M.add_box("b", position=[0, 0, 1])
    assert box["geometry"]["kind"] == "box"
    assert M.list_objects()["entities"] == ["g", "b"]
    assert "entities" in M.get_scene()
    assert M.move("b", [0, 0, 2])["transform"]["position"] == [0, 0, 2]
    assert M.remove("b") == {"removed": "b"}


def test_command_log_and_replay_via_mcp():
    M.reset_scene("t")
    M.add_plane("g")
    M.add_box("b", position=[0, 0, 1])
    ops = [e["op"] for e in M.get_log()]
    assert ops[:3] == ["reset", "add_plane", "add_box"]
    assert M.replay_log()["entities"] == ["g", "b"]


def test_render_returns_image():
    pytest.importorskip("mujoco")
    M.reset_scene("r")
    M.add_plane("g")
    M.add_box("b", position=[0, 0, 0.3])
    M.step(dt=0.05, steps=10)
    img = M.render(160, 120)
    assert type(img).__name__ == "Image"  # FastMCP image content, not prose


def test_place_object_composes_scene_via_mcp():
    # place_object composes many DISTINCT objects into ONE legible op-log (not replace)
    M.new_model()
    assert M.place_object(program=[{"op": "cube", "size": 1.0}], at=[0, 0, 0],
                          material={"color": [1, 0, 0]})["ok"]
    r = M.place_object(program=[{"op": "cube", "size": 1.0}], at=[2, 0, 0],
                       material={"color": [0, 1, 0]})
    assert r["ok"]
    assert [o["op"] for o in M.get_mesh_program()] == ["place", "place"]
    assert r["stats"]["faces"] == 12          # 6 + 6 disjoint, not one replaced cube
    assert len(r["components"]) == 2          # two separate objects in one model
