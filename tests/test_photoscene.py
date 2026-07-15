"""photo -> scene: the schema, the decomposition, and the measured/authored split.

The tests worth having here are the ones that pin the DISCIPLINE, not the plumbing: that a
guessed camera is recorded as guessed, that tracing a footprint really does recover a
placement, and that the per-object score ranks the worst object first — because that ranking
is the whole reason the decomposition exists.
"""
import json

import numpy as np
import pytest
from PIL import Image

from mirage.meshlang import MeshProgram
from mirage.photoscene import PhotoScene
from mirage.solve import Camera, project, place_from_footprint, solve_sun, estimate_sun_env_ratio


def _photo(tmp_path, w=320, h=180):
    p = tmp_path / "src.png"
    Image.fromarray(np.full((h, w, 3), 120, np.uint8)).save(p)
    return p


def _scene(tmp_path):
    s = PhotoScene.new(tmp_path / "sc", _photo(tmp_path))
    # a 4 x 6 m rectangle on the ground, seen as some quad
    s.set_ground([[60, 150], [260, 150], [230, 90], [90, 90]],
                 [[0, 0], [4, 0], [4, 6], [0, 6]], note="a test slab")
    return s


# --- schema / round-trip ---------------------------------------------------- #
def test_new_scene_round_trips(tmp_path):
    s = PhotoScene.new(tmp_path / "sc", _photo(tmp_path))
    s.set_camera_manual((0, -8, 3), (0, 0, 0.5), 0.9, why="test")
    s.add_object("cube", MeshProgram().cube(size=1.0), at=(1, 2, 0.5))
    s.save()

    t = PhotoScene.load(tmp_path / "sc")
    assert t.data["schema"] == 1
    assert [o["name"] for o in t.data["objects"]] == ["cube"]
    assert t.camera.fov_y == pytest.approx(0.9)
    assert (tmp_path / "sc" / "objects" / "cube" / "oplog.json").exists()


def test_each_object_is_its_own_directory(tmp_path):
    """The decomposition, concretely: refining one object touches one file. That is what
    makes 'this one deserves ten times the detail' a thing you can actually do."""
    s = _scene(tmp_path)
    s.add_object("a", MeshProgram().cube(size=1.0), detail=1)
    s.add_object("b", MeshProgram().uv_sphere(segments=8, rings=6), detail=5)
    assert (s.root / "objects" / "a" / "oplog.json").exists()
    assert (s.root / "objects" / "b" / "oplog.json").exists()
    before = (s.root / "objects" / "a" / "oplog.json").read_bytes()
    s.add_object("b", MeshProgram().uv_sphere(segments=64, rings=48), detail=9)   # refine b only
    assert (s.root / "objects" / "a" / "oplog.json").read_bytes() == before
    assert next(o for o in s.data["objects"] if o["name"] == "b")["detail"] == 9


def test_build_compiles_to_one_oplog(tmp_path):
    s = _scene(tmp_path)
    s.add_object("a", MeshProgram().cube(size=1.0), at=(0, 0, 0.5))
    s.add_object("b", MeshProgram().cube(size=1.0), at=(3, 0, 0.5))
    m = s.build().build()
    assert m.is_closed_manifold()
    assert len(m.faces) == 12          # two cubes, disjoint-unioned by `place`
    xs = [v.co[0] for v in m.verts]
    assert min(xs) < 0 and max(xs) > 2.5


def test_missing_oplog_is_skipped_not_fatal(tmp_path):
    s = _scene(tmp_path)
    s.add_object("real", MeshProgram().cube(size=1.0))
    s.data["objects"].append({"name": "todo", "oplog": "objects/todo/oplog.json",
                              "at": [0, 0, 0], "rotate": [0, 0, 0]})
    assert len(s.build().build().faces) == 6      # a not-yet-modelled object doesn't break the scene


# --- measured vs authored, kept apart --------------------------------------- #
def test_a_guessed_camera_says_it_is_guessed(tmp_path):
    """The discipline: a guess must never read as a measurement in the file."""
    s = _scene(tmp_path)
    s.set_camera_manual((0, -8, 3), (0, 0, 0.5), 0.9, why="no correspondences yet")
    assert s.data["camera"]["solved"] is False
    assert "why" in s.data["camera"]
    assert "solve" not in s.data["camera"]


def test_a_solved_camera_carries_its_residual(tmp_path):
    s = PhotoScene.new(tmp_path / "sc", _photo(tmp_path, 640, 360))
    truth = Camera(eye=(1.0, -7.0, 3.0), target=(0.5, 1.0, 0.4), fov_y=0.95)
    rng = np.random.default_rng(0)
    world = np.column_stack([rng.uniform(-3, 3, 14), rng.uniform(0, 8, 14), rng.uniform(0, 2, 14)])
    img = project(truth, world, 640, 360)
    ok = ~np.isnan(img).any(1)
    cam, info = s.set_camera(img[ok], world[ok], fit_lens=False)
    assert s.data["camera"]["solved"] is True
    assert info["rms_px"] < 0.5
    assert s.data["camera"]["solve"]["rms_px"] == info["rms_px"]
    assert np.allclose(cam.eye, truth.eye, atol=0.05)
    # the correspondences are kept, so the solve can be re-run and audited later
    assert len(s.data["camera"]["solve"]["correspondences"]["img"]) == int(ok.sum())


def test_ground_records_what_calibrated_it(tmp_path):
    s = _scene(tmp_path)
    g = s.data["ground"]
    assert g["solved"] is True
    assert g["calibrated_by"]["note"] == "a test slab"
    assert g["calibrated_by"]["fit_residual_m"] < 1e-9


# --- placing by measurement ------------------------------------------------- #
def test_place_from_footprint_recovers_position_and_yaw():
    # a 2 x 1 m rectangle, rotated 30 deg, centred at (5, 3) — seen through a homography
    yaw = 30.0
    c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    R = np.array([[c, -s_], [s_, c]])
    local = np.array([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]])
    world = local @ R.T + [5, 3]
    H = np.eye(3)                                  # image == world, so the answer is exact
    p = place_from_footprint(world, H)
    assert p["at"][:2] == pytest.approx([5, 3], abs=1e-9)
    assert p["rotate"][2] == pytest.approx(yaw, abs=1e-6)
    assert p["size"] == pytest.approx([2.0, 1.0], abs=1e-9)
    assert p["squareness"] < 1e-9


def test_place_object_writes_the_trace_it_used(tmp_path):
    s = _scene(tmp_path)
    s.place_object("van", [[100, 140], [200, 140], [190, 110], [110, 110]],
                   program=MeshProgram().cube(size=1.0), height=2.0)
    e = next(o for o in s.data["objects"] if o["name"] == "van")
    assert "placed_from_footprint" in e            # auditable: the trace is kept
    assert e["at"][2] == pytest.approx(1.0)        # height/2 -> it stands ON the ground
    assert e["placed_from_footprint"]["size_m"][2] == 2.0


def test_footprint_needs_four_corners():
    with pytest.raises(ValueError):
        place_from_footprint([[0, 0], [1, 0], [1, 1]], np.eye(3))


# --- lighting, solved ------------------------------------------------------- #
def test_solve_sun_from_a_shadow():
    # A 3 m pole at world (0,0) throwing its shadow tip to (4,0). The shadow falls AWAY from
    # the sun, so the sun is at -x: azimuth 180, elevation atan(3/4) = 36.87, and the
    # direction (pointing toward it) is (-0.8, 0, 0.6). Getting this backwards puts the key
    # light on the wrong side of the scene and every shadow in the render points the wrong way.
    H = np.eye(3)
    r = solve_sun(base_px=[0, 0], shadow_tip_px=[4, 0], height=3.0, H=H)
    assert r["elevation_deg"] == pytest.approx(36.8699, abs=1e-3)
    assert r["azimuth_deg"] == pytest.approx(180.0, abs=1e-6)
    assert r["sun_dir"] == pytest.approx([-0.8, 0.0, 0.6], abs=1e-6)
    assert r["shadow_len_m"] == pytest.approx(4.0)


def test_sun_is_opposite_the_shadow():
    # the sign convention, stated once: sun_dir . (shadow direction) must be NEGATIVE
    H = np.eye(3)
    for tip in ([4, 0], [-3, 2], [0, -5], [2.5, 2.5]):
        r = solve_sun([0, 0], tip, 2.0, H)
        shadow = np.array([tip[0], tip[1], 0.0])
        assert np.dot(r["sun_dir"], shadow) < 0, f"sun on the same side as the shadow for {tip}"


def test_sun_azimuth_does_not_depend_on_the_height_guess():
    """The good failure mode: a wrong height tilts the sun but cannot swing it around the
    compass, because the azimuth comes from the ground vector alone."""
    H = np.eye(3)
    a = solve_sun([1, 1], [4, 5], 2.0, H)
    b = solve_sun([1, 1], [4, 5], 6.0, H)          # 3x the height
    assert a["azimuth_deg"] == pytest.approx(b["azimuth_deg"], abs=1e-9)
    assert b["elevation_deg"] > a["elevation_deg"]


def test_solve_sun_rejects_a_zero_length_shadow():
    with pytest.raises(ValueError):
        solve_sun([2, 2], [2, 2], 3.0, np.eye(3))


def test_sun_env_ratio_tracks_contrast():
    dark = estimate_sun_env_ratio(lit_luma=0.22, shadow_luma=0.20, n_dot_l=0.7)
    hard = estimate_sun_env_ratio(lit_luma=0.90, shadow_luma=0.12, n_dot_l=0.7)
    assert hard["sun_over_env"] > dark["sun_over_env"] * 5
    assert dark["contrast"] < hard["contrast"]


def test_overcast_reports_no_key_rather_than_a_silly_number():
    r = estimate_sun_env_ratio(lit_luma=0.19, shadow_luma=0.19, n_dot_l=0.7)
    assert r["sun_over_env"] == 0.0
    assert "overcast" in r["note"]


# --- the score, and the work order ------------------------------------------ #
def test_score_ranks_the_worst_object_first(tmp_path):
    """The reason the decomposition exists. 'It's all a bit coarse' is not actionable;
    'dispenser 0.31, wall 0.04' is a work order."""
    s = PhotoScene.new(tmp_path / "sc", _photo(tmp_path, 200, 100))
    ref = np.zeros((100, 200, 3), np.uint8)
    ref[:, :100] = (200, 60, 40)          # left half: orange
    ref[:, 100:] = (120, 120, 120)        # right half: grey
    Image.fromarray(ref).save(s.reference)

    ren = ref.copy()
    ren[:, :100] = (40, 200, 60)          # left badly wrong, right exact
    Image.fromarray(ren).save(s.root / "render.png")

    s.add_object("left", crop=(0, 0, 0.5, 1.0))
    s.add_object("right", crop=(0.5, 0, 1.0, 1.0))
    m = s.score()
    assert [r["object"] for r in m["worst_first"]] == ["left", "right"]
    assert m["worst_first"][0]["err"] > m["worst_first"][1]["err"] * 10
    assert (s.root / "score.json").exists()


def test_report_says_solved_or_guessed(tmp_path):
    s = PhotoScene.new(tmp_path / "sc", _photo(tmp_path, 200, 100))
    Image.open(s.reference).save(s.root / "render.png")
    s.set_camera_manual((0, -8, 3), (0, 0, 0), 0.9, why="eyeballed")
    s.add_object("thing", crop=(0, 0, 0.5, 1))
    txt = s.report(s.score())
    assert "GUESSED" in txt and "eyeballed" in txt
    assert "work order" in txt
