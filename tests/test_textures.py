"""The procedural PBR map library — generation, and the cache that serves it.

The maps are gitignored and regenerated on demand, so the on-disk cache is load-bearing:
if it goes stale, a recipe edit has no visible effect and the only symptom is a render that
stubbornly looks wrong while the code reads right.
"""
import pytest

from mirage import textures as T


def test_library_generates_three_maps(tmp_path):
    p = T.generate("leather", tmp_path)
    assert set(p) == {"albedo", "rough", "normal"}
    for f in p.values():
        assert f.exists() and f.stat().st_size > 0
    # albedo/normal are P6 rgb, roughness is P5 gray — the reader in raytrace.cpp takes both
    assert p["albedo"].read_bytes()[:2] == b"P6"
    assert p["normal"].read_bytes()[:2] == b"P6"
    assert p["rough"].read_bytes()[:2] == b"P5"


def test_generation_is_deterministic(tmp_path):
    a = T.generate("wood_veneer", tmp_path / "a")["albedo"].read_bytes()
    b = T.generate("wood_veneer", tmp_path / "b")["albedo"].read_bytes()
    assert a == b, "seeded generation must be reproducible"


def test_maps_tile_seamlessly(tmp_path):
    # Sampled triplanar with no UVs, so every map repeats across a surface: if the first
    # column doesn't meet the last, the seam shows up as a grid over the whole model.
    import numpy as np
    from PIL import Image
    for name in ("leather", "wood_veneer"):
        a = np.asarray(Image.open(T.generate(name, tmp_path / name)["albedo"]), float)
        wrap = np.abs(a[:, 0] - a[:, -1]).mean()
        inner = np.abs(a[:, 0] - a[:, a.shape[1] // 2]).mean()
        assert wrap < inner, f"{name} does not tile: its edges differ more than unrelated columns"


def test_ensure_textures_caches(tmp_path):
    first = T.ensure_textures(["plaster"], tmp_path)["plaster"]
    mtimes = {k: v.stat().st_mtime_ns for k, v in first.items()}
    again = T.ensure_textures(["plaster"], tmp_path)["plaster"]
    assert {k: v.stat().st_mtime_ns for k, v in again.items()} == mtimes, "should not regenerate"


def test_ensure_textures_regenerates_when_the_recipe_changes(tmp_path):
    """The bug this guards: 'the file exists' was the whole cache key, so editing a recipe
    left the old map on disk and the render silently kept using it."""
    T.ensure_textures(["leather"], tmp_path)
    before = (tmp_path / "leather_albedo.ppm").read_bytes()
    stamp = (tmp_path / "leather.recipe").read_text()

    real = T._LIBRARY["leather"]
    try:
        T._LIBRARY["leather"] = lambda: T._leather(T.RES, 59, (0.9, 0.1, 0.1))  # a real edit
        assert T._recipe_id("leather") != stamp, "the digest must notice a constant changing"
        T.ensure_textures(["leather"], tmp_path)
        assert (tmp_path / "leather_albedo.ppm").read_bytes() != before, "served a stale map"
        assert (tmp_path / "leather.recipe").read_text() != stamp
    finally:
        T._LIBRARY["leather"] = real


def test_recipe_id_is_stable_across_calls():
    assert T._recipe_id("leather") == T._recipe_id("leather")
    assert T._recipe_id("leather") != T._recipe_id("wood_veneer")


def test_unknown_texture_raises():
    with pytest.raises(KeyError):
        T.generate("no_such_material")
