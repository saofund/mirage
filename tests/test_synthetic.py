import json

import pytest

pytest.importorskip("mujoco")
from mirage.synthetic import generate_dataset, random_scene
import numpy as np


def test_random_scene_is_valid():
    s = random_scene(np.random.default_rng(1))
    assert "ground" in s.entity_names()
    assert len(s.entity_names()) >= 4  # ground + >=3 objects


def test_generate_dataset(tmp_path):
    summary = generate_dataset(tmp_path, n=2, seed=0, width=160, height=120)
    assert summary["samples"] == 2
    assert (tmp_path / "annotations.json").exists()
    assert (tmp_path / "rgb_00.png").exists()
    coco = json.loads((tmp_path / "annotations.json").read_text())
    assert len(coco["images"]) == 2
    assert len(coco["categories"]) == 3
    assert summary["boxes"] >= 1  # at least some objects detected & boxed
    for a in coco["annotations"]:
        assert len(a["bbox"]) == 4
        assert a["category"] in ("box", "sphere", "cylinder")
