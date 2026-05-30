"""Synthetic-data generation (Replicator-style) — a first-class Mirage API.

Procedurally randomizes object classes/poses/colors, lighting and camera, renders
**RGB + segmentation** through MuJoCo, derives **2D bounding boxes** from the
segmentation, and writes a COCO-style annotation file. This is the robotics /
embodied-AI payoff of the stack, exposed for reuse (the ``05_synthetic_data``
demo is a thin wrapper over ``generate_dataset``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scene import Scene, Entity, Transform, Geometry, Material, PhysicsBody, Light

DEFAULT_CLASSES = ("box", "sphere", "cylinder")


def _geometry(cls: str, rng) -> Geometry:
    if cls == "box":
        return Geometry("box", {"size": [float(rng.uniform(0.2, 0.4))] * 3})
    if cls == "sphere":
        return Geometry("sphere", {"radius": float(rng.uniform(0.12, 0.22))})
    if cls == "cylinder":
        return Geometry("cylinder", {"radius": float(rng.uniform(0.1, 0.16)), "height": float(rng.uniform(0.3, 0.5))})
    raise ValueError(f"unknown class {cls!r}")


def random_scene(rng, classes=DEFAULT_CLASSES, n_range=(3, 6), region=0.6) -> Scene:
    """A randomized scene: a ground plane + N randomly-placed, colored primitives."""
    s = Scene(name="dr")
    s.add(Entity(name="ground", geometry=Geometry("plane", {"size": [4, 4]}),
                 physics=PhysicsBody(kind="static")))
    for i in range(int(rng.integers(*n_range))):
        cls = classes[int(rng.integers(0, len(classes)))]
        s.add(Entity(
            name=f"{cls}_{i}",
            transform=Transform(position=[float(rng.uniform(-region, region)),
                                          float(rng.uniform(-region, region)),
                                          float(rng.uniform(0.5, 0.9))]),
            geometry=_geometry(cls, rng),
            material=Material(base_color=[*rng.uniform(0.15, 0.95, 3).tolist(), 1.0]),
            physics=PhysicsBody(kind="dynamic"),
        ))
    s.add(Light(name="sun", kind="sun",
                color=[float(rng.uniform(0.7, 1.0)) for _ in range(3)],
                intensity=float(rng.uniform(0.6, 1.2))))
    return s


def _label_for(model, gid: int, classes):
    if gid < 0 or gid >= model.ngeom:
        return None
    name = model.geom(int(gid)).name or ""
    base = name[:-2] if name.endswith("_g") else name
    cls = base.split("_")[0]
    return cls if cls in classes else None


def generate_dataset(out_dir, n: int = 10, seed: int = 0, width: int = 640, height: int = 480,
                     settle: float = 0.4, classes=DEFAULT_CLASSES, save_overlays: bool = True) -> dict:
    """Generate ``n`` labeled samples into ``out_dir``. Returns a summary dict and
    writes ``rgb_*.png`` (+ ``boxes_*.png``) and a COCO ``annotations.json``."""
    from .mujoco_backend import MujocoSim
    from .imaging import save_png, seg_ids, overlay_boxes

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cats = sorted(classes)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c} for i, c in enumerate(cats)]}
    ann_id = 0

    for i in range(n):
        scene = random_scene(rng, classes=tuple(classes))
        sim = MujocoSim.from_scene(scene)
        sim.step_for(settle)
        view = dict(lookat=[0, 0, 0.2], distance=float(rng.uniform(2.3, 3.0)),
                    azimuth=float(rng.uniform(70, 160)), elevation=float(rng.uniform(-35, -15)))
        imgs = sim.render(width, height, modalities=("rgb", "segmentation"), **view)
        rgb, ids = imgs["rgb"], seg_ids(imgs["segmentation"])

        boxes = {}
        for gid in np.unique(ids):
            cls = _label_for(sim.model, int(gid), cats)
            if cls is None:
                continue
            ys, xs = np.where(ids == gid)
            if xs.size < 12:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            boxes[int(gid)] = (x0, y0, x1, y1)
            coco["annotations"].append({"id": ann_id, "image_id": i, "category_id": cats.index(cls),
                                        "category": cls, "bbox": [x0, y0, x1 - x0, y1 - y0]})
            ann_id += 1

        fn = f"rgb_{i:02d}.png"
        save_png(rgb, out / fn)
        if save_overlays:
            save_png(overlay_boxes(rgb, boxes), out / f"boxes_{i:02d}.png")
        coco["images"].append({"id": i, "file_name": fn, "width": width, "height": height})

    (out / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return {"samples": n, "boxes": ann_id, "classes": cats, "out_dir": str(out)}
