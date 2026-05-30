"""Shared helpers for the demo cases — thin wrappers over ``mirage.imaging``."""
from pathlib import Path

from mirage.imaging import (  # noqa: F401  (re-exported for the cases)
    save_png, save_gif, colorize_depth, colorize_seg, seg_ids, bboxes_from_seg, overlay_boxes,
)

OUTPUTS = Path(__file__).parent / "outputs"


def outdir(name: str) -> Path:
    p = OUTPUTS / name
    p.mkdir(parents=True, exist_ok=True)
    return p
