"""Record the airliner being MODELLED — the making-of, in the real viewport.

A thin caller over :mod:`mirage.capture`: assemble the jet's op-log one build step at a
time and hand the growing stages to ``record_build``, which films each in the real
``mirage_viewer`` GUI. Fixed 3/4 view while it assembles (a build, not a showcase), then
a gentle closing swing shows the finished jet off. Every frame is a headless screenshot
of the shipping editor, so this is the tool building the model, not a mock-up.

    uv run python docs/gallery/render_viewer_build.py
    ANIM_QUICK=1 uv run python docs/gallery/render_viewer_build.py   # fast smoke test

Needs mirage_viewer.exe built (-DMIRAGE_BUILD_VIEWER=ON) and ffmpeg on PATH.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "src"))
from airplane import part_steps, assemble  # noqa: E402
from mirage.capture import record_build  # noqa: E402

QUICK = os.environ.get("ANIM_QUICK") == "1"

STEPS = part_steps()
NP = len(STEPS)
# stages[i] = the model exactly as it stood after i+1 operators
stages = [assemble(STEPS, i) for i in range(1, NP + 1)]
NVERTS = len(stages[-1].v)

# one caption per stage — the operator that lands it (the final step reads "complete")
CAPTIONS = [
    "fuselage  ·  surface of revolution",
    "wings  ·  lofted, mirrored",
    "winglets  ·  swept fins",
    "tailplane  ·  lofted, mirrored",
    "vertical fin",
    "engines  ·  revolved, mirrored",
    f"complete  ·  one op-log, {NVERTS} verts",
]

record_build(
    stages,
    "airplane_assembly",
    captions=CAPTIONS,
    view=(float(os.environ.get("ANIM_YAW", 2.30)),
          float(os.environ.get("ANIM_PITCH", 0.36)),
          float(os.environ.get("ANIM_DIST", 9.0))),
    size=(854, 480) if QUICK else (1280, 720),
    fps=int(os.environ.get("ANIM_FPS", 24)),
    per=int(os.environ.get("ANIM_PER", 6 if QUICK else 13)),
    hold=int(os.environ.get("ANIM_HOLD", 10 if QUICK else 24)),
    reveal=float(os.environ.get("ANIM_REVEAL", 0.4 if QUICK else 1.0)),
    reveal_sweep=float(os.environ.get("ANIM_SWEEP", 0.32)),
    gif_w=int(os.environ.get("ANIM_GIF_W", 600)),
    gif_fps=int(os.environ.get("ANIM_GIF_FPS", 12)),   # mp4 stays 24; keep the gif lean
    tmp=Path(os.environ["ANIM_TMP"]) if os.environ.get("ANIM_TMP") else None,
)
