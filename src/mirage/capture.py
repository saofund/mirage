"""mirage.capture — film the engine modelling, headlessly, in its own real viewport.

The op-log *is* the model, so the truest "making-of" is the shipping GUI replaying a
growing op-log: hand :func:`record_build` a sequence of stages (the model after each
build step) and it drives ``mirage_viewer`` headlessly, one screenshot per stage, into
an ``.mp4`` (crisp, for video) and a ``.gif`` (inline, for README/chat). Every frame is
a real screenshot of the native editor — the tool building the model, not a mock-up.

It is cheap by construction:

* **Fixed camera during the build** → every held frame of a stage is identical, so a
  stage is rendered *once* and reused across its hold; the ``.gif`` then dedupes the
  static tool panel / floor down to almost nothing.
* **A render cache keyed on (stage, camera)** → the dwell at the end costs no renders,
  and the optional closing *reveal* (a gentle camera swing to show the finished model
  off) is symmetric, so its mirror-image poses collapse to the same render.

Minimal use (see ``docs/gallery/render_viewer_build.py`` for the real caller)::

    from mirage.capture import record_build
    stages = [assemble(steps, i) for i in range(1, len(steps) + 1)]
    record_build(stages, "airplane_assembly", captions=[...], reveal=1.3)

A *stage* may be anything the engine can turn into an op-log: a :class:`MeshProgram`,
an object exposing ``.v`` / ``.f`` / ``.m`` (verts / faces / per-face materials), or a
``(verts, faces, materials)`` tuple. Needs ``mirage_viewer`` built
(``-DMIRAGE_BUILD_VIEWER=ON``) and ``ffmpeg`` on ``PATH``.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    # src/mirage/capture.py -> <repo>
    return Path(__file__).resolve().parents[2]


def default_viewer() -> Path:
    """Where the native viewer lands in a standard Release build."""
    import os
    env = os.environ.get("MIRAGE_VIEWER")
    if env:
        return Path(env)
    return _repo_root() / "core" / "build" / "Release" / "mirage_viewer.exe"


# --------------------------------------------------------------------------- #
# Turning an arbitrary "stage" into an op-log the viewer can load, and reading
# its geometry back for framing.
# --------------------------------------------------------------------------- #
def _stage_oplog_json(stage) -> str:
    from .meshlang import MeshProgram
    if isinstance(stage, MeshProgram):
        return stage.to_json()
    prog = MeshProgram()
    if hasattr(stage, "v") and hasattr(stage, "f"):
        prog.mesh(stage.v, stage.f, face_materials=getattr(stage, "m", None))
    else:
        v, f, *rest = stage
        prog.mesh(v, f, face_materials=(rest[0] if rest else None))
    return prog.to_json()


def _stage_verts(stage):
    if hasattr(stage, "v"):
        return list(stage.v)
    from .meshlang import MeshProgram
    if isinstance(stage, MeshProgram):
        return [tuple(vert.co) for vert in stage.build().verts]
    return list(stage[0])


# --------------------------------------------------------------------------- #
# Caption overlay (a subtle step index + operator name, bottom-left, clear of
# the viewer's tool panel).
# --------------------------------------------------------------------------- #
def _fonts():
    from PIL import ImageFont
    def pick(size):
        for name in ("segoeui.ttf", "seguisb.ttf", "arial.ttf"):
            p = Path("C:/Windows/Fonts") / name
            if p.exists():
                try:
                    return ImageFont.truetype(str(p), size)
                except Exception:
                    pass
        return ImageFont.load_default()
    return pick(23), pick(20)   # (label, index)


def _caption(img, label, idx, pos, fonts):
    from PIL import ImageDraw
    f_label, f_index = fonts
    x, y = pos
    d = ImageDraw.Draw(img, "RGBA")
    # teal step index, then a light operator label; a soft shadow keeps both
    # legible over any tone the studio backdrop happens to be behind them.
    for dx, dy in ((1, 1), (2, 2)):
        d.text((x + dx, y + dy), idx, font=f_index, fill=(0, 0, 0, 150))
    d.text((x, y), idx, font=f_index, fill=(70, 200, 214, 255))
    ix = x + int(d.textlength(idx, font=f_index)) + 14
    for dx, dy in ((1, 1), (2, 2)):
        d.text((ix + dx, y + dy + 1), label, font=f_label, fill=(0, 0, 0, 150))
    d.text((ix, y + 1), label, font=f_label, fill=(232, 236, 240, 255))


def _encode(ffmpeg, frames_dir: Path, out_dir: Path, base: str, fps: int,
            gif_w: int, gif_fps: int, tmp: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / f"{base}.mp4"
    gif = out_dir / f"{base}.gif"
    pat = str(frames_dir / "f%04d.png")
    # the .mp4 keeps full frame rate — H.264 swallows the closing motion for almost
    # nothing; the .gif can drop to gif_fps so a moving tail doesn't bloat it (gif has
    # no interframe motion coding, so every changed pixel of every kept frame costs).
    gscale = f"scale={gif_w}:-1:flags=lanczos"
    if gif_fps and gif_fps != fps:
        gscale = f"fps={gif_fps}," + gscale
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True)
    # two-pass palette: a diff-optimised palette, then a light bayer dither so the
    # gradients in the studio backdrop don't band, at rectangle diff for small files.
    pal = tmp / "pal.png"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat,
                    "-vf", f"{gscale},palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat,
                    "-i", str(pal), "-lavfi",
                    f"{gscale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                    str(gif)], check=True)
    return mp4, gif


# --------------------------------------------------------------------------- #
# The one entry point.
# --------------------------------------------------------------------------- #
def record_build(stages, out_base, *, out_dir=None, captions=None,
                 view=(2.30, 0.36, 9.0), size=(1280, 720), fps=24,
                 per=13, hold=24, reveal=0.0, reveal_sweep=0.5, gif_w=760,
                 gif_fps=None, target=None, floorz=None, viewer=None, tmp=None,
                 caption_pos=None, quiet=False):
    """Film ``stages`` assembling in the real viewer; write ``<out_base>.mp4`` + ``.gif``.

    stages       ordered models (MeshProgram | .v/.f/.m object | (v,f,m) tuple); each is
                 the model as it stood after one more build step.
    captions     optional list aligned with ``stages`` — the operator name shown as each
                 lands (a ``k/N`` index is prepended). Falsy → no captions.
    view         the fixed ``(yaw, pitch, dist)`` the build is shot from.
    per / hold   frames each stage holds during the build / frames to dwell at the end.
    reveal       seconds of closing camera swing on the finished model (0 → none); the
                 swing eases out to ``reveal_sweep`` radians of yaw and back to ``view``,
                 so the clip's tail shows the model off without a full turntable.
    target/floorz  orbit centre / studio floor Z; default to the *final* model's bbox so
                 neither framing nor ground drifts as parts appear. Pass to override.
    gif_w/gif_fps  the .gif's width and (optionally lower) frame rate; the .mp4 always
                 keeps ``size`` / ``fps``. Drop ``gif_fps`` when a moving reveal would
                 otherwise inflate the .gif.

    Returns ``(mp4_path, gif_path)``. Renders are cached on (stage, camera), so a build
    of N stages held H frames each with an R-frame reveal costs ~``N + R/2`` viewer runs,
    not ``N*H + R``.
    """
    from PIL import Image

    stages = list(stages)
    NP = len(stages)
    if NP == 0:
        raise ValueError("record_build: need at least one stage")

    viewer = Path(viewer) if viewer else default_viewer()
    if not viewer.exists():
        raise FileNotFoundError(
            f"mirage_viewer not built (-DMIRAGE_BUILD_VIEWER=ON): {viewer}")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    out_dir = Path(out_dir) if out_dir else (_repo_root() / "docs" / "gallery")
    tmp = Path(tmp) if tmp else (out_dir / "_build")
    W, H = size
    yaw0, pitch0, dist0 = view

    # framing: pin the orbit centre + floor to the FINAL model so nothing jumps
    if target is None or floorz is None:
        vs = _stage_verts(stages[-1])
        xs = [p[0] for p in vs]; ys = [p[1] for p in vs]; zs = [p[2] for p in vs]
        if target is None:
            target = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2)
        if floorz is None:
            floorz = min(zs) - 0.01
    tx, ty, tz = target

    cache_dir = tmp / "cache"
    frames_dir = tmp / "frames"
    oplog_dir = tmp / "oplog"
    for d in (cache_dir, frames_dir, oplog_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    fonts = _fonts()
    cpos = caption_pos or (408, H - 56)   # clears the ~392px tool panel at any width

    # one op-log per stage, written once
    _oplogs = {}
    def oplog(i):
        if i not in _oplogs:
            p = oplog_dir / f"stage_{i}.json"
            p.write_text(_stage_oplog_json(stages[i]))
            _oplogs[i] = p
        return _oplogs[i]

    # render cache: (stage, cam-signature) -> captioned PNG. Identical held frames and
    # symmetric reveal poses hit the cache instead of re-running the viewer.
    _cache = {}
    def base_png(stage, cam):
        key = (stage, "%.4f_%.4f_%.4f" % cam)
        if key in _cache:
            return _cache[key]
        ppm = tmp / "_cur.ppm"
        if ppm.exists():
            ppm.unlink()
        r = subprocess.run(
            [str(viewer), "--oplog", str(oplog(stage)), "--winsize", str(W), str(H),
             "--cam", "%.5f" % cam[0], "%.5f" % cam[1], "%.5f" % cam[2],
             "--target", "%.5f" % tx, "%.5f" % ty, "%.5f" % tz,
             "--floorz", "%.5f" % floorz, "--nohighlight", "--screenshot", str(ppm)],
            capture_output=True, text=True,
        )
        if not ppm.exists():
            raise RuntimeError(
                f"viewer produced no frame (stage {stage}, rc={r.returncode}):\n{r.stderr[-400:]}")
        img = Image.open(ppm).convert("RGB")
        if captions:
            _caption(img, captions[min(stage, len(captions) - 1)], f"{stage + 1}/{NP}", cpos, fonts)
        out = cache_dir / f"r{len(_cache):04d}.png"
        img.save(out)
        _cache[key] = out
        return out

    # the frame plan: build (fixed cam, stage advances) -> reveal swing -> end dwell
    plan = []                                        # (stage, cam) per frame
    for f in range(per * NP):
        plan.append((min(NP - 1, f // per), (yaw0, pitch0, dist0)))
    R = round(reveal * fps)
    for i in range(R):                               # a single eased swing out and back
        yaw = yaw0 + reveal_sweep * math.sin(math.pi * i / R)
        plan.append((NP - 1, (yaw, pitch0, dist0)))
    for _ in range(hold):
        plan.append((NP - 1, (yaw0, pitch0, dist0)))

    for n, (stage, cam) in enumerate(plan):
        shutil.copyfile(base_png(stage, cam), frames_dir / f"f{n:04d}.png")

    mp4, gif = _encode(ffmpeg, frames_dir, out_dir, out_base, fps, gif_w, gif_fps or fps, tmp)
    if not quiet:
        print(f"[capture] {out_base}: {len(plan)} frames, {len(_cache)} renders "
              f"({W}x{H}@{fps}fps)  ->  {mp4.name} {mp4.stat().st_size // 1024} KB, "
              f"{gif.name} {gif.stat().st_size // 1024} KB")
    return mp4, gif
