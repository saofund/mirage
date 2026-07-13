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


def _bin(name: str) -> Path:
    """Locate a built binary cross-platform: MSVC drops it in ``core/build/Release/`` with
    a ``.exe`` suffix; a single-config Linux/make build has no suffix (and may or may not
    use a ``Release/`` subdir). Returns the first that exists, else the Windows default."""
    base = _repo_root() / "core" / "build"
    for p in (base / "Release" / f"{name}.exe", base / "Release" / name,
              base / f"{name}.exe", base / name):
        if p.exists():
            return p
    return base / "Release" / f"{name}.exe"


def default_viewer() -> Path:
    """Where the native viewer lands (cross-platform; ``MIRAGE_VIEWER`` overrides)."""
    import os
    env = os.environ.get("MIRAGE_VIEWER")
    return Path(env) if env else _bin("mirage_viewer")


def default_render() -> Path:
    """Where the headless path tracer lands (cross-platform; ``MIRAGE_RENDER`` overrides)."""
    import os
    env = os.environ.get("MIRAGE_RENDER")
    return Path(env) if env else _bin("mirage_render")


def _orbit_eye(cam, tgt):
    """The world-space eye for an orbit pose — matches the viewer's ``orbit_eye`` exactly,
    so a path-traced frame frames identically to the viewer frame at the same (yaw, pitch,
    dist) about ``tgt``. Lets the tracer stand in for the viewport with the same camera."""
    yaw, pitch, dist = cam
    cx, cy, cz = tgt
    return (cx + dist * math.cos(pitch) * math.sin(yaw),
            cy - dist * math.cos(pitch) * math.cos(yaw),
            cz + dist * math.sin(pitch))


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


def _interp_keyframes(kfs, t):
    """A smooth camera pose at clip-time ``t`` in [0,1] from keyframes, smoothstep-eased
    between the two bracketing keys. Each key is ``(t, yaw, pitch, dist)`` or, to pull the
    aim as well, ``(t, yaw, pitch, dist, tx, ty, tz)``. Returns ``(yaw, pitch, dist,
    target)`` where ``target`` is a 3-tuple or ``None`` (meaning: use the fixed default).
    Drives a moving camera (a dolly / orbit / focus-pull) instead of one fixed viewpoint."""
    def pose(k):
        return (k[1], k[2], k[3], (k[4], k[5], k[6]) if len(k) >= 7 else None)
    if t <= kfs[0][0]:
        return pose(kfs[0])
    if t >= kfs[-1][0]:
        return pose(kfs[-1])
    for i in range(len(kfs) - 1):
        a, b = kfs[i], kfs[i + 1]
        if a[0] <= t <= b[0]:
            u = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            u = u * u * (3.0 - 2.0 * u)              # smoothstep ease
            y = a[1] + (b[1] - a[1]) * u
            p = a[2] + (b[2] - a[2]) * u
            d = a[3] + (b[3] - a[3]) * u
            tgt = None
            if len(a) >= 7 and len(b) >= 7:          # both keys carry an aim -> pull it too
                tgt = tuple(a[4 + k] + (b[4 + k] - a[4 + k]) * u for k in range(3))
            return (y, p, d, tgt)
    return pose(kfs[-1])


def _flat_tail_t(kfs):
    """The clip-time ``t`` at which the camera stops moving — the start of the final
    equal-pose (yaw/pitch/dist/aim) segment, or 1.0 if there is none. Lets the caller
    path-trace ONLY the static held frames (one render) and leave the motion to the fast
    real-time renderer (where per-frame path-tracing would be slow and would shimmer)."""
    last = tuple(kfs[-1][1:])
    for i in range(len(kfs) - 2, -1, -1):
        if tuple(kfs[i][1:]) != last:
            return kfs[i + 1][0]
    return kfs[0][0]


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
                 caption_pos=None, automode=False, keyframes=None, smooth=False,
                 renderer="viewer", trace_hold=False, trace_spp=96, trace_threads=None,
                 trace_knobs=None, trace_denoise=0, cam_fov=0.9, render=None, quiet=False):
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
    automode     film in the viewer's AI "AUTO" mode — the tool panel is hidden and a
                 top-left status HUD names what's being built (each stage's caption),
                 so the frame is all model. The clip then reads as the AI driving the
                 editor. Without it, the real tool panel is shown (a human-facing tour).
    keyframes    optional camera path as ``[(t, yaw, pitch, dist), ...]`` with ``t`` in
                 [0,1] over the whole clip — a moving shot (dolly / orbit) instead of the
                 fixed ``view``. Smoothstep-eased between keys; make the last two keys
                 equal so the final dwell is static (and cache-cheap). Overrides ``view``
                 / ``reveal``. Motion means each frame is its own render (no cam cache).
    smooth       pass ``--smooth`` to the viewer (smooth shading, no facets) — nicer for
                 organic/curved models in beauty frames.
    renderer     ``"viewer"`` (default, fast real-time raster) or ``"raytrace"`` — render
                 EVERY frame with the path tracer (``mirage_render``) for promo-grade
                 footage (global illumination, soft shadows, sky+sun). Far slower, and a
                 moving camera at low spp can shimmer, so raise ``trace_spp``.
    trace_hold   with ``renderer="viewer"``: path-trace just the final dwell (the static
                 hold) — one render, held — so the clip *ends* on a Cycles-class beauty
                 frame while the build stays cheap real-time. The best of both.
    trace_spp / trace_threads / trace_knobs / trace_denoise / cam_fov
                 the tracer's samples-per-pixel, worker cap, extra ``--k v`` knobs (e.g.
                 ``{"sun": 1.2, "env": 1.15, "exposure": 1.1}``), denoise passes (>0 runs
                 the edge-avoiding a-trous filter so a LOW-spp traced clip comes out clean —
                 the practical way to path-trace an animation), and vertical FOV (match the
                 viewer's 0.9 so framing is identical).

    Returns ``(mp4_path, gif_path)``. Renders are cached on (stage, camera), so a build
    of N stages held H frames each with an R-frame reveal costs ~``N + R/2`` viewer runs,
    not ``N*H + R``.
    """
    from PIL import Image

    stages = list(stages)
    NP = len(stages)
    if NP == 0:
        raise ValueError("record_build: need at least one stage")

    # a fully path-traced pass (renderer="raytrace") never launches the viewer, so it can
    # run headless (e.g. on a build/render box with -DMIRAGE_BUILD_VIEWER=OFF).
    want_viewer = renderer != "raytrace"
    viewer = Path(viewer) if viewer else default_viewer()
    if want_viewer and not viewer.exists():
        raise FileNotFoundError(
            f"mirage_viewer not built (-DMIRAGE_BUILD_VIEWER=ON): {viewer}")
    # the path tracer is only needed if some frame is rendered with it
    want_trace = renderer == "raytrace" or trace_hold
    render_exe = (Path(render) if render else default_render()) if want_trace else None
    if want_trace and not render_exe.exists():
        raise FileNotFoundError(f"mirage_render not built: {render_exe}")
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

    # AUTO-mode HUD line per stage, written UTF-8 (argv is ANSI-mangled on Windows, so
    # the viewer reads the caption from a file to keep the "·" and any Unicode intact).
    _capfiles = {}
    def capfile(i):
        if i not in _capfiles:
            p = oplog_dir / f"cap_{i}.txt"
            txt = f"{i + 1}/{NP}  ·  {captions[min(i, len(captions) - 1)]}" if captions else ""
            p.write_text(txt, encoding="utf-8")
            _capfiles[i] = p
        return _capfiles[i]

    # render cache: (stage, cam, aim, renderer) -> captioned PNG. Identical held frames and
    # symmetric reveal poses hit the cache instead of re-rendering.
    _cache = {}
    def _run_viewer(stage, cam, tgt, ppm):
        args = [str(viewer), "--oplog", str(oplog(stage)), "--winsize", str(W), str(H),
                "--cam", "%.5f" % cam[0], "%.5f" % cam[1], "%.5f" % cam[2],
                "--target", "%.5f" % tgt[0], "%.5f" % tgt[1], "%.5f" % tgt[2],
                "--floorz", "%.5f" % floorz, "--nohighlight", "--screenshot", str(ppm)]
        if smooth:
            args.append("--smooth")
        if automode:  # hide the panel; the top-left HUD names the stage instead
            args.append("--automode")
            if captions:
                args += ["--autocap-file", str(capfile(stage))]
        r = subprocess.run(args, capture_output=True, text=True)
        return r, "viewer"

    def _run_tracer(stage, cam, tgt, ppm):
        eye = _orbit_eye(cam, tgt)  # same orbit pose the viewer would use -> identical framing
        args = [str(render_exe), "--oplog", str(oplog(stage)), "--out", str(ppm),
                "--w", str(W), "--h", str(H), "--spp", str(trace_spp), "--bounce", "8",
                "--cam-eye", *("%.5f" % v for v in eye),
                "--cam-target", *("%.5f" % v for v in tgt), "--cam-fov", "%.5f" % cam_fov]
        if trace_threads:
            args += ["--threads", str(trace_threads)]
        if trace_denoise:
            args += ["--denoise", str(trace_denoise)]
        for k, v in (trace_knobs or {}).items():
            # a multi-value knob (e.g. sun-dir X Y Z) is given as a list/tuple
            args += [f"--{k}", *([str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)])]
        r = subprocess.run(args, capture_output=True, text=True)
        return r, "tracer"

    def base_png(stage, cam, tgt=None, mode="viewer"):
        if tgt is None:
            tgt = (tx, ty, tz)
        key = (stage, "%.4f_%.4f_%.4f" % cam, "%.4f_%.4f_%.4f" % tgt, mode)
        if key in _cache:
            return _cache[key]
        ppm = tmp / "_cur.ppm"
        if ppm.exists():
            ppm.unlink()
        r, who = (_run_tracer if mode == "raytrace" else _run_viewer)(stage, cam, tgt, ppm)
        if not ppm.exists():
            raise RuntimeError(
                f"{who} produced no frame (stage {stage}, rc={r.returncode}):\n{r.stderr[-400:]}")
        img = Image.open(ppm).convert("RGB")
        # AUTO mode's HUD carries the caption in the viewer; the tracer draws no HUD, so a
        # traced AUTO frame (e.g. the money-shot hold) stays clean. Otherwise caption in PIL.
        if captions and not automode:
            _caption(img, captions[min(stage, len(captions) - 1)], f"{stage + 1}/{NP}", cpos, fonts)
        out = cache_dir / f"r{len(_cache):04d}.png"
        img.save(out)
        _cache[key] = out
        return out

    # the frame plan: (stage, cam, target, mode) per frame. mode picks the renderer:
    # "raytrace" for all frames (renderer="raytrace"), or just the STATIC closing dwell
    # (trace_hold) — a single traced beauty frame held, while the motion stays real-time.
    def frame_mode(is_static_tail):
        if renderer == "raytrace":
            return "raytrace"
        if trace_hold and is_static_tail:
            return "raytrace"
        return "viewer"

    plan = []
    if keyframes:
        # a moving camera: interpolate the keyframe path across the whole clip while the
        # build stages advance underneath it; a flat final key-segment gives a static
        # dwell. Motion means most frames are their own render (no fixed-cam cache reuse).
        F = per * NP + hold
        flat_t = _flat_tail_t(keyframes)          # where the camera stops moving
        for n in range(F):
            stage = min(NP - 1, n // per)
            t = n / max(1, F - 1)
            y, p, d, tgt = _interp_keyframes(keyframes, t)
            plan.append((stage, (y, p, d), tgt, frame_mode(t >= flat_t - 1e-9)))
    else:
        # fixed camera during the build -> optional eased reveal swing -> end dwell
        R = round(reveal * fps)
        for f in range(per * NP):
            plan.append((min(NP - 1, f // per), (yaw0, pitch0, dist0), None, frame_mode(False)))
        for i in range(R):                           # a single eased swing out and back
            yaw = yaw0 + reveal_sweep * math.sin(math.pi * i / R)
            plan.append((NP - 1, (yaw, pitch0, dist0), None, frame_mode(False)))
        for _ in range(hold):                        # the static end dwell
            plan.append((NP - 1, (yaw0, pitch0, dist0), None, frame_mode(True)))

    for n, (stage, cam, tgt, mode) in enumerate(plan):
        shutil.copyfile(base_png(stage, cam, tgt, mode), frames_dir / f"f{n:04d}.png")

    mp4, gif = _encode(ffmpeg, frames_dir, out_dir, out_base, fps, gif_w, gif_fps or fps, tmp)
    if not quiet:
        print(f"[capture] {out_base}: {len(plan)} frames, {len(_cache)} renders "
              f"({W}x{H}@{fps}fps)  ->  {mp4.name} {mp4.stat().st_size // 1024} KB, "
              f"{gif.name} {gif.stat().st_size // 1024} KB")
    return mp4, gif


def crossfade_clip(images, out_base, *, out_dir=None, hold=22, fade=12, fps=24,
                   gif_w=560, gif_fps=12, tmp=None, quiet=False):
    """Turn a few key stills into a smooth clip: hold each, cross-fade between consecutive
    ones. ``images`` are same-size PIL Images (already captioned). Cheap — no re-rendering,
    just blends — so a diff/refinement *story* becomes an animated showcase. Writes
    ``<out_base>.mp4`` + ``.gif`` into ``out_dir`` (default docs/gallery)."""
    import shutil
    from PIL import Image

    out_dir = Path(out_dir) if out_dir else (_repo_root() / "docs" / "gallery")
    tmp = Path(tmp) if tmp else (out_dir / "_clip")
    frames_dir = tmp / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

    imgs = [im.convert("RGB") for im in images]
    n = 0

    def put(im):
        nonlocal n
        im.save(frames_dir / f"f{n:04d}.png"); n += 1

    for i, im in enumerate(imgs):
        for _ in range(hold):
            put(im)
        if i + 1 < len(imgs) and fade > 0:                    # cross-fade into the next still
            for k in range(1, fade + 1):
                put(Image.blend(im, imgs[i + 1], k / (fade + 1)))
    mp4, gif = _encode(ffmpeg, frames_dir, out_dir, out_base, fps, gif_w, gif_fps, tmp)
    if not quiet:
        print(f"[clip] {out_base}: {n} frames  ->  {mp4.name} {mp4.stat().st_size // 1024} KB, "
              f"{gif.name} {gif.stat().st_size // 1024} KB")
    return mp4, gif
