"""Record the airliner being MODELLED — the making-of, in the real viewport.

Not a glamour turntable: this is the construction itself. Each frame is a headless
screenshot of the actual `mirage_viewer` GUI (tool panel and all), fed a growing
op-log so the jet assembles operator by operator in the live viewport —
fuselage -> wings -> winglets -> tail -> fin -> engines -> pylons — exactly the
sequence the op-log lays down. The camera holds one fixed 3/4 (a build, not a
showcase); the orbit centre and studio floor are pinned to the finished model's
bbox (viewer's --target / --floorz) so nothing jumps as parts appear. A subtle
caption names the operator adding each part.

Because the camera is fixed, every held frame is identical, so the clip is one
screenshot per part-count (7 renders) reused across the holds — fast, and the
.gif dedupes the static tool panel / floor down to almost nothing.

Output is an .mp4 (crisp, for video/Twitter) and a .gif (inline README / chat).

    uv run python docs/gallery/render_viewer_build.py
    ANIM_QUICK=1 uv run python docs/gallery/render_viewer_build.py   # fast smoke test

Needs mirage_viewer.exe built (-DMIRAGE_BUILD_VIEWER=ON) and ffmpeg on PATH.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "src"))
from airplane import part_steps, assemble  # noqa: E402
from mirage.meshlang import MeshProgram  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

VIEWER = ROOT / "core" / "build" / "Release" / "mirage_viewer.exe"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
GALLERY = ROOT / "docs" / "gallery"
TMP = Path(os.environ.get("ANIM_TMP", str(GALLERY / "_build")))

QUICK = os.environ.get("ANIM_QUICK") == "1"
W    = int(os.environ.get("ANIM_W", 854 if QUICK else 1280))
H    = int(os.environ.get("ANIM_H", 480 if QUICK else 720))
FPS  = int(os.environ.get("ANIM_FPS", 24))
PER  = int(os.environ.get("ANIM_PER", 6 if QUICK else 13))   # frames each part holds before the next
HOLD = int(os.environ.get("ANIM_HOLD", 12 if QUICK else 30)) # frames to dwell on the finished jet
GIF_W = int(os.environ.get("ANIM_GIF_W", 760))
CAPTION = os.environ.get("ANIM_CAPTION", "1") == "1"

# one fixed 3/4 viewpoint (the viewer's orbit yaw/pitch/dist)
YAW   = float(os.environ.get("ANIM_YAW", 2.30))
PITCH = float(os.environ.get("ANIM_PITCH", 0.36))
DIST  = float(os.environ.get("ANIM_DIST", 9.0))

STEPS = part_steps()
NP = len(STEPS)
_full = assemble(STEPS)
_xs = [v[0] for v in _full.v]
_ys = [v[1] for v in _full.v]
_zs = [v[2] for v in _full.v]
CX, CY, CZ = (min(_xs) + max(_xs)) / 2, (min(_ys) + max(_ys)) / 2, (min(_zs) + max(_zs)) / 2
FLOORZ = min(_zs) - 0.01
NVERTS = len(_full.v)

N = PER * NP + HOLD  # total frames

# what each part is, for the caption (indexed by how many parts are shown)
CAPTIONS = [
    "fuselage  ·  surface of revolution",
    "wings  ·  lofted, mirrored",
    "winglets  ·  swept fins",
    "tailplane  ·  lofted, mirrored",
    "vertical fin",
    "engines  ·  revolved, mirrored",
    f"complete  ·  one op-log, {NVERTS} verts",
]


def font(size):
    for name in ("segoeui.ttf", "seguisb.ttf", "arial.ttf"):
        p = Path("C:/Windows/Fonts") / name
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


F_LABEL = font(23)
F_INDEX = font(20)


def parts_at(f):
    return min(NP, 1 + f // PER)


def oplog_for(nparts):
    """Write (once) the op-log for the first `nparts` build steps; return its path."""
    p = TMP / f"parts_{nparts}.json"
    if not p.exists():
        acc = assemble(STEPS, nparts)
        prog = MeshProgram()
        prog.mesh(acc.v, acc.f, face_materials=acc.m)
        p.write_text(prog.to_json())
    return p


def caption(img, nparts):
    if not CAPTION:
        return
    d = ImageDraw.Draw(img, "RGBA")
    x, y = 408, H - 56                     # fixed offset: clears the 376px tool panel (right edge ~392)
    idx = f"{nparts}/{NP}"
    label = CAPTIONS[nparts - 1]
    # teal step index, then a light label; a soft shadow keeps both legible on any tone
    for dx, dy in ((1, 1), (2, 2)):
        d.text((x + dx, y + dy), idx, font=F_INDEX, fill=(0, 0, 0, 150))
    d.text((x, y), idx, font=F_INDEX, fill=(70, 200, 214, 255))
    ix = x + int(d.textlength(idx, font=F_INDEX)) + 14
    for dx, dy in ((1, 1), (2, 2)):
        d.text((ix + dx, y + dy + 1), label, font=F_LABEL, fill=(0, 0, 0, 150))
    d.text((ix, y + 1), label, font=F_LABEL, fill=(232, 236, 240, 255))


def render_base(nparts, out_png):
    """One captioned screenshot of the viewport with `nparts` parts placed."""
    oj = oplog_for(nparts)
    ppm = TMP / "_cur.ppm"
    r = subprocess.run(
        [str(VIEWER), "--oplog", str(oj), "--winsize", str(W), str(H),
         "--cam", f"{YAW:.5f}", f"{PITCH:.5f}", f"{DIST:.5f}",
         "--target", f"{CX:.5f}", f"{CY:.5f}", f"{CZ:.5f}",
         "--floorz", f"{FLOORZ:.5f}", "--nohighlight", "--screenshot", str(ppm)],
        capture_output=True, text=True,
    )
    if not ppm.exists():
        sys.exit(f"viewer failed for {nparts} parts (rc={r.returncode}):\n{r.stderr[-500:]}")
    img = Image.open(ppm).convert("RGB")
    caption(img, nparts)
    img.save(out_png)


def encode(frames_dir, base):
    mp4 = GALLERY / f"{base}.mp4"
    gif = GALLERY / f"{base}.gif"
    pat = str(frames_dir / "f%04d.png")
    scale = f"scale={GIF_W}:-1:flags=lanczos"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pat,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True)
    pal = TMP / "pal.png"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pat,
                    "-vf", f"{scale},palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", pat,
                    "-i", str(pal), "-lavfi",
                    f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                    str(gif)], check=True)
    print(f"  {mp4.name}  {mp4.stat().st_size / 1024:.0f} KB")
    print(f"  {gif.name}  {gif.stat().st_size / 1024:.0f} KB")


def main():
    if not VIEWER.exists():
        sys.exit(f"viewer not built (-DMIRAGE_BUILD_VIEWER=ON): {VIEWER}")
    fdir = TMP / "frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    print(f"making-of: {W}x{H} fps={FPS}  {NP} parts x {PER}f + {HOLD}f hold = {N} frames"
          f"{'  [QUICK]' if QUICK else ''}")
    print(f"target ({CX:.2f},{CY:.2f},{CZ:.2f})  floorz {FLOORZ:.2f}")

    # fixed camera -> render one captioned base image per part-count, reuse for holds
    bases = {}
    for nparts in range(1, NP + 1):
        bp = TMP / f"base_{nparts}.png"
        render_base(nparts, bp)
        bases[nparts] = bp
        sys.stdout.write(f"\r  rendered part-count {nparts}/{NP}   ")
        sys.stdout.flush()
    print()
    for f in range(N):
        shutil.copyfile(bases[parts_at(f)], fdir / f"f{f:04d}.png")

    print("encoding (ffmpeg)...")
    encode(fdir, "airplane_assembly")
    print("done.")


if __name__ == "__main__":
    main()
