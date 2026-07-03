"""Render a promo animation of the airliner, straight from its op-log.

Two clips, both path-traced by mirage_render.exe (no external DCC):

  * airplane_build      — the jet ASSEMBLES from its parts (fuselage -> wings ->
                          winglets -> tail -> engines) while it turns, then keeps
                          spinning. The op-log thesis made visible: a model is a
                          sequence of operations.
  * airplane_turntable  — the finished jet, a seamless 360 deg turntable loop.

Each is written as an .mp4 (crisp, for Twitter/video) and a .gif (for inline
README / chat). Frames are the model rotated about +Z (the camera is fixed, so
the geometry spins); a full 360 deg loops perfectly. A fixed ground slab keeps the
floor from jumping as low parts (the engines) appear mid-build.

    uv run python docs/gallery/render_airplane_anim.py
    ANIM_QUICK=1 uv run python docs/gallery/render_airplane_anim.py   # fast smoke test

Needs mirage_render.exe built and ffmpeg on PATH.
"""
import math
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
from PIL import Image  # noqa: E402

RENDER = ROOT / "core" / "build" / "Release" / "mirage_render.exe"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
GALLERY = ROOT / "docs" / "gallery"
TMP = Path(os.environ.get("ANIM_TMP", str(GALLERY / "_anim")))

QUICK = os.environ.get("ANIM_QUICK") == "1"
W       = int(os.environ.get("ANIM_W", 640 if QUICK else 960))
H       = int(os.environ.get("ANIM_H", 360 if QUICK else 540))
SPP     = int(os.environ.get("ANIM_SPP", 48 if QUICK else 192))
FPS     = int(os.environ.get("ANIM_FPS", 24))
BUILD   = int(os.environ.get("ANIM_BUILD", 18 if QUICK else 72))
SPIN    = int(os.environ.get("ANIM_SPIN", 12 if QUICK else 48))
REVEAL  = int(os.environ.get("ANIM_REVEAL", 2 if QUICK else 4))   # frames between parts
SCALE   = float(os.environ.get("ANIM_SCALE", 0.9))                # shrink to keep it in frame
YAW_OFF = float(os.environ.get("ANIM_YAWOFF", 35.0))              # turntable start angle
GIF_W   = int(os.environ.get("ANIM_GIF_W", 720))

# a matte floor matching the renderer's own ground albedo, so our fixed slab is
# indistinguishable from it (but stays put across frames)
GROUND = {"color": [0.40, 0.42, 0.46], "metallic": 0.0, "roughness": 0.92}

STEPS = part_steps()
_full = assemble(STEPS)
_xs = [v[0] for v in _full.v]
_ys = [v[1] for v in _full.v]
_zs = [v[2] for v in _full.v]
CX = (min(_xs) + max(_xs)) / 2.0
CY = (min(_ys) + max(_ys)) / 2.0
ZMIN = min(_zs)


def xform(verts, yaw_deg):
    """Uniform-scale about (CX, CY, ZMIN) then yaw about the vertical through
    (CX, CY): the jet shrinks toward the floor and spins in place."""
    a = math.radians(yaw_deg)
    c, s = math.cos(a), math.sin(a)
    out = []
    for (x, y, z) in verts:
        x = CX + (x - CX) * SCALE
        y = CY + (y - CY) * SCALE
        z = ZMIN + (z - ZMIN) * SCALE
        dx, dy = x - CX, y - CY
        out.append((CX + dx * c - dy * s, CY + dx * s + dy * c, z))
    return out


def ground_slab():
    L, z = 40.0, ZMIN - 0.02
    v = [(CX - L, CY - L, z), (CX + L, CY - L, z), (CX + L, CY + L, z), (CX - L, CY + L, z)]
    return v, [[0, 1, 2, 3]], [GROUND]


def frame_mesh(nsteps, yaw):
    acc = assemble(STEPS, nsteps)
    v = xform(acc.v, yaw)
    gv, gf, gm = ground_slab()          # slab first so its faces/mats stay aligned
    off = len(gv)
    verts = gv + v
    faces = gf + [[i + off for i in face] for face in acc.f]
    mats = gm + acc.m
    return verts, faces, mats


def render_frame(verts, faces, mats, png):
    prog = MeshProgram()
    prog.mesh(verts, faces, face_materials=mats)
    (TMP / "_cur.json").write_text(prog.to_json())
    r = subprocess.run(
        [str(RENDER), "--oplog", str(TMP / "_cur.json"), "--out", str(TMP / "_cur.ppm"),
         "--spp", str(SPP), "--w", str(W), "--h", str(H),
         "--env", "1.0", "--sun", "3.0", "--exposure", "1.15", "--clamp", "8"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"render failed:\n{r.stderr}")
    Image.open(TMP / "_cur.ppm").convert("RGB").save(png)


def reveal(k):
    return min(len(STEPS), 1 + k // REVEAL)


def clean_dir(d):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


def render_clip(name, n, yaw_of, steps_of):
    d = TMP / name
    clean_dir(d)
    for k in range(n):
        render_frame(*frame_mesh(steps_of(k), yaw_of(k)), d / f"f{k:04d}.png")
        sys.stdout.write(f"\r  {name}: frame {k + 1}/{n}  (parts {steps_of(k)}/{len(STEPS)})   ")
        sys.stdout.flush()
    print()
    return d


def encode(frames_dir, base, fps):
    mp4 = GALLERY / f"{base}.mp4"
    gif = GALLERY / f"{base}.gif"
    pat = str(frames_dir / "f%04d.png")
    scale = f"scale={GIF_W}:-1:flags=lanczos"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True)
    pal = TMP / "pal.png"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat,
                    "-vf", f"{scale},palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pat,
                    "-i", str(pal), "-lavfi",
                    f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                    str(gif)], check=True)
    print(f"  {mp4.name}  {mp4.stat().st_size/1024:.0f} KB")
    print(f"  {gif.name}  {gif.stat().st_size/1024:.0f} KB")


def main():
    if not RENDER.exists():
        sys.exit(f"renderer not built: {RENDER}")
    TMP.mkdir(parents=True, exist_ok=True)
    print(f"animation: {W}x{H} spp={SPP} fps={FPS}  build={BUILD} spin={SPIN}"
          f"{'  [QUICK]' if QUICK else ''}")

    bd = render_clip("build", BUILD, lambda k: 360.0 * k / BUILD, reveal)
    sd = render_clip("spin", SPIN, lambda k: YAW_OFF + 360.0 * k / SPIN, lambda k: len(STEPS))

    print("encoding (ffmpeg)...")
    encode(bd, "airplane_build", FPS)
    encode(sd, "airplane_turntable", FPS)
    print("done.")


if __name__ == "__main__":
    main()
