"""Case 19 — a PARAMETRIC tower: the op-log as a re-runnable generator.

This is the thing a puppet-an-app MCP can't do. The model isn't a bag of geometry
you poke at — it's a legible *program* with parameters and a ``repeat`` loop, so one
number rebuilds the whole structure. ``floors`` stacks more storeys, ``twist`` spirals
them, ``taper`` pinches the silhouette, ``cols`` sets the colonnade density — each a
parameter, each read by an expression inside the op-log:

    place(cylinder, at=["(w*taper^i - 0.06) * cos(tau*j/cols + twist*i*pi/180)", ...])
                                       #  floor i, column j — the loop indices ARE the form

    uv run python examples/cases/19_parametric_tower.py --hero     # one tower -> docs/gallery
    uv run python examples/cases/19_parametric_tower.py --grid     # a 4x4 design space (16 towers)
    uv run python examples/cases/19_parametric_tower.py --morph    # a parameter sweep, path-traced

Each variant is a different setting of the SAME parametric op-log, resolved to a plain
op-log and path-traced by the first-party renderer (``--denoise`` keeps low-spp clean).
Needs ``mirage_render.exe`` and Pillow.
"""
import sys
import time
import math
import subprocess
from pathlib import Path

from mirage.meshlang import MeshProgram

ROOT = Path(__file__).resolve().parents[2]
RENDER = ROOT / "core" / "build" / "Release" / "mirage_render.exe"
OUT = Path(__file__).resolve().parent / "outputs" / "19_parametric_tower"
GALLERY = ROOT / "docs" / "gallery"


def mat(color, metallic=0.0, roughness=0.5):
    return {"color": list(color), "metallic": metallic, "roughness": roughness}

BASE   = mat((0.40, 0.37, 0.33), 0.0, 0.75)   # dark plinth stone
STONE  = mat((0.70, 0.64, 0.54), 0.0, 0.65)   # warm limestone slabs
COLUMN = mat((0.78, 0.72, 0.62), 0.0, 0.55)   # pale columns
CORE   = mat((0.26, 0.47, 0.53), 0.0, 0.14)   # a glazed teal core
SPIRE  = mat((0.83, 0.62, 0.30), 1.0, 0.28)   # brass finial


def _cube():
    return MeshProgram().cube(1.0)


def tower(floors=12, twist=9.0, taper=0.94, fh=0.60, w=1.0, cols=6):
    """A parametric tower as ONE op-log: a plinth + glazed core, then ``floors``
    storeys (each a slab ringed by ``cols`` columns) placed by expressions of the
    loop indices, capped by a brass spire. Every knob is a parameter."""
    p = MeshProgram()
    p.params(floors=floors, twist=twist, taper=taper, fh=fh, w=w, cols=cols)

    p.place(obj=_cube(), at=[0, 0, -0.13], scale=["w*1.4", "w*1.4", 0.26], material=BASE)   # plinth
    p.place(obj=MeshProgram().cylinder(sides=28, radius=0.5, height=1.0),                    # glazed core
            at=[0, 0, "fh*floors*0.5"], scale=["w*0.34", "w*0.34", "fh*floors"], material=CORE)

    # one storey: a slab, then a ring of columns rotated with the storey's twist
    storey = MeshProgram()
    storey.place(obj=_cube(), at=[0, 0, "fh*i + fh*0.5"], rotate=[0, 0, "twist*i"],
                 scale=["w*taper^i", "w*taper^i", 0.10], material=STONE)
    column = MeshProgram().place(
        obj=MeshProgram().cylinder(sides=10, radius=0.055, height=1.0),
        at=["(w*taper^i - 0.07) * cos(tau*j/cols + twist*i*pi/180)",
            "(w*taper^i - 0.07) * sin(tau*j/cols + twist*i*pi/180)",
            "fh*i + fh*0.5"],
        scale=[1, 1, "fh"], material=COLUMN)
    storey.repeat("cols", column, index="j")
    p.repeat("floors", storey, index="i")

    p.place(obj=MeshProgram().cone(sides=10, radius=0.5, height=1.0),                        # spire
            at=[0, 0, "fh*floors + w*taper^floors*0.7"], rotate=[0, 0, "twist*floors"],
            scale=["w*taper^floors*0.8", "w*taper^floors*0.8", "w*1.5"], material=SPIRE)
    return p


# ---- render (resolve the parametric op-log, then path-trace it) ---------------- #
def trace(prog, png, w=820, h=900, spp=96, denoise=5, eye=None, target=None, fov=0.64,
          top_z=7.2, threads=16):
    if not RENDER.exists():
        print(f"  ! mirage_render not built — {RENDER}")
        return False
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / (png.stem + ".json")
    jp.write_text(prog.resolved_json(indent=None))       # the PLAIN op-log the tracer builds
    ppm = OUT / (png.stem + ".ppm")
    eye = eye or (top_z * 1.25, -top_z * 1.25, top_z * 0.56)   # pull back to fit the full height
    target = target or (0.0, 0.0, top_z * 0.44)
    cmd = [str(RENDER), "--oplog", str(jp), "--out", str(ppm), "--w", str(w), "--h", str(h),
           "--spp", str(spp), "--bounce", "8", "--threads", str(threads), "--denoise", str(denoise),
           "--sun", "1.1", "--env", "1.05", "--exposure", "0.88",
           "--cam-eye", *map(str, eye), "--cam-target", *map(str, target), "--cam-fov", str(fov)]
    subprocess.run(cmd, check=True)
    try:
        from PIL import Image
    except ImportError:
        print(f"  wrote {ppm} (install pillow for png)")
        return True
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ppm).save(png)
    return True


def _top(floors, fh=0.60):
    return fh * floors + 1.2


def render_hero():
    p = tower(floors=13, twist=10.0, taper=0.93, cols=6)
    m = p.build()
    print(f"  parametric tower: {len(p.ops)} ops -> {len(p.resolved())} resolved ops, {len(m.faces)} faces")
    trace(p, GALLERY / "parametric_tower.png", w=900, h=1040, spp=140, top_z=_top(13))
    print(f"  wrote {GALLERY/'parametric_tower.png'}")


# ---- a 4x4 design space: sweep two parameters over one program ---------------- #
def render_grid():
    from PIL import Image
    OUT.mkdir(parents=True, exist_ok=True)
    twists = [0.0, 8.0, 16.0, 26.0]
    floorset = [7, 11, 15, 19]
    cell = 300
    sheet = Image.new("RGB", (cell * 4, cell * 4), (16, 17, 20))
    t0 = time.perf_counter()
    for r, fl in enumerate(floorset):
        for c, tw in enumerate(twists):
            taper = 0.90 + 0.06 * (c / 3)
            p = tower(floors=fl, twist=tw, taper=taper, cols=6)
            png = OUT / f"cell_{r}{c}.png"
            trace(p, png, w=cell, h=cell, spp=80, denoise=5, fov=0.66, top_z=_top(fl))
            sheet.paste(Image.open(png), (c * cell, r * cell))
            print(f"  [{r*4+c+1}/16] floors={fl} twist={tw}  [{time.perf_counter()-t0:.0f}s]")
    GALLERY.mkdir(parents=True, exist_ok=True)
    sheet.save(GALLERY / "parametric_grid.png")
    print(f"  wrote {GALLERY/'parametric_grid.png'}  ({time.perf_counter()-t0:.0f}s total)")


# ---- a morph: sweep parameters smoothly, path-trace each frame ---------------- #
def render_morph():
    import shutil
    from PIL import Image
    frames_dir = OUT / "morph_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    quick = "--quick" in sys.argv
    N = 24 if quick else 60
    W, H = (480, 560) if quick else (720, 840)
    top = _top(16)
    for k in range(N):
        u = k / (N - 1)
        s = 0.5 - 0.5 * math.cos(math.pi * u)                    # ease 0..1
        p = tower(floors=16, twist=2.0 + 26.0 * s, taper=0.985 - 0.11 * s, cols=6)
        trace(p, frames_dir / f"f{k:04d}.png", w=W, h=H, spp=72, denoise=5, fov=0.60, top_z=top)
        print(f"  morph {k+1}/{N}")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    mp4 = GALLERY / "parametric_morph.mp4"
    gif = GALLERY / "parametric_morph.gif"
    pat = str(frames_dir / "f%04d.png")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "24", "-i", pat,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True)
    pal = frames_dir / "pal.png"
    scale = "scale=460:-1:flags=lanczos"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "24", "-i", pat,
                    "-vf", f"fps=12,{scale},palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "24", "-i", pat, "-i", str(pal),
                    "-lavfi", f"fps=12,{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    str(gif)], check=True)
    print(f"  wrote {mp4} + {gif}")


def main():
    if "--grid" in sys.argv:
        render_grid()
    elif "--morph" in sys.argv:
        render_morph()
    else:
        render_hero()


if __name__ == "__main__":
    main()
