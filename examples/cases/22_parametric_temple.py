"""Case 22 — a parametric temple: the whole peristyle from one op-log, lit by a raking sun.

The most complex showcase yet, and it leans on everything: `params` + expressions + nested
`repeat` generate a full classical temple — a stepped stylobate, a peristyle of columns
around all four sides (each a base/shaft/echinus/abacus), an entablature, gabled roof and
pediments — and the first-party path tracer renders it under a LOW sun so the colonnade
throws long raking shadows across the stone (the shot a rasteriser can't fake). Change
`cols`/`rows`/`sp` and the entire temple rebuilds.

    uv run python examples/cases/22_parametric_temple.py --hero    # docs/gallery/parametric_temple.png
    uv run python examples/cases/22_parametric_temple.py --orbit   # a cinematic path-traced orbit (mp4)

Needs mirage_render (with --sun-dir / --denoise) and Pillow; --orbit needs ffmpeg.
"""
import sys
import math
import json
import shutil
import subprocess
from pathlib import Path

from mirage.meshlang import MeshProgram
from mirage.capture import default_render

ROOT = Path(__file__).resolve().parents[2]
RENDER = default_render()
OUT = Path(__file__).resolve().parent / "outputs" / "22_parametric_temple"
GALLERY = ROOT / "docs" / "gallery"


def mat(c, rough=0.62, metallic=0.0):
    return {"color": list(c), "metallic": metallic, "roughness": rough}

STEP = mat((0.66, 0.60, 0.50))          # weathered base stone
STONE = mat((0.80, 0.74, 0.62))         # warm limestone (columns, entablature)
ROOF = mat((0.55, 0.40, 0.33), 0.72)    # terracotta roof
PED = mat((0.83, 0.77, 0.66))           # bright pediment stone


def _cyl(sides, r, h):
    return MeshProgram().cylinder(sides=sides, radius=r, height=h)


def column(cr=0.24, ch=3.4):
    """One Doric-ish column (concrete geometry): base drum, shaft, echinus, square abacus."""
    p = MeshProgram()
    p.place(obj=_cyl(18, cr * 1.35, 0.18), at=(0, 0, 0.09), material=STONE)                 # base
    p.place(obj=_cyl(24, cr, ch), at=(0, 0, 0.18 + ch / 2), material=STONE)                 # shaft
    p.place(obj=_cyl(24, cr * 1.4, 0.16), at=(0, 0, 0.18 + ch + 0.08), material=STONE)       # echinus
    p.place(obj=MeshProgram().cube(1.0), at=(0, 0, 0.18 + ch + 0.21),
            scale=(cr * 3.1, cr * 3.1, 0.12), material=STONE)                                # abacus
    return p


def prism(hw, h, d):
    """A triangular gable (pediment): a triangle in x-z, extruded depth ``d`` in y."""
    v = [(-hw, -d / 2, 0), (hw, -d / 2, 0), (0, -d / 2, h),
         (-hw, d / 2, 0), (hw, d / 2, 0), (0, d / 2, h)]
    f = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (0, 3, 5, 2), (1, 2, 5, 4)]
    return v, f


def temple(cols=8, rows=5, sp=1.15, cr=0.24, ch=3.4):
    """A whole temple as ONE parametric op-log. cols/rows/sp are parameters — the peristyle,
    base, entablature and roof all size themselves off them via expressions."""
    p = MeshProgram()
    p.params(cols=cols, rows=rows, sp=sp)
    col = column(cr, ch)
    HX = "(cols-1)*sp*0.5"          # column-ring half extents (expressions over the params)
    HY = "(rows-1)*sp*0.5"
    top = 0.18 + ch + 0.27          # top of the columns (abacus top)

    # --- stepped stylobate (3 courses, widest at the bottom; top course at z=0) ---
    for pad, z, h in [(1.15, -0.40, 0.16), (0.78, -0.24, 0.16), (0.44, -0.08, 0.16)]:
        p.place(obj=MeshProgram().cube(1.0), at=[0, 0, z],
                scale=[f"2*({HX} + {pad})", f"2*({HY} + {pad})", h], material=STEP)

    # --- peristyle: columns on all four sides (nested/paired repeats over the params) ---
    p.repeat("cols", MeshProgram().place(obj=col, at=["(i-(cols-1)*0.5)*sp", HY, 0]))          # front row
    p.repeat("cols", MeshProgram().place(obj=col, at=["(i-(cols-1)*0.5)*sp", f"-({HY})", 0]))  # back row
    p.repeat("rows-2", MeshProgram().place(obj=col, at=[f"-({HX})", "(i+1-(rows-1)*0.5)*sp", 0]))  # left
    p.repeat("rows-2", MeshProgram().place(obj=col, at=[HX, "(i+1-(rows-1)*0.5)*sp", 0]))          # right

    # --- entablature: a frame of four beams resting on the columns ---
    bz = top + 0.28
    p.place(obj=MeshProgram().cube(1.0), at=[0, HY, bz], scale=[f"2*({HX})+{cr*2}+0.5", 0.5, 0.56], material=STONE)
    p.place(obj=MeshProgram().cube(1.0), at=[0, f"-({HY})", bz], scale=[f"2*({HX})+{cr*2}+0.5", 0.5, 0.56], material=STONE)
    p.place(obj=MeshProgram().cube(1.0), at=[HX, 0, bz], scale=[0.5, f"2*({HY})+0.5", 0.56], material=STONE)
    p.place(obj=MeshProgram().cube(1.0), at=[f"-({HX})", 0, bz], scale=[0.5, f"2*({HY})+0.5", 0.56], material=STONE)

    # --- gabled roof: two pediments (front/back) + two sloped roof planes. Concrete
    #     geometry off the layout (the peristyle above is the parametric part). ---
    rz = bz + 0.28
    ph = 1.05                                    # gable height
    HXf, HYf = (cols - 1) * sp * 0.5, (rows - 1) * sp * 0.5
    ped_w = HXf + cr + 0.55                       # gable half-width (temple front)
    roof_d = HYf + 0.45                           # eave half-depth (overhang)
    pv, pf = prism(ped_w, ph, 0.55)
    p.place(verts=pv, faces=pf, at=[0, HYf + 0.18, rz], material=PED)       # front pediment
    p.place(verts=pv, faces=pf, at=[0, -(HYf + 0.18), rz], material=PED)    # back pediment
    pitch = math.degrees(math.atan2(ph, roof_d))
    slen = math.hypot(roof_d, ph) + 0.32          # a little long so the planes overlap at the ridge
    for sgn in (1, -1):                           # roof planes: ridge (y=0, high) -> eave (low)
        p.place(obj=MeshProgram().cube(1.0), at=[0, sgn * roof_d * 0.5, rz + ph * 0.5],
                rotate=[-sgn * pitch, 0, 0], scale=[2 * ped_w + 0.3, slen, 0.10], material=ROOF)
    p.place(obj=MeshProgram().cube(1.0), at=[0, 0, rz + ph + 0.0],
            scale=[2 * ped_w + 0.4, 0.5, 0.16], material=ROOF)   # ridge cap over the seam
    return p


def trace(prog, png, w=1280, h=760, spp=140, denoise=5, sun_dir=(0.82, 0.32, 0.26),
          eye=None, target=None, fov=0.62, threads=16):
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / (png.stem + ".json"); jp.write_text(prog.resolved_json(indent=None))
    ppm = OUT / (png.stem + ".ppm")
    eye = eye or (11.0, -13.0, 6.4)
    target = target or (0.0, 0.0, 2.3)
    cmd = [str(RENDER), "--oplog", str(jp), "--out", str(ppm), "--w", str(w), "--h", str(h),
           "--spp", str(spp), "--bounce", "8", "--threads", str(threads), "--denoise", str(denoise),
           "--sun", "1.25", "--env", "0.95", "--exposure", "0.92",
           "--sun-dir", *map(str, sun_dir),
           "--cam-eye", *map(str, eye), "--cam-target", *map(str, target), "--cam-fov", str(fov)]
    subprocess.run(cmd, check=True)
    from PIL import Image
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ppm).save(png)


def render_hero():
    p = temple(cols=8, rows=5, sp=1.15)
    m = p.build()
    print(f"  temple: {len(p.ops)} ops -> {len(p.resolved())} resolved ops, {len(m.faces)} faces")
    trace(p, GALLERY / "parametric_temple.png", w=1280, h=800, spp=200)
    print(f"  wrote {GALLERY / 'parametric_temple.png'}")


def render_orbit(frames=100, w=1280, h=720, spp=110, denoise=5):
    """A full 360 path-traced orbit. The sun is fixed (raking), so as the camera circles
    the temple you sweep from the sunlit colonnade to the shadowed side — cinematic."""
    quick = "--quick" in sys.argv
    if quick:
        frames, w, h, spp = 40, 640, 380, 80
    fdir = OUT / "orbit_frames"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    p = temple(cols=8, rows=5, sp=1.15)
    R, H, tz = 15.5, 6.6, 2.3
    for k in range(frames):
        a = 2 * math.pi * k / frames + math.radians(200)     # start on the sunlit 3/4
        trace(p, fdir / f"f{k:04d}.png", w=w, h=h, spp=spp, denoise=denoise,
              eye=(R * math.cos(a), R * math.sin(a), H), target=(0, 0, tz), fov=0.60)
        print(f"orbit {k + 1}/{frames}")
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    mp4, gif = GALLERY / "parametric_temple.mp4", GALLERY / "parametric_temple.gif"
    pat = str(fdir / "f%04d.png")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "24", "-i", pat,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-movflags", "+faststart", str(mp4)], check=True)
    pal, scale = fdir / "pal.png", "scale=560:-1:flags=lanczos"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "24", "-i", pat,
                    "-vf", f"fps=12,{scale},palettegen=stats_mode=diff", str(pal)], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", "24", "-i", pat, "-i", str(pal),
                    "-lavfi", f"fps=12,{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3", str(gif)], check=True)
    print(f"wrote {mp4} + {gif}")


def main():
    if "--orbit" in sys.argv:
        render_orbit()
    else:
        render_hero()


if __name__ == "__main__":
    main()
