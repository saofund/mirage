"""Render each part ALONE, on a neutral backdrop, into one contact sheet.

Building a scene part by part only works if you can look at a part by itself. In the full
shot the fire bucket is forty pixels across and half in shadow — you cannot tell a good
one from a bad one there, and a fault in it is indistinguishable from a fault in the
lighting, the camera or the thing standing in front of it. So each part gets its own
studio render, framed to its own bounding box, on the same grey.

    uv run python -m forecourt.sheet             # from examples/cases
    uv run python -m forecourt.sheet dispenser   # just one, bigger

Needs mirage_render + Pillow.
"""
import subprocess
import sys
from pathlib import Path

from mirage.capture import default_render
from mirage.meshlang import MeshProgram

from . import parts as P
from .materials import mat

OUT = Path(__file__).resolve().parents[1] / "outputs" / "26_forecourt" / "parts"
BACKDROP = mat((0.32, 0.325, 0.33), 0.0, 0.62)

# name -> (builder, "how big is it", camera yaw in degrees). The yaw picks the three-quarter
# view that shows the part's working face — its front is -y by convention.
CATALOG = {
    "grate_plinth":   (P.grate_plinth, 34),
    "dispenser":      (P.dispenser, 32),
    "nozzle":         (P.nozzle, 40),
    "hose_tangle":    (lambda: P.hose_tangle(1), 34),
    "fire_cabinet":   (P.fire_cabinet, 30),
    "fire_bucket":    (P.fire_bucket, 30),
    "hazard_rail":    (P.hazard_rail, 26),
    "pump_sign":      (P.pump_sign, 22),
    "clad_column":    (lambda: P.clad_column(h=3.0), 30),
    "island":         (P.island, 32),
    "roller_shutter": (P.roller_shutter, 26),
    "bollard":        (P.bollard, 30),
    "wet_floor_sign": (P.wet_floor_sign, 34),
    "speed_hump":     (P.speed_hump, 40),
    "drain_channel":  (lambda: P.drain_channel(3.0), 40),
    "wheel":          (P.wheel, 34),
    "van":            (P.van, 34),
    "suv":            (P.suv, 34),
    "person":         (P.person, 26),
    "bin_":           (P.bin_, 30),
}


def bounds(prog):
    m = prog.build()
    lo = [min(v.co[k] for v in m.verts) for k in range(3)]
    hi = [max(v.co[k] for v in m.verts) for k in range(3)]
    return lo, hi, len(m.faces)


def render_part(name, spp=64, w=520, h=620):
    import math

    build, yaw = CATALOG[name]
    part = build()
    lo, hi, faces = bounds(part)
    ctr = [(lo[k] + hi[k]) / 2 for k in range(3)]
    size = max(hi[k] - lo[k] for k in range(3))
    p = MeshProgram()
    p.place(P.box(size * 6, size * 6, size * 0.4),
            at=[ctr[0], ctr[1], lo[2] - size * 0.2], material=BACKDROP)
    p.place(part)
    OUT.mkdir(parents=True, exist_ok=True)
    js = OUT / f"{name}.json"
    js.write_text(p.to_json())
    ppm = OUT / f"{name}.ppm"
    d = size * 1.85
    a = math.radians(yaw)
    eye = [ctr[0] + d * math.sin(a) * 0.9, ctr[1] - d * math.cos(a), ctr[2] + size * 0.55]
    subprocess.run([str(default_render()), "--oplog", str(js), "--out", str(ppm), "--spp", str(spp),
                    "--w", str(w), "--h", str(h), "--threads", "14",
                    "--cam-eye", *[f"{v:.4f}" for v in eye],
                    "--cam-target", *[f"{v:.4f}" for v in ctr],
                    "--cam-fov", "0.62", "--env", "0.85", "--sun", "0.30",
                    "--sun-dir", "0.35", "-0.5", "0.79", "--exposure", "1.25",
                    "--clamp", "2.0", "--denoise", "3"], check=True,
                   stdout=subprocess.DEVNULL)
    from PIL import Image
    png = OUT / f"{name}.png"
    Image.open(ppm).save(png)
    return png, faces


def main():
    from PIL import Image, ImageDraw
    names = sys.argv[1:] or list(CATALOG)
    tiles = []
    for n in names:
        png, faces = render_part(n, spp=96 if len(names) == 1 else 56)
        print(f"  {n:16s} {faces:7,d} faces -> {png.name}")
        tiles.append((n, faces, Image.open(png).convert("RGB")))
    if len(tiles) == 1:
        return
    cols = 5
    tw, th = tiles[0][2].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    for i, (n, faces, im) in enumerate(tiles):
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(im, (x, y))
        d.rectangle([x, y, x + tw - 1, y + 20], fill=(12, 12, 14))
        d.text((6 + x, 5 + y), f"{n}   {faces:,} faces", fill=(238, 238, 240))
    p = OUT / "sheet.png"
    sheet.save(p)
    print("wrote", p)


if __name__ == "__main__":
    main()
