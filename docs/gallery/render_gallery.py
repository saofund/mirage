"""Regenerate the docs/gallery images from op-logs — end to end, no fakes.

Each panel is one op-log replayed through the kernel and shot with the native
path tracer (core/build/Release/mirage_render.exe). Run from the repo root:

    uv run python docs/gallery/render_gallery.py

Requires the render executable to be built:

    cmake --build core/build --config Release --target mirage_render

The four panels are exactly the features that landed together:
    1. screw      — the helical sweep (spring / thread / auger)
    2. selectors  — by-curvature selection (selection-as-query, deepened)
    3. profile    — an open wire revolved into a single-walled, hollow vase
    4. boolean    — real BSP mesh-mesh CSG (cube minus a cylinder bore)
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mirage import meshlang as ml  # noqa: E402
from mirage.meshlang import Sel  # noqa: E402
import mirage.kernel as K  # noqa: E402

GALLERY = ROOT / "docs" / "gallery"
RENDER = ROOT / "core" / "build" / "Release" / "mirage_render.exe"


def programs():
    # 1) SCREW — a small off-axis section climbs 4 turns into a spring.
    screw = ml.MeshProgram()
    screw.profile([[0.5, 0.0], [0.62, 0.0]], plane="xz")
    screw.screw(axis="z", steps=64, turns=4, height=1.6, angle=360.0)
    screw.solidify(0.06)
    screw.material(Sel.all(), color=[0.95, 0.78, 0.22], metallic=1.0, roughness=0.25)

    # 2) SELECTORS — curvature splits the flat-ish cap from the round body.
    sel = ml.MeshProgram()
    sel.uv_sphere(segments=32, rings=20, radius=0.6)
    sel.material(Sel.all(), color=[0.2, 0.5, 0.9], metallic=0.0, roughness=0.4)
    sel.material(Sel.curvature(min=0.0, max=8.0), color=[0.9, 0.3, 0.2], metallic=0.0, roughness=0.5)

    # 3) PROFILE — an OPEN polyline (wire, no face) revolved => single wall.
    vase = ml.MeshProgram()
    vase.profile(
        [[0.15, -0.7], [0.45, -0.55], [0.55, -0.2], [0.35, 0.15], [0.28, 0.5], [0.42, 0.75]],
        plane="xz",
    )
    vase.spin(axis="z", steps=48, angle=360.0)
    vase.solidify(0.04)
    vase.material(Sel.all(), color=[0.25, 0.55, 0.75], metallic=0.0, roughness=0.35)

    # 4) BOOLEAN — real BSP CSG: cube minus a tall cylinder = a bored block.
    drill = K.make_cylinder_ngon(24, 0.28, 3.0)
    boolean = ml.MeshProgram()
    boolean.cube(1.0)
    boolean.boolean("difference", drill)
    boolean.material(Sel.all(), color=[0.9, 0.5, 0.15], metallic=0.3, roughness=0.4)

    return {
        "1_screw_spring": screw,
        "2_selectors": sel,
        "3_profile_vase": vase,
        "4_boolean_bore": boolean,
    }


def main():
    if not RENDER.exists():
        sys.exit(f"render exe not found: {RENDER}\nbuild it: cmake --build core/build --config Release --target mirage_render")

    imgs = []
    for name, prog in programs().items():
        oplog = GALLERY / f"{name}.json"
        ppm = GALLERY / f"{name}.ppm"
        oplog.write_text(prog.to_json())
        stats = prog.build().stats()
        print(f"{name:14s} V={stats['verts']:4d} E={stats['edges']:4d} F={stats['faces']:4d}")
        subprocess.run(
            [str(RENDER), "--oplog", str(oplog), "--out", str(ppm),
             "--spp", "64", "--w", "640", "--h", "480",
             "--env", "--sun", "--exposure", "1.1", "--clamp"],
            check=True,
        )
        imgs.append((name, ppm))

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow not installed; wrote .ppm panels but skipped .png + contact sheet.")
        return

    titles = {
        "1_screw_spring": "1. SCREW  - helical sweep (spring / thread)",
        "2_selectors": "2. SELECTORS - by curvature (red = flat cap)",
        "3_profile_vase": "3. PROFILE - open wire revolved (hollow vase)",
        "4_boolean_bore": "4. BOOLEAN - cube minus cylinder (BSP CSG)",
    }
    loaded = []
    for name, ppm in imgs:
        im = Image.open(ppm).convert("RGB")
        im.save(GALLERY / f"{name}.png")
        ppm.unlink()  # keep only the .png in the repo
        loaded.append((name, im))

    w, h = loaded[0][1].size
    pad, lab = 12, 28
    sheet = Image.new("RGB", (w * 2 + pad * 3, (h + lab) * 2 + pad * 3), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(loaded):
        r, c = divmod(i, 2)
        x = pad + c * (w + pad)
        y = pad + r * (h + lab + pad)
        draw.text((x + 2, y), titles[name], fill=(230, 230, 235))
        sheet.paste(im, (x, y + lab))
    sheet.save(GALLERY / "showcase.png")
    print("wrote", GALLERY / "showcase.png")


if __name__ == "__main__":
    main()
