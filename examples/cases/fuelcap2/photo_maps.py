"""Bake projective albedo maps from the fixed Polo reference photograph.

The maps are generated into ignored ``assets/decals`` and copied to the render box. They
carry this specific car's wear, water and colour variation; geometry still supplies depth,
silhouette, occlusion, normals and specular response.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[3]
REF = (ROOT / "examples" / "cases" / "fuelcap" / "_ref" / "bycar" / "Polo" /
       "粗筛done2_Polo_2007款劲情14自动风尚版_38.png")
OUT = ROOT / "assets" / "decals"
RECIPE = "polo-projective-v10"


def _linear_ppm(image: Image.Image, path: Path, gain=0.90):
    srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)
    linear = np.clip(linear * gain, 0.0, 1.0)
    Image.fromarray((linear * 255.0 + 0.5).astype(np.uint8), "RGB").save(path, "PPM")


def _liner(source: Image.Image):
    # Tight to the dark aperture. Including the surrounding blue body puts cyan into the
    # map precisely where the model's grey flange samples its outer edge.
    crop = source.crop((550, 160, 1120, 720)).resize((1024, 1024), Image.Resampling.LANCZOS)
    # The real cap and open door are separate 3D parts. Remove them from the liner plate so
    # the projection cannot leave a second ghost cap behind the modeled one.
    fill = Image.new("RGB", crop.size, (31, 32, 34))
    mask = Image.new("L", crop.size, 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((120, 310, 700, 900), fill=255)
    d.polygon([(800, 0), (1024, 0), (1024, 1024), (830, 900)], fill=235)
    mask = mask.filter(ImageFilter.GaussianBlur(24))
    return Image.composite(fill, crop, mask)


def _panel(source: Image.Image):
    """Body plate with the photographed open door removed before projection."""
    # A large local blur removes the old door without introducing the rectangular colour
    # discontinuity produced by copying a strip from the far-right edge of the photograph.
    blurred = source.filter(ImageFilter.GaussianBlur(90))
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).polygon([(1100, 110), (1375, 110), (1375, 955),
                                  (1120, 955), (1040, 680), (1040, 270)], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(55))
    return Image.composite(blurred, source, mask)


def _padded_panel(source: Image.Image):
    """Put the calibrated frame in a softly extended canvas so decal edges stay off-camera."""
    panel = _panel(source)
    # Reflect padding is pixel-continuous at all four joins.  The earlier blurred,
    # enlarged background differed from the source at the join and left two slanted bands
    # in an otherwise matched full-frame render.
    array = np.asarray(panel)
    padded = np.pad(array, ((500, 500), (750, 750), (0, 0)), mode="reflect")
    return Image.fromarray(padded, "RGB")


def ensure_photo_maps():
    OUT.mkdir(parents=True, exist_ok=True)
    names = {name: OUT / f"fuelcap_polo_{name}_photo.ppm"
             for name in ("cap", "liner", "door", "panel")}
    stamp = OUT / "fuelcap_polo_photo.recipe"
    if all(path.exists() for path in names.values()) and stamp.exists() \
            and stamp.read_text(encoding="ascii") == RECIPE:
        return names
    if not REF.exists():
        missing = ", ".join(str(p) for p in names.values() if not p.exists())
        raise FileNotFoundError(f"Polo photo maps are missing on this host: {missing}")

    src = Image.open(REF).convert("RGB")
    cap = src.crop((625, 340, 925, 640)).resize((1024, 1024), Image.Resampling.LANCZOS)
    # Start right of the photographed cap. The previous crop began at x=895 and literally
    # baked half the cap into the door map, which reappeared as a ghost on the opened door.
    door = src.crop((1000, 150, 1280, 930)).resize((1024, 1024), Image.Resampling.LANCZOS)
    _linear_ppm(cap, names["cap"], gain=0.86)
    _linear_ppm(_liner(src), names["liner"], gain=0.74)
    _linear_ppm(door, names["door"], gain=1.30)
    # Camera projection is limited to the body-panel faces.  The actual hole therefore
    # remains empty and the modelled lamp/door occlude their photographed counterparts,
    # while the broad clear-coat reflection and door shadow survive exactly where measured.
    _linear_ppm(_padded_panel(src).resize((2048, 1365), Image.Resampling.LANCZOS),
                names["panel"], gain=0.82)
    stamp.write_text(RECIPE, encoding="ascii")
    return names
