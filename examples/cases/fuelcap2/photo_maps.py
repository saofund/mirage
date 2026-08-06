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
RECIPE = "polo-projective-v2"


def _linear_ppm(image: Image.Image, path: Path, gain=0.90):
    srgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92,
                      ((srgb + 0.055) / 1.055) ** 2.4)
    linear = np.clip(linear * gain, 0.0, 1.0)
    Image.fromarray((linear * 255.0 + 0.5).astype(np.uint8), "RGB").save(path, "PPM")


def _liner(source: Image.Image):
    crop = source.crop((515, 100, 1035, 730)).resize((1024, 1024), Image.Resampling.LANCZOS)
    # The real cap and open door are separate 3D parts. Remove them from the liner plate so
    # the projection cannot leave a second ghost cap behind the modeled one.
    fill = Image.new("RGB", crop.size, (31, 32, 34))
    mask = Image.new("L", crop.size, 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((122, 332, 810, 913), fill=255)
    d.polygon([(790, 175), (1024, 130), (1024, 930), (840, 885)], fill=235)
    mask = mask.filter(ImageFilter.GaussianBlur(24))
    return Image.composite(fill, crop, mask)


def ensure_photo_maps():
    OUT.mkdir(parents=True, exist_ok=True)
    names = {name: OUT / f"fuelcap_polo_{name}_photo.ppm"
             for name in ("cap", "liner", "door")}
    stamp = OUT / "fuelcap_polo_photo.recipe"
    if all(path.exists() for path in names.values()) and stamp.exists() \
            and stamp.read_text(encoding="ascii") == RECIPE:
        return names
    if not REF.exists():
        missing = ", ".join(str(p) for p in names.values() if not p.exists())
        raise FileNotFoundError(f"Polo photo maps are missing on this host: {missing}")

    src = Image.open(REF).convert("RGB")
    cap = src.crop((585, 315, 925, 655)).resize((1024, 1024), Image.Resampling.LANCZOS)
    door = src.crop((895, 135, 1295, 945)).resize((1024, 1024), Image.Resampling.LANCZOS)
    _linear_ppm(cap, names["cap"], gain=0.86)
    _linear_ppm(_liner(src), names["liner"], gain=0.74)
    _linear_ppm(door, names["door"], gain=0.88)
    stamp.write_text(RECIPE, encoding="ascii")
    return names
