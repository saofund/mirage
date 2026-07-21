"""Procedural PBR texture-set generator — the starter material library.

Each material is written as three image maps the path tracer samples (triplanar, no UVs):

    <name>_albedo.ppm    base colour        (P6 / sRGB)
    <name>_rough.ppm     roughness          (P5 / linear, 0=mirror .. 255=matte)
    <name>_normal.ppm    tangent-space normal (P6 / linear, derived from a height field)

They are plain uncompressed PPM so the C++ core reads them with a tiny parser and no image
decoder — and when real CC0 PBR sets are dropped in (same three files), they work unchanged.
The maps are deterministic (seeded) and tileable, so triplanar projection doesn't seam.

    uv run python -m mirage.textures            # (re)generate the whole library -> assets/textures/
    from mirage.textures import ensure_textures  # generate any missing sets on demand

Needs numpy + Pillow.
"""
from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TEX_DIR = ROOT / "assets" / "textures"

RES = 1024   # map resolution


# --------------------------------------------------------------------------- #
# tileable value-noise fBm (periodic lattice so the maps wrap seamlessly)
# --------------------------------------------------------------------------- #
def _periodic_noise(res: int, period: int, seed: int) -> np.ndarray:
    """A single octave of tileable value noise at integer `period` cells across the map."""
    rng = np.random.default_rng(seed)
    lattice = rng.random((period, period), dtype=np.float64)
    # sample coords in [0,period) with wrap
    t = (np.arange(res) / res * period)
    xi = np.floor(t).astype(int)
    xf = t - xi
    x0, x1 = xi % period, (xi + 1) % period
    sx = xf * xf * (3 - 2 * xf)                 # smoothstep
    # bilinear on the wrapped lattice: element [i,j] blends the 4 cells around (i,j)
    Ax = lattice[x0][:, x0]
    Bx = lattice[x1][:, x0]
    Ay = lattice[x0][:, x1]
    By = lattice[x1][:, x1]
    sxx, syy = sx[:, None], sx[None, :]
    top = Ax * (1 - sxx) + Bx * sxx
    bot = Ay * (1 - sxx) + By * sxx
    return top * (1 - syy) + bot * syy


def _fbm(res: int, base_period: int, octaves: int, seed: int, gain: float = 0.5) -> np.ndarray:
    out = np.zeros((res, res), dtype=np.float64)
    amp, tot, p = 1.0, 0.0, base_period
    for o in range(octaves):
        out += amp * _periodic_noise(res, p, seed + o * 101)
        tot += amp
        amp *= gain
        p *= 2
    return out / max(tot, 1e-9)


def _normal_from_height(h: np.ndarray, strength: float) -> np.ndarray:
    """Tangent-space normal map (RGB in [0,1]) from a height field, via wrapped gradients."""
    dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * 0.5
    dy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) * 0.5
    nx, ny, nz = -dx * strength, -dy * strength, np.ones_like(h)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx * inv, ny * inv, nz * inv
    return np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)


def _write_ppm_rgb(path: Path, rgb01: np.ndarray) -> None:
    arr = (np.clip(rgb01, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr, "RGB").save(path, format="PPM")


def _write_ppm_gray(path: Path, g01: np.ndarray) -> None:
    arr = (np.clip(g01, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(arr, "L").save(path, format="PPM")


def _lerp(a, b, t):
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# the materials
# --------------------------------------------------------------------------- #
def _wood(res: int, seed: int, col_a, col_b, plank=6, warp_amt=2.2, grain_freq=26):
    """Planked wood: grain streaks along y, warped by noise, with plank seams + tone variation.
    warp_amt controls how figured the grain is (low = straight planks, high = burl/walnut)."""
    y = np.linspace(0, 1, res)[:, None] * np.ones((1, res))
    x = np.ones((res, 1)) * np.linspace(0, 1, res)[None, :]
    warp = _fbm(res, 4, 4, seed) - 0.5
    # grain: many fine rings along x, warped by noise
    grain = np.sin((x * grain_freq + warp * warp_amt) * np.pi * 2)
    grain = 0.5 + 0.5 * grain
    grain = grain ** 1.6
    fine = _fbm(res, 64, 3, seed + 7)
    # planks: bands along x, each plank a slightly different tone + offset
    plank_id = np.floor(x * plank).astype(int)
    ptone = (np.sin(plank_id * 12.9898) * 43758.5)
    ptone = (ptone - np.floor(ptone))
    seam = np.abs(((x * plank) % 1.0) - 0.5) * 2  # 0 at seam center.. 1 mid-plank
    seam_line = np.clip((0.06 - (1 - seam)) * 16, 0, 1)  # dark thin groove at seams
    tone = _lerp(0.82, 1.06, ptone)
    base = np.stack(col_a, -1)[None, None] * (1 - grain[..., None]) + np.stack(col_b, -1)[None, None] * grain[..., None]
    base = base * (tone[..., None]) * (0.9 + 0.2 * fine[..., None])
    base = base * (1 - 0.55 * seam_line[..., None])
    albedo = np.clip(base, 0, 1)
    # roughness: varnished (low) but grain + seams rougher
    rough = 0.32 + 0.20 * grain + 0.28 * seam_line + 0.06 * (fine - 0.5)
    rough = np.clip(rough, 0.12, 0.85)
    # height: grain grooves + seam grooves
    height = grain * 0.35 + (1 - seam_line) * 0 - seam_line * 0.8 + fine * 0.1
    normal = _normal_from_height(height, strength=2.2)
    return albedo, rough, normal


def _veneer(res: int, seed: int, col_a, col_b, grain_freq=16, warp_amt=1.5, figure=0.6):
    """Sliced wood veneer: continuous figured grain with NO plank seams — what a piece of
    moulded plywood is actually faced with. `_wood` lays down planks and seam grooves,
    which is right for a floor and wrong for a shell (it reads as tiger stripes).

    A low-frequency 'figure' term drifts the grain spacing, giving the cathedral flare
    that makes walnut read as walnut rather than as a sine wave.
    """
    x = np.ones((res, 1)) * np.linspace(0, 1, res)[None, :]
    warp = _fbm(res, 3, 5, seed) - 0.5
    fig = _fbm(res, 2, 3, seed + 13) - 0.5
    grain = np.sin((x * grain_freq + warp * warp_amt + fig * figure * 3.0) * np.pi * 2)
    grain = (0.5 + 0.5 * grain) ** 1.9
    fine = _fbm(res, 90, 3, seed + 7)
    t = np.clip(grain * 0.85 + 0.15 * fine, 0, 1)
    base = (np.stack(col_a, -1)[None, None] * (1 - t[..., None])
            + np.stack(col_b, -1)[None, None] * t[..., None])
    base = base * (0.93 + 0.14 * fine[..., None])
    albedo = np.clip(base, 0, 1)
    rough = np.clip(0.26 + 0.16 * t + 0.05 * (fine - 0.5), 0.14, 0.55)   # satin lacquer
    height = t * 0.25 + fine * 0.08
    normal = _normal_from_height(height, strength=0.9)
    return albedo, rough, normal


def _fabric(res: int, seed: int, col, weave=180):
    """Woven fabric: over/under thread pattern (bumpy normal) + fuzz mottle, matte."""
    x = np.ones((res, 1)) * np.linspace(0, 1, res)[None, :]
    y = np.linspace(0, 1, res)[:, None] * np.ones((1, res))
    warp = np.sin(x * weave * np.pi)          # vertical threads
    weft = np.sin(y * weave * np.pi)          # horizontal threads
    # weave height: warp on top where warp>weft, else weft
    over = (warp >= weft)
    height = np.where(over, 0.5 + 0.5 * warp, 0.5 + 0.5 * weft)
    fuzz = _fbm(res, 96, 4, seed)
    height = height * 0.8 + fuzz * 0.2
    base = np.stack(col, -1)[None, None] * (0.82 + 0.36 * fuzz[..., None])
    # threads catch light slightly differently by direction
    base = base * (0.9 + 0.12 * np.where(over, warp, weft)[..., None])
    albedo = np.clip(base, 0, 1)
    rough = np.clip(0.82 + 0.10 * (fuzz - 0.5) - 0.06 * height, 0.6, 0.98)
    normal = _normal_from_height(height, strength=1.6)
    return albedo, rough, normal


def _plaster(res: int, seed: int, col):
    """Wall plaster: near-flat colour, subtle orange-peel surface, matte."""
    mott = _fbm(res, 24, 5, seed)
    micro = _fbm(res, 160, 3, seed + 5)
    base = np.stack(col, -1)[None, None] * (0.94 + 0.10 * mott[..., None])
    albedo = np.clip(base, 0, 1)
    rough = np.clip(0.86 + 0.08 * (mott - 0.5), 0.7, 0.96) * np.ones((res, res))
    height = mott * 0.5 + micro * 0.5
    normal = _normal_from_height(height, strength=0.7)
    return albedo, rough, normal


def _leather(res: int, seed: int, col, grain=88):
    """Aniline leather: a pebbled grain of soft cells divided by darker creases, over a
    broad sag. The creases are what sell it — they sit lower, take a rougher sheen and
    catch less light, which is the whole difference between leather and dark plastic.

    The cell divisions come from RIDGED noise (1 - |2n-1|), whose valleys form the
    connected crease network a Voronoi would give, without needing one.
    """
    cell = _fbm(res, grain, 3, seed)                                       # the pebbles
    ridged = 1.0 - np.abs(2.0 * _fbm(res, max(grain // 2, 2), 2, seed + 7) - 1.0)
    crease = np.clip(ridged, 0, 1) ** 2.4                                  # crease network
    wrinkle = _fbm(res, 7, 4, seed + 3)                                    # broad sag
    micro = _fbm(res, 300, 2, seed + 11)
    height = 0.60 * cell + 0.24 * wrinkle + 0.16 * micro - 0.55 * crease
    height = (height - height.min()) / (height.max() - height.min() + 1e-9)
    base = np.stack(col, -1)[None, None] * (0.78 + 0.46 * height[..., None])
    albedo = np.clip(base, 0, 1)
    rough = np.clip(0.32 + 0.28 * crease + 0.12 * (1.0 - height), 0.20, 0.74)
    normal = _normal_from_height(height, strength=1.5)
    return albedo, rough, normal


def _marble(res: int, seed: int, col_a, col_b):
    """Polished marble: turbulent veins, low roughness."""
    turb = _fbm(res, 6, 6, seed)
    x = np.ones((res, 1)) * np.linspace(0, 1, res)[None, :]
    vein = np.sin((x * 5 + turb * 4.5) * np.pi * 2)
    vein = np.abs(vein) ** 0.35
    t = np.clip(1 - vein, 0, 1)
    base = np.stack(col_a, -1)[None, None] * (1 - t[..., None]) + np.stack(col_b, -1)[None, None] * t[..., None]
    albedo = np.clip(base * (0.95 + 0.1 * _fbm(res, 40, 3, seed + 3)[..., None]), 0, 1)
    rough = np.clip(0.16 + 0.12 * t, 0.1, 0.4)
    height = t * 0.3 + turb * 0.1
    normal = _normal_from_height(height, strength=0.5)
    return albedo, rough, normal


# --------------------------------------------------------------------------- #
# outdoor / forecourt materials — the wet petrol-station floor
# --------------------------------------------------------------------------- #
def _crack_net(res: int, seed: int, period: int, thresh: float, sharp: float = 12.0):
    """A connected network of thin cracks in [0,1] (1 = crack). The valleys of ridged noise
    form a branching web the way a real crack does — one line splitting, not scattered dots."""
    ridged = 1.0 - np.abs(2.0 * _fbm(res, period, 4, seed) - 1.0)
    return np.clip((ridged - thresh) * sharp, 0, 1)


def _concrete(res: int, seed: int, col, crack=0.7, stain=0.8, wet=0.7, rough_base=0.72):
    """Damp concrete apron: broad tonal mottle, fine aggregate, a web of hairline cracks and
    dark oil/water staining pooled in the low spots. `wet` drops roughness inside the stains so
    they read as a wet sheen rather than more grey paint — the contrast is the wetness.
    `rough_base` sets the dry matte level: ~0.72 for concrete, lower for painted cladding."""
    mott = _fbm(res, 6, 4, seed)                       # slab-scale tone variation (gentle)
    fine = _fbm(res, 150, 3, seed + 5)                 # aggregate speckle
    cracks = _crack_net(res, seed + 9, period=5, thresh=0.91) * crack
    st = _fbm(res, 4, 4, seed + 17)                    # damp areas
    # A real wet-concrete apron is nearly uniform grey: the variation you see is the SKY
    # mirrored in the damp patches, i.e. a roughness cue, not a dark-blob albedo camo.
    wet_patch = np.clip((0.44 - st) * 1.8, 0, 1) ** 1.5
    tone = 0.92 + 0.13 * mott + 0.06 * (fine - 0.5)
    base = np.stack(col, -1)[None, None] * tone[..., None]
    base = base * (1 - 0.14 * stain * wet_patch[..., None])   # damp only slightly darker in albedo
    base = base * (1 - 0.50 * cracks[..., None])              # joints/cracks darkest
    albedo = np.clip(base, 0, 1)
    rough = rough_base - 0.06 * (mott - 0.5) - wet * 0.50 * wet_patch + 0.06 * cracks
    rough = np.clip(rough, 0.10, 0.98)
    height = mott * 0.18 + fine * 0.16 - cracks * 0.8
    normal = _normal_from_height(height, strength=0.9)
    return albedo, rough, normal


def _asphalt(res: int, seed: int, col):
    """Wet asphalt road: fine light aggregate flecked through a near-black matrix, low
    roughness so the overcast sky mirrors in it as a cool sheen."""
    grain = _fbm(res, 210, 2, seed)                    # tight speckle = the stones
    stones = np.clip((grain - 0.62) * 4.0, 0, 1)
    mott = _fbm(res, 6, 3, seed + 3)
    base = np.stack(col, -1)[None, None] * (0.7 + 0.6 * mott[..., None])
    base = base + stones[..., None] * 0.055            # grey stone flecks lift the black
    albedo = np.clip(base, 0, 1)
    rough = np.clip(0.14 + 0.50 * stones - 0.06 * (mott - 0.5), 0.06, 0.60)
    height = grain * 0.40 + mott * 0.10
    normal = _normal_from_height(height, strength=0.7)
    return albedo, rough, normal


def _painted_bay(res: int, seed: int, paint, concrete, wet=0.6, faded=0.0):
    """A weathered painted forecourt bay. The paint is worn through to the concrete in patches,
    fine-cracked, and — the point of the whole exercise — stained with big ORGANIC dark wet
    lobes pooled in the low spots. The photo's blue bay is that: irregular standing water, not a
    rectangle nested inside a rectangle. `wet` sets how glossy the pools are, `faded` dulls the
    dry paint. All the weathering lives here so the scene can lay ONE slab, not a stack."""
    mott = _fbm(res, 12, 4, seed)                      # paint laid unevenly
    fine = _fbm(res, 150, 3, seed + 5)
    wear = np.clip((_fbm(res, 18, 4, seed + 11) - 0.64) * 3.0, 0, 1) ** 1.5   # rubbed to concrete
    # ONE big soft pool per tile, like the photo's dark wet SHEET across the blue bay -- not a
    # scatter of little puddles. Low frequency (period 2) gives a single dominant lobe.
    st = _fbm(res, 2, 4, seed + 17)
    wet_mask = np.clip((0.55 - st) * 2.1, 0, 1) ** 1.2
    # only a few hairline cracks on a bay — the WET SHEET is the story, not a mud-crack web
    cracks = _crack_net(res, seed + 23, period=8, thresh=0.91) * 0.55
    paint_c = np.stack(paint, -1)[None, None]
    conc_c = np.stack(concrete, -1)[None, None]
    col = _lerp(paint_c, conc_c, wear[..., None])                   # worn paint -> concrete
    col = col * (0.86 + 0.28 * mott[..., None]) * (0.94 + 0.12 * (fine[..., None] - 0.5))
    col = _lerp(col, col * 0.26, wet_mask[..., None])              # the dark wet sheet
    col = col * (1 - faded * 0.28)
    col = col * (1 - 0.42 * cracks[..., None])
    albedo = np.clip(col, 0, 1)
    rough = 0.60 + 0.18 * wear - 0.10 * (mott - 0.5)               # matte paint, rougher where worn
    rough = _lerp(rough, np.full_like(rough, 0.09), wet * wet_mask)  # near-mirror wet sheet
    rough = np.clip(rough, 0.07, 0.92)
    height = mott * 0.20 + fine * 0.15 - cracks * 0.6 - wet_mask * 0.30 - wear * 0.10
    normal = _normal_from_height(height, strength=1.0)
    return albedo, rough, normal


# name -> generator thunk
_LIBRARY = {
    "wood_floor":  lambda: _wood(RES, 11, (0.30, 0.18, 0.09), (0.52, 0.34, 0.18), plank=7, warp_amt=0.7, grain_freq=34),
    "wood_walnut": lambda: _wood(RES, 23, (0.16, 0.09, 0.05), (0.34, 0.20, 0.11), plank=4, warp_amt=1.9),
    "wood_oak":    lambda: _wood(RES, 31, (0.40, 0.28, 0.15), (0.62, 0.46, 0.27), plank=5, warp_amt=1.0, grain_freq=30),
    "fabric_sofa": lambda: _fabric(RES, 41, (0.32, 0.40, 0.32)),
    "fabric_cush": lambda: _fabric(RES, 47, (0.72, 0.54, 0.24)),
    "fabric_rug":  lambda: _fabric(RES, 53, (0.46, 0.24, 0.20), weave=120),
    # Real black leather sits around 0.04 albedo — its grain reads through the SPECULAR,
    # not the base colour. Anything lighter tonemaps to a flat mid-grey under a bright sky.
    "leather":     lambda: _leather(RES, 59, (0.045, 0.040, 0.043)),
    # Walnut veneer wants a NARROW tonal range — the grain is a whisper, not a zebra. A wide
    # col_a..col_b spread reads as painted stripes however good the normal map is.
    "wood_veneer": lambda: _veneer(RES, 83, (0.105, 0.052, 0.028), (0.175, 0.093, 0.050)),
    "plaster":     lambda: _plaster(RES, 61, (0.84, 0.80, 0.73)),
    "marble":      lambda: _marble(RES, 71, (0.86, 0.85, 0.82), (0.42, 0.44, 0.48)),
    # the wet petrol-station forecourt (case 26). Albedos are honest surface colours; the
    # scene's dark wet look comes from the roughness pools mirroring a bright overcast sky.
    "forecourt_concrete": lambda: _concrete(RES, 101, (0.30, 0.31, 0.32), crack=0.35, stain=0.5, wet=0.7),
    "asphalt_wet":        lambda: _asphalt(RES, 107, (0.050, 0.055, 0.063)),
    "bay_blue":           lambda: _painted_bay(RES, 113, (0.075, 0.105, 0.185), (0.17, 0.18, 0.19), wet=0.85),
    "bay_orange":         lambda: _painted_bay(RES, 127, (0.355, 0.150, 0.065), (0.32, 0.29, 0.25), wet=0.40, faded=0.40),
    # painted metal cladding for the canopy column / facade: light cool grey, faint panel seams
    # (the concrete crack net, kept sparse), a semi-gloss sheen rather than matte concrete.
    "clad_panel":         lambda: _concrete(RES, 137, (0.52, 0.535, 0.55), crack=0.30, stain=0.22, wet=0.30, rough_base=0.50),
}


def _paths(name: str, tex_dir: Path) -> dict:
    return {k: tex_dir / f"{name}_{k}.ppm" for k in ("albedo", "rough", "normal")}


def generate(name: str, tex_dir: Path = TEX_DIR) -> dict:
    """Generate one material's three maps; return {'albedo':path,'rough':path,'normal':path}."""
    if name not in _LIBRARY:
        raise KeyError(f"unknown texture '{name}'; have {sorted(_LIBRARY)}")
    tex_dir.mkdir(parents=True, exist_ok=True)
    albedo, rough, normal = _LIBRARY[name]()
    p = _paths(name, tex_dir)
    _write_ppm_rgb(p["albedo"], albedo)
    _write_ppm_gray(p["rough"], rough)
    _write_ppm_rgb(p["normal"], normal)
    return p


def _recipe_id(name: str) -> str:
    """A short digest of the recipe that produced a map set — the generator's own bytecode
    plus this entry's constants. Changes iff the recipe changes."""
    fn = _LIBRARY[name]
    parts = [name, str(RES)]
    # the thunk's captured constants (colours, seeds, plank counts...)
    parts += [repr(c) for c in (fn.__code__.co_consts or ()) if c is not None]
    # and the generator it calls, so editing _leather() alone still invalidates
    for gen in (_wood, _veneer, _fabric, _plaster, _leather, _marble, _concrete, _asphalt,
                _painted_bay, _crack_net, _normal_from_height, _fbm):
        parts.append(gen.__name__)
        parts.append(hashlib.sha1(gen.__code__.co_code).hexdigest()[:8])
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def ensure_textures(names, tex_dir: Path = TEX_DIR) -> dict:
    """Generate any of `names` whose maps are missing OR STALE; return {name: {map: path}}.

    Staleness matters more than it sounds. These maps are gitignored and regenerated on
    demand, so "the file exists" was the only cache key — and editing a recipe then left the
    OLD map on disk, silently. Changing the leather from 0.16 to 0.045 albedo and seeing no
    difference in the render is a genuinely baffling half hour: the code is right, the
    picture is wrong, and nothing anywhere says why. The recipe's digest is stored beside
    the maps, so a recipe edit regenerates them.
    """
    out = {}
    for name in names:
        p = _paths(name, tex_dir)
        stamp = tex_dir / f"{name}.recipe"
        want = _recipe_id(name)
        fresh = (all(f.exists() for f in p.values())
                 and stamp.exists() and stamp.read_text().strip() == want)
        if not fresh:
            p = generate(name, tex_dir)
            stamp.write_text(want)
        out[name] = p
    return out


def main():
    for name in _LIBRARY:
        p = generate(name)
        print(f"  {name:14s} -> {p['albedo'].name}, {p['rough'].name}, {p['normal'].name}")
    print(f"  wrote {len(_LIBRARY)} material sets to {TEX_DIR}")


if __name__ == "__main__":
    main()
