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


# name -> generator thunk
_LIBRARY = {
    "wood_floor":  lambda: _wood(RES, 11, (0.30, 0.18, 0.09), (0.52, 0.34, 0.18), plank=7, warp_amt=0.7, grain_freq=34),
    "wood_walnut": lambda: _wood(RES, 23, (0.16, 0.09, 0.05), (0.34, 0.20, 0.11), plank=4, warp_amt=1.9),
    "wood_oak":    lambda: _wood(RES, 31, (0.40, 0.28, 0.15), (0.62, 0.46, 0.27), plank=5, warp_amt=1.0, grain_freq=30),
    "fabric_sofa": lambda: _fabric(RES, 41, (0.32, 0.40, 0.32)),
    "fabric_cush": lambda: _fabric(RES, 47, (0.72, 0.54, 0.24)),
    "fabric_rug":  lambda: _fabric(RES, 53, (0.46, 0.24, 0.20), weave=120),
    "plaster":     lambda: _plaster(RES, 61, (0.84, 0.80, 0.73)),
    "marble":      lambda: _marble(RES, 71, (0.86, 0.85, 0.82), (0.42, 0.44, 0.48)),
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


def ensure_textures(names, tex_dir: Path = TEX_DIR) -> dict:
    """Generate any of `names` whose maps are missing; return {name: {map: path}}."""
    out = {}
    for name in names:
        p = _paths(name, tex_dir)
        if not all(f.exists() for f in p.values()):
            p = generate(name, tex_dir)
        out[name] = p
    return out


def main():
    for name in _LIBRARY:
        p = generate(name)
        print(f"  {name:14s} -> {p['albedo'].name}, {p['rough'].name}, {p['normal'].name}")
    print(f"  wrote {len(_LIBRARY)} material sets to {TEX_DIR}")


if __name__ == "__main__":
    main()
