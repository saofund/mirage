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


def _streaks(res: int, seed: int, period: int, length: int, angle_frac: float = 0.18):
    """High-frequency noise SMEARED along one direction — tyre tracks, broom finish, drag
    marks. The thing a stack of isotropic fBm octaves cannot make and a real slab is covered
    in: its fine detail has a grain, because everything that made it was moving."""
    n = _fbm(res, period, 2, seed)
    # Smeared along the texture's Y, which triplanar maps to world Y — the direction cars
    # drive in this scene. Smearing along X put the tracks across the traffic, which is a
    # detail nobody would name and everybody would feel.
    out = np.zeros_like(n)
    for k in range(length):
        out += np.roll(np.roll(n, k, axis=0), int(k * angle_frac), axis=1)
    return out / length


def _concrete(res: int, seed: int, col, crack=0.7, stain=0.8, wet=0.7, rough_base=0.72,
              contrast=1.0):
    """Damp concrete apron: pour-to-pour patchiness, staining, aggregate, hairline cracks.

    `contrast` scales the whole tonal spread, and it exists because the first version of
    this generator had almost none. Measured against the reference over the same patch of
    ground, the render carried 13% of the photograph's tonal VARIATION — mean 0.558 against
    0.518, which is right, on a standard deviation of 0.013 against 0.101, which is a flat
    plane. Nothing about the colour was wrong. There simply was not any.

    Real ground varies at three scales and the old `0.92 + 0.13*mott` had only one: whole
    slabs poured on different days, staining and traffic over the top of that, and aggregate
    under both. `wet` still drops roughness inside the damp areas so they read as sky
    mirrored rather than as grey paint — that part was always right."""
    # Four bands, and the MIDDLE one does the visible work. A map's own standard deviation
    # is measured over its whole tile; what reaches the screen over a three-metre patch is
    # only the part of it whose features are smaller than three metres. Turning up a period-3
    # term (three-metre blotches at this uv_scale) barely moved the render at all.
    pour = _fbm(res, 3, 2, seed + 31)                  # slab-to-slab: different days, different mixes
    mott = _fbm(res, 9, 4, seed)                       # staining and traffic
    patch = _fbm(res, 26, 3, seed + 53)                # repairs, spills, scuffs: ~20 cm
    fine = _fbm(res, 150, 3, seed + 5)                 # aggregate speckle
    # SPILLS have edges. Smooth noise, however much of it, reads as cloud or marble — a
    # real oil stain has a boundary where the liquid stopped, and thresholding a noise field
    # is what puts one there. Two scales: big spills and a scatter of drips.
    spill = np.clip((_fbm(res, 7, 3, seed + 71) - 0.60) * 9.0, 0, 1)
    # Thresholded from a STREAKED field, not a round one. Cut from isotropic fBm these came
    # out as a scatter of dark ovals -- which read as fallen leaves, and became the loudest
    # thing on the ground the moment the blob terms were turned down around them. What marks
    # a forecourt is dragged, not dropped.
    drips = np.clip((_streaks(res, seed + 89, 44, 34, angle_frac=0.42) - 0.522) * 16.0, 0, 1)
    cracks = _crack_net(res, seed + 9, period=5, thresh=0.91) * crack
    st = _fbm(res, 4, 4, seed + 17)                    # damp areas
    wet_patch = np.clip((0.44 - st) * 1.8, 0, 1) ** 1.5
    # THE SHAPE MATTERS MORE THAN THE AMOUNT, which took two rounds to learn. First the
    # energy was in 1-3 m blobs and the ground was leopard print. Then the blobs came down,
    # the streaks went up, and a band-by-band measurement said the render now carried
    # 1.07/0.99/1.11 of the photograph's variation at 3-8, 8-20 and 20-50 px. The right
    # amount, and it still looked like granite -- because that statistic cannot tell a long
    # thin mark from a round blob, and the reference's variation is almost entirely LINEAR:
    # tyre tracks a couple of metres long, hairline cracks, straight slab joints, drag marks.
    # Isotropic fBm cannot make any of those however carefully it is weighted, so the three
    # blob terms drop to a fifth and the streaks get long enough to read as tracks: at this
    # resolution and uv_scale, 17 cm wide and about 1.7 m long.
    scratch = _streaks(res, seed + 61, 150, 70)
    tracks = _streaks(res, seed + 97, 26, 210, angle_frac=0.35)
    tone = (1.0 + contrast * (0.030 * (pour - 0.5)
                              + 0.030 * stain * (mott - 0.5)
                              + 0.035 * stain * (patch - 0.5)
                              + 0.110 * (fine - 0.5)
                              + 0.360 * (scratch - 0.5)
                              + 0.440 * stain * (tracks - 0.5)))
    base = np.stack(col, -1)[None, None] * np.clip(tone, 0.35, 1.9)[..., None]
    base = base * (1 - 0.22 * stain * wet_patch[..., None])
    base = base * (1 - 0.46 * stain * spill[..., None])       # oil, with an edge
    base = base * (1 - 0.14 * drips[..., None])
    base = base * (1 - 0.50 * cracks[..., None])              # joints/cracks darkest
    albedo = np.clip(base, 0, 1)
    # MOST OF THE VARIATION LIVES HERE, not in the albedo. On a wet apron under an overcast
    # dome the specular term is large and — if the roughness is uniform — the SAME everywhere,
    # so it adds a constant that dilutes whatever contrast the albedo has. Measured: the map
    # carried 12% relative spread in albedo and the render showed 4% on screen. What makes a
    # real wet forecourt blotchy is that some of it is wetter than the rest, which is a
    # roughness field, and it modulates the big term instead of the small one.
    rough = (rough_base - 0.06 * (mott - 0.5) - 0.14 * (patch - 0.5)
             - 0.16 * (tracks - 0.5) - wet * 0.55 * wet_patch
             - 0.28 * spill - 0.16 * drips + 0.06 * cracks)
    rough = np.clip(rough, 0.10, 0.98)
    height = mott * 0.10 + fine * 0.22 + scratch * 0.18 + tracks * 0.12 - cracks * 0.8
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


def _painted_bay(res: int, seed: int, paint, concrete, wet=0.6, faded=0.0, wear_amt=1.0,
                 wet_cover=0.44):
    """A weathered painted forecourt bay. The paint is worn through to the concrete in patches,
    fine-cracked, and — the point of the whole exercise — stained with big ORGANIC dark wet
    lobes pooled in the low spots. The photo's blue bay is that: irregular standing water, not a
    rectangle nested inside a rectangle. `wet` sets how glossy the pools are, `faded` dulls the
    dry paint. All the weathering lives here so the scene can lay ONE slab, not a stack."""
    mott = _fbm(res, 12, 4, seed)                      # paint laid unevenly
    patch = _fbm(res, 30, 3, seed + 53)                # scuffs and spills, ~20 cm across
    fine = _fbm(res, 150, 3, seed + 5)
    # rubbed through to concrete. `wear_amt` matters more than it looks: the pale patches it
    # leaves catch the sky at low roughness and read as light BLUE flecks, so a bay with the
    # wear turned up comes out tie-dyed rather than worn.
    wear = np.clip((_fbm(res, 18, 4, seed + 11) - 0.64) * 3.0, 0, 1) ** 1.5 * wear_amt
    # ONE big soft pool per tile, like the photo's dark wet SHEET across the blue bay -- not a
    # scatter of little puddles. Low frequency (period 2) gives a single dominant lobe.
    st = _fbm(res, 2, 4, seed + 17)
    # Softer and less contrasty than it was: at (0.55-st)*2.1 darkening to 26%% the lobes
    # read as marbling — tie-dye — rather than as water lying on paint.
    # `wet_cover` is the fraction of the tile the water reaches. The blue bay in the
    # reference is not a bay with puddles ON it -- it is a bay UNDER a sheet, edge to edge,
    # with the shop reflected in it. One lobe covering 40%% of the tile cannot say that.
    wet_mask = np.clip((wet_cover - st) * 1.3, 0, 1) ** 1.5
    # HAIRLINE. At period 8 and a 0.91 threshold the cells are half a metre across and the
    # lines between them are centimetres WIDE, which is not a crack network -- it is marble
    # veining, and magnified beside the reference it was the loudest thing on the bay. A real
    # bay's cracks are sub-millimetre; what the photograph actually shows at this scale is
    # paint PEELING, and that is the `wear` term above, not this one.
    cracks = _crack_net(res, seed + 23, period=8, thresh=0.964) * 0.55
    # Hard-edged marks: a tyre scuff and a spill both END somewhere. Thresholded from a
    # STREAKED field for the same reason the concrete's drips are -- cut from round noise
    # they are a scatter of ovals, and paint is marked by things sliding across it.
    marks = np.clip((_streaks(res, seed + 67, 20, 120, angle_frac=0.30) - 0.516) * 13.0, 0, 1)
    paint_c = np.stack(paint, -1)[None, None]
    conc_c = np.stack(concrete, -1)[None, None]
    # TYRE TRACKS. Cars enter a bay along its length and always in the same two ruts, so a
    # painted bay is darker in two broad bands and cleaner between them. Nothing stochastic
    # produces that, and its absence is a large part of why a bay reads as a coloured
    # rectangle: the wear on a real one has a DIRECTION.
    u = np.linspace(0.0, 1.0, res)[None, :] * np.ones((res, 1))
    ruts = (np.exp(-((u - 0.30) / 0.085) ** 2) + np.exp(-((u - 0.68) / 0.085) ** 2))
    ruts = ruts * (0.55 + 0.45 * _fbm(res, 5, 3, seed + 41))       # they fade in and out
    col = _lerp(paint_c, conc_c, wear[..., None])                   # worn paint -> concrete
    # Three scales, not one. Measured against the reference the old single mottle carried
    # 13% of the photograph's tonal variation over the same patch of bay — the right colour
    # spread across a plane with nothing on it.
    # A map's contrast is roughly HALVED on its way to the screen: the ACES curve compresses
    # around mid grey, and a uniform sky specular adds a constant on top that dilutes what is
    # left. Measured end to end, an 11.5% relative spread in the map arrived as 4.6% in the
    # render. So the map has to be authored louder than the answer, and the factor is about
    # 2.5. (The denoiser was the obvious suspect and was innocent: rendered at 900 spp with
    # the filter OFF the ground's spread was 0.0201 against 0.0238 with it ON — slightly
    # LOWER, because demodulate/remodulate sharpens albedo edges rather than blurring them.)
    # ...and the same correction the concrete needed, for the same reason. Getting the
    # AMOUNT of variation right left the bay marbled, because fBm's variation is round and a
    # painted bay's is dragged: ruts, scuffs, sweep marks, the edge where a squeegee stopped.
    # So the two blob terms drop to a third and the streak terms carry it, long enough to
    # cross a good part of the bay.
    bstreak = _streaks(res, seed + 61, 110, 90)
    bdrag = _streaks(res, seed + 79, 30, 170, angle_frac=0.22)
    col = col * (1.0 + 0.075 * (mott[..., None] - 0.5) + 0.085 * (patch[..., None] - 0.5)
                 + 0.150 * (fine[..., None] - 0.5) + 0.300 * (bstreak[..., None] - 0.5)
                 + 0.260 * (bdrag[..., None] - 0.5))
    col = col * (1 - 0.26 * ruts[..., None])                        # the ruts
    col = col * (1 - 0.24 * marks[..., None])                       # scuffs, with edges
    # The darkening now scales with `wet`. It was a flat 0.52 whatever the bay, so a DRY
    # terracotta got the same soft dark lobes as the standing water on the blue one, and read
    # as cloud shadows on paint. At wet=0.88 this is what it always was; at 0.45 it is half.
    col = _lerp(col, col * (1.0 - 0.55 * wet), wet_mask[..., None])   # the dark wet sheet
    col = col * (1 - faded * 0.28)
    col = col * (1 - 0.20 * cracks[..., None])
    albedo = np.clip(col, 0, 1)
    rough = (0.60 + 0.18 * wear - 0.14 * (mott - 0.5) - 0.30 * (patch - 0.5)
             + 0.10 * ruts)                                     # matte paint, rougher where worn
    rough = _lerp(rough, np.full_like(rough, 0.09), wet * wet_mask)  # near-mirror wet sheet
    rough = np.clip(rough, 0.07, 0.92)
    height = mott * 0.20 + fine * 0.15 - cracks * 0.6 - wet_mask * 0.30 - wear * 0.10 - ruts * 0.12
    normal = _normal_from_height(height, strength=1.0)
    return albedo, rough, normal


def _painted_metal(res: int, seed: int, col, dirt=0.35, rough_base=0.38):
    """Painted sheet-metal cladding — a canopy column, a shop front.

    Deliberately almost featureless: the *only* things a painted panel has are a faint
    roll of the sheet, a semi-gloss sheen and grime washed down it by rain. Reaching for
    the concrete generator here was a mistake worth naming — its crack network reads as
    marble veining at any scale a 0.7 m column is tiled at, so a painted steel column came
    out looking like a quarried one."""
    roll = _fbm(res, 4, 3, seed)                       # the shallow waviness of a sheet
    fine = _fbm(res, 90, 2, seed + 7)
    streak = _fbm(res, 3, 4, seed + 13)                # rain-washed vertical grime
    streak = np.clip((streak - 0.48) * 1.7, 0, 1) * dirt
    streak = streak * (0.35 + 0.65 * np.linspace(1.0, 0.0, res)[:, None])   # heavier low down
    base = np.stack(col, -1)[None, None] * (0.96 + 0.09 * roll[..., None])
    base = base * (0.985 + 0.03 * (fine[..., None] - 0.5))
    albedo = np.clip(base * (1.0 - 0.22 * streak[..., None]), 0, 1)
    rough = np.clip(rough_base + 0.10 * (roll - 0.5) + 0.22 * streak, 0.16, 0.85)
    normal = _normal_from_height(roll * 0.30 + fine * 0.06, strength=0.45)
    return albedo, rough, normal


def _moulded_plastic(res: int, seed: int, col, rough_base=0.68, wear=0.35):
    """Fine injection-moulded grain with handling polish and settled pocket dust.

    This is intentionally a micro-surface, not stone noise painted black. Automotive
    mouldings are nearly uniform in albedo; their scale comes from the broken specular
    lobe and from a little dust caught in the grain. Broad colour clouds make a small
    part look marbled, while a flat roughness makes the same part look like CAD rubber.
    """
    grain = _fbm(res, 72, 3, seed)
    fine = _fbm(res, 150, 2, seed + 7)
    handling = _fbm(res, 5, 4, seed + 19)
    dust = np.clip((_fbm(res, 11, 3, seed + 31) - 0.58) * 3.2, 0, 1) * wear
    base = np.stack(col, -1)[None, None]
    albedo = base * (0.94 + 0.10 * grain[..., None] + 0.04 * fine[..., None])
    # Pocket dust shifts black plastic only slightly. At the old 0.018 addition each
    # three-millimetre patch doubled the base albedo and read as dried mud.
    albedo += dust[..., None] * np.array((0.0045, 0.0040, 0.0032))[None, None]
    rough = rough_base + 0.12 * (grain - 0.5) + 0.18 * dust - 0.13 * handling
    height = 0.34 * grain + 0.20 * fine + 0.06 * dust
    return np.clip(albedo, 0, 1), np.clip(rough, 0.34, 0.92), \
        _normal_from_height(height, strength=1.15)


def _automotive_paint(res: int, seed: int, col, rough_base=0.16):
    """Subtle orange peel and metallic-flake variation for a clear-coated body panel."""
    peel = _fbm(res, 46, 3, seed)
    fine = _fbm(res, 130, 2, seed + 11)
    flake = np.clip((fine - 0.57) * 4.0, 0, 1)
    base = np.stack(col, -1)[None, None]
    albedo = base * (0.985 + 0.025 * (peel[..., None] - 0.5))
    albedo += flake[..., None] * 0.018
    rough = rough_base + 0.035 * (peel - 0.5) + 0.020 * flake
    height = 0.30 * peel + 0.04 * fine
    return np.clip(albedo, 0, 1), np.clip(rough, 0.09, 0.25), \
        _normal_from_height(height, strength=0.42)



def _beaded_water(res: int, seed: int, col, rough_base=0.66, count=190, r_lo=0.004,
                  r_hi=0.016, coverage=0.55):
    """A matt surface with WATER BEADS standing on it.

    The Polo reference was photographed in the rain, and the beads are not a detail of that
    picture — they are most of its surface character: a black moulding covered in a few
    hundred tiny near-mirrors, each one returning a hard white point where the substrate
    around it returns almost nothing.

    They cannot be painted into an albedo map. What a bead does is local *roughness* and
    local *normal*: the substrate is 0.66 rough and the water is 0.06, so the bead catches
    a specular highlight the moulding cannot, and its curvature aims that highlight
    somewhere different from its neighbours. Albedo barely changes at all — which is why a
    map that only paints light dots reads as dirt rather than as rain.

    Beads are placed on a jittered grid rather than by rejection sampling, so coverage is
    controllable and the result has no clumps: real beading on a vertical panel is fairly
    even until it starts to run.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32) / res
    height = np.zeros((res, res), np.float32)
    wet = np.zeros((res, res), np.float32)
    n = int(count ** 0.5) + 1
    for gy in range(n):
        for gx in range(n):
            if rng.random() > coverage:
                continue
            cx = (gx + 0.5 + rng.uniform(-0.42, 0.42)) / n
            cy = (gy + 0.5 + rng.uniform(-0.42, 0.42)) / n
            r = float(rng.uniform(r_lo, r_hi))
            dx = np.abs(xx - cx)
            dy = np.abs(yy - cy)
            dx = np.minimum(dx, 1.0 - dx)          # wrap, so the map tiles
            dy = np.minimum(dy, 1.0 - dy)
            d2 = (dx * dx + dy * dy) / (r * r)
            m = d2 < 1.0
            if not m.any():
                continue
            # a spherical cap, not a gaussian: a bead has a definite edge and a flat
            # contact circle, and the edge is where the ring highlight comes from
            height[m] = np.maximum(height[m], np.sqrt(1.0 - d2[m]) * r * 0.62)
            wet[m] = 1.0

    grain = _fbm(res, 90, 3, seed + 5)
    base = np.stack(col, -1)[None, None]
    albedo = base * (0.95 + 0.09 * grain[..., None])
    # Water darkens what it sits on very slightly and adds a trace of its own scatter.
    albedo = albedo * (1.0 - 0.10 * wet[..., None]) + wet[..., None] * 0.004
    rough = rough_base + 0.10 * (grain - 0.5)
    rough = rough * (1.0 - wet) + 0.06 * wet
    # `_normal_from_height` differentiates with a ONE-PIXEL roll, so its input has to be
    # in pixel-comparable units. `height` here is in the same 0..1 units as the bead
    # radii, so a 10 px bead 0.006 tall gives a per-pixel slope of 0.0006 and the map
    # comes out perfectly flat -- which is exactly what it did. Multiplying by `res` puts
    # the height in pixels, where a 10 px bead is 6 px tall and a dome is a dome.
    return np.clip(albedo, 0, 1), np.clip(rough, 0.05, 0.95), \
        _normal_from_height(height * res, strength=1.0)

def _wall_tile(res: int, seed: int, col, grout, tiles=10, grout_w=0.055):
    """A tiled shop front: a real grout GRID, per-tile tone variation, and dirt in the
    joints. A tiled wall's signature is the grid — no amount of noise substitutes for it."""
    u = (np.arange(res) + 0.5) / res * tiles
    gx = np.minimum(u % 1.0, 1.0 - u % 1.0)[None, :]
    gy = np.minimum(u % 1.0, 1.0 - u % 1.0)[:, None]
    joint = np.clip(1.0 - np.minimum(gx, gy) / grout_w, 0, 1) ** 0.7      # 1 in the joint
    rng = np.random.default_rng(seed)
    tone = rng.normal(1.0, 0.035, (tiles, tiles))
    tone = np.repeat(np.repeat(tone, res // tiles + 1, 0), res // tiles + 1, 1)[:res, :res]
    grime = _fbm(res, 5, 3, seed + 3)
    base = np.stack(col, -1)[None, None] * tone[..., None] * (0.93 + 0.14 * grime[..., None])
    albedo = np.clip(_lerp(base, np.stack(grout, -1)[None, None], joint[..., None]), 0, 1)
    rough = np.clip(_lerp(np.full((res, res), 0.30), np.full((res, res), 0.82), joint)
                    + 0.10 * (grime - 0.5), 0.12, 0.92)
    normal = _normal_from_height(-joint * 0.55 + grime * 0.05, strength=1.5)
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
    # A wet forecourt is DARK. The first pass set the concrete at 0.30 linear, which is a
    # dry pavement in bright sun -- against the reference it read as a white floor with
    # brown worms crawling over it, and no amount of roughness work fixes a base colour
    # that is twice as light as the thing it is copying. 0.15 with faint cracks is a damp
    # apron; the contrast then comes, correctly, from the sky mirrored in the wet patches.
    # rough_base 0.46, not the 0.72 default. The whole apron is WET, and a wet surface
    # brightens into the distance because Fresnel climbs steeply at grazing angles — which
    # is exactly what the photograph does (0.53 near, 0.80 at fifteen metres) and exactly
    # what a matte 0.72 cannot do, however dark or light its albedo is set.
    "forecourt_concrete": lambda: _concrete(RES, 101, (0.200, 0.205, 0.208), crack=0.22, stain=0.95, wet=0.90, rough_base=0.46, contrast=2.6),
    # Standing water on the SAME concrete, same seed, so the aggregate and the joints run
    # continuously under it. A puddle modelled as an opaque grey slab reads as a sheet of
    # plastic dropped on the floor -- water does not cover a surface, it darkens it and
    # glazes it, and you go on seeing what is underneath.
    "forecourt_wet":      lambda: _concrete(RES, 101, (0.088, 0.091, 0.096), crack=0.22, stain=0.95, wet=1.0, rough_base=0.085, contrast=1.5),
    "asphalt_wet":        lambda: _asphalt(RES, 107, (0.095, 0.100, 0.110)),
    # Sampled inside the bay itself, photograph against render: display luma 0.455
    # against 0.222 and r-b -0.26 against -0.19. Both the level and the saturation, so
    # the paint goes up 1.8x and stays blue rather than being greyed toward the middle.
    "bay_blue":           lambda: _painted_bay(RES, 113, (0.066, 0.134, 0.390), (0.222, 0.239, 0.266), wet=0.88, wear_amt=0.40, wet_cover=0.72),
    # The narrow lane to its left is NOT the same blue, which is what modelling it as one
    # assumed: sampled inside it the reference reads luma 0.606 and r-b -0.15, against the
    # middle lane's 0.455 and -0.26. Half again as bright and half as saturated -- an older
    # coat, weathered most of the way back to the concrete under it.
    "bay_blue_faded":     lambda: _painted_bay(RES, 151, (0.300, 0.375, 0.500), (0.330, 0.336, 0.345), wet=0.55, faded=0.45, wear_amt=0.75),
    # wear_amt 0.80, not 0.35: magnified, the reference's terracotta is more peeled than
    # painted -- irregular patches rubbed back to a concrete LIGHTER than the paint over it.
    "bay_orange":         lambda: _painted_bay(RES, 127, (0.442, 0.121, 0.038), (0.300, 0.265, 0.230), wet=0.45, faded=0.22, wear_amt=0.80),
    # The strip beyond the far white line is the SAME paint bleached almost out of
    # existence: the reference reads 0.858 0.700 0.649 there, a pale warm grey, where a
    # full-strength bay reads 0.727 0.403 0.265. It had been painted at full strength and
    # came out 0.30 too dark -- the largest single colour error left in the frame.
    "bay_bleached":       lambda: _painted_bay(RES, 139, (0.62, 0.45, 0.39), (0.400, 0.395, 0.385), wet=0.35, faded=0.62, wear_amt=0.85),
    # The shop frontage is a DIFFERENT SLAB, poured lighter. The photograph runs 0.68 to
    # 0.97 there against 0.53 near the bays, and that gap is albedo, not reflection: no
    # roughness makes a mid-grey pour read as white concrete at fifteen metres.
    "apron_light":        lambda: _concrete(RES, 103, (0.360, 0.365, 0.368), crack=0.16, stain=0.75, wet=0.80, rough_base=0.44, contrast=2.2),
    # painted metal cladding for the canopy column: light cool grey, semi-gloss, streaked
    # by rain rather than cracked (see _painted_metal on why concrete was the wrong base).
    "clad_panel":         lambda: _painted_metal(RES, 137, (0.40, 0.405, 0.41), dirt=0.85, rough_base=0.36),
    "shop_tile":          lambda: _wall_tile(RES, 149, (0.72, 0.725, 0.71), (0.30, 0.30, 0.295), tiles=6),
    # A cast/machined aluminium cap: fine directional turning marks under a coarse
    # casting grain, plus beading. Photographed caps are visibly GRANULAR at the
    # magnification these are judged at, and a perfectly smooth disc is the single
    # thing that most makes one read as a render.
    "fuelcap_cast_cap":   lambda: _beaded_water(RES, 241, (0.058, 0.059, 0.062),
                                                rough_base=0.46, count=120,
                                                r_lo=0.004, r_hi=0.017, coverage=0.34),
    "fuelcap_wet_liner":  lambda: _beaded_water(RES, 211, (0.024, 0.025, 0.027),
                                                rough_base=0.70, count=260,
                                                r_lo=0.0035, r_hi=0.013, coverage=0.42),
    "fuelcap_wet_cap":    lambda: _beaded_water(RES, 223, (0.060, 0.062, 0.066),
                                                rough_base=0.58, count=150,
                                                r_lo=0.005, r_hi=0.020, coverage=0.50),
    "fuelcap_plastic":    lambda: _moulded_plastic(RES, 181, (0.0165, 0.0165, 0.018), rough_base=0.68, wear=0.18),
    "fuelcap_white_paint": lambda: _automotive_paint(RES, 191, (0.76, 0.765, 0.76), rough_base=0.15),
    "fuelcap_polo_blue_paint": lambda: _automotive_paint(
        RES, 197, (0.065, 0.245, 0.405), rough_base=0.16),
    "fuelcap_polo_liner": lambda: _moulded_plastic(
        RES, 199, (0.038, 0.040, 0.043), rough_base=0.70, wear=0.58),
    "fuelcap_polo_cap": lambda: _moulded_plastic(
        RES, 211, (0.072, 0.074, 0.077), rough_base=0.50, wear=0.68),
    # The painted lines were the last flat surface in the frame -- one constant colour over
    # their whole length, where the reference has them clean at one end of the array (luma
    # 0.836) and trodden grey at the other (0.669). Same generator as a bay, because a line
    # IS a bay: paint on concrete, worn through, dirty in the ruts.
    # wear 0.25, not 0.70: at 0.70 the patches rubbed back to concrete are the size of the
    # LINE and it comes out spotted, which is worse than the flat colour it replaced.
    "line_paint":         lambda: _painted_bay(RES, 167, (0.505, 0.510, 0.502), (0.300, 0.297, 0.290), wet=0.30, faded=0.18, wear_amt=0.25),
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
                _painted_bay, _crack_net, _painted_metal, _wall_tile, _streaks,
                _normal_from_height, _fbm):
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
