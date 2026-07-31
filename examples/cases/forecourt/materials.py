"""The forecourt's material set — flat PBR, tiled map sets, and pinned decals.

Three kinds of surface appear here, and the difference matters:

* **flat** — a colour + metallic + roughness. Right for small painted parts whose whole
  face is one colour (a nozzle boot, a bollard's band).
* **tiled** — a triplanar PBR map set from :mod:`mirage.textures`, sized in world metres
  per tile. Right for anything large and stochastic: concrete, asphalt, cladding.
* **decal** — artwork from :mod:`mirage.decals` pinned to ONE rectangle of ONE part. Right
  for everything printed: the price board, the extinguisher cabinet, a number plate. A
  tiled map cannot do this — it has no anchor, so it would repeat the sign up the column.

The decal frame is authored in the coordinate frame of the geometry it is stuck to, and
the `place` op carries it along, so a labelled part stays labelled wherever it is put.
"""
from mirage.decals import ensure_decals
from mirage.textures import ensure_textures


def mat(c, metallic=0.0, roughness=0.5, emission=None, maps=None, uv_scale=1.0):
    """A flat PBR material, optionally carrying a TILED triplanar map set."""
    m = {"color": list(c), "metallic": metallic, "roughness": roughness}
    if emission:
        m["emission"] = list(emission)
    if maps:
        m["albedo_map"] = str(maps["albedo"])
        if "rough" in maps:
            m["roughness_map"] = str(maps["rough"])
        if "normal" in maps:
            m["normal_map"] = str(maps["normal"])
        m["uv_scale"] = uv_scale
    return m


def face_decal(art, w, h, depth, face="-y", centre=(0.0, 0.0, 0.0),
               base=(0.55, 0.55, 0.55), metallic=0.0, roughness=0.32):
    """Artwork pinned to one face of a part, in that part's own coordinates.

    `w`/`h` are the printed area and `depth` the panel's thickness. By default the artwork
    is centred on the part's origin, which is where a ``cbox`` sits; `centre` moves it, for
    a part whose geometry does not straddle its own origin — a lathed bucket standing on
    z=0, say. Outside the rectangle the tracer falls back to `base`, so the graphic ends at
    the panel's edge instead of tiling."""
    du = [w, 0.0, 0.0]
    dv = [0.0, 0.0, h]
    off = depth / 2 + 1e-4
    origin = {"-y": [-w / 2, -off, -h / 2], "+y": [-w / 2, off, -h / 2],
              "-x": [0.0, -w / 2, -h / 2], "+x": [0.0, -w / 2, -h / 2]}[face]
    if face in ("-x", "+x"):
        du = [0.0, w, 0.0]
        origin[0] = -off if face == "-x" else off
    origin = [origin[k] + centre[k] for k in range(3)]
    m = mat(base, metallic, roughness)
    m["albedo_map"] = str(art["albedo"])
    m["decal_origin"] = origin
    m["decal_du"] = du
    m["decal_dv"] = dv
    return m


def top_decal(art, w, d, height, base=(0.3, 0.3, 0.3), metallic=0.0, roughness=0.5):
    """Artwork pinned to the TOP face of a centred box — a painted road marking, the
    checker of a speed hump."""
    m = mat(base, metallic, roughness)
    m["albedo_map"] = str(art["albedo"])
    m["decal_origin"] = [-w / 2, -d / 2, height / 2 + 1e-4]
    m["decal_du"] = [w, 0.0, 0.0]
    m["decal_dv"] = [0.0, d, 0.0]
    return m


TEX = ensure_textures(["forecourt_concrete", "asphalt_wet", "bay_blue", "bay_orange",
                       "bay_slate", "apron_light", "clad_panel", "shop_tile"])
DEC = ensure_decals(["pump_sign", "fire_cabinet", "fire_bucket", "wet_floor", "wash_banner",
                     "promo_banner", "repair_sign", "plate", "shutter_slat"])

# ---- ground (tiled) ------------------------------------------------------------- #
APRON    = mat((0.20, 0.205, 0.208), 0.0, 0.6, maps=TEX["forecourt_concrete"], uv_scale=5.5)
ROAD     = mat((0.095, 0.10, 0.11), 0.0, 0.25, maps=TEX["asphalt_wet"], uv_scale=4.0)
APRON_LT = mat((0.36, 0.365, 0.368), 0.0, 0.55, maps=TEX["apron_light"], uv_scale=5.0)
BAY_BLUE = mat((0.03, 0.058, 0.15), 0.0, 0.5, maps=TEX["bay_blue"], uv_scale=4.0)
BAY_ORNG = mat((0.395, 0.108, 0.034), 0.0, 0.5, maps=TEX["bay_orange"], uv_scale=4.5)
BAY_SLATE = mat((0.09, 0.10, 0.115), 0.0, 0.5, maps=TEX["bay_slate"], uv_scale=4.5)
LINE_W   = mat((0.40, 0.405, 0.40), 0.0, 0.60)      # damp, grimy paint — never bright white
YELLOWP  = mat((0.52, 0.36, 0.04), 0.0, 0.68)
CONCRETE = mat((0.17, 0.175, 0.170), 0.0, 0.72)

# ---- structure ------------------------------------------------------------------ #
CLAD      = mat((0.415, 0.417, 0.413), 0.0, 0.38, maps=TEX["clad_panel"], uv_scale=1.5)
WALL_TILE = mat((0.86, 0.865, 0.85), 0.0, 0.35, maps=TEX["shop_tile"], uv_scale=1.15)
SEAM      = mat((0.24, 0.24, 0.24), 0.0, 0.45)      # the shadow line between clad panels
KERB      = mat((0.24, 0.24, 0.235), 0.0, 0.66)

# ---- painted metal and plastic --------------------------------------------------- #
PUMP_BL   = mat((0.070, 0.105, 0.315), 0.0, 0.30)   # the dispenser's blue
PUMP_BL_D = mat((0.046, 0.068, 0.205), 0.0, 0.34)
PANEL_WH  = mat((0.80, 0.80, 0.79), 0.0, 0.34)
RED       = mat((0.50, 0.045, 0.030), 0.0, 0.40)
RED_D     = mat((0.30, 0.028, 0.020), 0.0, 0.44)
ORANGE_S  = mat((0.72, 0.26, 0.03), 0.0, 0.42)
YELLOW    = mat((0.74, 0.55, 0.03), 0.0, 0.44)
BLACK     = mat((0.058, 0.058, 0.062), 0.0, 0.52)
RUBBER    = mat((0.020, 0.020, 0.022), 0.0, 0.78)
# A speed hump lives outdoors and is chalked pale by sun and grit; sharing the fresh-tyre
# black had it reading 0.451 against the photograph's 0.588.
HUMP      = mat((0.115, 0.115, 0.120), 0.0, 0.70)
# Fuel hose is near-black rubber. It was lifted to 0.125 chasing a tone target that was
# really a blend of hose and forecourt (see critique._fill) and came out grey.
HOSE      = mat((0.048, 0.048, 0.051), 0.0, 0.55)
STEEL     = mat((0.52, 0.52, 0.53), 1.0, 0.36)
# The island's grating stands in the rain. Bright dry steel turned it into a chrome
# serving tray; wet steel is darker and glossier, and reads as the thing in the photo.
WET_STEEL = mat((0.60, 0.605, 0.60), 1.0, 0.30)
GALV      = mat((0.46, 0.47, 0.48), 1.0, 0.44)      # galvanised: duller than bright steel
CHROME    = mat((0.62, 0.63, 0.64), 1.0, 0.18)
WHITE     = mat((0.78, 0.78, 0.77), 0.0, 0.42)
BODY_WH   = mat((0.83, 0.835, 0.84), 0.0, 0.22)     # vehicle paint: brighter and glossier
GLASS     = mat((0.05, 0.06, 0.07), 0.0, 0.07)
SHUTTER   = mat((0.42, 0.43, 0.435), 0.35, 0.40)
SHUTTER_D = mat((0.24, 0.245, 0.255), 0.35, 0.46)
NAVY      = mat((0.190, 0.215, 0.40), 0.0, 0.44)
TAIL_RED  = mat((0.42, 0.03, 0.02), 0.0, 0.22)
LAMP      = mat((0.70, 0.68, 0.62), 0.0, 0.14)

# ---- decals (printed artwork, pinned) -------------------------------------------- #
# Sized to the sign's visible APERTURE, not to its outer panel: the frame extrusions stand
# proud of the face and would otherwise crop 3 cm of artwork off each edge.
SIGN_FACE  = face_decal(DEC["pump_sign"], 0.915, 2.06, 0.02, base=(0.60, 0.60, 0.59),
                        roughness=0.26)
SIGN_FACE["emission"] = [0.045, 0.045, 0.044]  # a LIGHTBOX: lit from inside, not lit on.
# 0.16 overshot to sRGB 0.646 against the photo's 0.400. Measured back down, not guessed.
FIREBOX_F  = face_decal(DEC["fire_cabinet"], 0.38, 0.70, 0.012, base=(0.50, 0.045, 0.030),
                        roughness=0.40)
BUCKET_F   = face_decal(DEC["fire_bucket"], 0.28, 0.20, 0.30, centre=(0.0, 0.0, 0.175),
                        base=(0.46, 0.47, 0.48), metallic=0.7, roughness=0.45)
WETSIGN_F  = face_decal(DEC["wet_floor"], 0.34, 0.62, 0.024, base=(0.60, 0.47, 0.04),
                        roughness=0.45)
WASH_F     = face_decal(DEC["wash_banner"], 1.30, 1.76, 0.02, base=(0.03, 0.055, 0.20),
                        roughness=0.48)
PROMO_F    = face_decal(DEC["promo_banner"], 0.55, 2.75, 0.02, base=(0.03, 0.055, 0.20),
                        roughness=0.48)
REPAIR_F   = face_decal(DEC["repair_sign"], 0.396, 1.80, 0.06, base=(0.55, 0.27, 0.02),
                        roughness=0.45)
# The van's plate hangs on its REAR (a -x face), so the artwork's u axis has to run along
# world y — the one place in this scene where a decal is not on a -y face.
PLATE_F    = face_decal(DEC["plate"], 0.44, 0.15, 0.03, face="-x", base=(0.06, 0.14, 0.36),
                        roughness=0.30)
