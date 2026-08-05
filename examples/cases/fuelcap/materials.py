"""Materials for the fuel-filler pocket.

Colours are LINEAR albedo, the renderer's own space — not sRGB. A "black" plastic cap is
sRGB ~0.12, which is linear ~0.014; typing 0.12 here makes a cap that reads mid-grey, and
that single mistake is most of the gap between a synthetic pocket and a photographed one.
The real frames bear this out: the cap sits at sRGB 25–60 out of 255 in the orbbec set and
15–30 in the darker production set.

There is one deliberate exception to "measure it". The *recess* is not a material at all —
it is a light trap. Its walls are the same black plastic, but almost no sky reaches them,
so they photograph two to four stops below the cap even though their albedo is identical.
That is geometry doing the work, and the way to get it is to model the cavity, not to paint
the walls darker.
"""
from __future__ import annotations

import math

from mirage.textures import ensure_textures


def mat(color, metallic=0.0, roughness=0.5, maps=None, uv_scale=1.0):
    m = {"color": [float(c) for c in color], "metallic": float(metallic),
         "roughness": float(roughness)}
    if maps:
        m["albedo_map"] = str(maps["albedo"])
        m["roughness_map"] = str(maps["rough"])
        m["normal_map"] = str(maps["normal"])
        m["uv_scale"] = float(uv_scale)
    return m


TEX = ensure_textures(["fuelcap_plastic", "fuelcap_white_paint"])


# --------------------------------------------------------------------------- #
# the cap itself
# --------------------------------------------------------------------------- #
# Automotive black polypropylene. Not zero: a real black plastic still returns 2–5% and
# the shadow detail in the recess comes entirely from that couple of percent.
#
# ROUGHNESS matters more here than albedo does, and it was set too low. A fuel cap is
# injection-moulded polypropylene with a matt grain, not a polished part; at 0.42 the
# dielectric lobe is narrow enough that the handle's trough — a smooth concave surface —
# works as a cylindrical mirror and lays a hard white streak down the bar in every frame.
# The reference caps have a soft broad sheen and no streak at all. What reads as "the model
# is too bright" in a side-by-side is mostly this: the specular, not the base colour.
CAP_BLACK   = mat((0.019, 0.019, 0.021), 0.0, 0.58)
CAP_BLACK_G = mat((0.026, 0.026, 0.029), 0.0, 0.40)   # glossier, newer moulding
CAP_BLACK_M = mat((0.016, 0.016, 0.017), 0.0, 0.74)   # matt / weathered
CAP_GREY    = mat((0.055, 0.055, 0.058), 0.0, 0.60)   # the grey-plastic caps
# The metal caps — about one in ten of the by-car sample, not the two I first counted.
# They are BRUSHED aluminium, not chrome: a partial metal with a wide highlight, which is
# why a fully metallic mirror finish looks wrong on them. The dark variant is the anodised
# grey one; both are far darker than raw aluminium's 0.9 because they are anodised and
# have spent years in a filler pocket.
# Darker than the first pass by a factor of two. Rendered at the exposure this scene needs —
# one set for a black cap in a shadowed hole — a 0.26 metal cap comes out as a featureless
# white blob with no readable geometry at all, which is the same failure as the black cap
# that read mid-grey, from the other end.
CAP_ALU     = mat((0.135, 0.135, 0.132), 0.80, 0.44)
CAP_ALU_D   = mat((0.082, 0.082, 0.085), 0.76, 0.52)
CAP_CHROME  = mat((0.78, 0.78, 0.79), 1.0, 0.12)
CAP_METAL_FAMILY = (CAP_ALU, CAP_ALU_D)

CAP_FAMILY = (CAP_BLACK, CAP_BLACK_G, CAP_BLACK_M, CAP_GREY)

# --------------------------------------------------------------------------- #
# the pocket
# --------------------------------------------------------------------------- #
# Rough, and that is the point: the pocket is a matt moulding full of road dirt, and at 0.55
# the specular lobe was tight enough to lay an even sheen across four flat walls at once,
# which is what made a deep box render as a pale grey tray. A light trap is made of surfaces
# that scatter, not surfaces that reflect.
WELL_PLASTIC = mat((0.020, 0.020, 0.022), 0.0, 0.78)   # the moulded liner
WELL_TEXTURED = mat((0.0165, 0.0165, 0.018), 0.0, 0.68,
                    maps=TEX["fuelcap_plastic"], uv_scale=0.012)
PAINT_WHITE_TEXTURED = mat((0.76, 0.765, 0.76), 0.22, 0.15,
                           maps=TEX["fuelcap_white_paint"], uv_scale=0.055)
WELL_METAL   = mat((0.062, 0.062, 0.065), 0.30, 0.68)  # bare painted metal well
# The filler-neck ring, down a hole under a cap: it is bare steel but it is also the least
# lit surface in the pocket, so a "correct" bright steel albedo renders as the brightest
# object in a frame whose subject is black. Kept dark and rough — a used filler neck is.
NECK_STEEL   = mat((0.115, 0.113, 0.110), 0.75, 0.52)
SEAL_RUBBER  = mat((0.020, 0.019, 0.019), 0.0, 0.80)
SEAL_RED     = mat((0.115, 0.020, 0.016), 0.0, 0.70)   # the orange-red gasket
# Pocket hardware, all of it visible in the reference photographs and none of it modelled
# until now: the sprung catch that holds the door, the screws through the liner, the
# rubber grommet where a drain or a cable passes, the striker the door latches onto.
# Dimmer and rougher than a catalogue value for steel, because these are years old and in
# a pocket that fills with road dirt. At 0.30 / 0.26 a 7 mm screw head rendered as a white
# rectangle — the brightest thing in a frame whose subject is a black cap.
CATCH_STEEL  = mat((0.115, 0.115, 0.118), 0.72, 0.46)
SCREW_STEEL  = mat((0.150, 0.150, 0.152), 0.78, 0.40)
GROMMET      = mat((0.014, 0.014, 0.015), 0.0, 0.86)
TETHER       = mat((0.020, 0.020, 0.022), 0.0, 0.65)
GRIME        = mat((0.045, 0.042, 0.038), 0.0, 0.75)

# --------------------------------------------------------------------------- #
# body paint
# --------------------------------------------------------------------------- #
# Metallic car paint, sampled from the by-car set. `metallic` is deliberately partial:
# a clearcoat over a metallic basecoat is neither a dielectric nor a mirror, and pushing
# it to 1.0 tints the specular with the body colour, which is what makes CG cars look
# like painted metal foil.
def paint(color, metallic=0.55, roughness=0.16):
    return mat(color, metallic, roughness)


PAINTS = {
    "white":      paint((0.72, 0.72, 0.71), 0.30, 0.12),
    "pearl":      paint((0.66, 0.66, 0.68), 0.45, 0.10),
    "silver":     paint((0.34, 0.35, 0.36), 0.72, 0.14),
    "grey":       paint((0.14, 0.145, 0.152), 0.68, 0.18),
    "gunmetal":   paint((0.055, 0.058, 0.064), 0.70, 0.20),
    "black":      paint((0.017, 0.017, 0.019), 0.55, 0.09),
    "red":        paint((0.28, 0.020, 0.016), 0.50, 0.14),
    "darkred":    paint((0.115, 0.012, 0.012), 0.55, 0.15),
    "blue":       paint((0.020, 0.055, 0.240), 0.55, 0.13),
    "navy":       paint((0.012, 0.020, 0.075), 0.60, 0.15),
    "skyblue":    paint((0.075, 0.180, 0.330), 0.50, 0.13),
    "green":      paint((0.030, 0.095, 0.055), 0.55, 0.16),
    "beige":      paint((0.315, 0.270, 0.205), 0.40, 0.16),
    "brown":      paint((0.095, 0.060, 0.038), 0.50, 0.18),
    "yellow":     paint((0.62, 0.44, 0.020), 0.35, 0.14),
    "orange":     paint((0.52, 0.145, 0.016), 0.40, 0.15),
}
PAINT_NAMES = tuple(PAINTS)


def with_decal(base, art, w, h, origin_z, centre=(0.0, 0.0)):
    """`base` with artwork pinned to a +z-facing rectangle in the part's own frame.

    forecourt's ``face_decal`` only handles the four side faces; a fuel cap's printing is
    on its TOP, which is the one orientation that helper cannot express. Same mechanism —
    origin plus two edge vectors — turned to face +z."""
    m = dict(base)
    m["albedo_map"] = str(art["albedo"] if isinstance(art, dict) else art)
    m["decal_origin"] = [centre[0] - w / 2, centre[1] - h / 2, origin_z]
    m["decal_du"] = [w, 0.0, 0.0]
    m["decal_dv"] = [0.0, h, 0.0]
    return m


def with_face_decal(base, art, w, h, origin_z, centre=(0.0, 0.0)):
    """Artwork on a local xy face; useful for a fuel door's camera-facing inner panel."""
    return with_decal(base, art, w, h, origin_z, centre)


def jitter(m, rng, dc=0.18, dr=0.10):
    """A per-instance wobble on colour and roughness.

    Two cars are never the same black. Without this every synthetic cap is bit-identical in
    albedo and a network can learn that constant instead of the shape — the classic way a
    synthetic set trains a detector for its own renderer."""
    g = float(math.exp(rng.normal(0.0, dc)))
    return {"color": [max(0.0, c * g * float(math.exp(rng.normal(0.0, 0.05)))) for c in m["color"]],
            "metallic": m["metallic"],
            "roughness": float(min(0.95, max(0.04, m["roughness"] * math.exp(rng.normal(0.0, dr)))))}
