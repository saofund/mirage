"""Model a passenger jet from the engine's own geometry — a real test that the
kernel makes more than primitives.

Every part is built from mesh operators the engine already has: the fuselage is a
surface of revolution (the lathe / `spin`), the wings and stabilisers are lofted
swept-tapered solids, the engines are capped cylinders, and the whole thing is
made symmetric by mirroring the starboard parts. It is merged into ONE op-log
`mesh` op with per-face PBR materials (livery), then path-traced.

    uv run python examples/airplane.py            # writes airplane.json + renders it

Orientation is chosen for the renderer's fixed 3/4 camera: nose toward +X, wings
along +/-Y, belly down -Z.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mirage.meshlang import MeshProgram  # noqa: E402

TAU = math.pi * 2.0

# ---- materials (a clean white airliner, blue cheatline, red tail) -------------
FUS    = {"color": [0.90, 0.91, 0.93], "metallic": 0.15, "roughness": 0.22}
GLASS  = {"color": [0.03, 0.05, 0.09], "metallic": 0.40, "roughness": 0.06}
WING   = {"color": [0.72, 0.74, 0.78], "metallic": 0.55, "roughness": 0.30}
ENGINE = {"color": [0.24, 0.25, 0.29], "metallic": 0.70, "roughness": 0.26}
INTAKE = {"color": [0.04, 0.04, 0.05], "metallic": 0.30, "roughness": 0.20}
FIN    = {"color": [0.80, 0.16, 0.16], "metallic": 0.10, "roughness": 0.35}
STRIPE = {"color": [0.13, 0.26, 0.62], "metallic": 0.20, "roughness": 0.30}


class Mesh:
    """A tiny vert/face accumulator that also carries a per-face material."""

    def __init__(self):
        self.v, self.f, self.m = [], [], []

    def add(self, verts, faces, mat, mirror=False):
        """Add a part. `mat` is one material dict for all faces, or a list of one
        per face. All verts are added once (shared topology; nothing orphaned)."""
        base = len(self.v)
        if mirror:  # reflect across the XZ plane (y -> -y) and flip winding
            verts = [(x, -y, z) for (x, y, z) in verts]
            faces = [list(reversed(f)) for f in faces]
        self.v.extend(verts)
        for i, face in enumerate(faces):
            self.f.append([base + j for j in face])
            self.m.append(mat[i] if isinstance(mat, list) else mat)


def revolve(stations, seg=28):
    """Surface of revolution about the X axis. stations = [(x, radius, center_z)].
    A radius of 0 is a tip (nose/tail cone). Returns (verts, faces)."""
    verts, ring = [], []
    for (x, r, cz) in stations:
        ring.append(len(verts))
        if r <= 1e-6:
            verts.append((x, 0.0, cz))
        else:
            for k in range(seg):
                a = TAU * k / seg
                verts.append((x, r * math.cos(a), cz + r * math.sin(a)))
    faces = []
    for s in range(len(stations) - 1):
        r0, r1 = stations[s][1], stations[s + 1][1]
        a, b = ring[s], ring[s + 1]
        if r0 <= 1e-6:                       # nose tip fan
            for k in range(seg):
                faces.append([a, b + k, b + (k + 1) % seg])
        elif r1 <= 1e-6:                     # tail tip fan
            for k in range(seg):
                faces.append([a + k, a + (k + 1) % seg, b])
        else:                                # ring-to-ring quads
            for k in range(seg):
                k2 = (k + 1) % seg
                faces.append([a + k, b + k, b + k2, a + k2])
    return verts, faces


def loft(root, tip):
    """A lofted solid between two 4-point cross sections (a swept, tapered wing /
    stabiliser). Each section: dict(y, xle, xte, z, t) -> chord xle..xte, thick t."""
    def sect(s):
        h = s["t"] * 0.5
        return [(s["xle"], s["y"], s["z"] + h), (s["xte"], s["y"], s["z"] + h),
                (s["xte"], s["y"], s["z"] - h), (s["xle"], s["y"], s["z"] - h)]
    a, b = sect(root), sect(tip)
    verts = a + b
    faces = [[0, 1, 2, 3],           # root cap
             [7, 6, 5, 4],           # tip cap
             [0, 4, 5, 1],           # top
             [1, 5, 6, 2],           # trailing edge
             [2, 6, 7, 3],           # bottom
             [3, 7, 4, 0]]           # leading edge
    return verts, faces


def vfin(base, top):
    """A vertical fin: two cross sections stacked in Z, thin in Y. Sections:
    dict(z, xle, xte, y, t) -> chord xle..xte at height z, thickness t in Y."""
    def sect(s):
        h = s["t"] * 0.5
        return [(s["xle"], s["y"] + h, s["z"]), (s["xte"], s["y"] + h, s["z"]),
                (s["xte"], s["y"] - h, s["z"]), (s["xle"], s["y"] - h, s["z"])]
    a, b = sect(base), sect(top)
    verts = a + b
    faces = [[3, 2, 1, 0], [4, 5, 6, 7], [0, 1, 5, 4],
             [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return verts, faces


def cz_of(face_verts):
    return sum(v[2] for v in face_verts) / len(face_verts)


def build():
    m = Mesh()

    # -- fuselage: a lathe/surface-of-revolution, nose at +X, tail raised --------
    stations = [
        (-2.95, 0.02, 0.34), (-2.70, 0.09, 0.28), (-2.30, 0.16, 0.17),
        (-1.70, 0.22, 0.06), (-0.70, 0.25, 0.01), (0.70, 0.25, 0.0),
        (1.55, 0.245, 0.0), (2.10, 0.215, 0.01), (2.45, 0.175, 0.03),
        (2.68, 0.125, 0.05), (2.82, 0.06, 0.06), (2.90, 0.02, 0.07),  # rounded nose
    ]
    fv, ff = revolve(stations, seg=30)
    # paint the fuselage: windshield near the nose-top, a window cheatline down the
    # side, a livery stripe, else the white base.
    fmats = []
    for face in ff:
        pts = [fv[i] for i in face]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        cz = sum(p[2] for p in pts) / len(pts)
        windshield = cz > 0.15 and 2.05 < cx < 2.55       # cockpit glass, nose-top
        cabin = -2.0 < cx < 1.75 and abs(cy) > 0.16       # on the outer sides
        windows = cabin and 0.10 < cz < 0.17              # thin window row (upper side)
        cheat = cabin and 0.055 < cz <= 0.10              # thin cheatline below the windows
        if windshield:
            fmats.append(GLASS)
        elif windows:
            fmats.append(GLASS if (int((cx + 2.0) * 3.4) % 2 == 0) else FUS)  # window / pillar
        elif cheat:
            fmats.append(STRIPE)
        else:
            fmats.append(FUS)
    m.add(fv, ff, fmats)

    # -- main wings: low-mounted, swept, tapered; mirror for the port side -------
    wing_root = {"y": 0.22, "xle": 0.72, "xte": -1.05, "z": -0.11, "t": 0.14}
    wing_tip  = {"y": 2.15, "xle": -0.12, "xte": -0.92, "z": 0.05, "t": 0.05}
    wv, wf = loft(wing_root, wing_tip)
    m.add(wv, wf, WING)
    m.add(wv, wf, WING, mirror=True)

    # -- winglets: a swept fin turned up at each wingtip (modern-airliner tell) --
    wl_base = {"z": 0.03, "xle": -0.10, "xte": -0.90, "y": 2.13, "t": 0.045}
    wl_tip  = {"z": 0.34, "xle": -0.40, "xte": -0.86, "y": 2.13, "t": 0.03}
    lv, lf = vfin(wl_base, wl_tip)
    m.add(lv, lf, WING)
    m.add(lv, lf, WING, mirror=True)

    # -- horizontal stabilisers (tailplane) -------------------------------------
    stab_root = {"y": 0.16, "xle": -2.25, "xte": -2.80, "z": 0.16, "t": 0.06}
    stab_tip  = {"y": 0.95, "xle": -2.55, "xte": -2.88, "z": 0.22, "t": 0.03}
    sv, sf = loft(stab_root, stab_tip)
    m.add(sv, sf, WING)
    m.add(sv, sf, WING, mirror=True)

    # -- vertical fin (the red tail) --------------------------------------------
    fin_base = {"z": 0.28, "xle": -2.15, "xte": -2.82, "y": 0.0, "t": 0.09}
    fin_top  = {"z": 1.05, "xle": -2.55, "xte": -2.86, "y": 0.0, "t": 0.04}
    vv, vf = vfin(fin_base, fin_top)
    m.add(vv, vf, FIN)

    # -- podded engines under the wings: capped cylinders on pylons, mirrored ----
    ex, ey, ez = 0.30, 1.12, -0.30
    eng = [(ex - 0.55, 0.02, 0.0), (ex - 0.45, 0.14, 0.0), (ex - 0.30, 0.165, 0.0),
           (ex + 0.35, 0.165, 0.0), (ex + 0.48, 0.15, 0.0), (ex + 0.55, 0.11, 0.0)]
    ev, ef = revolve([(x, r, 0.0) for (x, r, _) in eng], seg=22)
    ev = [(x, y + ey, z + ez) for (x, y, z) in ev]
    # dark intake ring on the front-most faces (max x), else nacelle grey
    emats = [INTAKE if (sum(ev[i][0] for i in face) / len(face)) > ex + 0.42 else ENGINE
             for face in ef]
    m.add(ev, ef, emats)
    m.add(ev, ef, emats, mirror=True)

    # pylon: a short plate hanging the nacelle off the wing underside
    pylon_lo = {"z": ez + 0.15, "xle": ex + 0.10, "xte": ex - 0.30, "y": ey, "t": 0.05}
    pylon_hi = {"z": -0.11, "xle": ex + 0.05, "xte": ex - 0.28, "y": ey, "t": 0.05}
    pv, pf = vfin(pylon_lo, pylon_hi)
    m.add(pv, pf, ENGINE)
    m.add(pv, pf, ENGINE, mirror=True)

    return m


def main():
    m = build()
    prog = MeshProgram()
    prog.mesh(m.v, m.f, face_materials=m.m)
    out_json = ROOT / "examples" / "airplane.json"
    out_json.write_text(prog.to_json())
    s = prog.build().stats()
    print(f"airplane: {s['verts']} verts, {s['faces']} faces -> {out_json}")

    render = ROOT / "core" / "build" / "Release" / "mirage_render.exe"
    if not render.exists():
        print(f"(renderer not built: {render})")
        return
    out_png = Path(os.environ.get("AIRPLANE_OUT", ROOT / "docs" / "gallery" / "airplane.png"))
    out_ppm = out_png.with_suffix(".ppm")
    subprocess.run([str(render), "--oplog", str(out_json), "--out", str(out_ppm),
                    "--spp", "192", "--w", "960", "--h", "600",
                    "--env", "1.0", "--sun", "3.0", "--exposure", "1.15", "--clamp", "8"],
                   check=True)
    try:
        from PIL import Image
        Image.open(out_ppm).convert("RGB").save(out_png)
        out_ppm.unlink()
        print(f"rendered -> {out_png}")
    except ImportError:
        print(f"rendered -> {out_ppm} (Pillow absent; kept .ppm)")


if __name__ == "__main__":
    main()
