"""Case 17 — a whole SCENE at scale, path-traced from one merged op-log mesh.

Beyond a single model: a downtown of buildings composed into **one** ``mesh`` op
and rendered by the C++ path tracer (``mirage_render``) — the same ground-truth
pillar that shot the airliner. Also a stress test of how Mirage scales to large
scenes and where it bottlenecks (write-up: ``docs/scene-scaling.md``).

    uv run python examples/cases/17_city_scene.py            # render the hero city
    uv run python examples/cases/17_city_scene.py --bench    # the full scaling stress test
    uv run python examples/cases/17_city_scene.py --hero     # write docs/gallery/city.png

Needs ``mirage_render.exe`` built (``cmake --build core/build --config Release``)
and Pillow (``uv pip install pillow``). ``--bench`` also needs the usd+mujoco extras.
"""
import sys
import time
import math
import subprocess
from pathlib import Path

from mirage.meshlang import MeshProgram

ROOT = Path(__file__).resolve().parents[2]
RENDER = ROOT / "core" / "build" / "Release" / "mirage_render.exe"
OUT = Path(__file__).resolve().parent / "outputs" / "17_city_scene"


# -- deterministic geometry (no RNG state; a hashed sine gives repeatable jitter) -- #
def rnd(i, a=0.0, b=1.0):
    x = math.sin((i + 1) * 12.9898) * 43758.5453
    return a + (b - a) * (x - math.floor(x))


def box(cx, cy, cz, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    v = [(cx-hx,cy-hy,cz-hz),(cx+hx,cy-hy,cz-hz),(cx+hx,cy+hy,cz-hz),(cx-hx,cy+hy,cz-hz),
         (cx-hx,cy-hy,cz+hz),(cx+hx,cy-hy,cz+hz),(cx+hx,cy+hy,cz+hz),(cx-hx,cy+hy,cz+hz)]
    f = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]  # outward-wound
    return v, f


def build_city(g, extent, skyline=True):
    """A g x g grid of buildings merged into one (verts, faces, face_materials).

    Because meshlang primitives *replace* the running mesh, a multi-object scene
    can only reach the op-log as a single ``mesh`` op — we assemble the geometry
    here and hand it over whole. Per-face PBR gives each block its own material."""
    V, F, M = [], [], []
    step = (2 * extent) / g
    for ix in range(g):
        for iy in range(g):
            i = ix * g + iy
            cx = -extent + step * (ix + 0.5)
            cy = -extent + step * (iy + 0.5)
            core = max(0.0, 1.0 - math.hypot(cx, cy) / 2.0) if skyline else 0.5
            h = min(1.7, 0.22 + 1.5 * core * (0.55 + 0.6 * rnd(i)))
            w = step * (0.5 + 0.28 * rnd(i + 99))       # gaps between blocks read as streets
            v, f = box(cx, cy, h / 2, w, w, h)
            base = len(V)
            V.extend(v)
            F.extend([tuple(base + k for k in fc) for fc in f])
            col = [0.16 + 0.22*rnd(i+7), 0.24 + 0.30*rnd(i+11), 0.36 + 0.34*rnd(i+13)]
            if rnd(i + 5) > 0.85:
                col = [0.55 + 0.3*rnd(i), 0.42, 0.22]   # a few warm stone facades
            M.extend([{"color": col, "metallic": 0.22 + 0.5*rnd(i+41),
                       "roughness": 0.14 + 0.34*rnd(i+5)}] * len(f))
    return V, F, M


def trace(prog, png, spp=256, w=960, h=600, **knobs):
    """Path-trace a program to a PNG via mirage_render (writes an intermediate PPM)."""
    if not RENDER.exists():
        print(f"  ! mirage_render not built — expected {RENDER}\n"
              f"    build it:  cmake --build {ROOT/'core'/'build'} --config Release")
        return False
    OUT.mkdir(parents=True, exist_ok=True)
    jp = OUT / (png.stem + ".json")
    jp.write_text(prog.to_json(indent=None))
    ppm = OUT / (png.stem + ".ppm")
    cmd = [str(RENDER), "--oplog", str(jp), "--out", str(ppm),
           "--spp", str(spp), "--w", str(w), "--h", str(h), "--bounce", "6"]
    for k, val in knobs.items():
        cmd += [f"--{k}", str(val)]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    try:
        from PIL import Image
    except ImportError:
        print(f"  wrote {ppm} (install pillow to get a .png)")
        return True
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.open(ppm).save(png)
    print(f"  wrote {png}  [{time.perf_counter()-t0:.1f}s]")
    return True


def render_city(png, g=13, extent=1.55, spp=256):
    V, F, M = build_city(g, extent)
    prog = MeshProgram()
    prog.mesh(V, F, face_materials=M)
    t0 = time.perf_counter()
    mesh = prog.build()
    bt = time.perf_counter() - t0
    print(f"  {g*g} buildings  {len(V)} verts  {len(F)} faces  "
          f"build {bt*1000:.0f}ms  op-log {len(prog.to_json(indent=None))/1e6:.2f}MB")
    trace(prog, png, spp=spp, sun=1.15, env=1.0, exposure=1.05)


# -- the scaling stress test (reproduces docs/scene-scaling.md) ---------------- #
def bench():
    print("EXP 0 — does the legible op-log compose objects into a scene?")
    p = MeshProgram(); p.cube(1.0); p.cube(1.0); p.cube(1.0)
    print(f"  3x cube op -> {len(p.build().faces)} faces "
          f"({'REPLACE, no compose' if len(p.build().faces)==6 else 'appended?!'}); "
          f"the op-log is single-model.")

    print("\nEXP 1 — Session (many objects) renders through MuJoCo; render-per-edit is O(N^2):")
    try:
        from mirage.session import Session
    except Exception as e:
        print(f"  [skipped: {e}]")
    else:
        def author(n):
            s = Session(); g = max(1, int(round(n**0.5))); step = 5.0/g
            for ix in range(g):
                for iy in range(g):
                    i = ix*g+iy; hh = 0.25 + 1.6*rnd(i)**2
                    s.add_box(f"b{i}", position=[-2.5+step*(ix+.5), -2.5+step*(iy+.5), hh/2],
                              size=[step*0.6, step*0.6, hh], dynamic=False)
            return s, g*g
        probe = Session(); probe.add_box("p", position=[0,0,.5], dynamic=False)
        try:
            probe.render(width=32, height=32); look = lambda s: s.render(width=320, height=240)
        except Exception:
            look = lambda s: (s._invalidate(), s._ensure_sim())  # GL-less: the rebuild itself
        s, n = author(400); t0 = time.perf_counter(); look(s); once = time.perf_counter()-t0
        s2, _ = author(400); every = max(1, n//20)
        t0 = time.perf_counter(); k = 0; g = int(round(n**0.5)); step = 5.0/g
        s3 = Session()
        for ix in range(g):
            for iy in range(g):
                i = ix*g+iy; hh = 0.25 + 1.6*rnd(i)**2
                s3.add_box(f"b{i}", position=[-2.5+step*(ix+.5), -2.5+step*(iy+.5), hh/2],
                           size=[step*0.6, step*0.6, hh], dynamic=False)
                k += 1
                if k % every == 0: look(s3)
        inc = time.perf_counter()-t0
        print(f"  N=400: look ONCE {once:.2f}s vs look every {every} adds {inc:.2f}s "
              f"-> {inc/once:.0f}x for the SAME scene")

    print("\nEXP 2 — a whole scene as ONE merged mesh -> path tracer (scales sub-linearly in faces):")
    print(f"  {'bldgs':>6} {'faces':>7} {'build_ms':>9} {'json_MB':>8} {'trace_s':>8} (16spp)")
    for g in [10, 20, 30, 50, 80]:
        V, F, M = build_city(g, 2.5)
        prog = MeshProgram(); prog.mesh(V, F, face_materials=M)
        t0 = time.perf_counter(); prog.build(); bt = (time.perf_counter()-t0)*1000
        js = prog.to_json(indent=None)
        tr = float("nan")
        if RENDER.exists():
            OUT.mkdir(parents=True, exist_ok=True)
            jp = OUT / f"bench_{g*g}.json"; jp.write_text(js)
            op = OUT / f"bench_{g*g}.ppm"
            t0 = time.perf_counter()
            subprocess.run([str(RENDER), "--oplog", str(jp), "--out", str(op),
                            "--spp", "16", "--w", "480", "--h", "360"],
                           check=True, capture_output=True)
            tr = time.perf_counter()-t0
        print(f"  {g*g:>6} {len(F):>7} {bt:>9.0f} {len(js)/1e6:>8.2f} {tr:>8.1f}")


def main():
    if "--bench" in sys.argv:
        bench()
    elif "--hero" in sys.argv:
        render_city(ROOT / "docs" / "gallery" / "city.png", g=13, extent=1.55, spp=288)
    else:
        render_city(OUT / "city.png", g=13, extent=1.55, spp=200)
        print("  (a whole scene: 169 buildings, one merged op-log mesh, path-traced)")


if __name__ == "__main__":
    main()
