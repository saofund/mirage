"""photo -> scene: a general pipeline for rebuilding a real place from a photograph.

Not a case script. Point it at any photo, and it holds the whole reproduction: the solved
camera, the ground plane, the lighting, and one directory per OBJECT.

The decomposition is the load-bearing idea, and it comes straight out of a failure. Matching
a forecourt by hand produced a coarse scene, and the reason was not effort — the Eames chair
came out well because all the attention went to ONE object, and the forecourt spread the same
attention across fifteen. There was no way to say "the dispenser deserves ten times the
detail of that wall", and no number saying which one to spend on. So:

    scenes/<name>/
        reference.png
        scene.json          the whole reproduction, legibly: camera, ground, lighting, layout
        objects/<name>/oplog.json     one op-log per object — refine one, touch nothing else

`place` already made the op-log natively multi-object; a scene IS a list of `place` ops. This
just gives each of them a home, a crop of the photograph to answer to, and a score.

**The per-object score is the point.** "Everything is a bit coarse" is not actionable.
"dispenser 0.31, rail 0.18, wall 0.04" is: it says where the next hour goes, and it says
when to stop. Detail becomes a budget you allocate, instead of a mood.

What is MEASURED here and what is AUTHORED is kept strictly apart. The camera, the ground
plane, every object's position and yaw, and the sun are solved from the photograph and carry
their residuals. Only an object's own shape is authored. A field that was solved says so.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .solve import Camera, apply_h, homography, place_from_footprint, project, solve_camera

__all__ = ["PhotoScene"]

SCHEMA = 1


class PhotoScene:
    """A reproduction: a photograph, a solved camera, and a directory of objects."""

    def __init__(self, root, data=None):
        self.root = Path(root)
        self.data = data if data is not None else {
            "schema": SCHEMA, "reference": "reference.png",
            "camera": {}, "ground": {}, "lighting": {}, "objects": [],
            "render": {"w": 1600, "h": 900, "spp": 300, "denoise": 4},
        }

    # -- io ------------------------------------------------------------------ #
    @classmethod
    def new(cls, root, photo):
        root = Path(root)
        (root / "objects").mkdir(parents=True, exist_ok=True)
        from PIL import Image
        im = Image.open(photo).convert("RGB")
        im.save(root / "reference.png")
        s = cls(root)
        s.data["render"]["w"], s.data["render"]["h"] = im.size
        s.save()
        return s

    @classmethod
    def load(cls, root):
        root = Path(root)
        return cls(root, json.loads((root / "scene.json").read_text()))

    def save(self):
        (self.root / "scene.json").write_text(json.dumps(self.data, indent=2))
        return self

    @property
    def reference(self):
        return self.root / self.data["reference"]

    def ref_size(self):
        from PIL import Image
        with Image.open(self.reference) as im:
            return im.size

    # -- the measured half --------------------------------------------------- #
    def set_ground(self, img_quad, world_quad, note=""):
        """Calibrate the ground plane from one traced rectangle.

        Everything painted on that plane — every line, bay, kerb and footprint — becomes
        measurable from this one act. `world_quad` is usually the four corners of something
        whose real size you know or can assume (a parking bay, a paving slab, a doorway).
        """
        H = homography(np.asarray(img_quad, float), np.asarray(world_quad, float))
        resid = float(np.abs(apply_h(H, img_quad) - np.asarray(world_quad, float)).max())
        self.data["ground"] = {
            "H": H.tolist(), "solved": True,
            "calibrated_by": {"img": np.asarray(img_quad, float).tolist(),
                              "world": np.asarray(world_quad, float).tolist(),
                              "note": note, "fit_residual_m": resid},
        }
        return H

    @property
    def H(self):
        g = self.data.get("ground") or {}
        if "H" not in g:
            raise RuntimeError("no ground plane yet — call set_ground() with a known rectangle")
        return np.asarray(g["H"], float)

    def ground_to_world(self, img_pts):
        """Image pixels on the ground -> world metres. The workhorse."""
        return apply_h(self.H, img_pts)

    def set_camera(self, img_pts, world_pts, guess=None, fit_lens=True):
        """Resect the camera, and RECORD the residual with it.

        The residual is not decoration. A camera you have not reprojected is a camera you are
        guessing at, and every downstream placement inherits the error silently.
        """
        w, h = self.ref_size()
        cam, info = solve_camera(img_pts, world_pts, w, h, guess=guess, fit_lens=fit_lens)
        self.data["camera"] = {
            "eye": cam.eye.tolist(), "target": cam.target.tolist(), "up": cam.up.tolist(),
            "fov_y": cam.fov_y, "k1": cam.k1, "k2": cam.k2, "solved": True,
            "solve": {**info,
                      "correspondences": {"img": np.asarray(img_pts, float).tolist(),
                                          "world": np.asarray(world_pts, float).tolist()}},
        }
        return cam, info

    @property
    def camera(self):
        c = self.data.get("camera") or {}
        if not c:
            raise RuntimeError("no camera yet — call set_camera() or set_camera_manual()")
        return Camera(c["eye"], c["target"], c.get("up", (0, 0, 1)),
                      c["fov_y"], c.get("k1", 0.0), c.get("k2", 0.0))

    def set_camera_manual(self, eye, target, fov_y, up=(0, 0, 1), k1=0.0, k2=0.0, why=""):
        """An unsolved camera. Allowed, but it says so in the file — `solved: false` — so
        nobody later mistakes a guess for a measurement."""
        self.data["camera"] = {"eye": list(eye), "target": list(target), "up": list(up),
                               "fov_y": fov_y, "k1": k1, "k2": k2,
                               "solved": False, "why": why}
        return self.camera

    def set_lighting(self, sun_dir, sun=0.5, env=0.5, exposure=1.0, solve=None):
        self.data["lighting"] = {"sun_dir": [float(v) for v in sun_dir], "sun": float(sun),
                                 "env": float(env), "exposure": float(exposure),
                                 "solve": solve or {}}
        return self

    # -- the authored half --------------------------------------------------- #
    def add_object(self, name, program=None, at=(0, 0, 0), rotate=(0, 0, 0), scale=(1, 1, 1),
                   crop=None, material=None, note="", detail=1):
        """Add (or replace) an object. `program` is a MeshProgram or an op list.

        `crop` is its box in the reference, in FRACTIONS of the frame — that is what the
        object is scored against, and what makes "which object is worst" answerable.
        `detail` is a budget knob for the object's own script to read.
        """
        d = self.root / "objects" / name
        d.mkdir(parents=True, exist_ok=True)
        if program is not None:
            ops = program.ops if hasattr(program, "ops") else [dict(o) for o in program]
            (d / "oplog.json").write_text(json.dumps(ops, indent=1))
        entry = {"name": name, "oplog": f"objects/{name}/oplog.json",
                 "at": [float(v) for v in at], "rotate": [float(v) for v in rotate],
                 "scale": [float(v) for v in scale], "detail": detail, "note": note}
        if crop:
            entry["crop"] = [float(v) for v in crop]
        if material:
            entry["material"] = material
        self.data["objects"] = [o for o in self.data["objects"] if o["name"] != name] + [entry]
        return entry

    def place_object(self, name, img_footprint, height=None, **kw):
        """Add an object by TRACING its ground footprint in the photo.

        No nudging x / y / rotate and re-rendering to see where it went: the ground plane
        already knows. Returns the solved placement so it can be sanity-checked.
        """
        p = place_from_footprint(np.asarray(img_footprint, float), self.H, height=height)
        e = self.add_object(name, at=p["at"], rotate=p["rotate"], **kw)
        e["placed_from_footprint"] = {"img": np.asarray(img_footprint, float).tolist(),
                                      "size_m": p["size"], "squareness": p["squareness"]}
        return p

    # -- build / render / score ---------------------------------------------- #
    def build(self):
        """Compile the whole scene into ONE op-log — a legible list of `place` ops."""
        from .meshlang import MeshProgram
        p = MeshProgram()
        for o in self.data["objects"]:
            f = self.root / o["oplog"]
            if not f.exists():
                continue
            p.place(json.loads(f.read_text()), at=o["at"], rotate=o["rotate"],
                    scale=o.get("scale", [1, 1, 1]), material=o.get("material"))
        return p

    def render_flags(self):
        L = self.data.get("lighting") or {}
        r = self.data["render"]
        f = self.camera.render_flags() + ["--w", str(r["w"]), "--h", str(r["h"])]
        if L:
            f += ["--sun", str(L["sun"]), "--env", str(L["env"]),
                  "--exposure", str(L["exposure"]),
                  "--sun-dir", *(str(v) for v in L["sun_dir"])]
        if r.get("denoise"):
            f += ["--denoise", str(r["denoise"])]
        return f

    def render(self, out=None, spp=None, threads=None, render_exe=None):
        from .capture import default_render
        exe = Path(render_exe) if render_exe else default_render()
        out = Path(out or self.root / "render.png")
        js = self.root / "_build.json"
        js.write_text(self.build().to_json())
        ppm = out.with_suffix(".ppm")
        import os
        cmd = [str(exe), "--oplog", str(js), "--out", str(ppm),
               "--spp", str(spp or self.data["render"]["spp"]),
               "--threads", str(threads or os.environ.get("MIRAGE_THREADS", "14")),
               *self.render_flags()]
        subprocess.run(cmd, check=True)
        from PIL import Image
        Image.open(ppm).save(out)
        return out

    def score(self, render_path=None, plate=None):
        """Score the render — globally, and PER OBJECT.

        The per-object numbers are what turn "it's all a bit coarse" into a work order.
        Sorted worst-first, because that is the only order anybody actually wants.
        """
        from .photomatch import compare
        render_path = Path(render_path or self.root / "render.png")
        regions = {o["name"]: o["crop"] for o in self.data["objects"] if o.get("crop")}
        m = compare(render_path, self.reference, regions=regions,
                    plate=plate or self.root / "diff.png")
        ranked = sorted(((n, r["err"]) for n, r in m.get("regions", {}).items()),
                        key=lambda t: -t[1])
        m["worst_first"] = [{"object": n, "err": e} for n, e in ranked]
        (self.root / "score.json").write_text(json.dumps(m, indent=2))
        return m

    def report(self, m=None):
        """A human-readable verdict — what is solved, how well, and what to fix next."""
        m = m or json.loads((self.root / "score.json").read_text())
        cam = self.data.get("camera", {})
        out = [f"scene: {self.root.name}"]
        if cam.get("solved"):
            s = cam["solve"]
            out.append(f"  camera   SOLVED   rms {s['rms_px']:.2f} px  "
                       f"max {s['max_px']:.2f} px  over {s['n_points']} points")
        elif cam:
            out.append(f"  camera   GUESSED  ({cam.get('why', 'no correspondences')})")
        g = self.data.get("ground", {})
        if g.get("solved"):
            out.append(f"  ground   SOLVED   fit {g['calibrated_by']['fit_residual_m']*1000:.2f} mm"
                       f"   [{g['calibrated_by'].get('note', '')}]")
        e = m["exposure"]
        out.append(f"  exposure {e['stops_off']:+.2f} stops   "
                   f"(render {e['render_median_luma']:.3f} vs ref {e['ref_median_luma']:.3f})")
        out.append(f"  colour   linear RMSE {m['colour_rmse_linear']:.4f}")
        if m.get("worst_first"):
            out.append("  objects, worst first — this is the work order:")
            for r in m["worst_first"]:
                out.append(f"      {r['object']:<18s} {r['err']:.4f}")
        return "\n".join(out)
