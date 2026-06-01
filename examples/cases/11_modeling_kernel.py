"""Case 11 — Modeling from scratch on the mesh kernel (no trimesh, no CSG lib).

Every shape is built by *operators on owned topology* — extrude / inset /
Catmull-Clark subdivide — recorded as a program (the op sequence IS the model).
That op sequence is exactly what an AI would emit to model something.

    uv run python examples/cases/11_modeling_kernel.py
"""
from mirage.kernel import make_cube, extrude_faces, inset_faces, faces_by_normal, catmull_clark
from mirage import Session
from mirage.mujoco_backend import MujocoSim
from mirage.imaging import save_png
from _util import outdir


def render(m, name, color, view, out, metallic=0.3, rough=0.4):
    minz = min(v.co[2] for v in m.verts)
    obj = m.export_obj(str(out / f"{name}.obj"))
    s = Session(name=name)
    s.add_plane("ground", size=[8, 8])
    s.add_mesh(name, obj, position=[0, 0, -minz + 0.001], color=color, dynamic=False)
    s.set_material(name, metallic=metallic, roughness=rough)
    save_png(MujocoSim.from_scene(s.scene, quality="studio").render(960, 720, **view)["rgb"], out / f"{name}.png")
    return m.stats()


def ziggurat(levels=4):
    m, log = make_cube(1.6), ["cube(1.6)"]
    top = faces_by_normal(m, "z", 1.0)                 # the single starting top face
    for _ in range(levels):
        m = inset_faces(m, top, 0.26); log.append("inset(top, 0.26)")
        m = extrude_faces(m, [m.faces[-1]], 0.45); log.append("extrude(top, +0.45)")
        top = [m.faces[-1]]                            # the new cap is the next working top
    return m, log


def smooth_blob():
    m, log = make_cube(1.4), ["cube(1.4)"]
    for ax, sg in [("x", 1.0), ("x", -1.0), ("y", 1.0), ("y", -1.0)]:
        m = extrude_faces(m, faces_by_normal(m, ax, sg), 0.7)
        log.append(f"extrude({ax}{'+' if sg > 0 else '-'} face, +0.7)")
    m = catmull_clark(catmull_clark(m)); log += ["subdivide", "subdivide"]
    return m, log


def main():
    out = outdir("11_modeling_kernel")

    z, zlog = ziggurat(4)
    print("ziggurat:", render(z, "ziggurat", [0.78, 0.62, 0.4],
                              dict(lookat=[0, 0, 1.5], distance=5.4, azimuth=128, elevation=-14), out))
    print("  program:", "  ->  ".join(zlog))

    b, blog = smooth_blob()
    print("smooth_blob:", render(b, "smooth_blob", [0.5, 0.55, 0.72],
                                dict(lookat=[0, 0, 0.7], distance=4.6, azimuth=128, elevation=-8), out,
                                metallic=0.5, rough=0.25))
    print("  program:", "  ->  ".join(blog))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
