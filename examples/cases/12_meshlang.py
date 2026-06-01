"""Case 12 — meshlang: model by an op-log program (what an AI would emit).

A mug (cylinder -> inset rim -> extrude cavity) and a ziggurat (cube -> repeated
inset+extrude chained by `last_created`) are built from {op, on:<selector>, params}
command lists — never touching a vertex/face index. The program is JSON,
deterministic, and parametric (edit one number, rebuild). A bad selector fails
loudly with diagnostics (the guardrail).

    uv run python examples/cases/12_meshlang.py
"""
from mirage import Session
from mirage.meshlang import MeshProgram, Sel, SelectorEmpty
from mirage.mujoco_backend import MujocoSim
from mirage.imaging import save_png
from _util import outdir


def render(prog, name, color, view, out, metallic=0.4, rough=0.35):
    m = prog.build()
    minz = min(v.co[2] for v in m.verts)
    obj = m.export_obj(str(out / f"{name}.obj"))
    s = Session(name=name)
    s.add_plane("ground", size=[6, 6])
    s.add_mesh(name, obj, position=[0, 0, -minz + 0.001], color=color, dynamic=False)
    s.set_material(name, metallic=metallic, roughness=rough)
    save_png(MujocoSim.from_scene(s.scene, quality="studio").render(960, 720, **view)["rgb"], out / f"{name}.png")
    return m.stats()


def mug(sides=28):
    return (MeshProgram()
            .cylinder(sides=sides, radius=0.45, height=0.95, mark="body")
            .inset(on=Sel.extreme("z", "max"), thickness=0.16, mark="rim")
            .extrude(on=Sel.tag("rim"), distance=-0.8, mark="cavity")   # hollow downward
            .assert_(closed_manifold=True))


def ziggurat(levels=4):
    p = MeshProgram().cube(1.6)
    p.inset(on=Sel.normal("z", 1), thickness=0.26)
    p.extrude(on=Sel.last(), distance=0.45)
    for _ in range(levels - 1):
        p.inset(on=Sel.last(), thickness=0.26)
        p.extrude(on=Sel.last(), distance=0.45)
    return p


def main():
    out = outdir("12_meshlang")
    mug_view = dict(lookat=[0, 0, 0.45], distance=2.4, azimuth=125, elevation=-12)

    mp = mug()
    print("mug:", render(mp, "mug", [0.6, 0.62, 0.7], mug_view, out, metallic=0.5, rough=0.3))
    print("\n--- the mug as an op-log program (what an LLM emits) ---")
    print(mp.to_json())

    mp8 = MeshProgram.from_json(mp.to_json())   # parametric edit: octagonal mug
    mp8.ops[0]["sides"] = 8
    print("\noctagon mug (changed one number):",
          render(mp8, "mug_octagon", [0.72, 0.5, 0.4], mug_view, out, metallic=0.5, rough=0.3))

    print("ziggurat (cube + 4x inset/extrude via last_created):",
          render(ziggurat(4), "ziggurat", [0.78, 0.62, 0.4],
                 dict(lookat=[0, 0, 1.5], distance=5.4, azimuth=128, elevation=-14), out))

    try:                                         # guardrail: a bad selector fails loudly
        MeshProgram().cube().extrude(on=Sel.tag("no_such_tag")).build()
    except SelectorEmpty as e:
        print("\nbad selector -> SelectorEmpty (as designed):", e.diagnostics)

    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
