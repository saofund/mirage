"""Case 13 — a goblet, modeled the way an AI operates the kernel.

This is the program Claude emitted live over the MCP modeling tools
(``new_model`` / ``apply_mesh_op`` / ``render_model``): a chalice grown from a
disk foot, drawn up into a stem, flared into a tulip bowl by alternating
extrude + scale, then hollowed with an inset rim + downward extrude. Every step
is a selector-driven op — never a vertex/face index — so the whole thing stays a
closed 2-manifold (euler == 2) and survives a parametric edit (bump ``sides``).

    uv run python examples/cases/13_goblet.py
"""
from mirage import Session
from mirage.meshlang import MeshProgram, Sel
from mirage.mujoco_backend import MujocoSim
from mirage.imaging import save_png
from _util import outdir


def goblet(sides=32) -> MeshProgram:
    p = (MeshProgram()
         .cylinder(sides=sides, radius=0.62, height=0.10, mark="foot")    # base disk
         .inset(on=Sel.extreme("z", "max"), thickness=0.74, mark="seat")  # shrink to stem footprint
         .extrude(on=Sel.tag("seat"), distance=0.62, mark="stem")         # draw the stem up
         .extrude(on=Sel.tag("stem"), distance=0.04))                     # tiny node under the bowl
    # flare into a tulip bowl: alternate scale (widen) + extrude (rise), chained
    # purely by `last_created` so no index or tag bookkeeping is needed.
    p.scale(on=Sel.last(), by=[1.8, 1.8, 1]).extrude(on=Sel.last(), distance=0.14)
    p.scale(on=Sel.last(), by=[1.5, 1.5, 1]).extrude(on=Sel.last(), distance=0.16)
    p.scale(on=Sel.last(), by=[1.18, 1.18, 1]).extrude(on=Sel.last(), distance=0.16, mark="rim_top")
    # hollow it: inset the rim, then extrude that ring straight down into the bowl.
    p.inset(on=Sel.tag("rim_top"), thickness=0.10, mark="rim")
    p.extrude(on=Sel.tag("rim"), distance=-0.62, mark="cavity")
    return p.assert_(closed_manifold=True, euler=2)


def render(prog, name, color, out, metallic=0.45, rough=0.3):
    m = prog.build()
    minz = min(v.co[2] for v in m.verts)
    obj = m.export_obj(str(out / f"{name}.obj"))
    s = Session(name=name)
    s.add_plane("ground", size=[6, 6])
    s.add_mesh(name, obj, position=[0, 0, -minz + 0.001], color=color, dynamic=False)
    s.set_material(name, metallic=metallic, roughness=rough)
    view = dict(lookat=[0, 0, 0.55], distance=3.3, azimuth=130, elevation=-13)
    save_png(MujocoSim.from_scene(s.scene, quality="studio").render(760, 620, **view)["rgb"],
             out / f"{name}.png")
    return m.stats()


def main():
    out = outdir("13_goblet")
    gp = goblet()
    print("goblet:", render(gp, "goblet", [0.78, 0.7, 0.45], out))      # brass
    print("\n--- the goblet as an op-log program (what the AI emitted) ---")
    print(gp.to_json())

    g8 = MeshProgram.from_json(gp.to_json())                            # parametric edit
    g8.ops[0]["sides"] = 6
    print("\nhex-faceted goblet (changed one number):",
          render(g8, "goblet_hex", [0.6, 0.62, 0.72], out, metallic=0.5))

    print(f"\noutputs in {out}")


if __name__ == "__main__":
    main()
