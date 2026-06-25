"""Case 16 — the hard-surface + open-mesh operator set.

Everything here is one op-log (the single source of truth a human GUI and an AI
both edit), replayed by the kernel. The same JSON runs identically in the native
C++ engine (differential-tested). Run:

    uv run python examples/cases/16_hard_surface.py
"""
from mirage.meshlang import MeshProgram, Sel, ESel


def show(label, prog):
    s = prog.build().stats()
    print(f"  {label:34} v={s['verts']:3} e={s['edges']:3} f={s['faces']:3} "
          f"euler={s['euler']} {'closed' if s['closed_manifold'] else 'OPEN'}")


def main():
    print("hard-surface operators (selection-as-query, never an index):")

    # loop_cut: a watertight edge ring around a quad strip
    show("cube -> loop_cut(z) -> extrude band",
         MeshProgram().cube(1.0)
         .loop_cut(Sel.normal("x", 1.0), axis="z", mark="band")
         .extrude(Sel.tag("band"), 0.12))

    # edge_bevel: round EVERY sharp edge (full) -> a chamfered cube
    show("cube -> edge_bevel(sharp) [full]",
         MeshProgram().cube(1.0).edge_bevel(ESel.sharp(30), width=0.18))

    # edge_bevel: round ONLY the top loop (mixed valence) -> sharp sides
    show("cube -> edge_bevel(top loop) [mixed]",
         MeshProgram().cube(1.2).edge_bevel(ESel.on_face(Sel.normal("z", 1.0)), width=0.22))

    # edge_bevel a cylinder's top rim only (a loop)
    show("cyl -> edge_bevel(top rim) [mixed]",
         MeshProgram().cylinder(12, 0.5, 1.0).edge_bevel(ESel.on_face(Sel.normal("z", 1.0)), width=0.08))

    print("\nopen meshes (boundary edges are first-class):")

    # plane is an open mesh; delete opens a closed one; bridge tunnels; fill caps
    show("plane (open quad)", MeshProgram().plane(1.0))
    show("cube -> delete(top) [open box]",
         MeshProgram().cube(1.0).delete(Sel.normal("z", 1.0)))
    show("cube -> del sides -> bridge -> fill [closed box]",
         MeshProgram().cube(1.0)
         .delete(Sel.NOT(Sel.OR(Sel.normal("z", 1.0), Sel.normal("z", -1.0))))
         .bridge(Sel.all())
         .fill())

    print("\nper-face materials (the path tracer renders them; assign after geometry):")
    p = (MeshProgram().cube(1.2)
         .inset(Sel.normal("z", 1.0), 0.3).extrude(Sel.last(), 0.5)
         .material(Sel.last(), color=[1.0, 0.78, 0.34], metallic=1.0, roughness=0.18)  # gold metal boss
         .material(Sel.NOT(Sel.OR(Sel.normal("z", 1.0), Sel.normal("z", -1.0))),       # red base sides
                   color=[0.8, 0.12, 0.12], roughness=0.45))
    mesh = p.build()
    n_mat = sum(1 for f in mesh.faces if f.attrs.get("material"))
    print(f"  gold metal boss + red base: {n_mat}/{len(mesh.faces)} faces carry a material")
    print("  render it:  mirage_render --oplog mirage_oplog.json --out shot.ppm --spp 160")

    print("\nglTF export (the asset leaves the building — Blender/three.js/model-viewer):")
    from mirage.gltf_export import export_glb
    info = export_glb(mesh, "boss.glb")
    print(f"  wrote {info['path']}: {info['bytes']} bytes, {info['triangles']} tris, "
          f"{info['materials']} materials (per-face PBR carried through)")

    print("\none op-log, two engines, a human and an AI both editing it.")


if __name__ == "__main__":
    main()
