"""glTF import + round-trip. Importing baked geometry can't recover history, so we
lower it to one honest `mesh` op (raw verts/faces/materials) that welds the triangle
soup back into real topology. The headline guarantee: export -> import is lossless in
the ways that matter — vertex count (weld), manifoldness, orientation, materials."""
import pytest

from mirage.meshlang import MeshProgram, Sel
from mirage.gltf_export import mesh_to_glb
from mirage.gltf_import import glb_to_mesh_op, import_glb


def _roundtrip(prog):
    orig = prog.build()
    imported = MeshProgram([glb_to_mesh_op(mesh_to_glb(orig))]).build()
    return orig, imported


def test_cube_roundtrip_welds_back_to_real_topology():
    orig, imp = _roundtrip(MeshProgram().cube(1.0))
    imp.validate()
    # the soup welds back: 8 shared verts, euler 2, still a closed manifold
    assert imp.stats()["verts"] == 8
    assert imp.euler() == 2 and imp.is_closed_manifold()
    # quads were triangulated on export, so faces are triangles now (12, not 6)
    assert imp.stats()["faces"] == 12


def test_roundtrip_preserves_orientation_and_bounds():
    orig, imp = _roundtrip(MeshProgram().cylinder(12, 0.5, 1.4))
    def bbox(m):
        lo = [min(v.co[k] for v in m.verts) for k in range(3)]
        hi = [max(v.co[k] for v in m.verts) for k in range(3)]
        return [round(x, 5) for x in lo + hi]
    # Y-up export then Z-up import must land the model back where it started
    assert bbox(orig) == bbox(imp)


def test_roundtrip_carries_materials():
    prog = (MeshProgram().cube(1.2)
            .inset(Sel.normal("z", 1), 0.3).extrude(Sel.last(), 0.5)
            .material(Sel.last(), color=[1.0, 0.78, 0.34], metallic=1.0, roughness=0.18))
    _, imp = _roundtrip(prog)
    gold = [f.attrs["material"] for f in imp.faces
            if f.attrs.get("material") and f.attrs["material"]["metallic"] == 1.0]
    assert gold, "the gold metallic material should survive the round-trip"
    assert gold[0]["color"] == pytest.approx([1.0, 0.78, 0.34], abs=1e-4)


def test_import_glb_returns_single_mesh_op():
    prog = import_glb_from_blob(MeshProgram().uv_sphere(8, 6, 0.5))
    assert len(prog.ops) == 1 and prog.ops[0]["op"] == "mesh"
    prog.build().validate()


def import_glb_from_blob(src_prog):
    # helper: export to a temp .glb on disk, then import_glb it (exercises the file path)
    import tempfile, os
    blob = mesh_to_glb(src_prog.build())
    fd, path = tempfile.mkstemp(suffix=".glb")
    try:
        os.write(fd, blob); os.close(fd)
        return import_glb(path)
    finally:
        os.unlink(path)


def test_imported_geometry_is_editable():
    # an imported mesh is a first-class model: operators stack on it
    _, imp = _roundtrip(MeshProgram().plane(1.0))
    prog = MeshProgram([glb_to_mesh_op(mesh_to_glb(MeshProgram().plane(1.0).build()))])
    prog.extrude(Sel.all(), 0.4)            # extrude the imported quad into a slab
    m = prog.build()
    m.validate()
    assert m.stats()["faces"] > 2


def test_degenerate_glb_raises():
    with pytest.raises(ValueError):
        glb_to_mesh_op(b"not a glb at all")
