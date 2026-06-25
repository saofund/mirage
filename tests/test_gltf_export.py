"""glTF export: the .glb we emit must be a well-formed glTF 2.0 container with the
op-log's per-face materials carried through. We re-parse the binary here (no external
glTF lib) so the test is self-contained and asserts spec invariants directly."""
import json
import struct

import pytest

from mirage.meshlang import MeshProgram, Sel
from mirage.gltf_export import mesh_to_glb, export_glb

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942


def _parse_glb(blob):
    magic, ver, total = struct.unpack_from("<III", blob, 0)
    assert magic == GLB_MAGIC and ver == 2 and total == len(blob)
    jlen, jtype = struct.unpack_from("<II", blob, 12)
    assert jtype == JSON_CHUNK
    gltf = json.loads(blob[20:20 + jlen])
    blen, btype = struct.unpack_from("<II", blob, 20 + jlen)
    assert btype == BIN_CHUNK
    bin_blob = blob[20 + jlen + 8:20 + jlen + 8 + blen]
    return gltf, bin_blob


def _gold_boss():
    return (MeshProgram().cube(1.2)
            .inset(Sel.normal("z", 1), 0.3).extrude(Sel.last(), 0.5)
            .material(Sel.last(), color=[1.0, 0.78, 0.34], metallic=1.0, roughness=0.18)
            .material(Sel.NOT(Sel.OR(Sel.normal("z", 1), Sel.normal("z", -1))),
                      color=[0.8, 0.12, 0.12], roughness=0.45)).build()


def test_glb_is_well_formed():
    gltf, bin_blob = _parse_glb(mesh_to_glb(_gold_boss(), "boss"))
    assert gltf["asset"]["version"] == "2.0"
    assert len(gltf["meshes"]) == 1 and gltf["scene"] == 0
    # every accessor must decode within its bufferView, and views within the buffer
    for acc in gltf["accessors"]:
        bv = gltf["bufferViews"][acc["bufferView"]]
        comp = {5126: 4, 5125: 4, 5123: 2}[acc["componentType"]]
        ncomp = {"VEC3": 3, "SCALAR": 1}[acc["type"]]
        assert bv["byteLength"] >= acc["count"] * comp * ncomp
        assert bv["byteOffset"] + bv["byteLength"] <= len(bin_blob)


def test_per_face_materials_are_carried_through():
    gltf, _ = _parse_glb(mesh_to_glb(_gold_boss(), "boss"))
    # gold boss (metallic 1) + red base + neutral default = 3 distinct materials
    assert len(gltf["materials"]) == 3
    metals = sorted(m["pbrMetallicRoughness"]["metallicFactor"] for m in gltf["materials"])
    assert metals == [0.0, 0.0, 1.0]
    colors = [tuple(round(x, 2) for x in m["pbrMetallicRoughness"]["baseColorFactor"][:3])
              for m in gltf["materials"]]
    assert (1.0, 0.78, 0.34) in colors and (0.8, 0.12, 0.12) in colors
    # one primitive per material, each bound to a material index
    prims = gltf["meshes"][0]["primitives"]
    assert len(prims) == 3 and all("material" in p for p in prims)


def test_position_accessor_has_correct_bounds():
    gltf, bin_blob = _parse_glb(mesh_to_glb(MeshProgram().cube(2.0).build(), "cube"))
    prim = gltf["meshes"][0]["primitives"][0]
    acc = gltf["accessors"][prim["attributes"]["POSITION"]]
    bv = gltf["bufferViews"][acc["bufferView"]]
    pos = struct.unpack_from("<%df" % (acc["count"] * 3), bin_blob, bv["byteOffset"])
    assert acc["min"] == [min(pos[0::3]), min(pos[1::3]), min(pos[2::3])]
    assert acc["max"] == [max(pos[0::3]), max(pos[1::3]), max(pos[2::3])]
    assert acc["min"] == [-1.0, -1.0, -1.0] and acc["max"] == [1.0, 1.0, 1.0]


def test_zup_to_yup_conversion():
    # a face whose Mirage normal is +Z must become +Y in the exported (Y-up) glTF
    gltf, bin_blob = _parse_glb(mesh_to_glb(MeshProgram().plane(2.0).build(), "p"))
    prim = gltf["meshes"][0]["primitives"][0]
    nacc = gltf["accessors"][prim["attributes"]["NORMAL"]]
    bv = gltf["bufferViews"][nacc["bufferView"]]
    nrm = struct.unpack_from("<3f", bin_blob, bv["byteOffset"])
    assert nrm[1] == pytest.approx(1.0, abs=1e-5)   # +Z (Mirage) -> +Y (glTF)


def test_export_glb_writes_file(tmp_path):
    out = tmp_path / "m.glb"
    info = export_glb(MeshProgram().uv_sphere(8, 6, 0.5).build(), str(out))
    assert out.exists() and out.stat().st_size == info["bytes"]
    assert info["triangles"] > 0 and info["materials"] == 1


def test_export_grouped_triangle_count_matches_geometry(tmp_path):
    mesh = _gold_boss()
    info = export_glb(mesh, str(tmp_path / "t.glb"))
    # one triangle-fan per face: sum(verts-2)
    expect = sum(len(mesh.face_verts(f)) - 2 for f in mesh.faces)
    assert info["triangles"] == expect
