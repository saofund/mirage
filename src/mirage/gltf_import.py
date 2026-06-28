"""Import a glTF 2.0 binary (.glb) into Mirage as a replayable op-log op.

This is the other half of the interop seam (see :mod:`mirage.gltf_export`). The
catch: an op-log is Mirage's source of truth, but a glTF file is just baked
geometry — there is no history to recover. So we do NOT invent a fake history;
we lower the imported geometry to ONE honest op, ``mesh`` (raw verts + faces +
per-face materials), which both engines replay identically via ``from_pydata``.
The rest of the modeling operators then stack on top of it like any primitive.

Reconstruction:
* glTF is a triangle soup (often with per-triangle duplicated verts, as our own
  exporter emits for flat shading). We **weld** coincident positions back into
  shared vertices so the result has real topology (a welded cube is 8 verts /
  12 triangles / euler 2 again), then drop any degenerate triangle.
* **Y-up → Z-up**: the inverse of the exporter's axis swap, so a round-trip
  (export → import) lands the model back in its original orientation.
* Each primitive's glTF material (pbrMetallicRoughness) becomes the per-face
  material of that primitive's triangles.
"""
from __future__ import annotations

import json
import struct

from .meshlang import MeshProgram

GLB_MAGIC, JSON_CHUNK, BIN_CHUNK = 0x46546C67, 0x4E4F534A, 0x004E4942
_COMP = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2),
         5125: ("I", 4), 5126: ("f", 4)}            # componentType -> (struct code, bytes)
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _parse_glb(blob: bytes):
    magic, ver, total = struct.unpack_from("<III", blob, 0)
    if magic != GLB_MAGIC or ver != 2 or total != len(blob):
        raise ValueError("not a valid .glb (bad header)")
    jlen, jtype = struct.unpack_from("<II", blob, 12)
    if jtype != JSON_CHUNK:
        raise ValueError("first chunk is not JSON")
    gltf = json.loads(blob[20:20 + jlen])
    blen, btype = struct.unpack_from("<II", blob, 20 + jlen)
    bin_blob = blob[20 + jlen + 8:20 + jlen + 8 + blen] if btype == BIN_CHUNK else b""
    return gltf, bin_blob


def _read_accessor(gltf, bin_blob, acc_idx):
    acc = gltf["accessors"][acc_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    code, size = _COMP[acc["componentType"]]
    nc = _NCOMP[acc["type"]]
    base = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or (size * nc)
    out = []
    for i in range(acc["count"]):
        off = base + i * stride
        vals = struct.unpack_from("<%d%s" % (nc, code), bin_blob, off)
        out.append(vals if nc > 1 else vals[0])
    return out


def _material_of(gltf, prim):
    mi = prim.get("material")
    if mi is None:
        return None
    pbr = gltf["materials"][mi].get("pbrMetallicRoughness", {})
    c = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
    return {"color": [round(c[0], 6), round(c[1], 6), round(c[2], 6)],
            "metallic": round(pbr.get("metallicFactor", 1.0), 6),
            "roughness": round(pbr.get("roughnessFactor", 1.0), 6)}


def glb_to_mesh_op(blob: bytes, weld_tol: float = 1e-5) -> dict:
    """Parse a .glb blob and return a single ``mesh`` op dict (verts/faces/
    face_materials) — the op-log lowering of the imported geometry."""
    gltf, bin_blob = _parse_glb(blob)
    inv_tol = 1.0 / weld_tol
    verts: list[list[float]] = []
    vmap: dict = {}                                   # quantized position -> welded index
    faces: list[list[int]] = []
    face_materials: list = []

    def weld(p):
        key = (round(p[0] * inv_tol), round(p[1] * inv_tol), round(p[2] * inv_tol))
        idx = vmap.get(key)
        if idx is None:
            idx = vmap[key] = len(verts)
            verts.append([float(p[0]), float(p[1]), float(p[2])])
        return idx

    any_mat = False
    for m in gltf.get("meshes", []):
        for prim in m.get("primitives", []):
            if prim.get("mode", 4) != 4:              # only TRIANGLES
                continue
            pos = _read_accessor(gltf, bin_blob, prim["attributes"]["POSITION"])
            if "indices" in prim:
                idx = _read_accessor(gltf, bin_blob, prim["indices"])
            else:
                idx = list(range(len(pos)))
            mat = _material_of(gltf, prim)
            any_mat = any_mat or (mat is not None)
            for t in range(0, len(idx) - 2, 3):
                tri = []
                for k in range(3):
                    X, Y, Z = pos[idx[t + k]]
                    tri.append(weld((X, -Z, Y)))       # glTF Y-up -> Mirage Z-up
                if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                    continue                            # degenerate after welding
                faces.append(tri)
                face_materials.append(mat)

    if not faces:
        raise ValueError("glTF contained no triangle geometry")
    op = {"op": "mesh", "verts": verts, "faces": faces}
    if any_mat:
        op["face_materials"] = face_materials
    return op


def import_glb(path: str, weld_tol: float = 1e-5) -> MeshProgram:
    """Load ``path`` (.glb) into a fresh single-op MeshProgram (a ``mesh`` op)."""
    with open(path, "rb") as fh:
        blob = fh.read()
    return MeshProgram([glb_to_mesh_op(blob, weld_tol)])
