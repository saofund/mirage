"""Export a built :class:`~mirage.kernel.Mesh` to glTF 2.0 — the standard the rest
of the 3D world reads (Blender, three.js, Babylon, model-viewer, Unreal, Omniverse).

This is the *interop seam*: an op-log is Mirage's source of truth, but a finished
asset has to leave the building. We write a single self-contained ``.glb`` (binary
glTF) with **per-face PBR materials** carried straight through from the op-log's
``material`` op, so the gold-boss-on-red-base you authored renders the same in any
glTF viewer.

Dependency-free on purpose (just ``struct`` + ``json``), mirroring the kernel.

Design notes:
* **Faceted by construction.** Mirage meshes are hard-surface; each triangle gets
  its face's flat normal, so we emit per-triangle vertices (no smoothing seams to
  reason about). Faces are fan-triangulated from their first vertex.
* **One primitive per material.** glTF binds a material to a primitive, so faces
  are grouped by material; unmaterialed faces fall to a neutral default.
* **Z-up → Y-up.** Mirage (like Blender) is Z-up right-handed; glTF is Y-up. We
  apply the standard conversion ``(x, y, z) -> (x, z, -y)`` so assets land upright.
"""
from __future__ import annotations

import json
import struct

_DEFAULT_MAT = {"color": [0.8, 0.8, 0.8], "metallic": 0.0, "roughness": 0.5}


def _zup_to_yup(co):
    x, y, z = co
    return (x, z, -y)


def _face_normal_yup(verts):
    """Newell's method on the (already Y-up) corner positions."""
    n = [0.0, 0.0, 0.0]
    k = len(verts)
    for i in range(k):
        a, b = verts[i], verts[(i + 1) % k]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    m = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5 or 1.0
    return (n[0] / m, n[1] / m, n[2] / m)


def _mat_key(mat):
    c = mat.get("color", _DEFAULT_MAT["color"])
    return (round(c[0], 6), round(c[1], 6), round(c[2], 6),
            round(mat.get("metallic", 0.0), 6), round(mat.get("roughness", 0.5), 6))


def mesh_to_glb(mesh, name: str = "mirage_mesh") -> bytes:
    """Serialize ``mesh`` to a single binary-glTF (.glb) blob (bytes).

    Returns the full .glb container; write it verbatim to a ``.glb`` file."""
    # 1) group faces by material -> per-primitive flat triangle soup ------------- #
    groups: dict = {}        # mat_key -> {"mat": material, "pos": [...], "nrm": [...]}
    order: list = []         # stable material order
    for f in mesh.faces:
        mat = f.attrs.get("material") or _DEFAULT_MAT
        key = _mat_key(mat)
        g = groups.get(key)
        if g is None:
            g = groups[key] = {"mat": mat, "pos": [], "nrm": []}
            order.append(key)
        vs = [_zup_to_yup(v.co) for v in mesh.face_verts(f)]
        nrm = _face_normal_yup(vs)
        for i in range(1, len(vs) - 1):                  # fan-triangulate
            for corner in (vs[0], vs[i], vs[i + 1]):
                g["pos"].extend(corner)
                g["nrm"].extend(nrm)

    # 2) pack one binary buffer; build accessors/bufferViews/primitives --------- #
    bin_parts: list[bytes] = []
    offset = 0
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    materials: list[dict] = []
    primitives: list[dict] = []

    def add_view(blob: bytes, target: int) -> int:
        nonlocal offset
        # 4-byte align
        pad = (-len(blob)) % 4
        bin_parts.append(blob + b"\x00" * pad)
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(blob), "target": target}
        offset += len(blob) + pad
        buffer_views.append(bv)
        return len(buffer_views) - 1

    ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
    FLOAT, UINT = 5126, 5125

    for key in order:
        g = groups[key]
        pos, nrm = g["pos"], g["nrm"]
        nverts = len(pos) // 3
        if nverts == 0:
            continue
        pos_blob = struct.pack("<%df" % len(pos), *pos)
        nrm_blob = struct.pack("<%df" % len(nrm), *nrm)
        idx = list(range(nverts))
        idx_blob = struct.pack("<%dI" % nverts, *idx)

        pos_bv = add_view(pos_blob, ARRAY_BUFFER)
        nrm_bv = add_view(nrm_blob, ARRAY_BUFFER)
        idx_bv = add_view(idx_blob, ELEMENT_ARRAY_BUFFER)

        xs, ys, zs = pos[0::3], pos[1::3], pos[2::3]
        accessors.append({"bufferView": pos_bv, "componentType": FLOAT, "count": nverts,
                          "type": "VEC3", "min": [min(xs), min(ys), min(zs)],
                          "max": [max(xs), max(ys), max(zs)]})
        pos_acc = len(accessors) - 1
        accessors.append({"bufferView": nrm_bv, "componentType": FLOAT, "count": nverts, "type": "VEC3"})
        nrm_acc = len(accessors) - 1
        accessors.append({"bufferView": idx_bv, "componentType": UINT, "count": nverts, "type": "SCALAR"})
        idx_acc = len(accessors) - 1

        mat = g["mat"]
        c = mat.get("color", _DEFAULT_MAT["color"])
        materials.append({
            "name": "mat%d" % len(materials),
            "pbrMetallicRoughness": {
                "baseColorFactor": [float(c[0]), float(c[1]), float(c[2]), 1.0],
                "metallicFactor": float(mat.get("metallic", 0.0)),
                "roughnessFactor": float(mat.get("roughness", 0.5)),
            },
        })
        primitives.append({"attributes": {"POSITION": pos_acc, "NORMAL": nrm_acc},
                           "indices": idx_acc, "material": len(materials) - 1, "mode": 4})

    if not primitives:
        raise ValueError("mesh has no faces to export")

    bin_blob = b"".join(bin_parts)
    gltf = {
        "asset": {"version": "2.0", "generator": "Mirage glTF exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": primitives}],
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "materials": materials,
    }

    # 3) assemble the .glb container (header + JSON chunk + BIN chunk) ----------- #
    json_blob = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_blob += b" " * ((-len(json_blob)) % 4)          # pad with spaces
    bin_blob += b"\x00" * ((-len(bin_blob)) % 4)

    GLB_MAGIC, GLB_VERSION = 0x46546C67, 2
    JSON_CHUNK, BIN_CHUNK = 0x4E4F534A, 0x004E4942
    total = 12 + 8 + len(json_blob) + 8 + len(bin_blob)
    out = bytearray()
    out += struct.pack("<III", GLB_MAGIC, GLB_VERSION, total)
    out += struct.pack("<II", len(json_blob), JSON_CHUNK) + json_blob
    out += struct.pack("<II", len(bin_blob), BIN_CHUNK) + bin_blob
    return bytes(out)


def export_glb(mesh, path: str, name: str = "mirage_mesh") -> dict:
    """Write ``mesh`` to ``path`` as a .glb. Returns a small summary dict."""
    blob = mesh_to_glb(mesh, name=name)
    with open(path, "wb") as fh:
        fh.write(blob)
    n_tris = sum(1 for f in mesh.faces for _ in range(len(mesh.face_verts(f)) - 2))
    n_mats = len({_mat_key(f.attrs.get("material") or _DEFAULT_MAT) for f in mesh.faces})
    return {"path": path, "bytes": len(blob), "triangles": n_tris, "materials": n_mats}
