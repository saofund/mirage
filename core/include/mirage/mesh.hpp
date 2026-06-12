// mirage_core — the C++ topology kernel.
//
// A faithful port of the Python `mirage.kernel` radial-edge (BMesh-style) mesh:
// Vert / Edge / Loop / Face joined by three circular linked lists — the loop
// cycle (a face's boundary), the radial cycle (loops/faces around an edge, which
// is what permits non-manifold geometry), and the disk is implicit. The Python
// kernel is the executable spec; this C++ core is validated against it by
// differential testing.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace mirage {

struct Loop;
struct Face;

struct Vert {
    int id = 0;
    std::array<double, 3> co{0, 0, 0};
    Loop* loop = nullptr;  // one incident loop (entry into the disk)
};

struct Edge {
    int id = 0;
    Vert* v1 = nullptr;
    Vert* v2 = nullptr;
    Loop* loop = nullptr;  // one loop in this edge's radial cycle
    Vert* other(const Vert* v) const { return v == v1 ? v2 : v1; }
};

struct Loop {
    int id = 0;
    Vert* vert = nullptr;
    Edge* edge = nullptr;
    Face* face = nullptr;
    Loop* next = nullptr;          // loop cycle (face boundary, CCW)
    Loop* prev = nullptr;
    Loop* radial_next = nullptr;   // radial cycle (around `edge`)
    Loop* radial_prev = nullptr;
};

struct Face {
    int id = 0;
    Loop* loop = nullptr;  // entry into the loop cycle
    // Durable handles. Operators rebuild the mesh and renumber every element
    // (the Topological Naming Problem), so an index dies after one op — tags are
    // copied to descendant faces across rebuilds and are what selectors (`tag`,
    // `last_created`) resolve. Mirrors Python Face.attrs["tags"].
    std::vector<std::string> tags;
};

// A mesh owns its elements (vector<unique_ptr>) and links them by raw pointers,
// mirroring the Python object graph. Move-only.
class Mesh {
public:
    Mesh() = default;
    Mesh(Mesh&&) = default;
    Mesh& operator=(Mesh&&) = default;
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    Vert* add_vert(double x, double y, double z);
    // Create a face from an ordered, closed vertex loop. Throws std::invalid_argument
    // for < 3 verts or repeated verts (matches the Python kernel's add_face).
    Face* add_face(const std::vector<Vert*>& verts, std::vector<std::string> tags = {});

    std::vector<Loop*> face_loops(const Face* f) const;
    std::vector<Vert*> face_verts(const Face* f) const;
    std::vector<Loop*> edge_loops(const Edge* e) const;
    std::vector<Face*> edge_faces(const Edge* e) const;

    // Rebuild a mesh from positions + ngon face index lists (mirrors the Python
    // Mesh.from_pydata; the building block operators use to emit a fresh mesh).
    // `face_tags`, when non-empty, carries one tag list per face.
    static Mesh from_pydata(const std::vector<std::array<double, 3>>& positions,
                            const std::vector<std::vector<int>>& faces,
                            const std::vector<std::vector<std::string>>& face_tags = {});
    // A fresh, equivalent mesh (this type is move-only, so this is the deep copy).
    Mesh copy() const;

    // Invariants
    int euler() const {
        return static_cast<int>(verts_.size()) - static_cast<int>(edges_.size()) +
               static_cast<int>(faces_.size());
    }
    bool is_closed_manifold() const;     // every edge has exactly two incident loops
    void validate() const;               // throws std::runtime_error on any defect

    std::size_t num_verts() const { return verts_.size(); }
    std::size_t num_edges() const { return edges_.size(); }
    std::size_t num_faces() const { return faces_.size(); }
    const std::vector<std::unique_ptr<Vert>>& verts() const { return verts_; }
    const std::vector<std::unique_ptr<Edge>>& edges() const { return edges_; }
    const std::vector<std::unique_ptr<Face>>& faces() const { return faces_; }

private:
    Edge* get_edge(Vert* a, Vert* b);
    static void radial_insert(Edge* e, Loop* lp);

    std::vector<std::unique_ptr<Vert>> verts_;
    std::vector<std::unique_ptr<Edge>> edges_;
    std::vector<std::unique_ptr<Loop>> loops_;
    std::vector<std::unique_ptr<Face>> faces_;
    std::unordered_map<std::int64_t, Edge*> edge_map_;  // key = packed (min,max) vert id
    int loop_id_ = 0;
};

// Unit face normal via Newell's method (robust for non-planar polygons).
std::array<double, 3> face_normal(const Mesh& m, const Face* f);

// The face with the greatest centroid z — a convenient default selector until the
// full selection-as-query engine lands.
const Face* top_face(const Mesh& m);

// Centroid of a face, and the face whose centroid is nearest a point. A picked
// point is a re-evaluable selector: deterministic op-log replay reproduces the
// same geometry, so "nearest face to P" resolves to the same face each rebuild.
std::array<double, 3> face_centroid(const Mesh& m, const Face* f);
const Face* nearest_face(const Mesh& m, const std::array<double, 3>& p);

// Primitives.
// Cube: axis-aligned, centered at origin, outward-consistent winding (euler == 2).
Mesh make_cube(double size = 1.0);
// Cylinder: an n-gon prism (two `sides`-vertex rings + caps); a closed 2-manifold.
Mesh make_cylinder(int sides = 24, double radius = 0.5, double height = 1.0);

// Operators (built on the owned topology).
// One level of Catmull-Clark subdivision — the classic test that a radial-edge
// kernel actually works (face/edge/vertex points + the standard boundary rule,
// found by walking the topology, then rebuilt as quads).
Mesh catmull_clark(const Mesh& mesh);

// Region operators (emit a fresh mesh; descendant faces inherit their parent's
// tags). extrude: each region vertex moves along the average of its incident
// region-face normals, side walls bridge boundary edges, orphaned interior verts
// are compacted away; the lifted caps additionally get `mark` (when non-empty) —
// the durable handle the next op selects. inset: a centroid-proportional smaller
// copy of each face, ringed by border quads (thickness clamped to (0,1)); the
// inner faces get `mark`.
Mesh extrude(const Mesh& mesh, const std::vector<const Face*>& region, double distance = 0.5,
             const std::string& mark = "");
Mesh inset(const Mesh& mesh, const std::vector<const Face*>& region, double thickness = 0.3,
           const std::string& mark = "");
// bevel: an inset ring of `width` whose inner face is offset `depth` along the
// face normal — the border quads slant into a chamfer (the face-region analogue
// of an edge bevel; Blender's Inset Faces = Thickness + Depth). depth=0 is a
// plain inset. Same topology as inset, so euler/manifold are preserved.
Mesh bevel(const Mesh& mesh, const std::vector<const Face*>& region, double width = 0.2,
           double depth = 0.1, const std::string& mark = "");

}  // namespace mirage
