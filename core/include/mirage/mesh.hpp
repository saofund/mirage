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

// A PBR material assigned to a face (base colour + metallic + roughness). `set`
// false means "use the renderer's default material". Mirrors a face's
// attrs["material"] on the Python side.
struct Material {
    std::array<double, 3> color{0.8, 0.8, 0.8};
    double metallic = 0.0;
    double roughness = 0.5;
    bool set = false;
};

struct Face {
    int id = 0;
    Loop* loop = nullptr;  // entry into the loop cycle
    // Durable handles. Operators rebuild the mesh and renumber every element
    // (the Topological Naming Problem), so an index dies after one op — tags are
    // copied to descendant faces across rebuilds and are what selectors (`tag`,
    // `last_created`) resolve. Mirrors Python Face.attrs["tags"].
    std::vector<std::string> tags;
    Material material;  // per-face PBR material (assigned by the `material` op)
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
// Plane: a single quad in z=0 (an OPEN mesh: 4 boundary edges). size_y <= 0 => square.
Mesh make_plane(double size_x = 1.0, double size_y = -1.0);
// UV sphere: `segments` longitudinal slices, `rings` latitudinal bands. Triangle
// fans at the poles, quad strips between; closed 2-manifold (euler 2). Vertex order
// matches the Python kernel: north pole, rings-1 interior circles, south pole.
Mesh make_uv_sphere(int segments = 24, int rings = 16, double radius = 0.5);
// Cone: an `sides`-gon base (z=-h/2) capped by triangles meeting at the apex (z=+h/2).
Mesh make_cone(int sides = 24, double radius = 0.5, double height = 1.0);
// Torus: genus-1 closed manifold (euler 0). `major_segments` around the ring,
// `minor_segments` around the tube; vertex index = i*minor + j (matches Python).
Mesh make_torus(int major_segments = 24, int minor_segments = 12,
                double major_radius = 0.5, double minor_radius = 0.2);
// Grid: a subdivided quad in z=0 (an OPEN mesh). `x_div` by `y_div` cells; vertex
// index = iy*(x_div+1)+ix (matches Python). size_y <= 0 => square; y_div <= 0 => x_div.
Mesh make_grid(double size_x = 1.0, double size_y = -1.0, int x_div = 10, int y_div = -1);

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
// loop_cut: from a seed quad, walk the ring of quads whose shared edges run along
// `axis` and bisect each, threading a watertight loop of midpoint vertices (the
// classic hard-surface loop cut). N-gons (e.g. cylinder caps) stop the walk.
Mesh loop_cut(const Mesh& mesh, const std::vector<const Face*>& seed, const std::string& axis = "z",
              const std::string& mark = "");
// edge_bevel: round/chamfer the selected edges. Each face is shrunk at its
// bevelled corners, each bevelled edge becomes a chamfer quad, each bevelled
// vertex a corner face. Only fully-selected vertex stars are bevelled (a fixpoint
// keeps it watertight), so a cube with all edges selected -> a chamfered cube.
Mesh edge_bevel(const Mesh& mesh, const std::vector<Edge*>& edges, double width = 0.15,
                const std::string& mark = "");

// Open-mesh operators — boundary edges (1 incident loop) are first-class.
// delete_faces: remove the selected faces (opens the mesh); orphans compacted.
Mesh delete_faces(const Mesh& mesh, const std::vector<const Face*>& faces);
// fill_holes: cap every boundary loop with a single n-gon (close the holes).
Mesh fill_holes(const Mesh& mesh, const std::string& mark = "");
// bridge_faces: delete two vertex/edge-disjoint faces of equal vertex count and
// connect their rims with a ring of quads (a tunnel between separate openings).
Mesh bridge_faces(const Mesh& mesh, const std::vector<const Face*>& faces, const std::string& mark = "");

// Whole-mesh operators.
// solidify: give a surface thickness — an inner shell offset along the inverted
// vertex normals (reversed winding) + a wall quad per boundary edge, so an open
// surface (plane/grid/open box) becomes watertight. Verts: outer then inner.
Mesh solidify(const Mesh& mesh, double thickness = 0.1, const std::string& mark = "");
// mirror: reflect across the axis=0 plane and weld the seam (verts on the plane are
// shared; off-plane verts get a reflected, reversed-winding copy).
Mesh mirror(const Mesh& mesh, const std::string& axis = "x", const std::string& mark = "");
// array: `count` copies, copy c shifted by offset*c (disjoint; last copy gets mark).
Mesh array(const Mesh& mesh, int count = 3, const std::array<double, 3>& offset = {1.1, 0.0, 0.0},
           const std::string& mark = "");
// bisect: cut by a plane (point, normal); keep the half the normal points away from,
// clipping crossing faces (a shared vertex per crossing edge). fill caps the cut —
// the foundation of a plane/mesh boolean.
Mesh bisect(const Mesh& mesh, const std::array<double, 3>& point = {0, 0, 0},
            const std::array<double, 3>& normal = {0, 0, 1}, bool fill = false,
            const std::string& mark = "");
// spin (lathe): revolve a profile's boundary edges around an axis. Axis verts weld
// to a pole; angle>=360 wraps into a watertight surface of revolution, a partial
// angle leaves an open swept sheet.
Mesh spin(const Mesh& mesh, const std::string& axis = "z", int steps = 24,
          double angle = 360.0, const std::string& mark = "");
// screw (helical sweep — thread/spring/auger): like spin, but each angular step also
// advances along the axis, so the profile climbs into a helix. `turns` full revolutions,
// `height` is the axial rise per turn. Always open (the helix never wraps closed).
Mesh screw(const Mesh& mesh, const std::string& axis = "z", int steps = 24, int turns = 1,
           double height = 1.0, double angle = 360.0, const std::string& mark = "");

}  // namespace mirage
