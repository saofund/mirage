#include "mirage/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace mirage {

Vert* Mesh::add_vert(double x, double y, double z) {
    auto v = std::make_unique<Vert>();
    v->id = static_cast<int>(verts_.size());
    v->co = {x, y, z};
    Vert* raw = v.get();
    verts_.push_back(std::move(v));
    return raw;
}

Edge* Mesh::get_edge(Vert* a, Vert* b) {
    int lo = a->id < b->id ? a->id : b->id;
    int hi = a->id < b->id ? b->id : a->id;
    std::int64_t key = (static_cast<std::int64_t>(lo) << 32) | static_cast<std::int64_t>(hi);
    auto it = edge_map_.find(key);
    if (it != edge_map_.end()) return it->second;
    auto e = std::make_unique<Edge>();
    e->id = static_cast<int>(edges_.size());
    e->v1 = a;
    e->v2 = b;
    Edge* raw = e.get();
    edges_.push_back(std::move(e));
    edge_map_[key] = raw;
    return raw;
}

void Mesh::radial_insert(Edge* e, Loop* lp) {
    if (e->loop == nullptr) {
        e->loop = lp;
        lp->radial_next = lp;
        lp->radial_prev = lp;
    } else {
        Loop* tail = e->loop->radial_prev;
        tail->radial_next = lp;
        lp->radial_prev = tail;
        lp->radial_next = e->loop;
        e->loop->radial_prev = lp;
    }
}

Face* Mesh::add_face(const std::vector<Vert*>& verts, std::vector<std::string> tags) {
    const std::size_t n = verts.size();
    if (n < 3) throw std::invalid_argument("a face needs >= 3 verts");
    if (std::set<Vert*>(verts.begin(), verts.end()).size() != n)
        throw std::invalid_argument("face has repeated vertices");

    auto face = std::make_unique<Face>();
    face->id = static_cast<int>(faces_.size());
    face->tags = std::move(tags);
    Face* fraw = face.get();

    std::vector<Loop*> loops(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto lp = std::make_unique<Loop>();
        lp->id = loop_id_++;
        lp->vert = verts[i];
        lp->face = fraw;
        if (verts[i]->loop == nullptr) verts[i]->loop = lp.get();
        loops[i] = lp.get();
        loops_.push_back(std::move(lp));
    }
    for (std::size_t i = 0; i < n; ++i) {
        loops[i]->next = loops[(i + 1) % n];
        loops[i]->prev = loops[(i + n - 1) % n];
        Edge* e = get_edge(verts[i], verts[(i + 1) % n]);  // outgoing edge of this corner
        loops[i]->edge = e;
        radial_insert(e, loops[i]);
    }
    fraw->loop = loops[0];
    faces_.push_back(std::move(face));
    return fraw;
}

std::vector<Loop*> Mesh::face_loops(const Face* f) const {
    std::vector<Loop*> out;
    Loop* start = f->loop;
    Loop* lp = start;
    do {
        out.push_back(lp);
        lp = lp->next;
    } while (lp != start);
    return out;
}

std::vector<Vert*> Mesh::face_verts(const Face* f) const {
    std::vector<Vert*> out;
    for (Loop* lp : face_loops(f)) out.push_back(lp->vert);
    return out;
}

std::vector<Loop*> Mesh::edge_loops(const Edge* e) const {
    std::vector<Loop*> out;
    if (e->loop == nullptr) return out;
    Loop* lp = e->loop;
    do {
        out.push_back(lp);
        lp = lp->radial_next;
    } while (lp != e->loop);
    return out;
}

bool Mesh::is_closed_manifold() const {
    for (const auto& e : edges_)
        if (edge_loops(e.get()).size() != 2) return false;
    return true;
}

static double newell_sq(const std::vector<Vert*>& vs) {
    double nx = 0, ny = 0, nz = 0;
    const std::size_t n = vs.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = vs[i]->co;
        const auto& b = vs[(i + 1) % n]->co;
        nx += (a[1] - b[1]) * (a[2] + b[2]);
        ny += (a[2] - b[2]) * (a[0] + b[0]);
        nz += (a[0] - b[0]) * (a[1] + b[1]);
    }
    return nx * nx + ny * ny + nz * nz;
}

void Mesh::validate() const {
    auto fail = [](const std::string& msg) { throw std::runtime_error(msg); };

    for (const auto& f : faces_) {
        std::vector<Loop*> loops = face_loops(f.get());
        if (loops.size() < 3) fail("degenerate face");
        for (Loop* lp : loops) {
            if (lp->face != f.get()) fail("loop/face mismatch");
            if (lp->next->prev != lp) fail("loop cycle broken");
            if (lp->vert != lp->edge->v1 && lp->vert != lp->edge->v2) fail("loop/edge mismatch");
        }
    }
    for (const auto& e : edges_) {
        for (Loop* lp : edge_loops(e.get())) {
            if (lp->edge != e.get()) fail("radial/edge mismatch");
            if (lp->radial_next->radial_prev != lp) fail("radial cycle broken");
        }
    }
    std::set<int> referenced;
    for (const auto& f : faces_)
        for (Loop* lp : face_loops(f.get())) referenced.insert(lp->vert->id);
    for (const auto& v : verts_)
        if (referenced.find(v->id) == referenced.end())
            fail("orphan vertex " + std::to_string(v->id) + " (no incident face)");
    for (const auto& f : faces_)
        if (newell_sq(face_verts(f.get())) <= 1e-16)
            fail("degenerate face " + std::to_string(f->id));
}

Mesh make_cube(double size) {
    Mesh m;
    const double s = size / 2.0;
    Vert* v[8] = {
        m.add_vert(-s, -s, -s), m.add_vert(s, -s, -s), m.add_vert(s, s, -s), m.add_vert(-s, s, -s),
        m.add_vert(-s, -s, s),  m.add_vert(s, -s, s),  m.add_vert(s, s, s),  m.add_vert(-s, s, s),
    };
    // outward-consistent winding -> each undirected edge is used by exactly two faces
    m.add_face({v[0], v[3], v[2], v[1]});  // -z (bottom)
    m.add_face({v[4], v[5], v[6], v[7]});  // +z (top)
    m.add_face({v[0], v[1], v[5], v[4]});  // -y (front)
    m.add_face({v[1], v[2], v[6], v[5]});  // +x (right)
    m.add_face({v[2], v[3], v[7], v[6]});  // +y (back)
    m.add_face({v[3], v[0], v[4], v[7]});  // -x (left)
    return m;
}

std::vector<Face*> Mesh::edge_faces(const Edge* e) const {
    std::vector<Face*> out;
    for (Loop* lp : edge_loops(e)) out.push_back(lp->face);
    return out;
}

Mesh Mesh::from_pydata(const std::vector<std::array<double, 3>>& positions,
                       const std::vector<std::vector<int>>& faces,
                       const std::vector<std::vector<std::string>>& face_tags) {
    Mesh m;
    std::vector<Vert*> vs;
    vs.reserve(positions.size());
    for (const auto& p : positions) vs.push_back(m.add_vert(p[0], p[1], p[2]));
    for (std::size_t i = 0; i < faces.size(); ++i) {
        std::vector<Vert*> fv;
        fv.reserve(faces[i].size());
        for (int idx : faces[i]) fv.push_back(vs[idx]);
        m.add_face(fv, i < face_tags.size() ? face_tags[i] : std::vector<std::string>{});
    }
    return m;
}

std::array<double, 3> face_normal(const Mesh& m, const Face* f) {
    std::vector<Vert*> vs = m.face_verts(f);
    double nx = 0, ny = 0, nz = 0;
    const std::size_t n = vs.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = vs[i]->co;
        const auto& b = vs[(i + 1) % n]->co;
        nx += (a[1] - b[1]) * (a[2] + b[2]);
        ny += (a[2] - b[2]) * (a[0] + b[0]);
        nz += (a[0] - b[0]) * (a[1] + b[1]);
    }
    double mag = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (mag == 0.0) mag = 1.0;
    return {nx / mag, ny / mag, nz / mag};
}

double face_area(const Mesh& m, const Face* f) {
    std::vector<Vert*> vs = m.face_verts(f);
    double nx = 0, ny = 0, nz = 0;
    const std::size_t n = vs.size();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& a = vs[i]->co;
        const auto& b = vs[(i + 1) % n]->co;
        nx += (a[1] - b[1]) * (a[2] + b[2]);
        ny += (a[2] - b[2]) * (a[0] + b[0]);
        nz += (a[0] - b[0]) * (a[1] + b[1]);
    }
    return 0.5 * std::sqrt(nx * nx + ny * ny + nz * nz);
}

const Face* top_face(const Mesh& m) {
    const Face* best = nullptr;
    double bz = -1e30;
    for (const auto& f : m.faces()) {
        std::vector<Vert*> vs = m.face_verts(f.get());
        double cz = 0;
        for (Vert* v : vs) cz += v->co[2];
        cz /= static_cast<double>(vs.size());
        if (cz > bz) { bz = cz; best = f.get(); }
    }
    return best;
}

std::array<double, 3> face_centroid(const Mesh& m, const Face* f) {
    std::vector<Vert*> vs = m.face_verts(f);
    std::array<double, 3> c{0, 0, 0};
    for (Vert* v : vs) { c[0] += v->co[0]; c[1] += v->co[1]; c[2] += v->co[2]; }
    const double n = static_cast<double>(vs.size());
    return {c[0] / n, c[1] / n, c[2] / n};
}

const Face* nearest_face(const Mesh& m, const std::array<double, 3>& p) {
    const Face* best = nullptr;
    double bd = 1e30;
    for (const auto& f : m.faces()) {
        auto c = face_centroid(m, f.get());
        const double d = (c[0]-p[0])*(c[0]-p[0]) + (c[1]-p[1])*(c[1]-p[1]) + (c[2]-p[2])*(c[2]-p[2]);
        if (d < bd) { bd = d; best = f.get(); }
    }
    return best;
}

Mesh make_plane(double sx, double sy) {
    if (sy <= 0) sy = sx;
    const double hx = sx / 2.0, hy = sy / 2.0;
    std::vector<std::array<double, 3>> p = {{-hx, -hy, 0}, {hx, -hy, 0}, {hx, hy, 0}, {-hx, hy, 0}};
    return Mesh::from_pydata(p, {{0, 1, 2, 3}});
}

Mesh make_uv_sphere(int segments, int rings, double radius) {
    constexpr double PI = 3.14159265358979323846;
    const int S = std::max(segments, 3), R = std::max(rings, 2);
    Mesh m;
    Vert* north = m.add_vert(0.0, 0.0, radius);
    std::vector<std::vector<Vert*>> circ;       // interior rings 0..R-2
    for (int i = 1; i < R; ++i) {
        const double theta = PI * i / R;
        const double z = radius * std::cos(theta), rr = radius * std::sin(theta);
        std::vector<Vert*> ring;
        for (int j = 0; j < S; ++j) {
            const double a = 2.0 * PI * j / S;
            ring.push_back(m.add_vert(rr * std::cos(a), rr * std::sin(a), z));
        }
        circ.push_back(ring);
    }
    Vert* south = m.add_vert(0.0, 0.0, -radius);
    for (int j = 0; j < S; ++j) {               // north cap fan
        const int jn = (j + 1) % S;
        m.add_face({circ[0][j], circ[0][jn], north});
    }
    for (int i = 0; i < R - 2; ++i) {           // quad bands
        auto& up = circ[i]; auto& lo = circ[i + 1];
        for (int j = 0; j < S; ++j) {
            const int jn = (j + 1) % S;
            m.add_face({lo[j], lo[jn], up[jn], up[j]});
        }
    }
    for (int j = 0; j < S; ++j) {               // south cap fan
        const int jn = (j + 1) % S;
        m.add_face({south, circ[R - 2][jn], circ[R - 2][j]});
    }
    return m;
}

Mesh make_cone(int sides, double radius, double height) {
    constexpr double PI = 3.14159265358979323846;
    const double half = height / 2.0;
    Mesh m;
    std::vector<Vert*> vb;
    for (int i = 0; i < sides; ++i) {
        const double a = 2.0 * PI * i / sides;
        vb.push_back(m.add_vert(radius * std::cos(a), radius * std::sin(a), -half));
    }
    Vert* apex = m.add_vert(0.0, 0.0, half);
    std::vector<Vert*> base(vb.rbegin(), vb.rend());
    m.add_face(base);                            // base cap (normal -z)
    for (int i = 0; i < sides; ++i) {
        const int j = (i + 1) % sides;
        m.add_face({vb[i], vb[j], apex});        // side triangle
    }
    return m;
}

Mesh make_torus(int major_segments, int minor_segments, double major_radius, double minor_radius) {
    constexpr double PI = 3.14159265358979323846;
    const int M = std::max(major_segments, 3), N = std::max(minor_segments, 3);
    Mesh m;
    std::vector<std::vector<Vert*>> grid;
    for (int i = 0; i < M; ++i) {
        const double u = 2.0 * PI * i / M;
        std::vector<Vert*> row;
        for (int j = 0; j < N; ++j) {
            const double v = 2.0 * PI * j / N;
            const double cr = major_radius + minor_radius * std::cos(v);
            row.push_back(m.add_vert(cr * std::cos(u), cr * std::sin(u), minor_radius * std::sin(v)));
        }
        grid.push_back(row);
    }
    for (int i = 0; i < M; ++i) {
        const int ii = (i + 1) % M;
        for (int j = 0; j < N; ++j) {
            const int jn = (j + 1) % N;
            m.add_face({grid[i][j], grid[i][jn], grid[ii][jn], grid[ii][j]});
        }
    }
    return m;
}

Mesh make_grid(double size_x, double size_y, int x_div, int y_div) {
    if (size_y <= 0) size_y = size_x;
    const int nx = std::max(x_div, 1), ny = std::max(y_div <= 0 ? x_div : y_div, 1);
    const double hx = size_x / 2.0, hy = size_y / 2.0;
    std::vector<std::array<double, 3>> pos;
    for (int iy = 0; iy <= ny; ++iy)
        for (int ix = 0; ix <= nx; ++ix)
            pos.push_back({-hx + size_x * ix / nx, -hy + size_y * iy / ny, 0.0});
    std::vector<std::vector<int>> faces;
    for (int iy = 0; iy < ny; ++iy)
        for (int ix = 0; ix < nx; ++ix) {
            const int a = iy * (nx + 1) + ix;
            faces.push_back({a, a + 1, a + nx + 2, a + nx + 1});
        }
    return Mesh::from_pydata(pos, faces);
}

Mesh make_cylinder(int sides, double radius, double height) {
    constexpr double PI = 3.14159265358979323846;
    std::vector<std::array<double, 3>> pos;
    const double h = height / 2.0;
    for (int i = 0; i < sides; ++i) {  // bottom ring: indices 0 .. sides-1 (matches Python)
        const double a = 2.0 * PI * i / sides;
        pos.push_back({radius * std::cos(a), radius * std::sin(a), -h});
    }
    for (int i = 0; i < sides; ++i) {  // top ring: indices sides .. 2*sides-1
        const double a = 2.0 * PI * i / sides;
        pos.push_back({radius * std::cos(a), radius * std::sin(a), h});
    }
    std::vector<std::vector<int>> faces;
    std::vector<int> bot, top;
    for (int i = 0; i < sides; ++i) { bot.push_back(i); top.push_back(sides + i); }
    faces.emplace_back(bot.rbegin(), bot.rend());  // -z cap (reversed for outward normal)
    faces.push_back(top);                          // +z cap
    for (int i = 0; i < sides; ++i) {              // side quads (outward)
        const int j = (i + 1) % sides;
        faces.push_back({i, j, sides + j, sides + i});
    }
    return Mesh::from_pydata(pos, faces);
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------
namespace {
using A3 = std::array<double, 3>;
using Tags = std::vector<std::string>;
A3 a3add(const A3& a, const A3& b) { return {a[0] + b[0], a[1] + b[1], a[2] + b[2]}; }
A3 a3scale(const A3& a, double s) { return {a[0] * s, a[1] * s, a[2] * s}; }

// A face's tags, optionally extended by `mark` — the per-descendant copy that
// carries durable handles across an operator's rebuild (Python _copy_attrs).
Tags copy_tags(const Face* f, const std::string& mark = "") {
    Tags out = f->tags;
    if (!mark.empty()) out.push_back(mark);
    return out;
}

// Drop positions referenced by no face and remap indices (mirrors Python _compact),
// then rebuild — used by operators that orphan interior verts (extrude).
Mesh build_compact(const std::vector<A3>& pos, const std::vector<std::vector<int>>& faces,
                   const std::vector<Tags>& face_tags) {
    std::set<int> used;
    for (const auto& f : faces)
        for (int i : f) used.insert(i);
    std::unordered_map<int, int> remap;
    std::vector<A3> np;
    np.reserve(used.size());
    for (int old_i : used) { remap[old_i] = static_cast<int>(np.size()); np.push_back(pos[old_i]); }
    std::vector<std::vector<int>> nf;
    nf.reserve(faces.size());
    for (const auto& f : faces) {
        std::vector<int> g;
        g.reserve(f.size());
        for (int i : f) g.push_back(remap[i]);
        nf.push_back(g);
    }
    return Mesh::from_pydata(np, nf, face_tags);
}
}  // namespace

Mesh catmull_clark(const Mesh& mesh) {
    if (mesh.faces().empty()) return Mesh();

    // face points = face centroids
    std::unordered_map<const Face*, A3> fp;
    for (const auto& f : mesh.faces()) {
        A3 c{0, 0, 0};
        std::vector<Vert*> vs = mesh.face_verts(f.get());
        for (Vert* v : vs) c = a3add(c, v->co);
        fp[f.get()] = a3scale(c, 1.0 / static_cast<double>(vs.size()));
    }

    // edge midpoints, edge points, edge->faces adjacency
    std::unordered_map<const Edge*, A3> emid, ept;
    std::unordered_map<const Edge*, std::vector<Face*>> eadj;
    for (const auto& e : mesh.edges()) {
        std::vector<Face*> fs = mesh.edge_faces(e.get());
        eadj[e.get()] = fs;
        A3 mid = a3scale(a3add(e->v1->co, e->v2->co), 0.5);
        emid[e.get()] = mid;
        if (fs.size() == 2)
            ept[e.get()] = a3scale(a3add(a3add(e->v1->co, e->v2->co), a3add(fp[fs[0]], fp[fs[1]])), 0.25);
        else
            ept[e.get()] = mid;  // boundary edge
    }

    // per-vertex incident edges / faces
    std::unordered_map<const Vert*, std::vector<const Edge*>> ve;
    std::unordered_map<const Vert*, std::vector<const Face*>> vf;
    for (const auto& v : mesh.verts()) { ve[v.get()]; vf[v.get()]; }
    for (const auto& e : mesh.edges()) { ve[e->v1].push_back(e.get()); ve[e->v2].push_back(e.get()); }
    for (const auto& f : mesh.faces())
        for (Loop* lp : mesh.face_loops(f.get())) vf[lp->vert].push_back(f.get());

    // re-weighted vertex points
    std::unordered_map<const Vert*, A3> nv;
    for (const auto& vp : mesh.verts()) {
        const Vert* v = vp.get();
        const A3 P = v->co;
        std::vector<const Edge*> boundary;
        for (const Edge* e : ve[v]) if (eadj[e].size() < 2) boundary.push_back(e);
        if (!boundary.empty()) {  // standard cubic B-spline boundary rule: (6P + sum nb)/(6+k)
            A3 s = a3scale(P, 6.0);
            for (const Edge* e : boundary) s = a3add(s, e->other(v)->co);
            nv[v] = a3scale(s, 1.0 / (6.0 + static_cast<double>(boundary.size())));
        } else {
            const double n = static_cast<double>(ve[v].size());
            A3 F{0, 0, 0};
            for (const Face* f : vf[v]) F = a3add(F, fp[f]);
            F = a3scale(F, 1.0 / static_cast<double>(vf[v].size()));
            A3 R{0, 0, 0};
            for (const Edge* e : ve[v]) R = a3add(R, emid[e]);
            R = a3scale(R, 1.0 / static_cast<double>(ve[v].size()));
            nv[v] = a3scale(a3add(a3add(F, a3scale(R, 2.0)), a3scale(P, n - 3.0)), 1.0 / n);
        }
    }

    // assemble new mesh: vertex-points, then edge-points, then face-points; each
    // original corner becomes a quad [vertexPt, edgePt, facePt, prevEdgePt].
    std::vector<A3> pos;
    std::unordered_map<const Vert*, int> iv;
    std::unordered_map<const Edge*, int> ie;
    std::unordered_map<const Face*, int> jf;
    for (const auto& v : mesh.verts()) { iv[v.get()] = static_cast<int>(pos.size()); pos.push_back(nv[v.get()]); }
    for (const auto& e : mesh.edges()) { ie[e.get()] = static_cast<int>(pos.size()); pos.push_back(ept[e.get()]); }
    for (const auto& f : mesh.faces()) { jf[f.get()] = static_cast<int>(pos.size()); pos.push_back(fp[f.get()]); }

    std::vector<std::vector<int>> quads;
    std::vector<std::vector<std::string>> quad_tags;
    for (const auto& f : mesh.faces())
        for (Loop* lp : mesh.face_loops(f.get())) {  // child quads inherit the parent face's tags
            quads.push_back({iv[lp->vert], ie[lp->edge], jf[f.get()], ie[lp->prev->edge]});
            quad_tags.push_back(f->tags);
        }

    return Mesh::from_pydata(pos, quads, quad_tags);
}

Mesh Mesh::copy() const {
    std::vector<A3> pos;
    pos.reserve(verts_.size());
    for (const auto& v : verts_) pos.push_back(v->co);
    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    faces.reserve(faces_.size());
    tags.reserve(faces_.size());
    for (const auto& f : faces_) {
        std::vector<int> fv;
        for (Loop* lp : face_loops(f.get())) fv.push_back(lp->vert->id);
        faces.push_back(fv);
        tags.push_back(f->tags);
    }
    return from_pydata(pos, faces, tags);
}

// Region faces in a deterministic order (by id) — so a replayed op-log yields a
// byte-identical mesh every run, not one that depends on heap pointer values.
static std::vector<const Face*> region_in_id_order(const std::set<const Face*>& region) {
    std::vector<const Face*> v(region.begin(), region.end());
    std::sort(v.begin(), v.end(), [](const Face* a, const Face* b) { return a->id < b->id; });
    return v;
}

Mesh extrude(const Mesh& mesh, const std::vector<const Face*>& region_v, double distance,
             const std::string& mark) {
    std::set<const Face*> region(region_v.begin(), region_v.end());
    if (region.empty() || std::abs(distance) < 1e-9) return mesh.copy();

    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);

    // each region vertex accumulates the normals of its incident region faces
    std::unordered_map<const Face*, A3> fn;
    for (const Face* f : region) fn[f] = face_normal(mesh, f);
    std::unordered_map<int, A3> vacc;
    for (const Face* f : region)
        for (Vert* v : mesh.face_verts(f)) {
            auto it = vacc.find(v->id);
            vacc[v->id] = (it == vacc.end()) ? fn[f] : a3add(it->second, fn[f]);
        }

    std::vector<A3> new_pos = pos;
    std::unordered_map<int, int> newid;
    std::vector<int> vids;
    for (const auto& kv : vacc) vids.push_back(kv.first);
    std::sort(vids.begin(), vids.end());  // deterministic new-vertex ordering (matches Python)
    for (int vid : vids) {
        const A3 n = vacc[vid];
        const double nl = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        const A3 d = nl > 1e-9 ? a3scale(n, distance / nl) : A3{0, 0, 0};
        newid[vid] = static_cast<int>(new_pos.size());
        new_pos.push_back({pos[vid][0] + d[0], pos[vid][1] + d[1], pos[vid][2] + d[2]});
    }

    std::vector<std::vector<int>> new_faces;
    std::vector<Tags> new_tags;
    for (const auto& f : mesh.faces())  // untouched faces
        if (!region.count(f.get())) {
            std::vector<int> fv;
            for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
            new_faces.push_back(fv);
            new_tags.push_back(copy_tags(f.get()));
        }
    for (const auto& e : mesh.edges()) {  // side walls bridge edges with one region face
        std::vector<Face*> adj;
        for (Face* f : mesh.edge_faces(e.get())) if (region.count(f)) adj.push_back(f);
        if (adj.size() == 1) {
            Loop* lp = nullptr;
            for (Loop* l : mesh.face_loops(adj[0])) if (l->edge == e.get()) { lp = l; break; }
            const int a = lp->vert->id, b = lp->next->vert->id;
            new_faces.push_back({a, b, newid[b], newid[a]});
            new_tags.push_back(copy_tags(adj[0]));
        }
    }
    for (const Face* f : region_in_id_order(region)) {  // lifted caps, tagged with `mark`
        std::vector<int> cap;
        for (Loop* lp : mesh.face_loops(f)) cap.push_back(newid[lp->vert->id]);
        new_faces.push_back(cap);
        new_tags.push_back(copy_tags(f, mark));
    }
    return build_compact(new_pos, new_faces, new_tags);
}

Mesh inset(const Mesh& mesh, const std::vector<const Face*>& region_v, double thickness,
           const std::string& mark) {
    std::set<const Face*> region(region_v.begin(), region_v.end());
    if (region.empty()) return mesh.copy();
    thickness = std::min(std::max(thickness, 1e-3), 0.999);  // avoid degenerate / bowtie

    std::vector<A3> new_pos;
    for (const auto& v : mesh.verts()) new_pos.push_back(v->co);
    std::vector<std::vector<int>> new_faces;
    std::vector<Tags> new_tags;
    for (const auto& f : mesh.faces())  // untouched faces
        if (!region.count(f.get())) {
            std::vector<int> fv;
            for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
            new_faces.push_back(fv);
            new_tags.push_back(copy_tags(f.get()));
        }
    for (const Face* f : region_in_id_order(region)) {
        std::vector<int> vids;
        for (Loop* lp : mesh.face_loops(f)) vids.push_back(lp->vert->id);
        A3 c{0, 0, 0};
        for (int i : vids) c = a3add(c, new_pos[i]);
        c = a3scale(c, 1.0 / static_cast<double>(vids.size()));
        std::vector<int> inner;
        for (int i : vids) {
            const A3 p = new_pos[i];  // copy (push_back below may reallocate)
            inner.push_back(static_cast<int>(new_pos.size()));
            new_pos.push_back({p[0] + (c[0] - p[0]) * thickness,
                               p[1] + (c[1] - p[1]) * thickness,
                               p[2] + (c[2] - p[2]) * thickness});
        }
        const int n = static_cast<int>(vids.size());
        for (int k = 0; k < n; ++k) {  // border quads
            new_faces.push_back({vids[k], vids[(k + 1) % n], inner[(k + 1) % n], inner[k]});
            new_tags.push_back(copy_tags(f));
        }
        new_faces.push_back(inner);  // inner face, tagged with `mark`
        new_tags.push_back(copy_tags(f, mark));
    }
    return Mesh::from_pydata(new_pos, new_faces, new_tags);
}

// --- open-mesh operators -----------------------------------------------------
Mesh delete_faces(const Mesh& mesh, const std::vector<const Face*>& faces) {
    std::set<const Face*> rem(faces.begin(), faces.end());
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    std::vector<std::vector<int>> nf;
    std::vector<Tags> nt;
    for (const auto& f : mesh.faces())
        if (!rem.count(f.get())) {
            std::vector<int> fv;
            for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
            nf.push_back(fv); nt.push_back(copy_tags(f.get()));
        }
    if (nf.empty()) return Mesh();
    return build_compact(pos, nf, nt);  // drops the orphaned verts
}

// Chain the boundary edges (1 incident loop) into ordered vertex cycles, each
// wound opposite to the existing face (so a fill face is outward + manifold).
static std::vector<std::vector<int>> boundary_loops(const Mesh& mesh) {
    std::map<int, int> nxt;  // vert id -> next vert id around the hole (deterministic order)
    for (const auto& e : mesh.edges()) {
        auto loops = mesh.edge_loops(e.get());
        if (loops.size() == 1) nxt[loops[0]->next->vert->id] = loops[0]->vert->id;  // hole: b->a
    }
    std::vector<std::vector<int>> out;
    std::set<int> seen;
    for (const auto& kv : nxt) {
        int start = kv.first;
        if (seen.count(start)) continue;
        std::vector<int> loop;
        int cur = start;
        while (cur >= 0 && !seen.count(cur)) {
            seen.insert(cur); loop.push_back(cur);
            auto it = nxt.find(cur);
            cur = (it == nxt.end()) ? -1 : it->second;
            if (cur == start) break;
        }
        if (loop.size() >= 3) out.push_back(loop);
    }
    return out;
}

Mesh fill_holes(const Mesh& mesh, const std::string& mark) {
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    std::vector<std::vector<int>> nf;
    std::vector<Tags> nt;
    for (const auto& f : mesh.faces()) {
        std::vector<int> fv;
        for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
        nf.push_back(fv); nt.push_back(copy_tags(f.get()));
    }
    for (auto& loop : boundary_loops(mesh)) {
        nf.push_back(loop);
        nt.push_back(mark.empty() ? Tags{} : Tags{mark});
    }
    return Mesh::from_pydata(pos, nf, nt);
}

Mesh bridge_faces(const Mesh& mesh, const std::vector<const Face*>& faces, const std::string& mark) {
    if (faces.size() < 2) return mesh.copy();
    const Face* fa = faces[0];
    const Face* fb = faces[1];
    std::vector<int> la, lb;
    for (Loop* lp : mesh.face_loops(fa)) la.push_back(lp->vert->id);
    for (Loop* lp : mesh.face_loops(fb)) lb.push_back(lp->vert->id);
    const int n = static_cast<int>(la.size());
    if (fa == fb || static_cast<int>(lb.size()) != n) return mesh.copy();
    std::set<int> sa(la.begin(), la.end()), sb(lb.begin(), lb.end());
    for (int v : la) if (sb.count(v)) return mesh.copy();           // shared verts
    for (const auto& e : mesh.edges()) {                            // edge joining the rims
        const bool a1 = sa.count(e->v1->id), b1 = sb.count(e->v1->id);
        const bool a2 = sa.count(e->v2->id), b2 = sb.count(e->v2->id);
        if ((a1 && b2) || (b1 && a2)) return mesh.copy();
    }
    std::vector<int> lb_rev(lb.rbegin(), lb.rend());
    auto co = [&](int vid) -> const A3& { return mesh.verts()[vid]->co; };
    double best = 1e300;
    std::vector<int> pair(n);
    for (int off = 0; off < n; ++off) {                            // nearest correspondence
        double cost = 0;
        for (int i = 0; i < n; ++i) {
            const A3& A = co(la[i]); const A3& B = co(lb_rev[(i + off) % n]);
            cost += (A[0]-B[0])*(A[0]-B[0]) + (A[1]-B[1])*(A[1]-B[1]) + (A[2]-B[2])*(A[2]-B[2]);
        }
        if (cost < best) {
            best = cost;
            for (int i = 0; i < n; ++i) pair[i] = lb_rev[(i + off) % n];
        }
    }
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    std::vector<std::vector<int>> nf;
    std::vector<Tags> nt;
    for (const auto& f : mesh.faces())
        if (f.get() != fa && f.get() != fb) {
            std::vector<int> fv;
            for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
            nf.push_back(fv); nt.push_back(copy_tags(f.get()));
        }
    for (int i = 0; i < n; ++i) {  // wall quads, wound for outward normals
        const int j = (i + 1) % n;
        nf.push_back({la[j], la[i], pair[i], pair[j]});
        nt.push_back(copy_tags(fa, mark));
    }
    return build_compact(pos, nf, nt);
}

// Faces incident to vertex `vid`, in rotational (umbrella) order.
static std::vector<const Face*> faces_around_vertex(const Mesh& mesh, int vid) {
    struct Info { const Face* f; Edge* ein; Edge* eout; };
    std::unordered_map<const Face*, Info> info;
    std::unordered_map<Edge*, std::vector<const Face*>> e2f;
    for (const auto& fp : mesh.faces()) {
        const Face* f = fp.get();
        for (Loop* lp : mesh.face_loops(f))
            if (lp->vert->id == vid) {
                Info in{f, lp->prev->edge, lp->edge};
                info[f] = in;
                e2f[in.ein].push_back(f);
                e2f[in.eout].push_back(f);
                break;
            }
    }
    if (info.empty()) return {};
    const Face* start = info.begin()->first;
    std::vector<const Face*> ordered{start};
    std::set<const Face*> seen{start};
    Edge* bridge = info[start].eout;
    while (ordered.size() < info.size()) {
        const Face* nxt = nullptr;
        for (const Face* g : e2f[bridge]) if (!seen.count(g)) { nxt = g; break; }
        if (!nxt) break;
        seen.insert(nxt); ordered.push_back(nxt);
        const Info& gi = info[nxt];
        bridge = (gi.ein == bridge) ? gi.eout : gi.ein;
    }
    return ordered;
}

// The two edges of face f incident to vertex vid.
static std::set<Edge*> vertex_face_edges(const Mesh& mesh, const Face* f, int vid) {
    for (Loop* lp : mesh.face_loops(f))
        if (lp->vert->id == vid) return {lp->prev->edge, lp->edge};
    return {};
}

Mesh edge_bevel(const Mesh& mesh, const std::vector<Edge*>& edges, double width, const std::string& mark) {
    std::set<const Edge*> sel(edges.begin(), edges.end());
    if (sel.empty()) return mesh.copy();
    width = std::min(std::max(width, 1e-3), 0.49);

    // interior vertices only (a boundary vertex has an open umbrella the sector
    // walk can't handle); their edges are pruned so open meshes never crash.
    std::unordered_map<int, bool> vinterior;
    for (const auto& e : mesh.edges()) {
        const bool man = mesh.edge_faces(e.get()).size() == 2;
        for (int vid : {e->v1->id, e->v2->id}) {
            auto it = vinterior.find(vid);
            vinterior[vid] = (it == vinterior.end() ? true : it->second) && man;
        }
    }
    while (true) {  // prune edges with a lone-cut (<2) or boundary endpoint
        std::unordered_map<int, int> deg;
        for (const auto& e : mesh.edges())
            if (sel.count(e.get())) { deg[e->v1->id]++; deg[e->v2->id]++; }
        std::set<const Edge*> new_sel;
        for (const auto& e : mesh.edges())
            if (sel.count(e.get()) && vinterior[e->v1->id] && vinterior[e->v2->id] &&
                deg[e->v1->id] >= 2 && deg[e->v2->id] >= 2)
                new_sel.insert(e.get());
        if (new_sel == sel) break;
        sel = std::move(new_sel);
    }
    if (sel.empty()) return mesh.copy();

    std::set<int> bevel_vert;
    for (const auto& e : mesh.edges())
        if (sel.count(e.get())) { bevel_vert.insert(e->v1->id); bevel_vert.insert(e->v2->id); }

    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    std::map<std::pair<int, int>, int> sector_corner;          // (vid, sector) -> new vid
    std::map<std::pair<int, const Face*>, int> face_sector;    // (vid, face) -> sector
    std::map<int, int> vert_ns;                                // vid -> number of sectors
    std::map<int, const Face*> vert_repface;                   // vid -> a face (for the corner tag)

    for (int vid : bevel_vert) {  // split each star into sectors, one moved corner each
        auto ring = faces_around_vertex(mesh, vid);
        const int k = static_cast<int>(ring.size());
        if (k == 0) continue;
        std::vector<bool> cut(k);
        for (int i = 0; i < k; ++i) {
            auto a = vertex_face_edges(mesh, ring[i], vid);
            auto b = vertex_face_edges(mesh, ring[(i + 1) % k], vid);
            bool c = false;
            for (Edge* e : a) if (b.count(e) && sel.count(e)) c = true;
            cut[i] = c;
        }
        int start = 0;
        for (int i = 0; i < k; ++i) if (cut[(i - 1 + k) % k]) { start = i; break; }
        std::vector<std::vector<const Face*>> sectors;
        std::vector<const Face*> cur;
        int idx = start;
        for (int s = 0; s < k; ++s) {
            cur.push_back(ring[idx]);
            if (cut[idx]) { sectors.push_back(cur); cur.clear(); }
            idx = (idx + 1) % k;
        }
        if (!cur.empty()) sectors.push_back(cur);
        vert_ns[vid] = static_cast<int>(sectors.size());
        vert_repface[vid] = ring.front();
        const A3 vco = mesh.verts()[vid]->co;
        for (int si = 0; si < static_cast<int>(sectors.size()); ++si) {
            A3 avg{0, 0, 0};
            for (const Face* f : sectors[si]) avg = a3add(avg, face_centroid(mesh, f));
            avg = a3scale(avg, 1.0 / static_cast<double>(sectors[si].size()));
            sector_corner[{vid, si}] = static_cast<int>(pos.size());
            pos.push_back({vco[0] + (avg[0] - vco[0]) * width, vco[1] + (avg[1] - vco[1]) * width,
                           vco[2] + (avg[2] - vco[2]) * width});
            for (const Face* f : sectors[si]) face_sector[{vid, f}] = si;
        }
    }

    auto corner_of = [&](int vid, const Face* f) -> int {
        return bevel_vert.count(vid) ? sector_corner[{vid, face_sector[{vid, f}]}] : vid;
    };

    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    for (const auto& fp : mesh.faces()) {  // shrunk faces (corners moved per sector)
        const Face* f = fp.get();
        std::vector<int> poly;
        for (Loop* lp : mesh.face_loops(f)) poly.push_back(corner_of(lp->vert->id, f));
        faces.push_back(poly); tags.push_back(copy_tags(f));
    }
    for (const auto& ep : mesh.edges()) {  // chamfer quad per selected edge
        Edge* e = ep.get();
        if (!sel.count(e)) continue;
        std::vector<Face*> fs = mesh.edge_faces(e);
        if (fs.size() != 2) continue;
        Face* f1 = fs[0]; Face* f2 = fs[1];
        Loop* lp1 = nullptr;
        for (Loop* l : mesh.face_loops(f1)) if (l->edge == e) { lp1 = l; break; }
        const int u = lp1->vert->id, w = lp1->next->vert->id;
        faces.push_back({corner_of(w, f1), corner_of(u, f1), corner_of(u, f2), corner_of(w, f2)});
        tags.push_back(copy_tags(f1, mark));
    }
    for (int vid : bevel_vert) {  // corner face fills the sector-corner cycle (>=3 sectors)
        auto it = vert_ns.find(vid);
        if (it == vert_ns.end() || it->second < 3) continue;  // 2 sectors: chamfers share the cross-edge
        std::vector<int> poly;
        for (int si = it->second - 1; si >= 0; --si) poly.push_back(sector_corner[{vid, si}]);  // reversed -> outward
        faces.push_back(poly); tags.push_back(copy_tags(vert_repface[vid], mark));
    }
    return build_compact(pos, faces, tags);  // drop the now-orphaned original verts
}

Mesh loop_cut(const Mesh& mesh, const std::vector<const Face*>& seed_v, const std::string& axis,
              const std::string& mark) {
    if (seed_v.empty()) return mesh.copy();
    const Face* seed = seed_v.front();
    if (mesh.face_verts(seed).size() != 4) return mesh.copy();  // only a quad can seed
    const int ax = axis == "x" ? 0 : axis == "y" ? 1 : 2;

    auto quad_edges = [&](const Face* f) {
        std::vector<Edge*> e;
        for (Loop* lp : mesh.face_loops(f)) e.push_back(lp->edge);
        return e;
    };
    auto edge_index = [](const std::vector<Edge*>& es, Edge* e) {
        for (int k = 0; k < 4; ++k) if (es[k] == e) return k;
        return 0;
    };
    auto opposite_edge = [&](const Face* f, Edge* e) {
        auto es = quad_edges(f);
        return es[(edge_index(es, e) + 2) % 4];
    };
    auto other_face = [&](Edge* e, const Face* f) -> Face* {
        for (Face* x : mesh.edge_faces(e)) if (x != f) return x;
        return nullptr;
    };

    // seed's crossed edge-pair: the opposite pair most aligned with `axis`
    std::vector<Loop*> loops = mesh.face_loops(seed);
    std::vector<Edge*> es;
    std::vector<A3> dirs;
    for (Loop* lp : loops) {
        es.push_back(lp->edge);
        dirs.push_back({lp->next->vert->co[0] - lp->vert->co[0],
                        lp->next->vert->co[1] - lp->vert->co[1],
                        lp->next->vert->co[2] - lp->vert->co[2]});
    }
    auto axis_align = [&](int k) {
        double n = std::sqrt(dirs[k][0] * dirs[k][0] + dirs[k][1] * dirs[k][1] + dirs[k][2] * dirs[k][2]);
        return n > 0 ? std::fabs(dirs[k][ax]) / n : 0.0;
    };
    const int i0 = (axis_align(0) + axis_align(2)) >= (axis_align(1) + axis_align(3)) ? 0 : 1;

    std::unordered_map<const Face*, std::pair<Edge*, Edge*>> cross;
    cross[seed] = {es[i0], es[(i0 + 2) % 4]};
    std::set<const Face*> seen;
    std::set<Edge*> crossed;
    std::vector<const Face*> ring;
    std::vector<const Face*> stack{seed};
    while (!stack.empty()) {
        const Face* f = stack.back(); stack.pop_back();
        if (seen.count(f) || mesh.face_verts(f).size() != 4) continue;
        seen.insert(f);
        Edge* ea = cross[f].first; Edge* ec = cross[f].second;
        crossed.insert(ea); crossed.insert(ec); ring.push_back(f);
        for (Edge* e : {ea, ec}) {
            Face* nf = other_face(e, f);
            if (nf && !seen.count(nf) && mesh.face_verts(nf).size() == 4 && !cross.count(nf)) {
                cross[nf] = {e, opposite_edge(nf, e)};
                stack.push_back(nf);
            }
        }
    }

    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    std::map<Edge*, int> mid;
    for (Edge* e : crossed) {
        mid[e] = static_cast<int>(pos.size());
        pos.push_back(a3scale(a3add(e->v1->co, e->v2->co), 0.5));
    }

    std::set<const Face*> ring_set(ring.begin(), ring.end());
    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    for (const auto& fp : mesh.faces()) {
        const Face* f = fp.get();
        if (!ring_set.count(f)) {
            std::vector<int> v;
            for (Loop* lp : mesh.face_loops(f)) v.push_back(lp->vert->id);
            faces.push_back(v); tags.push_back(copy_tags(f));
            continue;
        }
        std::vector<int> vids;
        std::vector<Edge*> fe;
        for (Loop* lp : mesh.face_loops(f)) { vids.push_back(lp->vert->id); fe.push_back(lp->edge); }
        Edge* ea = cross[f].first; Edge* ec = cross[f].second;
        int i = 0;
        for (int k = 0; k < 4; ++k) if (fe[k] == ea || fe[k] == ec) { i = k; break; }
        const int mi = mid[fe[i]], mj = mid[fe[(i + 2) % 4]];
        faces.push_back({vids[i], mi, mj, vids[(i + 3) % 4]}); tags.push_back(copy_tags(f, mark));
        faces.push_back({mi, vids[(i + 1) % 4], vids[(i + 2) % 4], mj}); tags.push_back(copy_tags(f, mark));
    }
    return Mesh::from_pydata(pos, faces, tags);
}

Mesh solidify(const Mesh& mesh, double thickness, const std::string& mark) {
    const int n = static_cast<int>(mesh.num_verts());
    if (n == 0 || std::fabs(thickness) < 1e-9) return mesh.copy();
    std::vector<A3> acc(n, A3{0, 0, 0});
    for (const auto& f : mesh.faces()) {
        auto fn = face_normal(mesh, f.get());
        for (Vert* v : mesh.face_verts(f.get()))
            for (int k = 0; k < 3; ++k) acc[v->id][k] += fn[k];
    }
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    std::vector<A3> np = pos;
    for (int vid = 0; vid < n; ++vid) {  // inner verts: pos - normal*thickness
        const A3& a = acc[vid];
        double l = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        A3 d = l > 1e-9 ? A3{a[0] / l * thickness, a[1] / l * thickness, a[2] / l * thickness} : A3{0, 0, 0};
        np.push_back({pos[vid][0] - d[0], pos[vid][1] - d[1], pos[vid][2] - d[2]});
    }
    auto inner = [&](int i) { return i + n; };
    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    for (const auto& f : mesh.faces()) {  // outer shell
        std::vector<int> fv;
        for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
        faces.push_back(fv); tags.push_back(copy_tags(f.get()));
    }
    for (const auto& f : mesh.faces()) {  // inner shell (reversed winding)
        std::vector<int> ids;
        for (Loop* lp : mesh.face_loops(f.get())) ids.push_back(lp->vert->id);
        std::vector<int> fv;
        for (auto it = ids.rbegin(); it != ids.rend(); ++it) fv.push_back(inner(*it));
        faces.push_back(fv); tags.push_back(copy_tags(f.get(), mark));
    }
    for (const auto& e : mesh.edges()) {  // walls bridge the boundary (1-face edges)
        auto fs = mesh.edge_faces(e.get());
        if (fs.size() == 1) {
            Loop* lp = nullptr;
            for (Loop* l : mesh.face_loops(fs[0])) if (l->edge == e.get()) { lp = l; break; }
            int a = lp->vert->id, b = lp->next->vert->id;
            faces.push_back({b, a, inner(a), inner(b)});
            tags.push_back(copy_tags(fs[0], mark));
        }
    }
    return build_compact(np, faces, tags);
}

Mesh mirror(const Mesh& mesh, const std::string& axis, const std::string& mark) {
    const int k = axis == "x" ? 0 : axis == "y" ? 1 : 2;
    const double tol = 1e-6;
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    std::vector<A3> np = pos;
    std::vector<int> mir(pos.size());
    for (int vid = 0; vid < static_cast<int>(pos.size()); ++vid) {
        if (std::fabs(pos[vid][k]) < tol) {
            mir[vid] = vid;                       // on the plane -> shared seam vertex
        } else {
            mir[vid] = static_cast<int>(np.size());
            A3 p = pos[vid]; p[k] = -p[k]; np.push_back(p);
        }
    }
    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    for (const auto& f : mesh.faces()) {          // original half
        std::vector<int> fv;
        for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
        faces.push_back(fv); tags.push_back(copy_tags(f.get()));
    }
    for (const auto& f : mesh.faces()) {          // reflected half (reversed winding)
        std::vector<int> ids;
        for (Loop* lp : mesh.face_loops(f.get())) ids.push_back(lp->vert->id);
        std::vector<int> fv;
        for (auto it = ids.rbegin(); it != ids.rend(); ++it) fv.push_back(mir[*it]);
        faces.push_back(fv); tags.push_back(copy_tags(f.get(), mark));
    }
    return build_compact(np, faces, tags);
}

Mesh array(const Mesh& mesh, int count, const std::array<double, 3>& offset, const std::string& mark) {
    count = std::max(count, 1);
    const int n = static_cast<int>(mesh.num_verts());
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    std::vector<std::vector<int>> faces0;
    std::vector<const Face*> fptr;
    for (const auto& f : mesh.faces()) {
        std::vector<int> fv;
        for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
        faces0.push_back(fv); fptr.push_back(f.get());
    }
    std::vector<A3> np;
    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    for (int c = 0; c < count; ++c) {
        double ox = offset[0] * c, oy = offset[1] * c, oz = offset[2] * c;
        for (const auto& p : pos) np.push_back({p[0] + ox, p[1] + oy, p[2] + oz});
        for (std::size_t fi = 0; fi < faces0.size(); ++fi) {
            std::vector<int> fv;
            for (int i : faces0[fi]) fv.push_back(i + c * n);
            faces.push_back(fv);
            tags.push_back(c == count - 1 ? copy_tags(fptr[fi], mark) : copy_tags(fptr[fi]));
        }
    }
    return Mesh::from_pydata(np, faces, tags);
}

Mesh bisect(const Mesh& mesh, const std::array<double, 3>& point, const std::array<double, 3>& normal,
            bool fill, const std::string& mark) {
    double nx = normal[0], ny = normal[1], nz = normal[2];
    double nl = std::sqrt(nx * nx + ny * ny + nz * nz);
    if (nl == 0.0) nl = 1.0;
    nx /= nl; ny /= nl; nz /= nl;
    const double eps = 1e-7;
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    const int n = static_cast<int>(pos.size());
    std::vector<double> dist(n);
    std::vector<int> side(n);
    for (int i = 0; i < n; ++i) {
        dist[i] = (pos[i][0] - point[0]) * nx + (pos[i][1] - point[1]) * ny + (pos[i][2] - point[2]) * nz;
        side[i] = dist[i] > eps ? 1 : (dist[i] < -eps ? -1 : 0);
    }
    std::vector<A3> np = pos;
    std::unordered_map<const Edge*, int> cut;   // crossing edge -> shared intersection vertex
    for (const auto& e : mesh.edges()) {
        int a = e->v1->id, b = e->v2->id;
        if (side[a] * side[b] < 0) {
            double t = dist[a] / (dist[a] - dist[b]);
            cut[e.get()] = static_cast<int>(np.size());
            np.push_back({pos[a][0] + (pos[b][0] - pos[a][0]) * t,
                          pos[a][1] + (pos[b][1] - pos[a][1]) * t,
                          pos[a][2] + (pos[b][2] - pos[a][2]) * t});
        }
    }
    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    for (const auto& f : mesh.faces()) {
        std::vector<Loop*> loops = mesh.face_loops(f.get());
        std::vector<int> verts;
        for (Loop* lp : loops) verts.push_back(lp->vert->id);
        bool any_out = false, all_ge0 = true, all_le0 = true;
        for (int v : verts) { any_out |= side[v] > 0; all_ge0 &= side[v] >= 0; all_le0 &= side[v] <= 0; }
        if (all_ge0 && any_out) continue;       // wholly removed
        if (all_le0) {                          // wholly kept
            faces.push_back(verts); tags.push_back(copy_tags(f.get()));
            continue;
        }
        std::vector<int> poly;                  // clip to side <= 0
        for (Loop* lp : loops) {
            int a = lp->vert->id, b = lp->next->vert->id;
            if (side[a] <= 0) poly.push_back(a);
            if (side[a] * side[b] < 0) poly.push_back(cut[lp->edge]);
        }
        std::vector<int> clipped;
        for (int v : poly) if (clipped.empty() || clipped.back() != v) clipped.push_back(v);
        if (clipped.size() >= 2 && clipped.front() == clipped.back()) clipped.pop_back();
        if (clipped.size() >= 3) { faces.push_back(clipped); tags.push_back(copy_tags(f.get(), mark)); }
    }
    Mesh m = build_compact(np, faces, tags);
    if (fill) m = fill_holes(m, mark);
    return m;
}

Mesh spin(const Mesh& mesh, const std::string& axis, int steps, double angle, const std::string& mark) {
    constexpr double PI = 3.14159265358979323846;
    const int k = axis == "x" ? 0 : axis == "y" ? 1 : 2;
    const int i = (k + 1) % 3, j = (k + 2) % 3;   // axes perpendicular to `axis`
    steps = std::max(steps, 3);
    const bool full = angle >= 359.999;
    const int rings = full ? steps : steps + 1;
    const double eps = 1e-12;
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    const int nv = static_cast<int>(pos.size());
    std::vector<char> on_axis(nv);
    for (int v = 0; v < nv; ++v) on_axis[v] = (pos[v][i] * pos[v][i] + pos[v][j] * pos[v][j]) < eps;

    std::vector<A3> np;
    std::vector<int> base(nv);
    for (int v = 0; v < nv; ++v) {
        base[v] = static_cast<int>(np.size());
        if (on_axis[v]) { np.push_back(pos[v]); continue; }   // a pole: one shared copy
        for (int r = 0; r < rings; ++r) {
            const double theta = angle * PI / 180.0 * r / steps;
            const double c = std::cos(theta), s = std::sin(theta);
            A3 p = pos[v];
            p[i] = pos[v][i] * c - pos[v][j] * s;
            p[j] = pos[v][i] * s + pos[v][j] * c;
            np.push_back(p);
        }
    }
    auto outid = [&](int v, int r) {
        if (on_axis[v]) return base[v];
        return base[v] + (full ? r % steps : r);
    };

    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    for (const auto& e : mesh.edges()) {
        auto fs = mesh.edge_faces(e.get());
        if (fs.size() != 1) continue;             // boundary edges only (the silhouette)
        int a = e->v1->id, b = e->v2->id;
        if (on_axis[a] && on_axis[b]) continue;   // an edge on the axis sweeps nothing
        for (int r = 0; r < steps; ++r) {
            int quad[4] = {outid(a, r), outid(b, r), outid(b, r + 1), outid(a, r + 1)};
            std::vector<int> poly;
            for (int x : quad) if (poly.empty() || poly.back() != x) poly.push_back(x);
            if (poly.size() >= 2 && poly.front() == poly.back()) poly.pop_back();
            if (poly.size() >= 3) { faces.push_back(poly); tags.push_back(copy_tags(fs[0], mark)); }
        }
    }
    return build_compact(np, faces, tags);
}

Mesh screw(const Mesh& mesh, const std::string& axis, int steps, int turns, double height,
           double angle, const std::string& mark) {
    constexpr double PI = 3.14159265358979323846;
    const int k = axis == "x" ? 0 : axis == "y" ? 1 : 2;
    const int i = (k + 1) % 3, j = (k + 2) % 3;   // axes perpendicular to `axis`
    steps = std::max(steps, 3);
    turns = std::max(turns, 1);
    const int total = steps * turns;
    const int rings = total + 1;                  // always open: the helix never closes
    const double eps = 1e-12;
    std::vector<A3> pos;
    for (const auto& v : mesh.verts()) pos.push_back(v->co);
    const int nv = static_cast<int>(pos.size());
    std::vector<char> on_axis(nv);
    for (int v = 0; v < nv; ++v) on_axis[v] = (pos[v][i] * pos[v][i] + pos[v][j] * pos[v][j]) < eps;

    std::vector<A3> np;
    std::vector<int> base(nv);
    for (int v = 0; v < nv; ++v) {
        base[v] = static_cast<int>(np.size());
        const double ci = pos[v][i], cj = pos[v][j], axial0 = pos[v][k];
        for (int r = 0; r < rings; ++r) {
            const double theta = angle * PI / 180.0 * r / steps;
            const double c = std::cos(theta), s = std::sin(theta);
            A3 p{0, 0, 0};
            p[i] = ci * c - cj * s;
            p[j] = ci * s + cj * c;
            p[k] = axial0 + height * r / steps;   // the climb: rise of `height` per turn
            np.push_back(p);
        }
    }

    std::vector<std::vector<int>> faces;
    std::vector<Tags> tags;
    for (const auto& e : mesh.edges()) {
        auto fs = mesh.edge_faces(e.get());
        if (fs.size() != 1) continue;             // boundary edges only (the silhouette)
        int a = e->v1->id, b = e->v2->id;
        if (on_axis[a] && on_axis[b]) continue;   // an edge on the axis sweeps a zero-area strip
        for (int r = 0; r < total; ++r) {
            int quad[4] = {base[a] + r, base[b] + r, base[b] + r + 1, base[a] + r + 1};
            std::vector<int> poly;
            for (int x : quad) if (poly.empty() || poly.back() != x) poly.push_back(x);
            if (poly.size() >= 2 && poly.front() == poly.back()) poly.pop_back();
            if (poly.size() >= 3) { faces.push_back(poly); tags.push_back(copy_tags(fs[0], mark)); }
        }
    }
    return build_compact(np, faces, tags);
}

Mesh bevel(const Mesh& mesh, const std::vector<const Face*>& region_v, double width, double depth,
           const std::string& mark) {
    std::set<const Face*> region(region_v.begin(), region_v.end());
    if (region.empty()) return mesh.copy();
    width = std::min(std::max(width, 1e-3), 0.999);  // avoid degenerate / bowtie

    std::vector<A3> new_pos;
    for (const auto& v : mesh.verts()) new_pos.push_back(v->co);
    std::vector<std::vector<int>> new_faces;
    std::vector<Tags> new_tags;
    for (const auto& f : mesh.faces())  // untouched faces
        if (!region.count(f.get())) {
            std::vector<int> fv;
            for (Loop* lp : mesh.face_loops(f.get())) fv.push_back(lp->vert->id);
            new_faces.push_back(fv);
            new_tags.push_back(copy_tags(f.get()));
        }
    for (const Face* f : region_in_id_order(region)) {
        const A3 nrm = face_normal(mesh, f);
        std::vector<int> vids;
        for (Loop* lp : mesh.face_loops(f)) vids.push_back(lp->vert->id);
        A3 c{0, 0, 0};
        for (int i : vids) c = a3add(c, new_pos[i]);
        c = a3scale(c, 1.0 / static_cast<double>(vids.size()));
        std::vector<int> inner;
        for (int i : vids) {
            const A3 p = new_pos[i];  // copy (push_back below may reallocate)
            inner.push_back(static_cast<int>(new_pos.size()));
            new_pos.push_back({p[0] + (c[0] - p[0]) * width + nrm[0] * depth,   // inset, then lift along normal
                               p[1] + (c[1] - p[1]) * width + nrm[1] * depth,
                               p[2] + (c[2] - p[2]) * width + nrm[2] * depth});
        }
        const int n = static_cast<int>(vids.size());
        for (int k = 0; k < n; ++k) {  // slanted border quads (the chamfer)
            new_faces.push_back({vids[k], vids[(k + 1) % n], inner[(k + 1) % n], inner[k]});
            new_tags.push_back(copy_tags(f));
        }
        new_faces.push_back(inner);  // inner face, tagged with `mark`
        new_tags.push_back(copy_tags(f, mark));
    }
    return Mesh::from_pydata(new_pos, new_faces, new_tags);
}

}  // namespace mirage
