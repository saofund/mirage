#include "mirage/mesh.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>

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

}  // namespace mirage
