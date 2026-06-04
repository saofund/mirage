#include "mirage/mesh.hpp"

#include <set>
#include <stdexcept>
#include <string>

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

Face* Mesh::add_face(const std::vector<Vert*>& verts) {
    const std::size_t n = verts.size();
    if (n < 3) throw std::invalid_argument("a face needs >= 3 verts");
    if (std::set<Vert*>(verts.begin(), verts.end()).size() != n)
        throw std::invalid_argument("face has repeated vertices");

    auto face = std::make_unique<Face>();
    face->id = static_cast<int>(faces_.size());
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

}  // namespace mirage
