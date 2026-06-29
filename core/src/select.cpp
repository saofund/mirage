#include "mirage/select.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace mirage {

namespace {

int axis_index(const std::string& ax) {
    if (ax == "x") return 0;
    if (ax == "y") return 1;
    if (ax == "z") return 2;
    throw MeshLangError("unknown axis '" + ax + "' (expected x/y/z)");
}

struct BBox {
    std::array<double, 3> lo{0, 0, 0}, hi{0, 0, 0};
};

BBox mesh_bbox(const Mesh& mesh) {
    BBox b;
    if (mesh.verts().empty()) return b;
    b.lo = b.hi = mesh.verts().front()->co;
    for (const auto& v : mesh.verts())
        for (int k = 0; k < 3; ++k) {
            b.lo[k] = std::min(b.lo[k], v->co[k]);
            b.hi[k] = std::max(b.hi[k], v->co[k]);
        }
    return b;
}

bool has_tag(const Face* f, const std::string& name) {
    return std::find(f->tags.begin(), f->tags.end(), name) != f->tags.end();
}

std::array<double, 3> json_point(const json& j, const char* what) {
    if (!j.is_array() || j.size() != 3 || !j[0].is_number() || !j[1].is_number() || !j[2].is_number())
        throw MeshLangError(std::string(what) + " must be a [x, y, z] number triple, got " + j.dump());
    return {j[0].get<double>(), j[1].get<double>(), j[2].get<double>()};
}

// Faces of `mesh` (in mesh order) whose pointer is in `keep` — the and/or/not
// combinators work on identity sets exactly like the Python id() sets.
std::vector<const Face*> in_mesh_order(const Mesh& mesh, const std::unordered_set<const Face*>& keep,
                                       bool invert = false) {
    std::vector<const Face*> out;
    for (const auto& f : mesh.faces())
        if (keep.count(f.get()) != static_cast<std::size_t>(invert ? 1 : 0)) out.push_back(f.get());
    return out;
}

// Local curvature proxy: mean dihedral over a face's boundary edges (boundary = 180,
// flat = 0). Mirrors the Python _face_curvature exactly.
double face_curvature(const Mesh& mesh, const Face* f) {
    constexpr double PI = 3.14159265358979323846;
    auto loops = mesh.face_loops(f);
    if (loops.empty()) return 0.0;
    double sum = 0.0;
    for (Loop* lp : loops) {
        auto fs = mesh.edge_faces(lp->edge);
        if (fs.size() != 2) { sum += 180.0; continue; }
        auto n1 = face_normal(mesh, fs[0]), n2 = face_normal(mesh, fs[1]);
        double d = std::clamp(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2], -1.0, 1.0);
        sum += std::acos(d) * 180.0 / PI;
    }
    return sum / static_cast<double>(loops.size());
}

std::vector<const Face*> resolve_inner(const Mesh& mesh, const json& sel, const std::string& last_tag) {
    if (!sel.is_object()) throw MeshLangError("selector must be a dict, got " + sel.dump());

    if (sel.contains("and")) {
        std::unordered_set<const Face*> keep;
        bool first = true;
        for (const json& s : sel.at("and")) {
            std::vector<const Face*> part = resolve_inner(mesh, s, last_tag);
            std::unordered_set<const Face*> ps(part.begin(), part.end());
            if (first) { keep = std::move(ps); first = false; }
            else {
                std::unordered_set<const Face*> next;
                for (const Face* f : keep)
                    if (ps.count(f)) next.insert(f);
                keep = std::move(next);
            }
        }
        if (first) keep.clear();  // "and": [] -> empty (Python: intersection of no sets)
        return in_mesh_order(mesh, keep);
    }
    if (sel.contains("or")) {
        std::unordered_set<const Face*> keep;
        for (const json& s : sel.at("or"))
            for (const Face* f : resolve_inner(mesh, s, last_tag)) keep.insert(f);
        return in_mesh_order(mesh, keep);
    }
    if (sel.contains("not")) {
        std::vector<const Face*> excl_v = resolve_inner(mesh, sel.at("not"), last_tag);
        std::unordered_set<const Face*> excl(excl_v.begin(), excl_v.end());
        return in_mesh_order(mesh, excl, /*invert=*/true);
    }

    const std::string by = sel.value("by", "");
    if (by == "all") {
        std::vector<const Face*> out;
        for (const auto& f : mesh.faces()) out.push_back(f.get());
        return out;
    }
    if (by == "normal") {
        if (sel.contains("dir")) {
            std::array<double, 3> d = json_point(sel.at("dir"), "normal 'dir'");
            double m = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (m == 0.0) m = 1.0;
            for (double& c : d) c /= m;
            const double tol = sel.value("tol", 0.5);
            std::vector<const Face*> out;
            for (const auto& f : mesh.faces()) {
                const auto n = face_normal(mesh, f.get());
                if (n[0] * d[0] + n[1] * d[1] + n[2] * d[2] > 1.0 - tol) out.push_back(f.get());
            }
            return out;
        }
        const int k = axis_index(sel.value("axis", "z"));
        const double sign = sel.value("sign", 1.0);
        const double tol = sel.value("tol", 0.5);
        std::vector<const Face*> out;
        for (const auto& f : mesh.faces())
            if (face_normal(mesh, f.get())[k] * sign > tol) out.push_back(f.get());
        return out;
    }
    if (by == "tag") {
        if (!sel.contains("name")) throw MeshLangError("tag selector needs 'name': " + sel.dump());
        const std::string name = sel.at("name").get<std::string>();
        std::vector<const Face*> out;
        for (const auto& f : mesh.faces())
            if (has_tag(f.get(), name)) out.push_back(f.get());
        return out;
    }
    if (by == "last_created") {
        std::vector<const Face*> out;
        if (last_tag.empty()) return out;
        for (const auto& f : mesh.faces())
            if (has_tag(f.get(), last_tag)) out.push_back(f.get());
        return out;
    }
    if (by == "extreme") {
        const int ax = axis_index(sel.value("axis", "z"));
        const BBox b = mesh_bbox(mesh);
        double range = b.hi[ax] - b.lo[ax];
        if (range == 0.0) range = 1.0;
        const double tol = sel.value("tol", 0.02) * range;
        std::vector<std::pair<double, const Face*>> cents;
        for (const auto& f : mesh.faces()) cents.emplace_back(face_centroid(mesh, f.get())[ax], f.get());
        if (cents.empty()) return {};
        const bool want_max = sel.value("which", "max") == std::string("max");
        double target = cents.front().first;
        for (const auto& [c, f] : cents) target = want_max ? std::max(target, c) : std::min(target, c);
        std::vector<const Face*> out;
        for (const auto& [c, f] : cents)
            if (std::abs(c - target) <= tol) out.push_back(f);
        return out;
    }
    if (by == "side") {
        const int ax = axis_index(sel.value("axis", "x"));
        const BBox b = mesh_bbox(mesh);
        const double mid = (b.lo[ax] + b.hi[ax]) / 2.0;
        const double sign = sel.value("sign", 1.0);
        std::vector<const Face*> out;
        for (const auto& f : mesh.faces())
            if ((face_centroid(mesh, f.get())[ax] - mid) * sign > 0) out.push_back(f.get());
        return out;
    }
    if (by == "near") {
        if (!sel.contains("point")) throw MeshLangError("near selector needs 'point': " + sel.dump());
        const std::array<double, 3> p = json_point(sel.at("point"), "near 'point'");
        const Face* f = nearest_face(mesh, p);
        if (f == nullptr) return {};
        return {f};
    }
    if (by == "material") {
        const bool has_col = sel.contains("color");
        std::array<double, 3> col{0, 0, 0};
        if (has_col) col = json_point(sel.at("color"), "material 'color'");
        const double tol = sel.value("tol", 0.02);
        std::vector<const Face*> out;
        for (const auto& f : mesh.faces()) {
            const Material& m = f->material;
            if (!m.set) continue;
            if (!has_col || (std::abs(m.color[0] - col[0]) <= tol && std::abs(m.color[1] - col[1]) <= tol &&
                             std::abs(m.color[2] - col[2]) <= tol))
                out.push_back(f.get());
        }
        return out;
    }
    if (by == "connected") {
        // union-find over edge-shared faces (components keyed by their lowest face id,
        // matching the Python kernel's deterministic ordering + first-wins tie-break)
        std::unordered_map<int, int> parent;
        for (const auto& f : mesh.faces()) parent[f->id] = f->id;
        std::function<int(int)> find = [&](int x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        };
        for (const auto& e : mesh.edges()) {
            auto fs = mesh.edge_faces(e.get());
            for (std::size_t i = 1; i < fs.size(); ++i) {
                int ra = find(fs[0]->id), rb = find(fs[i]->id);
                if (ra != rb) parent[ra] = rb;
            }
        }
        std::unordered_map<int, std::vector<const Face*>> groups;
        std::unordered_map<int, int> minid;
        for (const auto& f : mesh.faces()) {
            int r = find(f->id);
            groups[r].push_back(f.get());
            auto it = minid.find(r);
            if (it == minid.end() || f->id < it->second) minid[r] = f->id;
        }
        std::vector<int> roots;
        for (auto& kv : groups) roots.push_back(kv.first);
        std::sort(roots.begin(), roots.end(), [&](int a, int b) { return minid[a] < minid[b]; });
        if (roots.empty()) return {};
        if (sel.contains("seed")) {
            std::vector<const Face*> seedv = resolve_inner(mesh, sel.at("seed"), last_tag);
            std::unordered_set<const Face*> seed(seedv.begin(), seedv.end());
            std::vector<const Face*> out;
            for (int r : roots) {
                bool hit = false;
                for (const Face* f : groups[r]) if (seed.count(f)) { hit = true; break; }
                if (hit) for (const Face* f : groups[r]) out.push_back(f);
            }
            return out;
        }
        const bool largest = sel.value("which", std::string("largest")) == "largest";
        int best = roots[0];
        for (int r : roots)  // strict comparison keeps the FIRST (lowest-min-id) on ties
            if (largest ? groups[r].size() > groups[best].size() : groups[r].size() < groups[best].size())
                best = r;
        return groups[best];
    }
    if (by == "box") {
        std::array<double, 3> lo{-1e30, -1e30, -1e30}, hi{1e30, 1e30, 1e30};
        if (sel.contains("min")) lo = json_point(sel.at("min"), "box 'min'");
        if (sel.contains("max")) hi = json_point(sel.at("max"), "box 'max'");
        std::vector<const Face*> out;
        for (const auto& f : mesh.faces()) {
            const auto c = face_centroid(mesh, f.get());
            if (c[0] >= lo[0] && c[0] <= hi[0] && c[1] >= lo[1] && c[1] <= hi[1] &&
                c[2] >= lo[2] && c[2] <= hi[2])
                out.push_back(f.get());
        }
        return out;
    }
    if (by == "area") {
        if (sel.contains("which")) {
            const bool largest = sel.at("which").get<std::string>() == "largest";
            const Face* best = nullptr;
            double bv = 0;
            for (const auto& f : mesh.faces()) {   // first-wins on ties (strict compare)
                const double a = face_area(mesh, f.get());
                if (best == nullptr || (largest ? a > bv : a < bv)) { best = f.get(); bv = a; }
            }
            if (best == nullptr) return {};
            return {best};
        }
        const double amin = sel.value("min", 0.0), amax = sel.value("max", 1e30);
        std::vector<const Face*> out;
        for (const auto& f : mesh.faces()) {
            const double a = face_area(mesh, f.get());
            if (a >= amin && a <= amax) out.push_back(f.get());
        }
        return out;
    }
    if (by == "curvature") {
        const double cmin = sel.value("min", 0.0), cmax = sel.value("max", 180.0);
        std::vector<const Face*> out;
        for (const auto& f : mesh.faces()) {
            const double c = face_curvature(mesh, f.get());
            if (c >= cmin && c <= cmax) out.push_back(f.get());
        }
        return out;
    }
    throw MeshLangError("unknown selector " + sel.dump());
}

}  // namespace

json selector_diagnostics(const Mesh& mesh) {
    const BBox b = mesh_bbox(mesh);
    std::set<std::string> tags;
    for (const auto& f : mesh.faces())
        for (const std::string& t : f->tags)
            if (t.rfind("__", 0) != 0) tags.insert(t);
    json hist = json::object();
    const char* axes = "xyz";
    for (const auto& f : mesh.faces()) {
        const auto n = face_normal(mesh, f.get());
        for (int k = 0; k < 3; ++k) {
            if (n[k] > 0.5) {
                const std::string key = std::string("+") + axes[k];
                hist[key] = hist.value(key, 0) + 1;
            } else if (n[k] < -0.5) {
                const std::string key = std::string("-") + axes[k];
                hist[key] = hist.value(key, 0) + 1;
            }
        }
    }
    return json{{"faces", mesh.faces().size()},
                {"bbox", json::array({json(b.lo), json(b.hi)})},
                {"tags", json(tags)},
                {"normal_histogram", hist}};
}

// --- edge selection (the parallel grammar for edges) -------------------------
namespace {

constexpr double PI = 3.14159265358979323846;

std::array<double, 3> edge_dir(const Edge* e) {
    std::array<double, 3> d{e->v2->co[0] - e->v1->co[0], e->v2->co[1] - e->v1->co[1],
                            e->v2->co[2] - e->v1->co[2]};
    double m = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    if (m == 0) m = 1;
    return {d[0] / m, d[1] / m, d[2] / m};
}

// Angle between the two faces at e (0 = flat, 90 = cube edge); boundary = sharp.
double dihedral_deg(const Mesh& mesh, const Edge* e) {
    std::vector<Face*> fs = mesh.edge_faces(e);
    if (fs.size() != 2) return 180.0;
    auto n1 = face_normal(mesh, fs[0]), n2 = face_normal(mesh, fs[1]);
    double d = std::clamp(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2], -1.0, 1.0);
    return std::acos(d) * 180.0 / PI;
}

std::vector<Edge*> edges_in_order(const Mesh& mesh, const std::unordered_set<const Edge*>& keep,
                                  bool invert = false) {
    std::vector<Edge*> out;
    for (const auto& e : mesh.edges())
        if (keep.count(e.get()) != static_cast<std::size_t>(invert ? 1 : 0)) out.push_back(e.get());
    return out;
}

std::vector<Edge*> resolve_edges_inner(const Mesh& mesh, const json& sel, const std::string& last_tag) {
    if (!sel.is_object()) throw MeshLangError("edge selector must be a dict, got " + sel.dump());
    if (sel.contains("and")) {
        std::unordered_set<const Edge*> keep;
        bool first = true;
        for (const json& s : sel.at("and")) {
            auto part = resolve_edges_inner(mesh, s, last_tag);
            std::unordered_set<const Edge*> ps(part.begin(), part.end());
            if (first) { keep = std::move(ps); first = false; }
            else {
                std::unordered_set<const Edge*> next;
                for (const Edge* e : keep) if (ps.count(e)) next.insert(e);
                keep = std::move(next);
            }
        }
        if (first) keep.clear();
        return edges_in_order(mesh, keep);
    }
    if (sel.contains("or")) {
        std::unordered_set<const Edge*> keep;
        for (const json& s : sel.at("or"))
            for (Edge* e : resolve_edges_inner(mesh, s, last_tag)) keep.insert(e);
        return edges_in_order(mesh, keep);
    }
    if (sel.contains("not")) {
        auto ex = resolve_edges_inner(mesh, sel.at("not"), last_tag);
        std::unordered_set<const Edge*> excl(ex.begin(), ex.end());
        return edges_in_order(mesh, excl, /*invert=*/true);
    }
    const std::string by = sel.value("by", "");
    std::vector<Edge*> out;
    if (by == "all") {
        for (const auto& e : mesh.edges()) out.push_back(e.get());
        return out;
    }
    if (by == "sharp") {
        const double ang = sel.value("angle", 30.0);
        for (const auto& e : mesh.edges()) if (dihedral_deg(mesh, e.get()) >= ang) out.push_back(e.get());
        return out;
    }
    if (by == "axis") {
        const int ax = axis_index(sel.value("axis", "z"));
        const double tol = sel.value("tol", 0.1);
        for (const auto& e : mesh.edges()) if (std::fabs(edge_dir(e.get())[ax]) > 1.0 - tol) out.push_back(e.get());
        return out;
    }
    if (by == "boundary") {
        for (const auto& e : mesh.edges()) if (mesh.edge_faces(e.get()).size() != 2) out.push_back(e.get());
        return out;
    }
    if (by == "on_face") {
        if (!sel.contains("face")) throw MeshLangError("on_face edge selector needs 'face': " + sel.dump());
        auto faces = resolve(mesh, sel.at("face"), last_tag);
        std::unordered_set<const Face*> fset(faces.begin(), faces.end());
        for (const auto& e : mesh.edges())
            for (Face* f : mesh.edge_faces(e.get()))
                if (fset.count(f)) { out.push_back(e.get()); break; }
        return out;
    }
    throw MeshLangError("unknown edge selector " + sel.dump());
}

}  // namespace

std::vector<Edge*> resolve_edges(const Mesh& mesh, const json& sel, const std::string& last_tag) {
    std::vector<Edge*> edges = resolve_edges_inner(mesh, sel, last_tag);
    std::unordered_set<const Edge*> seen;
    std::vector<Edge*> out;
    for (Edge* e : edges)
        if (seen.insert(e).second) out.push_back(e);
    if (out.empty()) throw SelectorEmpty(sel, selector_diagnostics(mesh));
    return out;
}

SelectorEmpty::SelectorEmpty(json sel_, json diagnostics_)
    : MeshLangError("selector matched 0 faces: " + sel_.dump() + " | available: " + diagnostics_.dump()),
      sel(std::move(sel_)),
      diagnostics(std::move(diagnostics_)) {}

std::vector<const Face*> resolve(const Mesh& mesh, const json& sel, const std::string& last_tag) {
    std::vector<const Face*> faces = resolve_inner(mesh, sel, last_tag);
    std::unordered_set<const Face*> seen;
    std::vector<const Face*> out;
    for (const Face* f : faces)
        if (seen.insert(f).second) out.push_back(f);
    if (out.empty()) throw SelectorEmpty(sel, selector_diagnostics(mesh));
    return out;
}

}  // namespace mirage
