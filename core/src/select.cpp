#include "mirage/select.hpp"

#include <algorithm>
#include <cmath>
#include <set>
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
