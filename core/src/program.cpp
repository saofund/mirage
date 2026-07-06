#include "mirage/program.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

namespace mirage {

namespace {

// Stamp a tag onto a face. resolve() hands back const Face* (read-only contract),
// but build() owns the mesh mutably, so this const_cast is honest.
void add_tag(const Face* f, const std::string& t) { const_cast<Face*>(f)->tags.push_back(t); }
bool has_tag(const Face* f, const std::string& t) {
    return std::find(f->tags.begin(), f->tags.end(), t) != f->tags.end();
}

std::vector<const Face*> faces_with_tag(const Mesh& m, const std::string& tag) {
    std::vector<const Face*> out;
    for (const auto& f : m.faces())
        if (has_tag(f.get(), tag)) out.push_back(f.get());
    return out;
}

// translate/scale: move the selected faces' verts in place (Python _transform).
void transform_region(const Mesh& m, const std::vector<const Face*>& faces, const std::string& op,
                      const std::array<double, 3>& by) {
    std::map<int, Vert*> verts;  // unique verts of the region, deterministic order
    for (const Face* f : faces)
        for (Vert* v : m.face_verts(f)) verts[v->id] = v;
    if (verts.empty()) return;
    if (op == "translate") {
        for (auto& [id, v] : verts)
            for (int k = 0; k < 3; ++k) v->co[k] += by[k];
    } else {  // scale about the region centroid
        std::array<double, 3> c{0, 0, 0};
        for (auto& [id, v] : verts)
            for (int k = 0; k < 3; ++k) c[k] += v->co[k];
        const double n = static_cast<double>(verts.size());
        for (double& x : c) x /= n;
        for (auto& [id, v] : verts)
            for (int k = 0; k < 3; ++k) v->co[k] = c[k] + (v->co[k] - c[k]) * by[k];
    }
}

void check_assert(const Mesh& m, const json& cmd) {
    if (cmd.value("closed_manifold", false) && !m.is_closed_manifold())
        throw MeshLangError("assert closed_manifold failed");
    if (cmd.contains("euler")) {
        const int want = cmd.at("euler").get<int>();
        if (m.euler() != want)
            throw MeshLangError("assert euler=" + std::to_string(want) + " failed (got " +
                                std::to_string(m.euler()) + ")");
    }
}

std::array<double, 3> json_by(const json& cmd, const std::array<double, 3>& dflt) {
    if (!cmd.contains("by")) return dflt;
    const json& b = cmd.at("by");
    if (!b.is_array() || b.size() != 3) throw MeshLangError("'by' must be a [x,y,z] triple: " + cmd.dump());
    return {b[0].get<double>(), b[1].get<double>(), b[2].get<double>()};
}

// -- the `place` op (scene composition) — byte-mirror of Python meshlang ------ //
constexpr double kPlacePi = 3.14159265358979323846;

std::array<double, 3> json_vec3(const json& cmd, const std::string& key,
                                const std::array<double, 3>& dflt) {
    if (!cmd.contains(key)) return dflt;
    const json& a = cmd.at(key);
    if (!a.is_array() || a.size() < 3) return dflt;
    return {a[0].get<double>(), a[1].get<double>(), a[2].get<double>()};
}

// Transform a point: scale -> rotate (Rz@Ry@Rx, degrees) -> translate.
std::array<double, 3> place_xform(std::array<double, 3> p, const std::array<double, 3>& t,
                                  const std::array<double, 3>& rot, const std::array<double, 3>& s) {
    const double rx = rot[0] * kPlacePi / 180.0, ry = rot[1] * kPlacePi / 180.0,
                 rz = rot[2] * kPlacePi / 180.0;
    double x = p[0] * s[0], y = p[1] * s[1], z = p[2] * s[2];
    const double cx = std::cos(rx), sx = std::sin(rx);
    const double y1 = y * cx - z * sx, z1 = y * sx + z * cx; y = y1; z = z1;
    const double cy = std::cos(ry), sy = std::sin(ry);
    const double x1 = x * cy + z * sy, z2 = -x * sy + z * cy; x = x1; z = z2;
    const double cz = std::cos(rz), sz = std::sin(rz);
    const double x2 = x * cz - y * sz, y2 = x * sz + y * cz; x = x2; y = y2;
    return {x + t[0], y + t[1], z + t[2]};
}

// A mesh flattened to (positions, ngon faces, per-face tags, per-face materials) —
// the operands the place-merge concatenates before one from_pydata rebuild.
struct MeshArrays {
    std::vector<std::array<double, 3>> verts;
    std::vector<std::vector<int>> faces;
    std::vector<std::vector<std::string>> tags;
    std::vector<Material> mats;
};

MeshArrays mesh_to_arrays(const Mesh& m) {
    MeshArrays a;
    std::map<int, int> id2idx;
    for (const auto& v : m.verts()) {
        id2idx[v->id] = static_cast<int>(a.verts.size());
        a.verts.push_back(v->co);
    }
    for (const auto& f : m.faces()) {
        std::vector<int> fi;
        for (const Loop* lp : m.face_loops(f.get())) fi.push_back(id2idx[lp->vert->id]);
        a.faces.push_back(std::move(fi));
        a.tags.push_back(f->tags);
        a.mats.push_back(f->material);
    }
    return a;
}

std::string num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", v);
    return buf;
}

// A short, human tag for the GUI showing what an op selects.
std::string on_suffix(const json& op) {
    if (!op.contains("on") || !op.at("on").is_object()) return "";
    const json& on = op.at("on");
    const std::string by = on.value("by", "");
    if (by == "near") return "  @pick";
    if (by == "last_created") return "  @last";
    if (by == "normal") return "  @" + on.value("axis", std::string("z")) + (on.value("sign", 1.0) < 0 ? "-" : "+");
    if (by == "extreme") return "  @" + on.value("which", std::string("max")) + " " + on.value("axis", std::string("z"));
    if (by == "tag") return "  @#" + on.value("name", std::string("?"));
    if (by == "side") return "  @side";
    if (!by.empty()) return "  @" + by;
    if (on.contains("and") || on.contains("or") || on.contains("not")) return "  @(combo)";
    return "";
}

}  // namespace

// --------------------------------------------------------------------------- //
// log editing
// --------------------------------------------------------------------------- //
// Any explicit edit invalidates the redo stack (you can't redo onto a new branch).
Program& Program::add(json cmd) { ops_.push_back(std::move(cmd)); redo_.clear(); return *this; }
Program& Program::insert(std::size_t i, json cmd) {
    ops_.insert(ops_.begin() + std::min(i, ops_.size()), std::move(cmd));
    redo_.clear();
    return *this;
}
Program& Program::replace(std::size_t i, json cmd) { ops_.at(i) = std::move(cmd); redo_.clear(); return *this; }
Program& Program::erase(std::size_t i) { ops_.erase(ops_.begin() + i); redo_.clear(); return *this; }
void Program::undo() { if (!ops_.empty()) { redo_.push_back(std::move(ops_.back())); ops_.pop_back(); } }
void Program::redo() { if (!redo_.empty()) { ops_.push_back(std::move(redo_.back())); redo_.pop_back(); } }
void Program::clear() { ops_.clear(); redo_.clear(); }

// --------------------------------------------------------------------------- //
// fluent builders
// --------------------------------------------------------------------------- //
Program& Program::cube(double size, const std::string& mark) {
    json c{{"op", "cube"}, {"size", size}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::cylinder(int sides, double radius, double height, const std::string& mark) {
    json c{{"op", "cylinder"}, {"sides", sides}, {"radius", radius}, {"height", height}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::plane(double size_x, double size_y, const std::string& mark) {
    json c{{"op", "plane"}, {"size_x", size_x}, {"size_y", size_y <= 0 ? size_x : size_y}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::uv_sphere(int segments, int rings, double radius, const std::string& mark) {
    json c{{"op", "uv_sphere"}, {"segments", segments}, {"rings", rings}, {"radius", radius}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::cone(int sides, double radius, double height, const std::string& mark) {
    json c{{"op", "cone"}, {"sides", sides}, {"radius", radius}, {"height", height}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::torus(int major_segments, int minor_segments, double major_radius,
                        double minor_radius, const std::string& mark) {
    json c{{"op", "torus"}, {"major_segments", major_segments}, {"minor_segments", minor_segments},
           {"major_radius", major_radius}, {"minor_radius", minor_radius}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::grid(double size_x, double size_y, int x_div, int y_div, const std::string& mark) {
    json c{{"op", "grid"}, {"size_x", size_x}, {"size_y", size_y <= 0 ? size_x : size_y},
           {"x_div", x_div}, {"y_div", y_div <= 0 ? x_div : y_div}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::mesh(const std::vector<std::array<double, 3>>& verts,
                       const std::vector<std::vector<int>>& faces, const std::string& mark) {
    json vj = json::array();
    for (const auto& v : verts) vj.push_back({v[0], v[1], v[2]});
    json fj = json::array();
    for (const auto& f : faces) fj.push_back(f);
    json c{{"op", "mesh"}, {"verts", std::move(vj)}, {"faces", std::move(fj)}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::profile(const std::vector<std::array<double, 2>>& points, const std::string& plane,
                          bool closed, const std::string& mark) {
    json pj = json::array();
    for (const auto& p : points) pj.push_back({p[0], p[1]});
    json c{{"op", "profile"}, {"points", std::move(pj)}, {"plane", plane}, {"closed", closed}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::boolean_op(const std::string& mode, const std::vector<std::array<double, 3>>& verts,
                             const std::vector<std::vector<int>>& faces, const std::string& mark) {
    json vj = json::array();
    for (const auto& v : verts) vj.push_back({v[0], v[1], v[2]});
    json fj = json::array();
    for (const auto& f : faces) fj.push_back(f);
    json c{{"op", "boolean"}, {"mode", mode}, {"verts", std::move(vj)}, {"faces", std::move(fj)}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::del(const json& on) { return add(json{{"op", "delete"}, {"on", on}}); }
Program& Program::bridge(const json& on, const std::string& mark) {
    json c{{"op", "bridge"}, {"on", on}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::fill(const std::string& mark) {
    json c{{"op", "fill"}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::extrude(const json& on, double distance, const std::string& mark) {
    json c{{"op", "extrude"}, {"on", on}, {"distance", distance}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::inset(const json& on, double thickness, const std::string& mark) {
    json c{{"op", "inset"}, {"on", on}, {"thickness", thickness}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::bevel(const json& on, double width, double depth, const std::string& mark) {
    json c{{"op", "bevel"}, {"on", on}, {"width", width}, {"depth", depth}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::loop_cut(const json& on, const std::string& axis, const std::string& mark) {
    json c{{"op", "loop_cut"}, {"on", on}, {"axis", axis}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::edge_bevel(const json& on, double width, const std::string& mark) {
    json c{{"op", "edge_bevel"}, {"on", on}, {"width", width}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::solidify(double thickness, const std::string& mark) {
    json c{{"op", "solidify"}, {"thickness", thickness}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::mirror(const std::string& axis, const std::string& mark) {
    json c{{"op", "mirror"}, {"axis", axis}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::array(int count, const std::array<double, 3>& offset, const std::string& mark) {
    json c{{"op", "array"}, {"count", count}, {"offset", {offset[0], offset[1], offset[2]}}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::bisect(const std::array<double, 3>& point, const std::array<double, 3>& normal,
                         bool fill, const std::string& mark) {
    json c{{"op", "bisect"}, {"point", {point[0], point[1], point[2]}},
           {"normal", {normal[0], normal[1], normal[2]}}, {"fill", fill}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::spin(const std::string& axis, int steps, double angle, const std::string& mark) {
    json c{{"op", "spin"}, {"axis", axis}, {"steps", steps}, {"angle", angle}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::screw(const std::string& axis, int steps, int turns, double height,
                        double angle, const std::string& mark) {
    json c{{"op", "screw"}, {"axis", axis}, {"steps", steps}, {"turns", turns},
           {"height", height}, {"angle", angle}};
    if (!mark.empty()) c["mark"] = mark;
    return add(std::move(c));
}
Program& Program::subdivide(int levels) { return add(json{{"op", "subdivide"}, {"levels", levels}}); }
Program& Program::tag(const json& on, const std::string& name) {
    return add(json{{"op", "tag"}, {"on", on}, {"name", name}});
}
Program& Program::material(const json& on, const std::array<double, 3>& color, double metallic, double roughness) {
    return add(json{{"op", "material"}, {"on", on}, {"color", {color[0], color[1], color[2]}},
                    {"metallic", metallic}, {"roughness", roughness}});
}
Program& Program::translate(const json& on, const std::array<double, 3>& by) {
    return add(json{{"op", "translate"}, {"on", on}, {"by", {by[0], by[1], by[2]}}});
}
Program& Program::scale(const json& on, const std::array<double, 3>& by) {
    return add(json{{"op", "scale"}, {"on", on}, {"by", {by[0], by[1], by[2]}}});
}
Program& Program::assert_(const json& cond) {
    json c = cond;
    c["op"] = "assert";
    return add(std::move(c));
}

// --------------------------------------------------------------------------- //
// replay (a faithful port of meshlang.MeshProgram.build)
// --------------------------------------------------------------------------- //
Mesh Program::build(std::string* last_tag_out) const {
    Mesh mesh;
    bool has = false;
    std::string last_tag;  // most recent "__out<i>" (empty = none) -> last_created

    for (std::size_t i = 0; i < ops_.size(); ++i) {
        const json& cmd = ops_[i];
        if (!cmd.is_object() || !cmd.contains("op"))
            throw MeshLangError("op #" + std::to_string(i) + " is not a command dict: " + cmd.dump());
        const std::string op = cmd.at("op").get<std::string>();
        const std::string out_tag = "__out" + std::to_string(i);
        std::vector<const Face*> outs;

        try {
            if (op == "cube") {
                mesh = make_cube(cmd.value("size", 1.0));
                has = true;
                for (const auto& f : mesh.faces()) outs.push_back(f.get());
            } else if (op == "cylinder") {
                mesh = make_cylinder(cmd.value("sides", 24), cmd.value("radius", 0.5), cmd.value("height", 1.0));
                has = true;
                for (const auto& f : mesh.faces()) outs.push_back(f.get());
            } else if (op == "plane") {
                mesh = make_plane(cmd.value("size_x", 1.0), cmd.value("size_y", -1.0));
                has = true;
                for (const auto& f : mesh.faces()) outs.push_back(f.get());
            } else if (op == "uv_sphere") {
                mesh = make_uv_sphere(cmd.value("segments", 24), cmd.value("rings", 16), cmd.value("radius", 0.5));
                has = true;
                for (const auto& f : mesh.faces()) outs.push_back(f.get());
            } else if (op == "cone") {
                mesh = make_cone(cmd.value("sides", 24), cmd.value("radius", 0.5), cmd.value("height", 1.0));
                has = true;
                for (const auto& f : mesh.faces()) outs.push_back(f.get());
            } else if (op == "torus") {
                mesh = make_torus(cmd.value("major_segments", 24), cmd.value("minor_segments", 12),
                                  cmd.value("major_radius", 0.5), cmd.value("minor_radius", 0.2));
                has = true;
                for (const auto& f : mesh.faces()) outs.push_back(f.get());
            } else if (op == "grid") {
                mesh = make_grid(cmd.value("size_x", 1.0), cmd.value("size_y", -1.0),
                                 cmd.value("x_div", 10), cmd.value("y_div", -1));
                has = true;
                for (const auto& f : mesh.faces()) outs.push_back(f.get());
            } else if (op == "mesh") {
                // inline geometry (the import seam): raw verts+faces via from_pydata,
                // with optional per-face materials. Byte-identical to the Python kernel.
                std::vector<std::array<double, 3>> verts;
                for (const auto& v : cmd.value("verts", json::array()))
                    verts.push_back({v.at(0).get<double>(), v.at(1).get<double>(), v.at(2).get<double>()});
                std::vector<std::vector<int>> faces;
                for (const auto& f : cmd.value("faces", json::array())) {
                    std::vector<int> fi;
                    for (const auto& idx : f) fi.push_back(idx.get<int>());
                    faces.push_back(std::move(fi));
                }
                mesh = Mesh::from_pydata(verts, faces);
                has = true;
                if (cmd.contains("face_materials") && cmd.at("face_materials").is_array()) {
                    const json& fms = cmd.at("face_materials");
                    std::size_t fi = 0;
                    for (const auto& f : mesh.faces()) {
                        if (fi >= fms.size()) break;
                        const json& fm = fms[fi++];
                        if (fm.is_object()) {
                            Material mtl;
                            auto col = fm.value("color", std::vector<double>{0.8, 0.8, 0.8});
                            mtl.color = {col[0], col[1], col[2]};
                            mtl.metallic = fm.value("metallic", 0.0);
                            mtl.roughness = fm.value("roughness", 0.5);
                            mtl.set = true;
                            const_cast<Face*>(f.get())->material = mtl;
                        }
                    }
                }
                for (const auto& f : mesh.faces()) outs.push_back(f.get());
            } else if (op == "profile") {
                // a first-class 2D generatrix (a wire polyline) — the real lathe input.
                std::vector<std::array<double, 2>> pts;
                for (const auto& p : cmd.value("points", json::array()))
                    pts.push_back({p.at(0).get<double>(), p.at(1).get<double>()});
                mesh = make_profile(pts, cmd.value("plane", std::string("xz")), cmd.value("closed", false));
                has = true;
                for (const auto& f : mesh.faces()) outs.push_back(f.get());   // wire: no faces
            } else if (op == "place") {
                // compose a sub-object at a transform (the scene op): build it (a nested
                // op-log, or inline verts/faces), transform, and DISJOINT-UNION onto the
                // running mesh — which it starts if this is the first op. So the op-log is
                // natively multi-object. Byte-mirror of the Python meshlang place branch.
                Mesh sub;
                if (cmd.contains("program")) {
                    sub = Program(cmd.at("program").get<std::vector<json>>()).build();
                } else {
                    std::vector<std::array<double, 3>> sv;
                    for (const auto& v : cmd.value("verts", json::array()))
                        sv.push_back({v.at(0).get<double>(), v.at(1).get<double>(), v.at(2).get<double>()});
                    std::vector<std::vector<int>> sf;
                    for (const auto& f : cmd.value("faces", json::array())) {
                        std::vector<int> fi;
                        for (const auto& idx : f) fi.push_back(idx.get<int>());
                        sf.push_back(std::move(fi));
                    }
                    sub = Mesh::from_pydata(sv, sf);
                }
                MeshArrays B = mesh_to_arrays(sub);
                const std::array<double, 3> tt = json_vec3(cmd, "translate", {0, 0, 0});
                const std::array<double, 3> rr = json_vec3(cmd, "rotate", {0, 0, 0});
                const std::array<double, 3> ss = json_vec3(cmd, "scale", {1, 1, 1});
                for (auto& p : B.verts) p = place_xform(p, tt, rr, ss);
                MeshArrays A;
                if (has) A = mesh_to_arrays(mesh);
                const std::size_t nA = A.faces.size();
                const int base = static_cast<int>(A.verts.size());
                std::vector<std::array<double, 3>> V = std::move(A.verts);
                for (const auto& p : B.verts) V.push_back(p);
                std::vector<std::vector<int>> F = std::move(A.faces);
                for (const auto& f : B.faces) {
                    std::vector<int> fi;
                    for (int k : f) fi.push_back(base + k);
                    F.push_back(std::move(fi));
                }
                std::vector<std::vector<std::string>> tags = std::move(A.tags);
                for (auto& t : B.tags) tags.push_back(std::move(t));
                mesh = Mesh::from_pydata(V, F, tags);
                has = true;
                // materials: A's preserved; placed faces get the place material (if given),
                // else keep the sub-object's own materials.
                const bool has_pm = cmd.contains("material") && cmd.at("material").is_object();
                Material pm;
                if (has_pm) {
                    const json& mj = cmd.at("material");
                    if (mj.contains("color")) {
                        const json& c = mj.at("color");
                        pm.color = {c[0].get<double>(), c[1].get<double>(), c[2].get<double>()};
                    }
                    pm.metallic = mj.value("metallic", 0.0);
                    pm.roughness = mj.value("roughness", 0.5);
                    pm.set = true;
                }
                std::vector<Material> mats = std::move(A.mats);
                for (std::size_t k = 0; k < B.faces.size(); ++k) mats.push_back(has_pm ? pm : B.mats[k]);
                std::size_t fi = 0;
                for (const auto& f : mesh.faces()) {
                    if (fi < mats.size()) const_cast<Face*>(f.get())->material = mats[fi];
                    if (fi >= nA) outs.push_back(f.get());
                    ++fi;
                }
            } else if (op == "boolean") {
                // current mesh = operand A; inline verts+faces = operand B (the tool/cutter)
                std::vector<std::array<double, 3>> bverts;
                for (const auto& v : cmd.value("verts", json::array()))
                    bverts.push_back({v.at(0).get<double>(), v.at(1).get<double>(), v.at(2).get<double>()});
                std::vector<std::vector<int>> bfaces;
                for (const auto& f : cmd.value("faces", json::array())) {
                    std::vector<int> fi;
                    for (const auto& idx : f) fi.push_back(idx.get<int>());
                    bfaces.push_back(std::move(fi));
                }
                mesh = mirage::boolean(mesh, Mesh::from_pydata(bverts, bfaces),
                                       cmd.value("mode", std::string("difference")));
                // a fresh welded mesh; last_created undefined (outs stays empty)
            } else if (!has) {
                throw MeshLangError("op '" + op + "' before any primitive");
            } else if (op == "extrude") {
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                mesh = mirage::extrude(mesh, seln, cmd.value("distance", 0.5), out_tag);  // free op, not Program::extrude
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "inset") {
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                mesh = mirage::inset(mesh, seln, cmd.value("thickness", 0.3), out_tag);  // free op, not Program::inset
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "bevel") {
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                mesh = mirage::bevel(mesh, seln, cmd.value("width", 0.2), cmd.value("depth", 0.1), out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "loop_cut") {
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                mesh = mirage::loop_cut(mesh, seln, cmd.value("axis", std::string("z")), out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "edge_bevel") {
                auto esel = resolve_edges(mesh, cmd.value("on", json{{"by", "all"}}), last_tag);
                mesh = mirage::edge_bevel(mesh, esel, cmd.value("width", 0.15), out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "delete") {
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                mesh = mirage::delete_faces(mesh, seln);  // outs stays empty (faces removed)
            } else if (op == "bridge") {
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                mesh = mirage::bridge_faces(mesh, seln, out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "fill") {
                mesh = mirage::fill_holes(mesh, out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "solidify") {
                mesh = mirage::solidify(mesh, cmd.value("thickness", 0.1), out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "mirror") {
                mesh = mirage::mirror(mesh, cmd.value("axis", std::string("x")), out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "array") {
                auto off = cmd.value("offset", std::vector<double>{1.1, 0.0, 0.0});
                std::array<double, 3> o{off.size() > 0 ? off[0] : 1.1, off.size() > 1 ? off[1] : 0.0,
                                        off.size() > 2 ? off[2] : 0.0};
                mesh = mirage::array(mesh, cmd.value("count", 3), o, out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "bisect") {
                auto pt = cmd.value("point", std::vector<double>{0.0, 0.0, 0.0});
                auto nm = cmd.value("normal", std::vector<double>{0.0, 0.0, 1.0});
                std::array<double, 3> p{pt.size() > 0 ? pt[0] : 0.0, pt.size() > 1 ? pt[1] : 0.0,
                                        pt.size() > 2 ? pt[2] : 0.0};
                std::array<double, 3> nrm{nm.size() > 0 ? nm[0] : 0.0, nm.size() > 1 ? nm[1] : 0.0,
                                          nm.size() > 2 ? nm[2] : 1.0};
                mesh = mirage::bisect(mesh, p, nrm, cmd.value("fill", false), out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "spin") {
                mesh = mirage::spin(mesh, cmd.value("axis", std::string("z")), cmd.value("steps", 24),
                                    cmd.value("angle", 360.0), out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "screw") {
                mesh = mirage::screw(mesh, cmd.value("axis", std::string("z")), cmd.value("steps", 24),
                                     cmd.value("turns", 1), cmd.value("height", 1.0),
                                     cmd.value("angle", 360.0), out_tag);
                outs = faces_with_tag(mesh, out_tag);
            } else if (op == "subdivide") {
                const int levels = cmd.value("levels", 1);
                for (int k = 0; k < levels; ++k) mesh = catmull_clark(mesh);
                // global op — last_created is undefined afterward (outs stays empty)
            } else if (op == "tag") {
                if (!cmd.contains("name")) throw MeshLangError("tag op needs 'name'");
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                const std::string name = cmd.at("name").get<std::string>();
                for (const Face* f : seln) add_tag(f, name);
                outs = seln;
            } else if (op == "material") {
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                Material m;
                if (cmd.contains("color")) {
                    const json& c = cmd.at("color");
                    m.color = {c[0].get<double>(), c[1].get<double>(), c[2].get<double>()};
                }
                m.metallic = cmd.value("metallic", 0.0);
                m.roughness = cmd.value("roughness", 0.5);
                m.set = true;
                for (const Face* f : seln) const_cast<Face*>(f)->material = m;
                outs = seln;
            } else if (op == "translate" || op == "scale") {
                auto seln = resolve(mesh, cmd.value("on", sel::all()), last_tag);
                const std::array<double, 3> dflt = op == "scale" ? std::array<double, 3>{1, 1, 1}
                                                                  : std::array<double, 3>{0, 0, 0};
                transform_region(mesh, seln, op, json_by(cmd, dflt));
                outs = seln;
            } else if (op == "assert") {
                check_assert(mesh, cmd);
            } else {
                throw MeshLangError("unknown op '" + op + "'");
            }
        } catch (const SelectorEmpty&) {
            throw;
        } catch (const MeshLangError&) {
            throw;
        } catch (const std::exception& e) {  // localise any kernel error to its op
            throw MeshLangError("op #" + std::to_string(i) + " '" + op + "': " + e.what());
        }

        for (const Face* f : outs)  // stamp the step's out tag (extrude/inset already did via mark)
            if (!has_tag(f, out_tag)) add_tag(f, out_tag);
        if (cmd.contains("mark") && !cmd.at("mark").is_null() && !outs.empty()) {
            const std::string m = cmd.at("mark").get<std::string>();
            for (const Face* f : outs) add_tag(f, m);
        }
        if (!outs.empty()) last_tag = out_tag;

        try {
            mesh.validate();
        } catch (const std::exception& e) {
            throw MeshLangError("op #" + std::to_string(i) + " '" + op + "' produced an invalid mesh: " +
                                e.what());
        }
    }
    if (!has) throw MeshLangError("empty program");
    if (last_tag_out) *last_tag_out = last_tag;
    return mesh;
}

// --------------------------------------------------------------------------- //
// JSON round-trip — the dual-operator bridge
// --------------------------------------------------------------------------- //
std::string Program::to_json(int indent) const { return json(ops_).dump(indent); }

Program Program::from_json(const std::string& s) {
    json j = json::parse(s);
    if (!j.is_array()) throw MeshLangError("op-log JSON must be an array of op dicts");
    return Program(j.get<std::vector<json>>());
}

std::string Program::label(const json& op) {
    const std::string k = op.value("op", std::string("?"));
    if (k == "cube") return "cube  size=" + num(op.value("size", 1.0));
    if (k == "cylinder")
        return "cylinder  n=" + std::to_string(op.value("sides", 24)) + " r=" + num(op.value("radius", 0.5)) +
               " h=" + num(op.value("height", 1.0));
    if (k == "inset") return "inset  t=" + num(op.value("thickness", 0.3)) + on_suffix(op);
    if (k == "bevel") return "bevel  w=" + num(op.value("width", 0.2)) + " d=" + num(op.value("depth", 0.1)) + on_suffix(op);
    if (k == "extrude") return "extrude  d=" + num(op.value("distance", 0.5)) + on_suffix(op);
    if (k == "loop_cut") return "loop_cut  " + op.value("axis", std::string("z")) + on_suffix(op);
    if (k == "edge_bevel") return "edge_bevel  w=" + num(op.value("width", 0.15)) + on_suffix(op);
    if (k == "plane") return "plane  " + num(op.value("size_x", 1.0)) + "x" + num(op.value("size_y", 1.0));
    if (k == "uv_sphere")
        return "uv_sphere  seg=" + std::to_string(op.value("segments", 24)) +
               " rings=" + std::to_string(op.value("rings", 16)) + " r=" + num(op.value("radius", 0.5));
    if (k == "cone")
        return "cone  n=" + std::to_string(op.value("sides", 24)) + " r=" + num(op.value("radius", 0.5)) +
               " h=" + num(op.value("height", 1.0));
    if (k == "torus")
        return "torus  M=" + std::to_string(op.value("major_segments", 24)) +
               " N=" + std::to_string(op.value("minor_segments", 12)) +
               " R=" + num(op.value("major_radius", 0.5)) + " r=" + num(op.value("minor_radius", 0.2));
    if (k == "grid")
        return "grid  " + std::to_string(op.value("x_div", 10)) + "x" + std::to_string(op.value("y_div", 10));
    if (k == "mesh") {
        const std::size_t nv = op.contains("verts") ? op.at("verts").size() : 0;
        const std::size_t nf = op.contains("faces") ? op.at("faces").size() : 0;
        return "mesh  " + std::to_string(nv) + "v " + std::to_string(nf) + "f (imported)";
    }
    if (k == "profile") {
        const std::size_t np = op.contains("points") ? op.at("points").size() : 0;
        return "profile  " + std::to_string(np) + "pt " + op.value("plane", std::string("xz")) +
               (op.value("closed", false) ? " closed" : "");
    }
    if (k == "boolean") {
        const std::size_t nf = op.contains("faces") ? op.at("faces").size() : 0;
        return "boolean  " + op.value("mode", std::string("difference")) + " (cutter " +
               std::to_string(nf) + "f)";
    }
    if (k == "delete") return "delete" + on_suffix(op);
    if (k == "bridge") return "bridge" + on_suffix(op);
    if (k == "fill") return "fill  (cap holes)";
    if (k == "solidify") return "solidify  t=" + num(op.value("thickness", 0.1));
    if (k == "mirror") return "mirror  " + op.value("axis", std::string("x"));
    if (k == "array") return "array  x" + std::to_string(op.value("count", 3));
    if (k == "bisect") return std::string("bisect") + (op.value("fill", false) ? " +fill" : "");
    if (k == "spin")
        return "spin  " + op.value("axis", std::string("z")) + " " + num(op.value("angle", 360.0)) +
               "deg x" + std::to_string(op.value("steps", 24));
    if (k == "screw")
        return "screw  " + op.value("axis", std::string("z")) + " x" +
               std::to_string(op.value("turns", 1)) + "turn h=" + num(op.value("height", 1.0));
    if (k == "subdivide") return "subdivide  x" + std::to_string(op.value("levels", 1));
    if (k == "tag") return "tag  #" + op.value("name", std::string("?")) + on_suffix(op);
    if (k == "material") return "material  m=" + num(op.value("metallic", 0.0)) + " r=" + num(op.value("roughness", 0.5)) + on_suffix(op);
    if (k == "translate") return "translate" + on_suffix(op);
    if (k == "scale") return "scale" + on_suffix(op);
    if (k == "assert") return "assert";
    return k;
}

}  // namespace mirage
