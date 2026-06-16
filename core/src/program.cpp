#include "mirage/program.hpp"

#include <algorithm>
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
Program& Program::subdivide(int levels) { return add(json{{"op", "subdivide"}, {"levels", levels}}); }
Program& Program::tag(const json& on, const std::string& name) {
    return add(json{{"op", "tag"}, {"on", on}, {"name", name}});
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
    if (k == "subdivide") return "subdivide  x" + std::to_string(op.value("levels", 1));
    if (k == "tag") return "tag  #" + op.value("name", std::string("?")) + on_suffix(op);
    if (k == "translate") return "translate" + on_suffix(op);
    if (k == "scale") return "scale" + on_suffix(op);
    if (k == "assert") return "assert";
    return k;
}

}  // namespace mirage
