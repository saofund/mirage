// mirage::Program — the op-log: the model is a replayable list of operators.
//
// This is the single source of truth a human (GUI) and an AI (MCP) both edit.
// Crucially it is a *twin* of the Python meshlang.MeshProgram: an op is the SAME
// JSON command dict — {"op", "on": <selector>, ...params, "mark"} — so one op-log
// loads and replays identically in either engine (to_json/from_json bridge them).
//
// build() replays the ops into a kernel Mesh, resolving each op's `on` selector
// against the LIVE mesh (selection-as-query, never a stored index — the
// Topological Naming Problem) and threading a durable "__out<i>" tag so
// `last_created` resolves after any op. There is no fragile "active face"
// heuristic: the selector IS the intent, and it re-evaluates on every rebuild.
#pragma once

#include <array>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "mirage/lint.hpp"
#include "mirage/mesh.hpp"
#include "mirage/select.hpp"

namespace mirage {

using json = nlohmann::json;

class Program {
public:
    Program() = default;
    explicit Program(std::vector<json> ops) : ops_(std::move(ops)) {}

    // -- log editing (mirrors Python MeshProgram.add/insert/replace/delete) ----
    Program& add(json cmd);
    Program& insert(std::size_t i, json cmd);
    Program& replace(std::size_t i, json cmd);
    Program& erase(std::size_t i);  // Python's `delete` (a C++ keyword)
    void undo();                    // pop the last op onto the redo stack
    void redo();                    // re-apply the last undone op
    bool can_redo() const { return !redo_.empty(); }
    void clear();

    // -- fluent builders (sugar; an AI just emits the op dicts) ----------------
    // `on` is a selector dict (see select.hpp). Default selectors are chosen so a
    // bare cube()/inset() does the obvious thing in a GUI.
    Program& cube(double size = 1.0, const std::string& mark = "");
    Program& cylinder(int sides = 24, double radius = 0.5, double height = 1.0, const std::string& mark = "");
    Program& plane(double size_x = 1.0, double size_y = -1.0, const std::string& mark = "");
    Program& uv_sphere(int segments = 24, int rings = 16, double radius = 0.5, const std::string& mark = "");
    Program& cone(int sides = 24, double radius = 0.5, double height = 1.0, const std::string& mark = "");
    Program& torus(int major_segments = 24, int minor_segments = 12, double major_radius = 0.5,
                   double minor_radius = 0.2, const std::string& mark = "");
    Program& grid(double size_x = 1.0, double size_y = -1.0, int x_div = 10, int y_div = -1,
                  const std::string& mark = "");
    Program& mesh(const std::vector<std::array<double, 3>>& verts,
                  const std::vector<std::vector<int>>& faces, const std::string& mark = "");
    Program& profile(const std::vector<std::array<double, 2>>& points, const std::string& plane = "xz",
                     bool closed = false, const std::string& mark = "");
    Program& del(const json& on);                  // delete the selected faces (open the mesh)
    Program& bridge(const json& on, const std::string& mark = "");  // tunnel between two openings
    Program& fill(const std::string& mark = "");   // cap every boundary loop
    Program& extrude(const json& on, double distance = 0.5, const std::string& mark = "");
    Program& inset(const json& on, double thickness = 0.3, const std::string& mark = "");
    Program& bevel(const json& on, double width = 0.2, double depth = 0.1, const std::string& mark = "");
    Program& loop_cut(const json& on, const std::string& axis = "z", const std::string& mark = "");
    Program& edge_bevel(const json& on, double width = 0.15, const std::string& mark = "");
    Program& solidify(double thickness = 0.1, const std::string& mark = "");
    Program& mirror(const std::string& axis = "x", const std::string& mark = "");
    Program& array(int count = 3, const std::array<double, 3>& offset = {1.1, 0.0, 0.0},
                   const std::string& mark = "");
    Program& bisect(const std::array<double, 3>& point = {0, 0, 0},
                    const std::array<double, 3>& normal = {0, 0, 1}, bool fill = false,
                    const std::string& mark = "");
    Program& spin(const std::string& axis = "z", int steps = 24, double angle = 360.0,
                  const std::string& mark = "");
    Program& screw(const std::string& axis = "z", int steps = 24, int turns = 1,
                   double height = 1.0, double angle = 360.0, const std::string& mark = "");
    Program& subdivide(int levels = 1);
    Program& tag(const json& on, const std::string& name);
    Program& material(const json& on, const std::array<double, 3>& color, double metallic = 0.0, double roughness = 0.5);
    Program& translate(const json& on, const std::array<double, 3>& by);
    Program& scale(const json& on, const std::array<double, 3>& by);
    Program& assert_(const json& cond);  // {"closed_manifold": true} and/or {"euler": n}

    bool empty() const { return ops_.empty(); }
    std::size_t size() const { return ops_.size(); }
    const std::vector<json>& ops() const { return ops_; }
    const json& op(std::size_t i) const { return ops_.at(i); }

    // Replay the op-log into a fresh, validated mesh. Throws MeshLangError /
    // SelectorEmpty (both derive from std::runtime_error) localised to the op.
    // `last_tag_out`, when non-null, receives the final "__out<i>" tag — what a
    // `last_created` selector resolves against (the GUI uses it to show/target
    // the most recent result).
    Mesh build(std::string* last_tag_out = nullptr) const;

    // Static lint of the op-log: silent traps that build but lose intent.
    std::vector<LintWarning> lint() const { return lint_program(ops_); }

    // The op-log IS JSON — round-trips with the Python meshlang dialect.
    std::string to_json(int indent = 2) const;
    static Program from_json(const std::string& s);

    // One-line human label for the GUI op-log view.
    static std::string label(const json& op);

private:
    std::vector<json> ops_;
    std::vector<json> redo_;  // ops popped by undo(), transient (not part of the saved SoT)
};

// Selector sugar (the native mirror of Python meshlang.Sel) — an AI emits the
// dicts directly; the GUI and tests use these to stay terse and correct.
namespace sel {
inline json all() { return json{{"by", "all"}}; }
inline json normal(const std::string& axis = "z", double sign = 1.0, double tol = 0.5) {
    return json{{"by", "normal"}, {"axis", axis}, {"sign", sign}, {"tol", tol}};
}
inline json tag(const std::string& name) { return json{{"by", "tag"}, {"name", name}}; }
inline json extreme(const std::string& axis = "z", const std::string& which = "max") {
    return json{{"by", "extreme"}, {"axis", axis}, {"which", which}};
}
inline json side(const std::string& axis = "x", double sign = 1.0) {
    return json{{"by", "side"}, {"axis", axis}, {"sign", sign}};
}
inline json last() { return json{{"by", "last_created"}}; }
inline json near(const std::array<double, 3>& p) {
    return json{{"by", "near"}, {"point", {p[0], p[1], p[2]}}};
}
inline json AND(std::initializer_list<json> s) { return json{{"and", json(s)}}; }
inline json OR(std::initializer_list<json> s) { return json{{"or", json(s)}}; }
inline json NOT(json s) { return json{{"not", std::move(s)}}; }
}  // namespace sel

}  // namespace mirage
