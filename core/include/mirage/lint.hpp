// Lint — the native port of meshlang's silent-trap detector.
//
// Some op-logs build cleanly yet quietly lose intent: a zero-distance extrude
// (a no-op whose mark lands on the un-extruded face), an inset/bevel value
// silently clamped, which != 'max' that the kernel treats as MIN, a
// `last_created` right after a primitive that resolves to the WHOLE surface.
// build() cannot catch these (the mesh is valid), so a static lint does. This is
// the same nervous system the AI gets over MCP (repair.lint_program), now on the
// native path so a human's loaded/edited op-log is checked too.
#pragma once

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace mirage {

using json = nlohmann::json;

struct LintWarning {
    int op_index;
    std::string code;
    std::string message;
    std::string suggestion;
};

// Static lint of an op-log. Mirrors Python repair.lint_program (same codes).
std::vector<LintWarning> lint_program(const std::vector<json>& ops);

}  // namespace mirage
