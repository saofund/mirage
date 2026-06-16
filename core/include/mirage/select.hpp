// Selection-as-query — the native port of meshlang's selector engine.
//
// A selector is a JSON dict re-evaluated against the LIVE mesh, never a stored
// index (kernel operators rebuild the mesh and renumber every element — the
// Topological Naming Problem — so the grammar has no field for a raw index):
//
//   {"by": "all" | "normal" | "tag" | "last_created" | "extreme" | "side" | "near"}
//   composable with {"and": [...]}, {"or": [...]}, {"not": {...}}.
//
// "near" {"by":"near","point":[x,y,z]} resolves to the face whose centroid is
// closest to the point — it is how a GUI click is recorded in the op-log (the
// pick survives replay because it re-resolves against the rebuilt mesh).
//
// This grammar is shared verbatim with the Python meshlang (src/mirage/
// meshlang.py): one selector dialect, two engines, differential-tested.
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "mirage/mesh.hpp"

namespace mirage {

using json = nlohmann::json;

// The meshlang-level error (bad op / bad selector / failed assert). Mirrors
// Python's MeshLangError.
struct MeshLangError : std::runtime_error {
    explicit MeshLangError(const std::string& msg) : std::runtime_error(msg) {}
};

// A selector matched zero faces — carries the selector and a diagnostics
// summary of what IS selectable, so an agent (or a user) can self-correct.
struct SelectorEmpty : MeshLangError {
    SelectorEmpty(json sel_, json diagnostics_);
    json sel;
    json diagnostics;
};

// What the mesh offers a selector right now: face count, bbox, public tags,
// and a per-axis normal histogram. (Python: meshlang._diagnostics.)
json selector_diagnostics(const Mesh& mesh);

// Resolve a selector to a list of faces (deduped, mesh order). Throws
// SelectorEmpty when nothing matches, MeshLangError on a malformed selector.
// `last_tag` is the replay's most recent "__out<i>" tag (for last_created);
// empty string = none.
std::vector<const Face*> resolve(const Mesh& mesh, const json& sel, const std::string& last_tag = "");

// Resolve an EDGE selector to a list of edges (deduped, mesh order). The parallel
// grammar for edges (the foundation for edge_bevel):
//   {"by": "all" | "sharp"(angle) | "axis"(axis,tol) | "boundary" | "on_face"(face)}
//   composable with and/or/not. `on_face` composes with the face grammar above.
std::vector<Edge*> resolve_edges(const Mesh& mesh, const json& sel, const std::string& last_tag = "");

}  // namespace mirage
