// pybind11 bindings — expose mirage_core to Python so the C++ kernel can be
// differential-tested against the Python kernel (the truth oracle). See
// tests/test_cpp_core.py.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mirage/mesh.hpp"
#include "mirage/program.hpp"
#include "mirage/select.hpp"

namespace py = pybind11;
using namespace mirage;  // top_face / make_cube / extrude / inset are provided by mirage_core

PYBIND11_MODULE(_mirage_core, mod) {
    mod.doc() = "mirage_core — native C++ topology kernel (radial-edge)";

    py::class_<Mesh>(mod, "Mesh")
        .def("validate", &Mesh::validate)
        .def("num_verts", &Mesh::num_verts)
        .def("num_edges", &Mesh::num_edges)
        .def("num_faces", &Mesh::num_faces)
        .def("euler", &Mesh::euler)
        .def("is_closed_manifold", &Mesh::is_closed_manifold)
        .def("stats", [](const Mesh& m) {
            py::dict d;
            d["verts"] = m.num_verts();
            d["edges"] = m.num_edges();
            d["faces"] = m.num_faces();
            d["euler"] = m.euler();
            d["closed_manifold"] = m.is_closed_manifold();
            return d;
        })
        // Vertex positions in mesh order. The kernels claim to be byte-identical, but with
        // no way to read a coordinate out of the C++ mesh that claim could only ever be
        // tested on element COUNTS — positional drift between the engines would pass every
        // test silently. This is what lets the differential tests compare actual geometry.
        .def("positions", [](const Mesh& m) {
            py::list out;
            for (const auto& v : m.verts()) out.append(py::make_tuple(v->co[0], v->co[1], v->co[2]));
            return out;
        })
        // Faces as vertex-index lists, in mesh order — so face order and winding are
        // testable too, not just the vertex cloud.
        .def("face_indices", [](const Mesh& m) {
            py::list out;
            for (const auto& f : m.faces()) {
                py::list fv;
                for (Loop* lp : m.face_loops(f.get())) fv.append(lp->vert->id);
                out.append(fv);
            }
            return out;
        })
        // Edge creases: (v1_id, v2_id, crease) for every edge with sharpness, in mesh order.
        .def("edge_creases", [](const Mesh& m) {
            py::list out;
            for (const auto& e : m.edges())
                if (e->crease > 0.0) out.append(py::make_tuple(e->v1->id, e->v2->id, e->crease));
            return out;
        })
        // Per-face material: [r, g, b, metallic, roughness, set?] for each face,
        // for differential-testing the `material` op against the Python engine.
        .def("face_materials", [](const Mesh& m) {
            py::list out;
            for (const auto& f : m.faces()) {
                const Material& mat = f->material;
                out.append(py::make_tuple(mat.color[0], mat.color[1], mat.color[2], mat.metallic,
                                          mat.roughness, mat.set));
            }
            return out;
        });

    mod.def("make_cube", &make_cube, py::arg("size") = 1.0);
    mod.def("make_cylinder", &make_cylinder, py::arg("sides") = 24, py::arg("radius") = 0.5,
            py::arg("height") = 1.0);
    mod.def("catmull_clark", static_cast<Mesh (*)(const Mesh&, int)>(&catmull_clark), py::arg("mesh"),
            py::arg("levels") = 1);
    // top-face convenience ops (the selection-as-query engine lands in the op-log layer)
    mod.def("extrude_top", [](const Mesh& m, double d) { return extrude(m, {top_face(m)}, d); },
            py::arg("mesh"), py::arg("distance") = 0.5);
    mod.def("inset_top", [](const Mesh& m, double t) { return inset(m, {top_face(m)}, t); },
            py::arg("mesh"), py::arg("thickness") = 0.3);

    // op-log (mirage::Program) — the C++ twin of Python meshlang.MeshProgram. It
    // speaks the SAME JSON dialect, so an op-log authored by either engine (a
    // human in the GUI or an AI over MCP) replays identically in the other. The
    // differential test (tests/test_cpp_program.py) builds the SAME JSON in both.
    py::register_exception<MeshLangError>(mod, "MeshLangError");
    py::class_<Program>(mod, "Program")
        .def(py::init<>())
        .def_static("from_json", &Program::from_json, py::arg("s"))
        .def("to_json", &Program::to_json, py::arg("indent") = 2)
        .def("size", &Program::size)
        .def("build", [](const Program& p) { return p.build(); });

    // Replay an op-log given as a JSON string straight to a kernel Mesh — the
    // one call a differential harness needs.
    mod.def("replay_json", [](const std::string& s) { return Program::from_json(s).build(); },
            py::arg("ops_json"));
    // Lint an op-log (JSON string) -> list of {op_index, code, message, suggestion}.
    // Differential-tested against Python repair.lint_program (tests/test_cpp_program).
    mod.def("lint_json", [](const std::string& s) {
        py::list out;
        for (const LintWarning& w : lint_program(json::parse(s).get<std::vector<json>>())) {
            py::dict d;
            d["op_index"] = w.op_index; d["code"] = w.code;
            d["message"] = w.message; d["suggestion"] = w.suggestion;
            out.append(d);
        }
        return out;
    }, py::arg("ops_json"));
    // Resolve a selector (JSON) against a mesh, returning the matched face count —
    // exercises the native selection-as-query engine directly.
    mod.def("selector_count",
            [](const Mesh& m, const std::string& sel_json) {
                return resolve(m, json::parse(sel_json)).size();
            },
            py::arg("mesh"), py::arg("selector_json"));
    // Same, for the edge-selection grammar (the foundation for edge_bevel).
    mod.def("edge_selector_count",
            [](const Mesh& m, const std::string& sel_json) {
                return resolve_edges(m, json::parse(sel_json)).size();
            },
            py::arg("mesh"), py::arg("edge_selector_json"));
}
