// pybind11 bindings — expose mirage_core to Python so the C++ kernel can be
// differential-tested against the Python kernel (the truth oracle). See
// tests/test_cpp_core.py.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mirage/mesh.hpp"

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
        });

    mod.def("make_cube", &make_cube, py::arg("size") = 1.0);
    mod.def("make_cylinder", &make_cylinder, py::arg("sides") = 24, py::arg("radius") = 0.5,
            py::arg("height") = 1.0);
    mod.def("catmull_clark", &catmull_clark, py::arg("mesh"));
    // top-face convenience ops (the selection-as-query engine lands in the op-log layer)
    mod.def("extrude_top", [](const Mesh& m, double d) { return extrude(m, {top_face(m)}, d); },
            py::arg("mesh"), py::arg("distance") = 0.5);
    mod.def("inset_top", [](const Mesh& m, double t) { return inset(m, {top_face(m)}, t); },
            py::arg("mesh"), py::arg("thickness") = 0.3);
}
