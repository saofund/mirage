#include "mirage/program.hpp"

#include <cstdio>

namespace mirage {

void Program::cube(double size) { ops_.push_back({OpKind::Cube, size, 0, 0}); }
void Program::cylinder(int sides, double radius, double height) {
    ops_.push_back({OpKind::Cylinder, static_cast<double>(sides), radius, height});
}
void Program::inset(double thickness) { ops_.push_back({OpKind::Inset, thickness, 0, 0}); }
void Program::extrude(double distance) { ops_.push_back({OpKind::Extrude, distance, 0, 0}); }
void Program::subdivide(int levels) { ops_.push_back({OpKind::Subdivide, static_cast<double>(levels), 0, 0}); }
void Program::undo() { if (!ops_.empty()) ops_.pop_back(); }
void Program::clear() { ops_.clear(); }

Mesh Program::build() const {
    Mesh mesh;
    const Face* active = nullptr;
    bool has = false;
    for (const Op& op : ops_) {
        switch (op.kind) {
            case OpKind::Cube:
                mesh = make_cube(op.a > 0 ? op.a : 1.0);
                active = top_face(mesh);
                has = true;
                break;
            case OpKind::Cylinder:
                mesh = make_cylinder(static_cast<int>(op.a), op.b, op.c);
                active = top_face(mesh);
                has = true;
                break;
            case OpKind::Inset:
                if (has && active) {
                    mesh = mirage::inset(mesh, {active}, op.a);  // free operator, not Program::inset
                    active = mesh.faces().empty() ? nullptr : mesh.faces().back().get();  // inner face
                }
                break;
            case OpKind::Extrude:
                if (has && active) {
                    mesh = mirage::extrude(mesh, {active}, op.a);  // free operator, not Program::extrude
                    active = mesh.faces().empty() ? nullptr : mesh.faces().back().get();  // lifted cap
                }
                break;
            case OpKind::Subdivide:
                if (has) {
                    for (int i = 0; i < static_cast<int>(op.a); ++i) mesh = catmull_clark(mesh);
                    active = top_face(mesh);
                }
                break;
        }
    }
    return mesh;
}

std::string Program::label(const Op& op) {
    char buf[80];
    switch (op.kind) {
        case OpKind::Cube: std::snprintf(buf, sizeof(buf), "cube  size=%.2f", op.a); break;
        case OpKind::Cylinder:
            std::snprintf(buf, sizeof(buf), "cylinder  n=%d r=%.2f h=%.2f", static_cast<int>(op.a), op.b, op.c);
            break;
        case OpKind::Inset: std::snprintf(buf, sizeof(buf), "inset  t=%.2f", op.a); break;
        case OpKind::Extrude: std::snprintf(buf, sizeof(buf), "extrude  d=%.2f", op.a); break;
        case OpKind::Subdivide: std::snprintf(buf, sizeof(buf), "subdivide  x%d", static_cast<int>(op.a)); break;
    }
    return std::string(buf);
}

}  // namespace mirage
