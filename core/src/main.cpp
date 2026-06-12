// mirage_stats — native smoke check for mirage_core: primitives + Catmull-Clark
// subdivision, each validated (euler == 2, closed 2-manifold). Counts are
// differential-checked against the Python kernel spec (see the build script).
#include "mirage/mesh.hpp"
#include "mirage/program.hpp"

#include <cstdio>

using namespace mirage;  // top_face provided by mirage_core

static bool report(const char* name, const Mesh& m) {
    const bool ok = (m.euler() == 2) && m.is_closed_manifold();
    std::printf("  %-22s v=%-4zu e=%-4zu f=%-4zu euler=%d manifold=%s  %s\n", name,
                m.num_verts(), m.num_edges(), m.num_faces(), m.euler(),
                m.is_closed_manifold() ? "true" : "false", ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    std::printf("mirage_core native kernel check:\n");

    Mesh cube = make_cube(1.0);
    cube.validate();
    ok &= report("cube", cube);

    Mesh cube_cc1 = catmull_clark(cube);
    cube_cc1.validate();
    ok &= report("cube + cc x1", cube_cc1);

    Mesh cube_cc2 = catmull_clark(cube_cc1);
    cube_cc2.validate();
    ok &= report("cube + cc x2", cube_cc2);

    Mesh cyl = make_cylinder(8, 0.5, 1.0);
    cyl.validate();
    ok &= report("cylinder(8)", cyl);

    Mesh cyl_cc1 = catmull_clark(cyl);
    cyl_cc1.validate();
    ok &= report("cylinder(8) + cc x1", cyl_cc1);

    Mesh ex = extrude(cube, {top_face(cube)}, 0.5);
    ex.validate();
    ok &= report("cube extrude top", ex);

    Mesh ins = inset(cube, {top_face(cube)}, 0.3);
    ins.validate();
    ok &= report("cube inset top", ins);

    Mesh ins2 = inset(cube, {top_face(cube)}, 0.3);  // inset -> extrude the inner face (a boss)
    Mesh boss = extrude(ins2, {ins2.faces().back().get()}, 0.5);
    boss.validate();
    ok &= report("cube inset+extrude boss", boss);

    // op-log (mirage::Program) — selectors + replay, the AI/GUI-shared SoT.
    std::printf("op-log replay (selection-as-query):\n");
    Program prog;
    prog.cube(1.0)
        .inset(sel::normal("z"), 0.3)   // the top face, by its normal
        .extrude(sel::last(), 0.6)      // the inner face the inset just made
        .tag(sel::extreme("z", "max"), "lid");
    Mesh m = prog.build();
    m.validate();
    ok &= report("program: boss + tag", m);

    // JSON round-trip — the dual-operator bridge (this op-log loads in Python too)
    const std::string js = prog.to_json(0);
    Mesh m2 = Program::from_json(js).build();
    const bool same = (m.num_verts() == m2.num_verts()) && (m.num_edges() == m2.num_edges()) &&
                      (m.num_faces() == m2.num_faces());
    std::printf("  %-22s %s  (%s)\n", "program JSON round-trip", same ? "OK" : "FAIL", js.c_str());
    ok &= same;

    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
