// mirage_stats — native smoke check for mirage_core: primitives + Catmull-Clark
// subdivision, each validated (euler == 2, closed 2-manifold). Counts are
// differential-checked against the Python kernel spec (see the build script).
#include "mirage/mesh.hpp"

#include <cstdio>

using namespace mirage;

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

    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
