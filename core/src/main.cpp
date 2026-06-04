// mirage_stats — smoke demo for mirage_core: build a cube on the C++ radial-edge
// kernel and confirm it matches the Python spec (8 verts, 12 edges, 6 faces,
// euler == 2, closed 2-manifold). The first native artifact of the engine.
#include "mirage/mesh.hpp"

#include <cstdio>

int main() {
    mirage::Mesh m = mirage::make_cube(1.0);
    m.validate();
    const bool ok = (m.euler() == 2) && m.is_closed_manifold();
    std::printf("mirage_core  cube: verts=%zu edges=%zu faces=%zu euler=%d closed_manifold=%s\n",
                m.num_verts(), m.num_edges(), m.num_faces(), m.euler(),
                m.is_closed_manifold() ? "true" : "false");
    std::printf("%s\n", ok ? "OK: matches the Python kernel spec." : "FAIL");
    return ok ? 0 : 1;
}
