// mirage::Program — the op-log (the model is a replayable list of operators).
//
// This is the seed of the single source of truth that a human (GUI) and an AI
// will both edit. Selection is currently simplified to "the active face" (the
// top face after a primitive/subdivide, or the face an inset/extrude just
// created), pending the full selection-as-query engine. build() replays the
// ops into a kernel Mesh — the same geometry-as-program idea as the Python
// meshlang, now native.
#pragma once

#include <string>
#include <vector>

#include "mirage/mesh.hpp"

namespace mirage {

enum class OpKind { Cube, Cylinder, Inset, Extrude, Subdivide };

struct Op {
    OpKind kind;
    double a = 0, b = 0, c = 0;  // params: cube{size}; cylinder{sides,r,h}; inset{t}; extrude{d}; subdivide{levels}
    bool has_target = false;     // inset/extrude: act on the face nearest `target`
    std::array<double, 3> target{0, 0, 0};
};

class Program {
public:
    void cube(double size = 1.0);
    void cylinder(int sides = 24, double radius = 0.5, double height = 1.0);
    void inset(double thickness = 0.3);
    void extrude(double distance = 0.5);
    void inset_at(const std::array<double, 3>& p, double thickness = 0.3);     // picked-face inset
    void extrude_at(const std::array<double, 3>& p, double distance = 0.5);    // picked-face extrude
    void subdivide(int levels = 1);
    void undo();
    void clear();

    bool empty() const { return ops_.empty(); }
    const std::vector<Op>& ops() const { return ops_; }

    // Replay the op-log into a fresh mesh. A primitive op restarts the mesh;
    // inset/extrude act on the active face and update it to the new one; the
    // active face becomes the top face after a primitive or subdivide.
    Mesh build() const;

    static std::string label(const Op& op);

private:
    std::vector<Op> ops_;
};

}  // namespace mirage
