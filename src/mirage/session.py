"""Session — the unified command vocabulary that the Python API and the MCP
server both speak.

One object owns a USD ``Scene`` and (when MuJoCo is available) a live simulation
built from it. Every *mutating* call is recorded to a replayable command log, so
a whole authoring+simulation session reproduces deterministically. Reads return
structured data (plain dicts), never prose. Editing the scene invalidates the
simulation, which is rebuilt lazily on the next ``step``/``render``.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Optional

from .scene import (
    Scene, Entity, Camera, Light, Transform, Geometry, Material, PhysicsBody,
    EntityNotFound,
)
from .backends import NullPhysics


def _rgba(color, default=(0.8, 0.8, 0.8, 1.0)) -> list:
    if color is None:
        return list(default)
    c = [float(v) for v in color]
    return c if len(c) == 4 else c + [1.0]


class Session:
    """Stateful control surface over a scene + simulation."""

    def __init__(self, name: str = "default", use_physics: bool = True):
        self.scene = Scene(name=name)
        self._log: list[dict] = []
        self._use_physics = use_physics
        self._sim = None
        self._dirty = True
        self.time = 0.0

    # -- internals ----------------------------------------------------------- #
    def _record(self, op: str, **args) -> None:
        self._log.append({"op": op, "args": args})

    def _invalidate(self) -> None:
        self._sim = None
        self._dirty = True

    def _mujoco(self) -> bool:
        if not self._use_physics:
            return False
        try:
            import mujoco  # noqa: F401
        except Exception:
            return False
        return True

    def _ensure_sim(self):
        if not self._mujoco():
            return None
        if self._sim is None or self._dirty:
            from .mujoco_backend import MujocoSim
            self._sim = MujocoSim.from_scene(self.scene)
            self._dirty = False
            self.time = 0.0
        return self._sim

    def _entity(self, name: str) -> dict:
        return asdict(self.scene.get_entity(name))

    # -- scene lifecycle ----------------------------------------------------- #
    def reset(self, name: str = "default") -> dict:
        self.scene = Scene(name=name)
        self._invalidate()
        self.time = 0.0
        self._record("reset", name=name)
        return {"ok": True, "scene": name}

    # -- authoring ----------------------------------------------------------- #
    def _add_body(self, name, geom, position, color, mass, dynamic) -> dict:
        self.scene.add(Entity(
            name=name,
            transform=Transform(position=list(position or [0.0, 0.0, 0.0])),
            geometry=geom,
            material=Material(base_color=_rgba(color)),
            physics=PhysicsBody(kind="dynamic" if dynamic else "static", mass=float(mass)),
        ))
        self._invalidate()
        return self._entity(name)

    def add_box(self, name, position=None, size=None, color=None, mass=1.0, dynamic=True) -> dict:
        size = [float(v) for v in (size or [1.0, 1.0, 1.0])]
        self._record("add_box", name=name, position=position, size=size, color=color, mass=mass, dynamic=dynamic)
        return self._add_body(name, Geometry("box", {"size": size}), position, color, mass, dynamic)

    def add_sphere(self, name, position=None, radius=0.5, color=None, mass=1.0, dynamic=True) -> dict:
        self._record("add_sphere", name=name, position=position, radius=radius, color=color, mass=mass, dynamic=dynamic)
        return self._add_body(name, Geometry("sphere", {"radius": float(radius)}), position, color, mass, dynamic)

    def add_cylinder(self, name, position=None, radius=0.5, height=1.0, color=None, mass=1.0, dynamic=True) -> dict:
        self._record("add_cylinder", name=name, position=position, radius=radius, height=height, color=color, mass=mass, dynamic=dynamic)
        return self._add_body(name, Geometry("cylinder", {"radius": float(radius), "height": float(height)}), position, color, mass, dynamic)

    def add_mesh(self, name, path, position=None, scale=None, color=None, mass=1.0, dynamic=True) -> dict:
        import os
        abspath = os.path.abspath(path).replace("\\", "/")
        scale = [float(v) for v in (scale or [1.0, 1.0, 1.0])]
        self._record("add_mesh", name=name, path=path, position=position, scale=scale, color=color, mass=mass, dynamic=dynamic)
        return self._add_body(name, Geometry("mesh", {"path": abspath, "scale": scale}), position, color, mass, dynamic)

    def add_part(self, name, part=None, features=None, position=None, color=None, scale=None, dynamic=False) -> dict:
        """Add a parametric `modeling.Part` as a meshed scene entity (CSG -> OBJ -> mesh)."""
        import os
        import tempfile
        from .modeling import Part
        if part is None:
            part = Part.from_features(features or [], name=name)
        cache = os.path.join(tempfile.gettempdir(), "mirage_parts")
        os.makedirs(cache, exist_ok=True)
        path = os.path.join(cache, f"{name}.obj").replace("\\", "/")
        mesh = part.build()
        mesh.export(path)
        lo, hi = mesh.bounds
        sc = [float(v) for v in (scale or [1.0, 1.0, 1.0])]
        self._record("add_part", name=name, features=part.features, position=position, color=color, scale=sc, dynamic=dynamic)
        return self._add_body(name, Geometry("mesh", {
            "path": path, "scale": sc,
            "bbox_lo": [float(v) for v in lo], "bbox_hi": [float(v) for v in hi],
        }), position, color, 1.0, dynamic)

    def add_plane(self, name="ground", position=None, size=None, color=None) -> dict:
        size = [float(v) for v in (size or [10.0, 10.0])]
        self._record("add_plane", name=name, position=position, size=size, color=color)
        self.scene.add(Entity(
            name=name, transform=Transform(position=list(position or [0.0, 0.0, 0.0])),
            geometry=Geometry("plane", {"size": size}), material=Material(base_color=_rgba(color)),
            physics=PhysicsBody(kind="static"),
        ))
        self._invalidate()
        return self._entity(name)

    def add_camera(self, name, position=None, width=640, height=480, modalities=None) -> dict:
        self._record("add_camera", name=name, position=position, width=width, height=height, modalities=modalities)
        self.scene.add(Camera(
            name=name, transform=Transform(position=list(position or [0.0, -5.0, 2.0])),
            width=int(width), height=int(height), modalities=list(modalities or ["rgb"]),
        ))
        return asdict(self.scene.get_camera(name))

    def add_light(self, name, position=None, kind="point", color=None, intensity=1.0) -> dict:
        self._record("add_light", name=name, position=position, kind=kind, color=color, intensity=intensity)
        self.scene.add(Light(
            name=name, kind=kind, transform=Transform(position=list(position or [0.0, 0.0, 5.0])),
            color=list(color or [1.0, 1.0, 1.0]), intensity=float(intensity),
        ))
        return asdict(self.scene.get_light(name))

    # -- edits --------------------------------------------------------------- #
    def move(self, name, position) -> dict:
        self._record("move", name=name, position=position)
        self.scene.move(name, position)
        self._invalidate()
        return self._entity(name)

    def set_transform(self, name, position=None, rotation=None, scale=None) -> dict:
        self._record("set_transform", name=name, position=position, rotation=rotation, scale=scale)
        self.scene.set_transform(name, position=position, rotation=rotation, scale=scale)
        self._invalidate()
        return self._entity(name)

    def set_material(self, name, color=None, metallic=None, roughness=None) -> dict:
        self._record("set_material", name=name, color=color, metallic=metallic, roughness=roughness)
        self.scene.set_material(
            name, base_color=_rgba(color) if color is not None else None,
            metallic=metallic, roughness=roughness,
        )
        self._invalidate()
        return self._entity(name)

    def set_velocity(self, name, linear=None, angular=None) -> dict:
        self._record("set_velocity", name=name, linear=linear, angular=angular)
        self.scene.set_velocity(name, linear=linear, angular=angular)
        self._invalidate()
        return self._entity(name)

    def remove(self, name) -> dict:
        self._record("remove", name=name)
        self.scene.remove(name)
        self._invalidate()
        return {"removed": name}

    def rename(self, old, new) -> dict:
        self._record("rename", old=old, new=new)
        self.scene.rename(old, new)
        self._invalidate()
        return {"renamed": [old, new]}

    # -- introspection (read-only; not logged) ------------------------------- #
    def get(self, name) -> dict:
        for kind, getter in (("entity", self.scene.get_entity),
                             ("camera", self.scene.get_camera),
                             ("light", self.scene.get_light)):
            try:
                return {"kind": kind, **asdict(getter(name))}
            except EntityNotFound:
                continue
        raise EntityNotFound(f"no object named '{name}'")

    def list(self) -> dict:
        return {
            "entities": self.scene.entity_names(),
            "cameras": self.scene.camera_names(),
            "lights": self.scene.light_names(),
        }

    def get_scene(self) -> dict:
        return self.scene.to_dict()

    def diff(self, other) -> dict:
        if isinstance(other, str):
            other = json.loads(other)
        return self.scene.diff(Scene.from_dict(other))

    # -- reproduce ----------------------------------------------------------- #
    def set_scene(self, data) -> dict:
        if isinstance(data, str):
            data = json.loads(data)
        self.scene = Scene.from_dict(data)
        self._invalidate()
        self._record("set_scene", data=data)
        return self.list()

    def save(self, path) -> dict:
        self.scene.export(path)
        return {"saved": path}

    def load(self, path) -> dict:
        self.scene = Scene.load(path)
        self._invalidate()
        self._record("load", path=path)
        return self.list()

    # -- simulate / render --------------------------------------------------- #
    def step(self, dt: float = 1.0 / 60.0, steps: int = 1) -> dict:
        sim = self._ensure_sim()
        if sim is not None:
            for _ in range(int(steps)):
                sim.step_for(dt)
                self.time += dt
            sim.sync_to_scene(self.scene)
            res = {"time": round(self.time, 4), "ncontact": sim.ncontact, "engine": "mujoco"}
        else:
            phys = NullPhysics()
            for _ in range(int(steps)):
                phys.step(self.scene, dt)
                self.time += dt
            res = {"time": round(self.time, 4), "engine": "null"}
        self._record("step", dt=dt, steps=steps)
        return res

    def render(self, width=640, height=480, modalities=("rgb",),
               lookat=None, distance=None, azimuth=None, elevation=None) -> dict:
        """Render the current state. Returns ``{modality: ndarray}`` under ``data``."""
        sim = self._ensure_sim()
        if sim is None:
            return {"data": {}, "summary": f"[null] {len(self.scene.entity_names())} entities "
                    "(install mirage[mujoco] to render pixels)"}
        imgs = sim.render(width=width, height=height, modalities=tuple(modalities),
                          lookat=lookat, distance=distance, azimuth=azimuth, elevation=elevation)
        return {"data": imgs, "width": width, "height": height, "modalities": list(modalities)}

    # -- spatial relations (the AI-native scene layer) ----------------------- #
    def relations(self, view: Optional[dict] = None) -> dict:
        """Spatial scene graph: what is on / in / next-to / left-of / aligned-with what."""
        from .relations import extract_relations
        return extract_relations(self.scene, view=view)

    def set_of_mark(self, view: Optional[dict] = None, width: int = 720, height: int = 540, quality: str = "basic"):
        """Grounding render with object-id overlays. Returns ``(image, boxes)``."""
        from .grounding import set_of_mark
        return set_of_mark(self.scene, view=view, width=width, height=height, quality=quality)

    def render_studio(self, view: Optional[dict] = None, width: int = 960, height: int = 720):
        """A clean studio-quality render: sky, soft shadows, reflective floor, glossy materials."""
        from .mujoco_backend import MujocoSim
        view = view or {"lookat": [0, 0, 0.2], "distance": 3.0, "azimuth": 120, "elevation": -20}
        return MujocoSim.from_scene(self.scene, quality="studio").render(width=width, height=height, **view)["rgb"]

    def place_on(self, a: str, b: str) -> dict:
        from .relations import place_on
        self._record("place_on", a=a, b=b)
        place_on(self.scene, a, b)
        self._invalidate()
        return self._entity(a)

    def place_beside(self, a: str, b: str, side: str = "left", gap: float = 0.04) -> dict:
        from .relations import place_beside
        self._record("place_beside", a=a, b=b, side=side, gap=gap)
        place_beside(self.scene, a, b, side=side, gap=gap)
        self._invalidate()
        return self._entity(a)

    def place_inside(self, a: str, b: str) -> dict:
        from .relations import place_inside
        self._record("place_inside", a=a, b=b)
        place_inside(self.scene, a, b)
        self._invalidate()
        return self._entity(a)

    def align_tops(self, names) -> dict:
        from .relations import align_tops
        self._record("align_tops", names=list(names))
        align_tops(self.scene, names)
        self._invalidate()
        return {"aligned": list(names)}

    def stack(self, names, base: str) -> dict:
        from .relations import stack
        self._record("stack", names=list(names), base=base)
        stack(self.scene, names, base)
        self._invalidate()
        return {"stacked": list(names), "base": base}

    # -- command log --------------------------------------------------------- #
    def get_log(self) -> list:
        return list(self._log)

    def replay(self, log: Optional[list] = None) -> "Session":
        """Re-apply a command log to a fresh Session (deterministic)."""
        log = self._log if log is None else log
        s = Session(use_physics=self._use_physics)
        for entry in log:
            getattr(s, entry["op"])(**entry.get("args", {}))
        return s
