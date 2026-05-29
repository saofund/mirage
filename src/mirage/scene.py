"""Mirage scene model — the single, serializable source of truth.

The scene is plain data: it serializes losslessly to/from JSON (and, later,
USD) so a coding agent can read it, diff it, edit it, and reproduce it
deterministically. Backends (renderer, physics) consume a ``Scene``; they never
own state that isn't expressible here. Objects are addressed by stable names,
not opaque handles, which keeps agent interactions explicit and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


def _vec3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> list:
    return [float(x), float(y), float(z)]


@dataclass
class Transform:
    position: list = field(default_factory=_vec3)
    rotation: list = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])  # quaternion (w, x, y, z)
    scale: list = field(default_factory=lambda: [1.0, 1.0, 1.0])


@dataclass
class Geometry:
    kind: str = "box"  # box | sphere | cylinder | plane | mesh
    params: dict = field(default_factory=dict)  # e.g. {"size": [1, 1, 1]} or {"path": "robot.glb"}


@dataclass
class Material:
    base_color: list = field(default_factory=lambda: [0.8, 0.8, 0.8, 1.0])  # rgba
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: list = field(default_factory=_vec3)


@dataclass
class PhysicsBody:
    kind: str = "dynamic"  # dynamic | static | kinematic
    mass: float = 1.0
    linear_velocity: list = field(default_factory=_vec3)
    angular_velocity: list = field(default_factory=_vec3)


@dataclass
class Entity:
    name: str
    transform: Transform = field(default_factory=Transform)
    geometry: Optional[Geometry] = None
    material: Optional[Material] = None
    physics: Optional[PhysicsBody] = None
    tags: list = field(default_factory=list)


@dataclass
class Camera:
    name: str
    transform: Transform = field(default_factory=Transform)
    fov_deg: float = 60.0
    width: int = 640
    height: int = 480
    modalities: list = field(default_factory=lambda: ["rgb"])  # rgb | depth | segmentation | normals


@dataclass
class Light:
    name: str
    kind: str = "point"  # point | sun | area
    transform: Transform = field(default_factory=Transform)
    color: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    intensity: float = 1.0


@dataclass
class Scene:
    name: str = "untitled"
    gravity: list = field(default_factory=lambda: [0.0, 0.0, -9.81])
    entities: dict = field(default_factory=dict)  # name -> Entity
    cameras: dict = field(default_factory=dict)   # name -> Camera
    lights: dict = field(default_factory=dict)    # name -> Light

    # -- mutation helpers (convenient for agents & tests) --
    def add(self, obj) -> "Scene":
        if isinstance(obj, Entity):
            self.entities[obj.name] = obj
        elif isinstance(obj, Camera):
            self.cameras[obj.name] = obj
        elif isinstance(obj, Light):
            self.lights[obj.name] = obj
        else:
            raise TypeError(f"cannot add {type(obj).__name__} to scene")
        return self

    # -- serialization --
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        scene = cls(
            name=d.get("name", "untitled"),
            gravity=list(d.get("gravity", [0.0, 0.0, -9.81])),
        )
        for name, e in d.get("entities", {}).items():
            scene.entities[name] = _entity_from_dict(e)
        for name, c in d.get("cameras", {}).items():
            scene.cameras[name] = _camera_from_dict(c)
        for name, light in d.get("lights", {}).items():
            scene.lights[name] = _light_from_dict(light)
        return scene

    @classmethod
    def from_json(cls, s: str) -> "Scene":
        return cls.from_dict(json.loads(s))


def _transform_from_dict(d: Optional[dict]) -> Transform:
    d = d or {}
    return Transform(
        position=list(d.get("position", [0.0, 0.0, 0.0])),
        rotation=list(d.get("rotation", [1.0, 0.0, 0.0, 0.0])),
        scale=list(d.get("scale", [1.0, 1.0, 1.0])),
    )


def _entity_from_dict(d: dict) -> Entity:
    geom = d.get("geometry")
    mat = d.get("material")
    phys = d.get("physics")
    return Entity(
        name=d["name"],
        transform=_transform_from_dict(d.get("transform")),
        geometry=Geometry(**geom) if geom else None,
        material=Material(**mat) if mat else None,
        physics=PhysicsBody(**phys) if phys else None,
        tags=list(d.get("tags", [])),
    )


def _camera_from_dict(d: dict) -> Camera:
    return Camera(
        name=d["name"],
        transform=_transform_from_dict(d.get("transform")),
        fov_deg=d.get("fov_deg", 60.0),
        width=d.get("width", 640),
        height=d.get("height", 480),
        modalities=list(d.get("modalities", ["rgb"])),
    )


def _light_from_dict(d: dict) -> Light:
    return Light(
        name=d["name"],
        kind=d.get("kind", "point"),
        transform=_transform_from_dict(d.get("transform")),
        color=list(d.get("color", [1.0, 1.0, 1.0])),
        intensity=d.get("intensity", 1.0),
    )
