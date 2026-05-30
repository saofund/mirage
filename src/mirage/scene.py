"""Mirage scene model — an OpenUSD stage is the single source of truth.

The scene is plain, serializable data: it lives in a USD stage (the SoT) and
round-trips losslessly to/from JSON and ``.usda``/``.usdc`` so a coding agent can
read it, diff it, edit it, and reproduce it deterministically. Entities are
addressed by stable names (USD prim names under ``/World``), not opaque handles.

Agents and tests interact through lightweight *DTO* dataclasses (``Entity``,
``Camera``, ``Light`` and friends); the facade converts those to/from USD prims.
Mirage semantics are stored on each prim as namespaced ``mirage:*`` attributes
(lossless), alongside best-effort standard USD typing (``UsdGeom`` gprims,
``UsdLux`` lights, ``UsdGeom.Camera``) so external USD tools can consume the stage
too. Real geometry/material/physics USD schemas are filled in by later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

try:  # USD is the scene source of truth; keep `import mirage` cheap when absent.
    from pxr import Usd, UsdGeom, UsdLux, Gf, Sdf, Tf
    _HAS_USD = True
except ImportError:  # pragma: no cover - exercised only without the [usd] extra
    _HAS_USD = False


def _require_usd() -> None:
    if not _HAS_USD:
        raise RuntimeError(
            "OpenUSD is required for the scene model: pip install \"mirage[usd]\" "
            "(or `uv pip install usd-core`)."
        )


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class MirageError(Exception):
    """Base class for Mirage scene errors."""


class EntityNotFound(MirageError):
    pass


class DuplicateName(MirageError):
    pass


class InvalidName(MirageError):
    pass


# --------------------------------------------------------------------------- #
# DTOs — the agent-facing value types (what you pass in / read back as JSON)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Scene — a thin, ergonomic facade over a USD stage (the source of truth)
# --------------------------------------------------------------------------- #
_ROOT = "/World"
_KIND_ENTITY, _KIND_CAMERA, _KIND_LIGHT = "entity", "camera", "light"
_DEFAULT_GRAVITY = [0.0, 0.0, -9.81]


class Scene:
    """A 3D world backed by an in-memory USD stage.

    Construct it, mutate it through the helpers (convenient for agents & tests),
    and (de)serialize it to JSON or USD. The USD stage is authoritative; the DTOs
    returned by the getters are snapshots.
    """

    def __init__(self, name: str = "untitled", gravity: Optional[list] = None):
        _require_usd()
        self.stage = Usd.Stage.CreateInMemory()
        world = UsdGeom.Xform.Define(self.stage, _ROOT)
        self.stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self._set_str(world.GetPrim(), "mirage:name", name)
        self._set_vec3(world.GetPrim(), "mirage:gravity", list(gravity or _DEFAULT_GRAVITY))

    # -- world-level metadata ------------------------------------------------ #
    def _world(self):
        return self.stage.GetPrimAtPath(_ROOT)

    @property
    def name(self) -> str:
        return self._get_str(self._world(), "mirage:name", "untitled")

    @name.setter
    def name(self, value: str) -> None:
        self._set_str(self._world(), "mirage:name", value)

    @property
    def gravity(self) -> list:
        return self._get_vec3(self._world(), "mirage:gravity", _DEFAULT_GRAVITY)

    @gravity.setter
    def gravity(self, value: list) -> None:
        self._set_vec3(self._world(), "mirage:gravity", list(value))

    # -- low-level attribute helpers ---------------------------------------- #
    @staticmethod
    def _set_str(prim, key: str, value: str) -> None:
        prim.CreateAttribute(key, Sdf.ValueTypeNames.String).Set(str(value))

    @staticmethod
    def _get_str(prim, key: str, default: str = "") -> str:
        attr = prim.GetAttribute(key)
        return attr.Get() if attr and attr.HasAuthoredValue() else default

    @staticmethod
    def _set_vec3(prim, key: str, value: list) -> None:
        prim.CreateAttribute(key, Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(*[float(v) for v in value]))

    @staticmethod
    def _get_vec3(prim, key: str, default: list) -> list:
        attr = prim.GetAttribute(key)
        if attr and attr.HasAuthoredValue():
            v = attr.Get()
            return [v[0], v[1], v[2]]
        return list(default)

    @staticmethod
    def _set_json(prim, key: str, value) -> None:
        prim.CreateAttribute(key, Sdf.ValueTypeNames.String).Set(json.dumps(value))

    @staticmethod
    def _get_json(prim, key: str, default=None):
        attr = prim.GetAttribute(key)
        if attr and attr.HasAuthoredValue():
            return json.loads(attr.Get())
        return default

    # -- transform (stored as standard USD xformOps: translate, orient, scale) #
    def _set_xform(self, prim, t: Transform) -> None:
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in t.position]))
        w, x, y, z = [float(v) for v in t.rotation]
        xf.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
        xf.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in t.scale]))

    def _read_xform(self, prim) -> Transform:
        pos, rot, scale = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
        for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
            ot, val = op.GetOpType(), op.Get()
            if val is None:
                continue
            if ot == UsdGeom.XformOp.TypeTranslate:
                pos = [val[0], val[1], val[2]]
            elif ot == UsdGeom.XformOp.TypeOrient:
                im = val.GetImaginary()
                rot = [val.GetReal(), im[0], im[1], im[2]]
            elif ot == UsdGeom.XformOp.TypeScale:
                scale = [val[0], val[1], val[2]]
        return Transform(position=pos, rotation=rot, scale=scale)

    def _translate_op(self, prim):
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return xf.AddTranslateOp()

    # -- prim addressing ----------------------------------------------------- #
    def _path(self, name: str) -> str:
        if not Tf.IsValidIdentifier(name):
            raise InvalidName(
                f"'{name}' is not a valid name (use letters, digits, underscores; "
                "must not start with a digit)"
            )
        return f"{_ROOT}/{name}"

    def _prim(self, name: str, kind: Optional[str] = None):
        prim = self.stage.GetPrimAtPath(f"{_ROOT}/{name}")
        if not prim or not prim.IsValid():
            raise EntityNotFound(f"no object named '{name}' (have: {self._all_names()})")
        if kind is not None and self._get_str(prim, "mirage:kind") != kind:
            raise EntityNotFound(f"'{name}' is not a {kind}")
        return prim

    def _exists(self, name: str) -> bool:
        prim = self.stage.GetPrimAtPath(f"{_ROOT}/{name}")
        return bool(prim and prim.IsValid())

    def _names_of_kind(self, kind: str) -> list:
        return [
            p.GetName()
            for p in self._world().GetChildren()
            if self._get_str(p, "mirage:kind") == kind
        ]

    def _all_names(self) -> list:
        return [p.GetName() for p in self._world().GetChildren()]

    # -- add ----------------------------------------------------------------- #
    def add(self, obj) -> "Scene":
        if isinstance(obj, Entity):
            self._write_entity(obj)
        elif isinstance(obj, Camera):
            self._write_camera(obj)
        elif isinstance(obj, Light):
            self._write_light(obj)
        else:
            raise TypeError(f"cannot add {type(obj).__name__} to scene")
        return self

    def _new_prim(self, name: str, kind: str, geom: Optional[Geometry] = None):
        if self._exists(name):
            raise DuplicateName(f"an object named '{name}' already exists")
        path = self._path(name)
        if kind == _KIND_ENTITY and geom is not None and geom.kind == "box":
            prim = UsdGeom.Cube.Define(self.stage, path).GetPrim()
        elif kind == _KIND_ENTITY and geom is not None and geom.kind == "sphere":
            prim = UsdGeom.Sphere.Define(self.stage, path).GetPrim()
        elif kind == _KIND_ENTITY and geom is not None and geom.kind == "cylinder":
            prim = UsdGeom.Cylinder.Define(self.stage, path).GetPrim()
        else:
            prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        self._set_str(prim, "mirage:kind", kind)
        return prim

    def _write_entity(self, e: Entity) -> None:
        prim = self._new_prim(e.name, _KIND_ENTITY, e.geometry)
        self._set_xform(prim, e.transform)
        if e.geometry is not None:
            self._set_json(prim, "mirage:geometry", asdict(e.geometry))
        if e.material is not None:
            self._set_json(prim, "mirage:material", asdict(e.material))
        if e.physics is not None:
            self._set_str(prim, "mirage:phys:kind", e.physics.kind)
            prim.CreateAttribute("mirage:phys:mass", Sdf.ValueTypeNames.Double).Set(float(e.physics.mass))
            self._set_vec3(prim, "mirage:phys:linear_velocity", e.physics.linear_velocity)
            self._set_vec3(prim, "mirage:phys:angular_velocity", e.physics.angular_velocity)
        if e.tags:
            self._set_json(prim, "mirage:tags", list(e.tags))

    def _write_camera(self, c: Camera) -> None:
        prim = self._new_prim(c.name, _KIND_CAMERA)
        UsdGeom.Camera.Define(self.stage, prim.GetPath())  # type the prim for USD tools
        self._set_xform(prim, c.transform)
        prim.CreateAttribute("mirage:cam:fov_deg", Sdf.ValueTypeNames.Double).Set(float(c.fov_deg))
        prim.CreateAttribute("mirage:cam:width", Sdf.ValueTypeNames.Int).Set(int(c.width))
        prim.CreateAttribute("mirage:cam:height", Sdf.ValueTypeNames.Int).Set(int(c.height))
        self._set_json(prim, "mirage:cam:modalities", list(c.modalities))

    def _write_light(self, light: Light) -> None:
        prim = self._new_prim(light.name, _KIND_LIGHT)
        path = prim.GetPath()
        if light.kind == "sun":
            UsdLux.DistantLight.Define(self.stage, path)
        elif light.kind == "area":
            UsdLux.RectLight.Define(self.stage, path)
        else:
            UsdLux.SphereLight.Define(self.stage, path)
        self._set_xform(prim, light.transform)
        self._set_str(prim, "mirage:light:kind", light.kind)
        self._set_vec3(prim, "mirage:light:color", list(light.color))
        prim.CreateAttribute("mirage:light:intensity", Sdf.ValueTypeNames.Double).Set(float(light.intensity))

    # -- read ---------------------------------------------------------------- #
    def get_entity(self, name: str) -> Entity:
        prim = self._prim(name, _KIND_ENTITY)
        geom = self._get_json(prim, "mirage:geometry")
        mat = self._get_json(prim, "mirage:material")
        phys = None
        if self._get_str(prim, "mirage:phys:kind"):
            mass_attr = prim.GetAttribute("mirage:phys:mass")
            phys = PhysicsBody(
                kind=self._get_str(prim, "mirage:phys:kind"),
                mass=mass_attr.Get() if mass_attr and mass_attr.HasAuthoredValue() else 1.0,
                linear_velocity=self._get_vec3(prim, "mirage:phys:linear_velocity", _vec3()),
                angular_velocity=self._get_vec3(prim, "mirage:phys:angular_velocity", _vec3()),
            )
        return Entity(
            name=name,
            transform=self._read_xform(prim),
            geometry=Geometry(**geom) if geom else None,
            material=Material(**mat) if mat else None,
            physics=phys,
            tags=self._get_json(prim, "mirage:tags", []),
        )

    def get_camera(self, name: str) -> Camera:
        prim = self._prim(name, _KIND_CAMERA)
        fov = prim.GetAttribute("mirage:cam:fov_deg")
        w = prim.GetAttribute("mirage:cam:width")
        h = prim.GetAttribute("mirage:cam:height")
        return Camera(
            name=name,
            transform=self._read_xform(prim),
            fov_deg=fov.Get() if fov and fov.HasAuthoredValue() else 60.0,
            width=w.Get() if w and w.HasAuthoredValue() else 640,
            height=h.Get() if h and h.HasAuthoredValue() else 480,
            modalities=self._get_json(prim, "mirage:cam:modalities", ["rgb"]),
        )

    def get_light(self, name: str) -> Light:
        prim = self._prim(name, _KIND_LIGHT)
        intensity = prim.GetAttribute("mirage:light:intensity")
        return Light(
            name=name,
            kind=self._get_str(prim, "mirage:light:kind", "point"),
            transform=self._read_xform(prim),
            color=self._get_vec3(prim, "mirage:light:color", [1.0, 1.0, 1.0]),
            intensity=intensity.Get() if intensity and intensity.HasAuthoredValue() else 1.0,
        )

    def entity_names(self) -> list:
        return self._names_of_kind(_KIND_ENTITY)

    def camera_names(self) -> list:
        return self._names_of_kind(_KIND_CAMERA)

    def light_names(self) -> list:
        return self._names_of_kind(_KIND_LIGHT)

    @property
    def entities(self) -> dict:
        return {n: self.get_entity(n) for n in self.entity_names()}

    @property
    def cameras(self) -> dict:
        return {n: self.get_camera(n) for n in self.camera_names()}

    @property
    def lights(self) -> dict:
        return {n: self.get_light(n) for n in self.light_names()}

    # -- edit / remove ------------------------------------------------------- #
    def remove(self, name: str) -> "Scene":
        self._prim(name)  # raise if missing
        self.stage.RemovePrim(self._path(name))
        return self

    def rename(self, old: str, new: str) -> "Scene":
        if self._exists(new):
            raise DuplicateName(f"an object named '{new}' already exists")
        # Re-create under the new name, then drop the old prim (USD has no cheap rename).
        kind = self._get_str(self._prim(old), "mirage:kind")
        obj = {_KIND_ENTITY: self.get_entity, _KIND_CAMERA: self.get_camera, _KIND_LIGHT: self.get_light}[kind](old)
        obj.name = new
        self.add(obj)
        self.remove(old)
        return self

    def set_transform(self, name, position=None, rotation=None, scale=None) -> "Scene":
        prim = self._prim(name)
        t = self._read_xform(prim)
        if position is not None:
            t.position = [float(v) for v in position]
        if rotation is not None:
            t.rotation = [float(v) for v in rotation]
        if scale is not None:
            t.scale = [float(v) for v in scale]
        self._set_xform(prim, t)
        return self

    def move(self, name, position) -> "Scene":
        return self.set_transform(name, position=position)

    def set_material(self, name, base_color=None, metallic=None, roughness=None, emissive=None) -> "Scene":
        prim = self._prim(name, _KIND_ENTITY)
        cur = self._get_json(prim, "mirage:material") or asdict(Material())
        if base_color is not None:
            cur["base_color"] = [float(v) for v in base_color]
        if metallic is not None:
            cur["metallic"] = float(metallic)
        if roughness is not None:
            cur["roughness"] = float(roughness)
        if emissive is not None:
            cur["emissive"] = [float(v) for v in emissive]
        self._set_json(prim, "mirage:material", cur)
        return self

    def set_velocity(self, name, linear=None, angular=None) -> "Scene":
        prim = self._prim(name, _KIND_ENTITY)
        if not self._get_str(prim, "mirage:phys:kind"):
            raise MirageError(f"'{name}' has no physics body")
        if linear is not None:
            self._set_vec3(prim, "mirage:phys:linear_velocity", linear)
        if angular is not None:
            self._set_vec3(prim, "mirage:phys:angular_velocity", angular)
        return self

    # -- physics accessors (used by physics backends) ------------------------ #
    def physics_kind(self, name: str) -> Optional[str]:
        return self._get_str(self._prim(name, _KIND_ENTITY), "mirage:phys:kind") or None

    def get_position(self, name: str) -> list:
        v = self._translate_op(self._prim(name)).Get()
        return [v[0], v[1], v[2]] if v is not None else [0.0, 0.0, 0.0]

    def set_position(self, name: str, position) -> None:
        self._translate_op(self._prim(name)).Set(Gf.Vec3d(*[float(v) for v in position]))

    def get_linear_velocity(self, name: str) -> list:
        return self._get_vec3(self._prim(name, _KIND_ENTITY), "mirage:phys:linear_velocity", _vec3())

    def set_linear_velocity(self, name: str, v) -> None:
        self._set_vec3(self._prim(name, _KIND_ENTITY), "mirage:phys:linear_velocity", v)

    # -- serialization ------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "gravity": self.gravity,
            "entities": {n: asdict(self.get_entity(n)) for n in self.entity_names()},
            "cameras": {n: asdict(self.get_camera(n)) for n in self.camera_names()},
            "lights": {n: asdict(self.get_light(n)) for n in self.light_names()},
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        scene = cls(name=d.get("name", "untitled"), gravity=list(d.get("gravity", _DEFAULT_GRAVITY)))
        for e in d.get("entities", {}).values():
            scene.add(_entity_from_dict(e))
        for c in d.get("cameras", {}).values():
            scene.add(_camera_from_dict(c))
        for light in d.get("lights", {}).values():
            scene.add(_light_from_dict(light))
        return scene

    @classmethod
    def from_json(cls, s: str) -> "Scene":
        return cls.from_dict(json.loads(s))

    def to_usda(self) -> str:
        return self.stage.GetRootLayer().ExportToString()

    def export(self, path: str) -> None:
        """Write the stage to ``path`` (``.usda`` text or ``.usdc`` binary)."""
        self.stage.GetRootLayer().Export(path)

    @classmethod
    def load(cls, path: str) -> "Scene":
        """Load a Mirage-authored USD file back into a Scene."""
        _require_usd()
        scene = cls.__new__(cls)
        scene.stage = Usd.Stage.Open(path)
        if scene._world() is None or not scene._world().IsValid():
            raise MirageError(f"'{path}' has no /World prim (not a Mirage scene?)")
        return scene

    # -- diff ---------------------------------------------------------------- #
    def diff(self, other: "Scene") -> dict:
        """Structured difference ``self`` -> ``other``; empty dict means equal."""
        a, b = self.to_dict(), other.to_dict()
        out: dict = {}
        if a["name"] != b["name"]:
            out["name"] = [a["name"], b["name"]]
        if a["gravity"] != b["gravity"]:
            out["gravity"] = [a["gravity"], b["gravity"]]
        for sec in ("entities", "cameras", "lights"):
            da, db, secdiff = a[sec], b[sec], {}
            for k in sorted(set(da) | set(db)):
                if k not in da:
                    secdiff[k] = {"added": db[k]}
                elif k not in db:
                    secdiff[k] = {"removed": da[k]}
                elif da[k] != db[k]:
                    secdiff[k] = {"from": da[k], "to": db[k]}
            if secdiff:
                out[sec] = secdiff
        return out


# --------------------------------------------------------------------------- #
# dict -> DTO helpers (used by Scene.from_dict)
# --------------------------------------------------------------------------- #
def _transform_from_dict(d: Optional[dict]) -> Transform:
    d = d or {}
    return Transform(
        position=list(d.get("position", [0.0, 0.0, 0.0])),
        rotation=list(d.get("rotation", [1.0, 0.0, 0.0, 0.0])),
        scale=list(d.get("scale", [1.0, 1.0, 1.0])),
    )


def _entity_from_dict(d: dict) -> Entity:
    geom, mat, phys = d.get("geometry"), d.get("material"), d.get("physics")
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
