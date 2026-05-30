"""MuJoCo backend — real physics (collision, joints, articulation, actuators)
and multimodal rendering (RGB / depth / segmentation), behind Mirage's tiny
backend interfaces.

``MujocoSim`` is the central wrapper. It can be built from raw **MJCF**
(``from_mjcf`` — the full-featured path: joints, contacts, actuators) or from a
Mirage **USD Scene** of primitives (``from_scene`` — authoring in Mirage, solving
in MuJoCo). ``MujocoPhysics`` / ``MujocoRenderer`` adapt it to the generic
``PhysicsBackend`` / ``RenderBackend`` so the standard ``Engine`` can drive a
primitive USD scene through MuJoCo.
"""
from __future__ import annotations

from typing import Optional, Sequence

try:
    import numpy as np
    import mujoco
    _HAS_MUJOCO = True
except ImportError:  # pragma: no cover
    _HAS_MUJOCO = False

from .scene import Scene
from .backends import RenderBackend, PhysicsBackend, RenderResult


def _require_mujoco() -> None:
    if not _HAS_MUJOCO:
        raise RuntimeError(
            "MuJoCo is required for this backend: pip install \"mirage[mujoco]\" "
            "(or `uv pip install mujoco numpy`)."
        )


# --------------------------------------------------------------------------- #
# Scene (USD primitives) -> MJCF
# --------------------------------------------------------------------------- #
def _fmt(seq) -> str:
    return " ".join(repr(float(v)) for v in seq)


def _geom_size_attr(kind: str, params: dict) -> str:
    """MuJoCo geom `size`. Mirage box `size` is full extents (MuJoCo wants half)."""
    if kind == "box":
        sx, sy, sz = (params.get("size") or [1.0, 1.0, 1.0])
        return f'size="{sx/2} {sy/2} {sz/2}"'
    if kind == "sphere":
        r = params.get("radius", (params.get("size", [1.0])[0] / 2))
        return f'size="{r}"'
    if kind == "cylinder":
        r = params.get("radius", 0.5)
        h = params.get("height", 1.0)
        return f'size="{r} {h/2}"'
    if kind == "plane":
        hx, hy = (params.get("size") or [10.0, 10.0])[:2]
        return f'size="{hx} {hy} 0.1"'
    return 'size="0.5 0.5 0.5"'


def _studio_material(en, mat) -> str:
    """Map a Mirage PBR material (base_color/metallic/roughness) to a MuJoCo
    material (rgba/specular/shininess/reflectance) for the studio preset."""
    base = _fmt(mat.base_color) if mat else "0.7 0.7 0.75 1"
    metallic = float(mat.metallic) if mat else 0.0
    rough = float(mat.roughness) if mat else 0.5
    spec = round(0.3 + 0.6 * metallic, 3)
    shin = round(min(0.95, max(0.0, 1.0 - 0.7 * rough)), 3)
    refl = round(0.45 * metallic, 3)
    return f'<material name="{en}_mat" rgba="{base}" specular="{spec}" shininess="{shin}" reflectance="{refl}"/>'


def scene_to_mjcf(scene: Scene, quality: str = "basic") -> str:
    """Translate a Mirage scene (primitives + OBJ/STL meshes) into MJCF.

    ``quality='studio'`` adds a sky gradient, soft shadows, a reflective floor,
    glossy materials derived from metallic/roughness, and MSAA — a much nicer
    render preset (visual only; physics is unchanged)."""
    studio = quality == "studio"
    mesh_assets, mat_assets, items = [], [], []
    for en in scene.entity_names():
        E = scene.get_entity(en)
        if E.geometry is None:
            continue
        kind = E.geometry.kind
        params = E.geometry.params or {}
        if kind == "plane":
            appearance = 'material="floor"'
        elif studio:
            mat_assets.append(_studio_material(en, E.material))
            appearance = f'material="{en}_mat"'
        else:
            appearance = f'rgba="{_fmt(E.material.base_color) if E.material else "0.7 0.7 0.75 1"}"'
        if kind == "mesh":
            scale = params.get("scale", [1.0, 1.0, 1.0])
            mesh_assets.append(f'<mesh name="{en}_mesh" file="{params["path"]}" scale="{_fmt(scale)}"/>')
            geom_def = f'type="mesh" mesh="{en}_mesh"'
        else:
            geom_def = f'type="{kind}" {_geom_size_attr(kind, params)}'
        pos, quat = E.transform.position, E.transform.rotation
        is_dynamic = E.physics is not None and E.physics.kind == "dynamic" and kind != "plane"
        if is_dynamic:
            mass = f' mass="{E.physics.mass}"' if E.physics.mass else ""
            items.append(f'<body name="{en}" pos="{_fmt(pos)}" quat="{_fmt(quat)}">'
                         f'<freejoint name="{en}"/><geom name="{en}_g" {geom_def} {appearance}{mass}/></body>')
        else:
            items.append(f'<geom name="{en}" {geom_def} pos="{_fmt(pos)}" quat="{_fmt(quat)}" {appearance}/>')

    if studio:
        lights = [
            '<light name="key" directional="true" pos="3 -3 5" dir="-0.5 0.5 -1" diffuse="0.9 0.88 0.85" specular="0.5 0.5 0.5" castshadow="true"/>',
            '<light name="fill" directional="true" pos="-4 -1 3" dir="0.7 0.2 -1" diffuse="0.25 0.28 0.34" castshadow="false"/>',
        ]
        visual = ('<visual><global offwidth="1920" offheight="1080"/>'
                  '<quality shadowsize="4096" numslices="28" offsamples="8"/>'
                  '<headlight ambient="0.32 0.32 0.35" diffuse="0.3 0.3 0.3" specular="0.3 0.3 0.3"/>'
                  '<map shadowclip="6"/></visual>')
        assets = ('<texture name="sky" type="skybox" builtin="gradient" rgb1="0.55 0.72 0.95" rgb2="0.08 0.11 0.2" width="256" height="512"/>'
                  '<texture name="grid" type="2d" builtin="checker" rgb1="0.32 0.34 0.38" rgb2="0.38 0.40 0.45" width="512" height="512" mark="edge" markrgb="0.5 0.5 0.55"/>'
                  '<material name="floor" texture="grid" texrepeat="10 10" reflectance="0.3" specular="0.5" shininess="0.6"/>')
    else:
        scene_lights = []
        for ln in scene.light_names():
            L = scene.get_light(ln)
            p = L.transform.position
            d = ' directional="true"' if L.kind == "sun" else ""
            scene_lights.append(f'<light name="{ln}"{d} pos="{_fmt(p)}" dir="0 0 -1" diffuse="{_fmt(L.color)}"/>')
        lights = scene_lights or ['<light name="key" directional="true" pos="0 0 4" dir="0 0 -1"/>']
        visual = '<visual><global offwidth="1920" offheight="1080"/></visual>'
        assets = ('<texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.25 0.35 0.45" width="300" height="300"/>'
                  '<material name="floor" texture="grid" texrepeat="8 8" reflectance="0.1"/>')

    return f"""<mujoco model="{scene.name}">
  <compiler angle="radian"/>
  <option gravity="{_fmt(scene.gravity)}"/>
  {visual}
  <asset>
    {assets}
    {chr(10).join('    ' + a for a in mat_assets)}
    {chr(10).join('    ' + a for a in mesh_assets)}
  </asset>
  <worldbody>
    {chr(10).join('    ' + l for l in lights)}
    {chr(10).join('    ' + it for it in items)}
  </worldbody>
</mujoco>"""


# --------------------------------------------------------------------------- #
# MujocoSim
# --------------------------------------------------------------------------- #
class MujocoSim:
    """Owns an ``mjModel`` + ``mjData``; steps physics and renders sensors."""

    def __init__(self, model):
        _require_mujoco()
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_forward(self.model, self.data)
        self._renderer = None
        self._rsize = None

    @classmethod
    def from_mjcf(cls, xml: str) -> "MujocoSim":
        _require_mujoco()
        return cls(mujoco.MjModel.from_xml_string(xml))

    @classmethod
    def from_mjcf_path(cls, path: str) -> "MujocoSim":
        _require_mujoco()
        return cls(mujoco.MjModel.from_xml_path(path))

    @classmethod
    def from_urdf(cls, src: str) -> "MujocoSim":
        """Load a robot from URDF — either an inline URDF string or a path.
        (MuJoCo's compiler parses URDF directly; meshes resolve relative to the file.)"""
        _require_mujoco()
        if "<robot" in src[:1024]:
            return cls(mujoco.MjModel.from_xml_string(src))
        return cls(mujoco.MjModel.from_xml_path(src))

    @classmethod
    def from_scene(cls, scene: Scene, quality: str = "basic") -> "MujocoSim":
        sim = cls.from_mjcf(scene_to_mjcf(scene, quality=quality))
        for en in scene.entity_names():  # honor initial velocities authored in the scene
            E = scene.get_entity(en)
            if E.physics and E.physics.kind == "dynamic":
                lv, av = list(E.physics.linear_velocity), list(E.physics.angular_velocity)
                if any(lv) or any(av):
                    try:
                        sim.joint(en).qvel[:] = [*lv, *av]
                    except (KeyError, ValueError):
                        pass
        mujoco.mj_forward(sim.model, sim.data)
        return sim

    # -- simulation ---------------------------------------------------------- #
    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> "MujocoSim":
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self

    def step(self, n: int = 1) -> "MujocoSim":
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)
        return self

    def step_for(self, seconds: float) -> "MujocoSim":
        return self.step(max(1, round(seconds / self.timestep)))

    @property
    def ncontact(self) -> int:
        return int(self.data.ncon)

    def joint(self, name: str):
        return self.data.joint(name)

    def body_pos(self, name: str):
        return np.array(self.data.body(name).xpos, dtype=float)

    def site_pos(self, name: str):
        return np.array(self.data.site(name).xpos, dtype=float)

    # -- rendering ----------------------------------------------------------- #
    def _renderer_for(self, height: int, width: int):
        # clamp to the model's offscreen framebuffer (externally-loaded models
        # like Menagerie robots often default to 640x480)
        height = min(height, int(self.model.vis.global_.offheight))
        width = min(width, int(self.model.vis.global_.offwidth))
        if self._renderer is None or self._rsize != (height, width):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height, width)
            self._rsize = (height, width)
        return self._renderer

    @staticmethod
    def _free_camera(lookat, distance, azimuth, elevation):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat if lookat is not None else [0.0, 0.0, 0.2]
        cam.distance = float(distance) if distance is not None else 3.0
        cam.azimuth = float(azimuth) if azimuth is not None else 120.0
        cam.elevation = float(elevation) if elevation is not None else -20.0
        return cam

    def render(
        self,
        width: int = 640,
        height: int = 480,
        camera=None,
        lookat: Optional[Sequence[float]] = None,
        distance: Optional[float] = None,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        modalities: Sequence[str] = ("rgb",),
    ) -> dict:
        """Render the current state; returns ``{modality: ndarray}``."""
        r = self._renderer_for(height, width)
        cam = camera if camera is not None else self._free_camera(lookat, distance, azimuth, elevation)
        out = {}
        if "rgb" in modalities:
            r.update_scene(self.data, camera=cam)
            out["rgb"] = r.render().copy()
        if "depth" in modalities:
            r.enable_depth_rendering()
            r.update_scene(self.data, camera=cam)
            out["depth"] = r.render().copy()
            r.disable_depth_rendering()
        if "segmentation" in modalities:
            r.enable_segmentation_rendering()
            r.update_scene(self.data, camera=cam)
            out["segmentation"] = r.render().copy()
            r.disable_segmentation_rendering()
        return out

    def sync_to_scene(self, scene: Scene) -> None:
        """Write dynamic body positions back to the USD scene (where names match)."""
        for name in scene.entity_names():
            try:
                b = self.data.body(name)
            except (KeyError, ValueError):
                continue
            scene.set_position(name, [float(v) for v in b.xpos])


# --------------------------------------------------------------------------- #
# Adapters to the generic Engine backends
# --------------------------------------------------------------------------- #
class MujocoPhysics(PhysicsBackend):
    """Drive a primitive USD scene's physics through MuJoCo. Builds the model
    from the scene on first ``step`` and syncs poses back each step."""

    name = "mujoco"

    def __init__(self, sim: Optional[MujocoSim] = None):
        _require_mujoco()
        self.sim = sim

    def step(self, scene: Scene, dt: float) -> None:
        if self.sim is None:
            self.sim = MujocoSim.from_scene(scene)
        self.sim.step_for(dt)
        self.sim.sync_to_scene(scene)


class MujocoRenderer(RenderBackend):
    """Render a USD scene through MuJoCo. Pass ``sim=`` to share an already-built
    (and possibly stepped) simulation; otherwise one is built from the scene."""

    name = "mujoco"

    def __init__(self, sim: Optional[MujocoSim] = None, **framing):
        _require_mujoco()
        self.sim = sim
        self.framing = framing  # lookat / distance / azimuth / elevation

    def render(self, scene: Scene, camera) -> RenderResult:
        if self.sim is None:
            self.sim = MujocoSim.from_scene(scene)
        imgs = self.sim.render(
            width=camera.width, height=camera.height,
            modalities=camera.modalities, **self.framing,
        )
        return RenderResult(
            camera=camera.name, width=camera.width, height=camera.height,
            modalities=list(camera.modalities), data=imgs,
            summary=f"[mujoco] rendered {list(camera.modalities)} at {camera.width}x{camera.height}",
        )
