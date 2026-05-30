"""Backend interfaces and reference *null* implementations.

Mirage separates the *scene* (a USD stage) from *backends* (behavior). A backend
consumes a ``Scene`` and does work: rendering it through a camera, or advancing
its physics. The null backends below have zero heavy dependencies so the full
agent loop runs anywhere; real backends (MuJoCo/Newton physics, a Hydra/Cycles
renderer) implement the same tiny interface and read/write the same stage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .scene import Scene, Camera


@dataclass
class RenderResult:
    camera: str
    width: int
    height: int
    modalities: list
    data: dict = field(default_factory=dict)  # modality -> pixels (filled by real backends)
    summary: str = ""


class RenderBackend(ABC):
    name = "abstract"

    @abstractmethod
    def render(self, scene: Scene, camera: Camera) -> RenderResult:
        ...


class PhysicsBackend(ABC):
    name = "abstract"

    @abstractmethod
    def step(self, scene: Scene, dt: float) -> None:
        ...


class NullRenderer(RenderBackend):
    """Produces no pixels — only a structured description of what *would* render.
    Lets the agent loop, MCP tools, and tests run before a real renderer (Hydra /
    Cycles, MuJoCo's renderer) is plugged in behind the same interface."""

    name = "null"

    def render(self, scene: Scene, camera: Camera) -> RenderResult:
        return RenderResult(
            camera=camera.name,
            width=camera.width,
            height=camera.height,
            modalities=list(camera.modalities),
            summary=(
                f"[null] would render {len(scene.entity_names())} entities and "
                f"{len(scene.light_names())} lights through camera '{camera.name}' "
                f"at {camera.width}x{camera.height}"
            ),
        )


class NullPhysics(PhysicsBackend):
    """Gravity-only, collision-free explicit integrator operating on the USD
    stage. Deterministic and dependency-free — enough to validate the step/render
    loop until a real engine (MuJoCo) lands behind the same ``step`` interface."""

    name = "null"

    def step(self, scene: Scene, dt: float) -> None:
        gx, gy, gz = scene.gravity
        for name in scene.entity_names():
            if scene.physics_kind(name) != "dynamic":
                continue
            vx, vy, vz = scene.get_linear_velocity(name)
            vx, vy, vz = vx + gx * dt, vy + gy * dt, vz + gz * dt
            scene.set_linear_velocity(name, [vx, vy, vz])
            px, py, pz = scene.get_position(name)
            scene.set_position(name, [px + vx * dt, py + vy * dt, pz + vz * dt])
