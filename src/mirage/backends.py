"""Backend interfaces and reference *null* implementations.

Mirage separates the *scene* (data) from *backends* (behavior). A backend is
anything that consumes a ``Scene`` and does work: rendering it through a camera,
or advancing its physics. The null backends below have zero heavy dependencies
so the full agent loop runs anywhere; real backends (e.g. a Cycles/Embree
renderer, a MuJoCo physics step) implement the same tiny interface.
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
    """Produces no pixels — only a structured description of what *would*
    render. Lets the agent loop, MCP tools, and tests run before a real
    renderer is plugged in."""

    name = "null"

    def render(self, scene: Scene, camera: Camera) -> RenderResult:
        return RenderResult(
            camera=camera.name,
            width=camera.width,
            height=camera.height,
            modalities=list(camera.modalities),
            summary=(
                f"[null] would render {len(scene.entities)} entities and "
                f"{len(scene.lights)} lights through camera '{camera.name}' "
                f"at {camera.width}x{camera.height}"
            ),
        )


class NullPhysics(PhysicsBackend):
    """Gravity-only, collision-free explicit integrator. Deterministic and
    dependency-free — enough to validate the step/render loop. Replace with a
    real engine behind the same ``step`` interface."""

    name = "null"

    def step(self, scene: Scene, dt: float) -> None:
        gx, gy, gz = scene.gravity
        for entity in scene.entities.values():
            body = entity.physics
            if body is None or body.kind != "dynamic":
                continue
            body.linear_velocity[0] += gx * dt
            body.linear_velocity[1] += gy * dt
            body.linear_velocity[2] += gz * dt
            entity.transform.position[0] += body.linear_velocity[0] * dt
            entity.transform.position[1] += body.linear_velocity[1] * dt
            entity.transform.position[2] += body.linear_velocity[2] * dt
