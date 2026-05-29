"""The engine ties a scene to its backends and drives the sim/render loop."""
from __future__ import annotations

from dataclasses import dataclass, field

from .scene import Scene
from .backends import (
    RenderBackend,
    PhysicsBackend,
    NullRenderer,
    NullPhysics,
    RenderResult,
)


@dataclass
class Engine:
    scene: Scene
    renderer: RenderBackend = field(default_factory=NullRenderer)
    physics: PhysicsBackend = field(default_factory=NullPhysics)
    time: float = 0.0

    def step(self, dt: float = 1.0 / 60.0, steps: int = 1) -> float:
        """Advance physics by ``steps`` increments of ``dt`` seconds."""
        for _ in range(steps):
            self.physics.step(self.scene, dt)
            self.time += dt
        return self.time

    def render(self, camera_name: str) -> RenderResult:
        if camera_name not in self.scene.cameras:
            raise KeyError(
                f"no camera named '{camera_name}' (have: {list(self.scene.cameras)})"
            )
        return self.renderer.render(self.scene, self.scene.cameras[camera_name])
