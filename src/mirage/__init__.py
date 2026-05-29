"""Mirage — an AI-native 3D renderer + lightweight physics simulator.

Designed to be driven by coding agents (e.g. Claude Code via MCP): the scene is
plain, serializable data; the engine is a tiny step/render loop; backends are
swappable behind minimal interfaces.
"""
from .scene import (
    Scene,
    Entity,
    Transform,
    Geometry,
    Material,
    PhysicsBody,
    Camera,
    Light,
)
from .backends import (
    RenderBackend,
    PhysicsBackend,
    NullRenderer,
    NullPhysics,
    RenderResult,
)
from .engine import Engine

__version__ = "0.0.1"

__all__ = [
    "Scene",
    "Entity",
    "Transform",
    "Geometry",
    "Material",
    "PhysicsBody",
    "Camera",
    "Light",
    "RenderBackend",
    "PhysicsBackend",
    "NullRenderer",
    "NullPhysics",
    "RenderResult",
    "Engine",
    "__version__",
]
