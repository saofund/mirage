"""Mirage — an AI-native, USD-centric control layer for 3D + physics.

Designed to be driven by coding agents (e.g. Claude Code via MCP): the scene is a
serializable USD stage (the source of truth); the engine is a tiny step/render
loop; backends are swappable behind minimal interfaces. See ``docs/design.md``.
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
    MirageError,
    EntityNotFound,
    DuplicateName,
    InvalidName,
)
from .backends import (
    RenderBackend,
    PhysicsBackend,
    NullRenderer,
    NullPhysics,
    RenderResult,
)
from .engine import Engine
from .session import Session

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
    "MirageError",
    "EntityNotFound",
    "DuplicateName",
    "InvalidName",
    "RenderBackend",
    "PhysicsBackend",
    "NullRenderer",
    "NullPhysics",
    "RenderResult",
    "Engine",
    "Session",
    "__version__",
]
