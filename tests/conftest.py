"""Test-session setup.

The suite is allowed to render locally. `capture.default_render()` refuses to hand back a
binary in a working copy marked `.render-remote-only` — a guard against "just a quick
preview" on a workstation that has a render farm behind it — but that rule is about
DELIVERABLES, not about correctness. A test render is 64x64, takes a moment, must run on any
machine including the box itself and a CI runner, and produces nothing anybody will look at.
So the tests opt out explicitly, here, once, where it can be read.
"""
import os

os.environ.setdefault("MIRAGE_ALLOW_LOCAL", "1")
