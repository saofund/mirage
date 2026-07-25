"""Case 26's parts kit: the forecourt modelled object by object.

`materials` is the palette (flat / tiled / decal), `parts` builds each object on its own,
and `sheet` renders every part alone so it can be judged before it is placed. The case file
(`26_forecourt.py`) then does only what a scene file should: layout, camera, render.
"""
from . import materials, parts  # noqa: F401
