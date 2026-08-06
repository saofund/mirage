"""Case 28: photograph-matched circular fuel-filler region from a 2007 Polo."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fuelcap2.sheet import main  # noqa: E402


if __name__ == "__main__":
    main()
