"""Stable import shim for map_measurement_efo.

This executes the repository script in an importable package module namespace so
`multiprocessing` process workers can import it on spawn-based platforms.
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_RELATIVE_PATH = Path("skills") / "pqtl-measurement-mapper" / "scripts" / "map_measurement_efo.py"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / SCRIPT_RELATIVE_PATH

if not SCRIPT_PATH.exists():
    raise FileNotFoundError(
        "Mapper script not found at expected path:\n"
        f"  {SCRIPT_PATH}\n\n"
        "Install from a cloned repository in editable mode:\n"
        "  python -m pip install -e ."
    )

_code = compile(SCRIPT_PATH.read_text(encoding="utf-8"), str(SCRIPT_PATH), "exec")
exec(_code, globals(), globals())

