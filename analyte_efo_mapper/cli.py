"""Console entrypoint for analyte-efo-mapper.

This package currently delegates to the repository script:
skills/pqtl-measurement-mapper/scripts/map_measurement_efo.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPT_RELATIVE_PATH = Path("skills") / "pqtl-measurement-mapper" / "scripts" / "map_measurement_efo.py"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_mapper_module(script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("analyte_efo_mapper._map_measurement_efo", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load mapper module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_mapper_main():
    root = _repo_root()
    script_path = root / SCRIPT_RELATIVE_PATH
    if not script_path.exists():
        raise FileNotFoundError(
            "Mapper script not found at expected path:\n"
            f"  {script_path}\n\n"
            "Install from a cloned repository in editable mode:\n"
            "  python -m pip install -e ."
        )
    module = _load_mapper_module(script_path)
    mapper_main = getattr(module, "main", None)
    if mapper_main is None:
        raise AttributeError(f"Expected `main()` in {script_path}")
    return mapper_main


def main() -> int:
    try:
        mapper_main = _load_mapper_main()
        return int(mapper_main())
    except Exception as exc:
        print(f"[ERROR] analyte-efo-mapper CLI failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
