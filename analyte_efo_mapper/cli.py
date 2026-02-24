"""Console entrypoint for analyte-efo-mapper."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from . import _map_measurement_efo as mapper_module

        mapper_main = getattr(mapper_module, "main", None)
        if mapper_main is None:
            raise AttributeError("Expected `main()` in analyte_efo_mapper._map_measurement_efo")
        return int(mapper_main())
    except Exception as exc:
        print(f"[ERROR] analyte-efo-mapper CLI failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
