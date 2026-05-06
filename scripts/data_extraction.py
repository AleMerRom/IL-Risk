"""Module 1 entrypoint.

Examples:
    PYTHONPATH=src python scripts/data_extraction.py extract all
    PYTHONPATH=src python scripts/data_extraction.py validate
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SRC_PATH = Path(__file__).resolve().parents[1] / "src" / "module1" / "data_extraction.py"
_SPEC = importlib.util.spec_from_file_location("module1_data_extraction", _SRC_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_SRC_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
app = _MODULE.app


if __name__ == "__main__":
    app()
