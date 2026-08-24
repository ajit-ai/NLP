"""Shared test helper: load example scripts by filename stem."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def load_example(stem: str):
    """Import ``examples/<stem>.py`` as a module object."""
    path = EXAMPLES_DIR / f"{stem}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(f"examples.{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_all_examples():
    """Load every example script (used by smoke tests)."""
    return {
        p.stem: load_example(p.stem)
        for p in sorted(EXAMPLES_DIR.glob("*.py"))
        if not p.name.startswith("__")
    }
