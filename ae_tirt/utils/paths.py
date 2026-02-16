"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def get_project_root() -> str:
    current_path = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    project_indicators = ["pyproject.toml", "ae_tirt", "model.py", "sim_data.py"]

    for _ in range(8):
        for indicator in project_indicators:
            if (current_path / indicator).exists():
                return str(current_path)
        parent = current_path.parent
        if parent == current_path:
            break
        current_path = parent
    return str(Path.cwd())


def get_safe_path(*path_parts, base_dir=None):
    base = get_project_root() if base_dir is None else base_dir
    return str(Path(base).joinpath(*path_parts))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
