"""Path helpers."""

from __future__ import annotations

from pathlib import Path

_MAX_ANCESTORS = 8
_PROJECT_INDICATORS = ("pyproject.toml", "ae_tirt", "model.py", "sim_data.py")


def get_project_root() -> str:
    current_path = Path(__file__).resolve().parent
    for _ in range(_MAX_ANCESTORS):
        for indicator in _PROJECT_INDICATORS:
            if (current_path / indicator).exists():
                return str(current_path)
        parent = current_path.parent
        if parent == current_path:
            break
        current_path = parent
    return str(Path.cwd())


def get_safe_path(*path_parts: str | Path, base_dir: str | Path | None = None) -> str:
    base = get_project_root() if base_dir is None else base_dir
    return str(Path(base).joinpath(*path_parts))


def ensure_dir(path: str | Path) -> str | Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path
