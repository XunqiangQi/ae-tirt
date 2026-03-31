"""Paper real-data example using bundled files under data/real."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "real"
    script_path = project_root / "scripts" / "run_real_data_analysis.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--weights",
        str(data_dir / "pretrained_model.pth"),
        "--data-dir",
        str(data_dir),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
