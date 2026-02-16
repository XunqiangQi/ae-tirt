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
        "--data-dir",
        str(data_dir),
        "--responses-file",
        "X_responses.csv",
        "--batch-size",
        "16",
        "--num-epochs",
        "500",
        "--learning-rate",
        "0.001",
        "--early-stopping-patience",
        "20",
        "--penalty-weight-factor",
        "1.0",
        "--optimizer",
        "adam",
        "--seed",
        "42",
        "--weight-constraint",
        "standardized",
        "--link-function",
        "probit",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
