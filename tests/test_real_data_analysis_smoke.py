"""Smoke tests for real-data pipeline: verify script and full analysis (when torch available)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_REAL = REPO_ROOT / "data" / "real"


def test_verify_real_data_script_passes():
    """verify_real_data.py must pass on bundled data/real."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "verify_real_data.py")],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "OK" in proc.stdout or "OK" in proc.stderr


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "real" / "X_responses.csv").exists(),
    reason="data/real not present",
)
def test_run_real_data_analysis_loads_and_validates():
    """Run analysis script with 2 epochs; must complete (validates pipeline). Requires torch."""
    pytest.importorskip("torch")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_real_data_analysis.py"),
            "--data-dir",
            str(DATA_REAL),
            "--responses-file",
            "X_responses.csv",
            "--num-epochs",
            "2",
            "--strict-real-data",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    assert "Training completed" in proc.stdout or "Persons=" in proc.stdout
