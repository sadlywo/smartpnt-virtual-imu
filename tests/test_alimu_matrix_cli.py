"""CLI contract tests for the manual ALiMU matrix entry point."""

from __future__ import annotations

import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_alimu_leave_sensor_matrix.py"
PINN_IMU_PYTHON = Path(r"D:\Anaconda3\envs\pinn_imu\python.exe")


def test_help_works_with_pinn_imu() -> None:
    result = subprocess.run(
        [str(PINN_IMU_PYTHON), str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--smoke" in result.stdout
    assert "--resume-run" in result.stdout


def test_cli_declares_default_config_and_path_preflight() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'default=Path("configs/alimu_leave_sensor_matrix_3seed.yaml")' in source
    assert "config_path.is_file()" in source
    assert "args.resume_run.is_dir()" in source
