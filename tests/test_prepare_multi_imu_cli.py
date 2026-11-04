from __future__ import annotations

from pathlib import Path

from scripts.prepare_multi_imu_strict import app
from typer.testing import CliRunner


def test_cli_help_lists_strict_preparation_options() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for option in (
        "--config",
        "--dataset",
        "--fetch-calibration",
        "--dry-run",
        "--max-duration-s",
    ):
        assert option in result.output


def test_cli_dry_run_uses_isolated_bounded_output_variant(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""seed: 1
strict_output_root: {tmp_path.as_posix()}/strict
diagnostic_output_root: {tmp_path.as_posix()}/diagnostics
preparation_pairs: []
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "--config",
            str(config),
            "--dataset",
            "penncosyvio",
            "--dry-run",
            "--max-duration-s",
            "120",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "bounded_120s" in result.output
    assert not (tmp_path / "strict" / "manifest.json").exists()
