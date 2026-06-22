from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import run_experiment


def main(
    train_config: Path = typer.Option(..., "--train-config"),
    test_input: Path = typer.Option(..., "--test"),
    baseline_train: Path | None = typer.Option(None, "--baseline-train"),
    device: str = typer.Option("auto", "--device"),
    publish: bool = typer.Option(True, "--publish/--no-publish"),
    baseline_neural_epochs: int = typer.Option(4, "--baseline-neural-epochs"),
    baseline_neural_window_size: int = typer.Option(128, "--baseline-neural-window-size"),
    baseline_neural_stride: int = typer.Option(64, "--baseline-neural-stride"),
) -> None:
    run_dir = run_experiment(
        train_config,
        test_input,
        baseline_train=baseline_train,
        device=device,
        publish=publish,
        baseline_neural_epochs=baseline_neural_epochs,
        baseline_neural_window_size=baseline_neural_window_size,
        baseline_neural_stride=baseline_neural_stride,
    )
    typer.echo(run_dir)


if __name__ == "__main__":
    typer.run(main)
