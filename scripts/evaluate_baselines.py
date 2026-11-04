from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.eval.baselines import evaluate_baselines


def main(
    input_path: Path = typer.Option(..., "--input"),
    out: Path = typer.Option(..., "--out"),
    train_path: Path = typer.Option(..., "--train"),
    predictions_dir: Path | None = typer.Option(None, "--predictions-dir"),
    navigation_predictions: list[Path] | None = typer.Option(None, "--navigation-prediction"),
    min_allan_duration_s: float = typer.Option(20.0, "--min-allan-duration-s"),
    moving_average_window: int = typer.Option(5, "--moving-average-window"),
    butterworth_cutoff_hz: float = typer.Option(12.0, "--butterworth-cutoff-hz"),
    savgol_window: int = typer.Option(9, "--savgol-window"),
    fir_taps: int = typer.Option(9, "--fir-taps"),
    neural_epochs: int = typer.Option(4, "--neural-epochs"),
    neural_window_size: int = typer.Option(128, "--neural-window-size"),
    neural_stride: int = typer.Option(64, "--neural-stride"),
) -> None:
    evaluate_baselines(
        input_path,
        out,
        train_path=train_path,
        predictions_dir=predictions_dir,
        navigation_predictions=navigation_predictions,
        min_allan_duration_s=min_allan_duration_s,
        moving_average_window=moving_average_window,
        butterworth_cutoff_hz=butterworth_cutoff_hz,
        savgol_window=savgol_window,
        fir_taps=fir_taps,
        neural_epochs=neural_epochs,
        neural_window_size=neural_window_size,
        neural_stride=neural_stride,
    )


if __name__ == "__main__":
    typer.run(main)
