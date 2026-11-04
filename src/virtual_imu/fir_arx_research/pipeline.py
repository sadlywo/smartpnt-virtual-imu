"""Experiment orchestration for frozen FIR/ARX reproduction runs."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from virtual_imu.fir_arx_research.data import (
    HIGH,
    PAIR_COLUMNS,
    UnifiedDataset,
    load_config,
    ordered_pairs,
    resolve,
    select_pair,
    sha256_file,
)
from virtual_imu.fir_arx_research.features import RawFeatureSource
from virtual_imu.fir_arx_research.metrics import signal_metrics
from virtual_imu.fir_arx_research.models import fit_models, model_metric_rows
from virtual_imu.fir_arx_research.viz import plot_pair_scatter, plot_protocol_improvements

LOGGER = logging.getLogger(__name__)


def feature_source_for_config(config: dict[str, Any]):
    """Branch extension point for preprocessing experiments."""
    name = config.get("features", {}).get("source", "raw")
    if name != "raw":
        raise ValueError(f"Unsupported feature source in base branch: {name}")
    return RawFeatureSource()


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("WRITES %s", path)


def _sequence_metric_rows(
    models,
    protocol: str,
    identity: tuple[str, str, str, str],
    test: pd.DataFrame,
) -> list[dict[str, Any]]:
    pair = dict(zip(PAIR_COLUMNS, identity, strict=True))
    rows = []
    for model in models:
        for sequence_id, positions in test.groupby("sequence_id", sort=False).indices.items():
            frame = test.iloc[positions].reset_index(drop=True)
            prediction = model.test_prediction[positions]
            metrics = signal_metrics(prediction, frame[HIGH].to_numpy(dtype=np.float64), frame=frame)
            rows.append(
                {
                    "protocol": protocol,
                    **pair,
                    "sequence_id": sequence_id,
                    "model": model.name,
                    "condition": model.condition,
                    **metrics.__dict__,
                }
            )
    return rows


def _macro_summary(pair_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [
        "validation_gyro_vector_rmse",
        "validation_accel_vector_rmse",
        "test_gyro_vector_rmse",
        "test_accel_vector_rmse",
        "test_gyro_vector_mae",
        "test_accel_vector_mae",
        "test_gyro_bias_norm",
        "test_accel_bias_norm",
        "test_gyro_highpass_rmse",
        "test_accel_highpass_rmse",
        "test_gyro_improvement_vs_identity_pct",
        "test_accel_improvement_vs_identity_pct",
        "feature_count",
        "component_count",
    ]
    dataset_summary = (
        pair_metrics.groupby(["protocol", "dataset_id", "condition", "model"], as_index=False)[numeric]
        .mean()
        .assign(aggregation_level="dataset_macro")
    )
    protocol_summary = (
        dataset_summary.groupby(["protocol", "condition", "model"], as_index=False)[numeric]
        .mean()
        .assign(aggregation_level="protocol_macro")
    )
    return dataset_summary, protocol_summary


def run_experiment(config_path: str | Path, *, run_id: str | None = None) -> Path:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[2] if config_path.parent.name == "fir_arx" else Path.cwd()
    # The package can be invoked from any cwd; locate pyproject.toml deterministically.
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "virtual_imu").is_dir():
            project_root = candidate
            break
    config = load_config(config_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    identifier = run_id or config["output"].get("run_id") or f"{config['experiment_id']}_{timestamp}"
    run_dir = resolve(project_root, Path(config["output"]["root"]) / identifier)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    states_dir = run_dir / "states"
    tables_dir.mkdir()
    figures_dir.mkdir()
    states_dir.mkdir()
    (run_dir / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    LOGGER.info("READS config %s", config_path)
    LOGGER.info("WRITES run directory %s", run_dir)

    dataset = UnifiedDataset(
        project_root,
        config["data"]["manifest"],
        verify_hashes=bool(config["data"].get("verify_hashes", True)),
    )
    source = feature_source_for_config(config)
    methods = list(config["search"].get("methods", ["identity", "mean_bias", "affine", "ridge_affine", "fir", "arx"]))
    pair_rows: list[dict[str, Any]] = []
    sequence_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    protocol_audits = []
    for protocol_id in config["protocols"]:
        LOGGER.info("READS and assembles protocol %s", protocol_id)
        protocol = dataset.protocol(
            protocol_id,
            min_segment_rows=int(config["data"].get("min_segment_rows", 320)),
            stride=int(config["data"].get("stride", 40)),
        )
        protocol_audits.append(protocol.audit)
        for identity in ordered_pairs(protocol.frames["test"]):
            LOGGER.info("FITS protocol=%s pair=%s", protocol_id, "::".join(identity))
            train = select_pair(protocol.frames["train"], identity)
            validation = select_pair(protocol.frames["validation"], identity)
            test = select_pair(protocol.frames["test"], identity)
            bank_train = source.transform(train)
            bank_validation = source.transform(validation)
            bank_test = source.transform(test)
            models = fit_models(
                condition=source.name,
                train=train,
                validation=validation,
                test=test,
                train_bank=bank_train,
                validation_bank=bank_validation,
                test_bank=bank_test,
                methods=methods,
                search=config["search"],
            )
            pair_rows.extend(model_metric_rows(models, protocol_id, identity, train, validation, test))
            sequence_rows.extend(_sequence_metric_rows(models, protocol_id, identity, test))
            for model in models:
                for trial in model.candidates:
                    trial_rows.append(
                        {
                            "protocol": protocol_id,
                            **dict(zip(PAIR_COLUMNS, identity, strict=True)),
                            **trial,
                        }
                    )
                state_path = states_dir / protocol_id / ("__".join(identity) + f"__{model.name}.json")
                _write_json(state_path, model.state)
        LOGGER.info("SKIPS test-based model selection for protocol %s", protocol_id)

    pair_metrics = pd.DataFrame(pair_rows)
    sequence_metrics = pd.DataFrame(sequence_rows)
    trials = pd.DataFrame(trial_rows)
    dataset_summary, protocol_summary = _macro_summary(pair_metrics)
    pair_metrics.to_csv(tables_dir / "metrics_by_pair.csv", index=False)
    sequence_metrics.to_csv(tables_dir / "metrics_by_sequence.csv", index=False)
    trials.to_csv(tables_dir / "trials.csv", index=False)
    dataset_summary.to_csv(tables_dir / "metrics_by_dataset.csv", index=False)
    protocol_summary.to_csv(tables_dir / "metrics_by_protocol.csv", index=False)
    _write_json(run_dir / "split_audit.json", protocol_audits)

    record_manifest = {
        "dataset_manifest": str(dataset.manifest_path),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "records": [
            {key: record[key] for key in ("dataset_id", "pair_id", "sequence_id", "path", "sha256", "split", "rows")}
            for record in dataset.records
            if record.get("task") == "paired_signal" and record.get("training_eligible")
        ],
    }
    _write_json(run_dir / "data_manifest.json", record_manifest)

    figures = [
        plot_protocol_improvements(pair_metrics, figures_dir),
        plot_pair_scatter(pair_metrics, figures_dir),
    ]
    provenance = {
        "experiment_id": config["experiment_id"],
        "run_id": identifier,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(project_root),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest": str(dataset.manifest_path),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "feature_source": source.name,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "protocols": list(config["protocols"]),
        "methods": methods,
        "seed": int(config["seed"]),
    }
    _write_json(run_dir / "provenance.json", provenance)
    _write_json(run_dir / "figures.json", figures)
    _write_report(run_dir, config, protocol_summary, figures, source.name)
    _write_manifest(run_dir)
    return run_dir


def _write_report(run_dir: Path, config: dict, summary: pd.DataFrame, figures, source_name: str) -> None:
    best = summary.sort_values("test_gyro_improvement_vs_identity_pct", ascending=False).head(8)
    lines = [
        "# FIR/ARX IMU Correction Reproduction",
        "",
        f"- Experiment: `{config['experiment_id']}`",
        f"- Feature condition: `{source_name}`",
        "- Selection: train fit, validation selection, one fixed test evaluation.",
        "- Metrics: vector RMSE in physical SI units; improvement is relative to uncorrected LOW.",
        "",
        "## Data and protocol",
        "",
        "The experiment uses the frozen `unified_imu_v2` paired-signal corpus and recording-disjoint protocols. "
        "All models are fit separately for each known LOW-to-HIGH sensor pair; dataset/protocol summaries are unweighted macro averages.",
        "",
        "## Best gyroscope results",
        "",
        "| protocol | model | gyro improvement % | accel improvement % |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"| {row.protocol} | {row.model} | {row.test_gyro_improvement_vs_identity_pct:.2f} | "
            f"{row.test_accel_improvement_vs_identity_pct:.2f} |"
        )
    lines += ["", "## Figures", ""]
    for record in figures:
        lines.append(f"- `figures/{record['stem']}.png`")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "These are retrospective fixed-protocol comparisons, not global SOTA claims. A positive result means lower "
        "signal RMSE against the measured HIGH reference on the specified recording-disjoint test split. HIGH is a "
        "measured device reference and is not treated as error-free ground truth.",
    ]
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest(run_dir: Path) -> None:
    outputs = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            outputs[str(path.relative_to(run_dir))] = sha256_file(path)
    _write_json(
        run_dir / "artifact_manifest.json",
        {"root": str(run_dir), "files": outputs},
    )
