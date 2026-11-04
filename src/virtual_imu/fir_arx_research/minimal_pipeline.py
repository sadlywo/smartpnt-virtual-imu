"""Scenario-aware minimal-component FIR/ARX experiment."""

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

from virtual_imu.fir_arx_research.data import HIGH, LOW, PAIR_COLUMNS, UnifiedDataset, load_config, ordered_pairs, resolve, select_pair, sha256_file
from virtual_imu.fir_arx_research.features import ComponentBank
from virtual_imu.fir_arx_research.metrics import signal_metrics
from virtual_imu.fir_arx_research.minimal import MinimalConfig, SceneRouter, annotate_scene, causal_band_split, scene_windows, select_minimal_prefix
from virtual_imu.fir_arx_research.models import fit_models
from virtual_imu.fir_arx_research.viz import plot_minimal_frontier, plot_minimal_vs_full

LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _prediction_by_window(values: np.ndarray, windows: list[pd.DataFrame]) -> dict[str, np.ndarray]:
    offset = 0
    result = {}
    for window in windows:
        count = len(window)
        result[str(window["window_id"].iloc[0])] = values[offset : offset + count]
        offset += count
    if offset != len(values):
        raise ValueError("Prediction support does not match test windows")
    return result


def _full_model_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    train_bank: ComponentBank,
    validation_bank: ComponentBank,
    test_bank: ComponentBank,
    *,
    condition: str,
    search: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    models = fit_models(
        condition=condition,
        train=train,
        validation=validation,
        test=test,
        train_bank=train_bank,
        validation_bank=validation_bank,
        test_bank=test_bank,
        methods=["fir", "arx"],
        search=search,
    )
    return {model.name: model for model in models}


def run_minimal_experiment(config_path: str | Path, *, run_id: str | None = None) -> Path:
    config_path = Path(config_path).resolve()
    project_root = Path.cwd().resolve()
    for candidate in [project_root, *project_root.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src/virtual_imu").is_dir():
            project_root = candidate
            break
    config = load_config(config_path)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    identifier = run_id or config["output"].get("run_id") or f"{config['experiment_id']}_{stamp}"
    run_dir = resolve(project_root, Path(config["output"]["root"]) / identifier)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    for name in ("tables", "figures", "states"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    dataset = UnifiedDataset(project_root, config["data"]["manifest"], verify_hashes=bool(config["data"].get("verify_hashes", True)))
    minimal_config = MinimalConfig(**config.get("minimal", {}))
    pair_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    protocol_audits = []
    for protocol_id in config["protocols"]:
        protocol = dataset.protocol(
            protocol_id,
            min_segment_rows=int(config["data"].get("min_segment_rows", 320)),
            stride=int(config["data"].get("stride", 40)),
        )
        protocol_audits.append(protocol.audit)
        for pair_identity in ordered_pairs(protocol.frames["test"]):
            LOGGER.info("MINIMAL protocol=%s pair=%s", protocol_id, "::".join(pair_identity))
            frames = {role: select_pair(protocol.frames[role], pair_identity) for role in ("train", "validation", "test")}
            banks = {role: causal_band_split(frame, minimal_config) for role, frame in frames.items()}
            windows = {role: scene_windows(frame, banks[role], minimal_config) for role, frame in frames.items()}
            router = SceneRouter.fit(windows["train"][1], minimal_config)
            labels = {role: router.predict(windows[role][1]) for role in ("train", "validation", "test")}
            annotated = {
                role: annotate_scene(frames[role], windows[role][0], labels[role])
                for role in ("train", "validation", "test")
            }
            reference = annotated["test"][HIGH].to_numpy(dtype=np.float64)
            identity = annotated["test"][LOW].to_numpy(dtype=np.float64)
            baseline = signal_metrics(identity, reference, frame=annotated["test"])
            window_banks = {role: causal_band_split(annotated[role], minimal_config) for role in annotated}
            raw_banks = {role: window_banks[role].subset(list(range(6))) for role in window_banks}
            all_indices = list(range(len(banks["train"].names)))
            all_banks = {role: window_banks[role].subset(all_indices) for role in window_banks}
            method_predictions: dict[str, np.ndarray] = {}
            for name, subset_banks in (("raw", raw_banks), ("full", all_banks)):
                models = _full_model_predictions(
                    annotated["train"],
                    annotated["validation"],
                    annotated["test"],
                    subset_banks["train"],
                    subset_banks["validation"],
                    subset_banks["test"],
                    condition=name,
                    search=config["search"],
                )
                for method, model in models.items():
                    # Refit/predict on the corresponding split bank for full baseline.
                    from virtual_imu.fir_arx_research.models import lag_matrix, ridge_predict

                    test_x = lag_matrix(annotated["test"], subset_banks["test"], taps=int(model.state["taps"]), include_delta=bool(model.state.get("include_delta", False)))
                    method_predictions[f"{name}_{method}"] = ridge_predict(test_x, np.asarray(model.state["weights"]), np.asarray(model.state["bias"]))
                    for scene in sorted(annotated["test"]["scene"].unique()):
                        mask = annotated["test"]["scene"].eq(scene).to_numpy()
                        scene_metrics = signal_metrics(method_predictions[f"{name}_{method}"][mask], reference[mask], frame=annotated["test"].loc[mask].reset_index(drop=True))
                        scenario_rows.append({
                            "protocol": protocol_id,
                            **dict(zip(PAIR_COLUMNS, pair_identity, strict=True)),
                            "condition": name,
                            "model": method,
                            "scene": int(scene),
                            "test_gyro_vector_rmse": scene_metrics.gyro_vector_rmse,
                            "test_accel_vector_rmse": scene_metrics.accel_vector_rmse,
                        })
            minimal_predictions = np.zeros((len(annotated["test"]), 6), dtype=np.float64)
            scene_offsets = {int(scene): 0 for scene in sorted(labels["test"].tolist())}
            for scene in sorted(np.unique(labels["train"]).tolist()):
                train_scene = annotated["train"][annotated["train"]["scene"].eq(scene)].reset_index(drop=True)
                validation_scene = annotated["validation"][annotated["validation"]["scene"].eq(scene)].reset_index(drop=True)
                test_mask = annotated["test"]["scene"].eq(scene).to_numpy()
                test_scene = annotated["test"][test_mask].reset_index(drop=True)
                scenario_rows.append({
                    "protocol": protocol_id,
                    **dict(zip(PAIR_COLUMNS, pair_identity, strict=True)),
                    "condition": "minimal",
                    "model": "fir_arx",
                    "scene": int(scene),
                    "train_rows": len(train_scene),
                    "validation_rows": len(validation_scene),
                    "test_rows": len(test_scene),
                })
                if len(train_scene) < minimal_config.min_scene_rows or len(validation_scene) < 200 or len(test_scene) < 200:
                    missing = test_scene
                    fallback_bank = causal_band_split(test_scene, minimal_config)
                    fallback_models = _full_model_predictions(
                        annotated["train"],
                        annotated["validation"],
                        test_scene,
                        all_banks["train"],
                        all_banks["validation"],
                        fallback_bank,
                        condition="fallback",
                        search=config["search"],
                    )
                    fallback = fallback_models["fir"]
                    from virtual_imu.fir_arx_research.models import lag_matrix, ridge_predict

                    values = ridge_predict(lag_matrix(test_scene, fallback_bank, taps=int(fallback.state["taps"]), include_delta=False), np.asarray(fallback.state["weights"]), np.asarray(fallback.state["bias"]))
                    minimal_predictions[test_mask] = values
                    selection_rows.append({
                        "protocol": protocol_id,
                        **dict(zip(PAIR_COLUMNS, pair_identity, strict=True)),
                        "scene": int(scene),
                        "fallback": True,
                        "component_count": 18,
                        "components": ",".join(all_banks["train"].names),
                    })
                    continue
                scene_banks = {
                    "train": causal_band_split(train_scene, minimal_config),
                    "validation": causal_band_split(validation_scene, minimal_config),
                    "test": causal_band_split(test_scene, minimal_config),
                }
                selected = select_minimal_prefix(
                    train_scene,
                    validation_scene,
                    test_scene,
                    scene_banks["train"].subset(all_indices),
                    scene_banks["validation"].subset(all_indices),
                    scene_banks["test"].subset(all_indices),
                    config=minimal_config,
                    search=config["search"],
                )
                minimal_predictions[test_mask] = selected["test_prediction"]
                selection_rows.append({
                    "protocol": protocol_id,
                    **dict(zip(PAIR_COLUMNS, pair_identity, strict=True)),
                    "scene": int(scene),
                    "fallback": False,
                    "component_count": selected["component_count"],
                    "components": ",".join(selected["components"]),
                    "taps": selected["taps"],
                    "alpha": selected["alpha"],
                    "include_delta": selected["include_delta"],
                    "validation_score": selected["validation_score"],
                    "frontier": json.dumps(selected["frontier"]),
                })
            method_predictions["minimal_fir_arx"] = minimal_predictions
            for name, values in method_predictions.items():
                metrics = signal_metrics(values, reference, frame=annotated["test"])
                pair_rows.append({
                    "protocol": protocol_id,
                    **dict(zip(PAIR_COLUMNS, pair_identity, strict=True)),
                    "model": name,
                    "train_rows": len(annotated["train"]),
                    "validation_rows": len(annotated["validation"]),
                    "test_rows": len(annotated["test"]),
                    "test_gyro_vector_rmse": metrics.gyro_vector_rmse,
                    "test_accel_vector_rmse": metrics.accel_vector_rmse,
                    "test_gyro_highpass_rmse": metrics.gyro_highpass_rmse,
                    "test_accel_highpass_rmse": metrics.accel_highpass_rmse,
                    "test_gyro_improvement_vs_identity_pct": 100.0 * (baseline.gyro_vector_rmse - metrics.gyro_vector_rmse) / max(baseline.gyro_vector_rmse, 1e-12),
                    "test_accel_improvement_vs_identity_pct": 100.0 * (baseline.accel_vector_rmse - metrics.accel_vector_rmse) / max(baseline.accel_vector_rmse, 1e-12),
                })
            _write_json(run_dir / "states" / protocol_id / ("__".join(pair_identity) + ".json"), {
                "router": {
                    "scaler_mean": router.scaler.mean_.tolist(),
                    "scaler_scale": router.scaler.scale_.tolist(),
                    "cluster_centers": router.kmeans.cluster_centers_.tolist(),
                },
                "scenario_counts": {str(k): int(v) for k, v in pd.Series(labels["train"]).value_counts().sort_index().items()},
                "selected": [row for row in selection_rows if row["protocol"] == protocol_id and all(row[key] == value for key, value in zip(PAIR_COLUMNS, pair_identity, strict=True))],
            })
    pair_metrics = pd.DataFrame(pair_rows)
    selections = pd.DataFrame(selection_rows)
    scenes = pd.DataFrame(scenario_rows)
    pair_metrics.to_csv(run_dir / "tables" / "metrics_by_pair.csv", index=False)
    selections.to_csv(run_dir / "tables" / "component_selection.csv", index=False)
    scenes.to_csv(run_dir / "tables" / "scenario_metrics.csv", index=False)
    protocol_summary = pair_metrics.groupby(["protocol", "model"], as_index=False).agg(
        test_gyro_improvement_vs_identity_pct=("test_gyro_improvement_vs_identity_pct", "mean"),
        test_accel_improvement_vs_identity_pct=("test_accel_improvement_vs_identity_pct", "mean"),
        test_gyro_vector_rmse=("test_gyro_vector_rmse", "mean"),
        test_accel_vector_rmse=("test_accel_vector_rmse", "mean"),
    )
    protocol_summary.to_csv(run_dir / "tables" / "metrics_by_protocol.csv", index=False)
    _write_json(run_dir / "split_audit.json", protocol_audits)
    _write_json(run_dir / "data_manifest.json", {
        "dataset_manifest": str(dataset.manifest_path),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "records": [{key: record[key] for key in ("dataset_id", "pair_id", "sequence_id", "path", "sha256", "split", "rows")} for record in dataset.records if record.get("task") == "paired_signal" and record.get("training_eligible")],
    })
    figures = [
        plot_minimal_vs_full(pair_metrics, run_dir / "figures"),
        plot_minimal_frontier(selections, run_dir / "figures"),
    ]
    _write_json(run_dir / "figures.json", figures)
    _write_json(run_dir / "provenance.json", {
        "experiment_id": config["experiment_id"],
        "run_id": identifier,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True).stdout.strip() or None,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_sha256": dataset.manifest_sha256,
        "feature_source": "causal_band_component_selection",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "minimal": minimal_config.__dict__,
    })
    (run_dir / "report.md").write_text(
        "# Scenario-aware Minimal FIR/ARX Input Experiment\n\n"
        "- Causal low/high band components are computed with a 0.5 Hz low-pass filter.\n"
        "- Scenes are learned on TRAIN windows only and assigned by high-frequency energy ratio.\n"
        "- Component ranking and minimal-prefix selection use VALIDATION only; TEST is fixed.\n"
        "- Comparisons retain raw FIR/ARX, all-component FIR/ARX, and scenario-routed minimal FIR/ARX.\n",
        encoding="utf-8",
    )
    files = {str(path.relative_to(run_dir)): sha256_file(path) for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json"}
    _write_json(run_dir / "artifact_manifest.json", {"root": str(run_dir), "files": files})
    return run_dir
