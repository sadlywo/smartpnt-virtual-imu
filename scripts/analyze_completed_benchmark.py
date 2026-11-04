"""Independently verify and tabulate the immutable retrospective IMU benchmark.

No neural inference, fitting, training, candidate selection or source-run mutation.
Classical predictions are replayed from saved train-only state to provide vector
MAE and segment diagnostics absent from the original compact CSVs.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from virtual_imu.eval.formal_evaluation import (
    HIGH,
    IDENTITY,
    LOW,
    VIRTUAL,
    physical_error_metrics,
    predict_pair_baselines,
)
from virtual_imu.eval.formal_navigation import navigation_error_metrics
from virtual_imu.models import make_model

SEEDS = [20261010, 20261011, 20261012]
AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
NAV_AXES = ["east", "north", "up"]
CLASSICAL = ["identity", "bias_mean", "affine", "ridge_affine", "learned_fir_causal", "arx_linear"]
ORDER = [
    *CLASSICAL,
    "tcn_original",
    "plain_cnn",
    "gru",
    "transformer",
    "cnn_bilstm",
    "deepgyro_adapter",
    "tlio_resnet18_adapter",
    "tartanimu_window_adapter",
]
LABEL = {
    "identity": "Identity",
    "bias_mean": "Mean bias",
    "affine": "Affine",
    "ridge_affine": "Ridge affine",
    "learned_fir_causal": "Causal FIR",
    "arx_linear": "ARX (LOW history)",
    "tcn_original": "TCN",
    "plain_cnn": "CNN",
    "gru": "GRU",
    "transformer": "Transformer",
    "cnn_bilstm": "CNN-BiLSTM",
    "deepgyro_adapter": "DeepGyro adapter",
    "tlio_resnet18_adapter": "TLIO ResNet18 adapter",
    "tartanimu_window_adapter": "TartanIMU window adapter",
    "zero": "Zero",
    "train_mean": "Train mean",
    "ridge": "Ridge",
}
MANIFEST = "5e7f198a57c9370734c4ecce092aa11f797530dded935957324c725c4c752260"
SOURCE_NEW = "a63ab02d355c04f7451bafac7576c80cfca503d86c7bc1652097b92d1e2b77a4"
SOURCE_OLD = "1a60647ce6ca7d84bf8729b032893a527af63bc731945f307d3b7aca8d8c4b2f"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, value):
    path.write_text(
        json.dumps(plain(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(actual, expected, label):
    if not np.allclose(
        np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), rtol=2e-9, atol=1e-11
    ):
        raise ValueError(f"Independent metric replay mismatch: {label}: {actual} != {expected}")


def compare_metrics(actual: dict, expected: dict, label: str) -> int:
    count = 0
    for key, value in expected.items():
        if key in actual and isinstance(value, (int, float, list)):
            assert_close(actual[key], value, f"{label}/{key}")
            count += 1
    return count


def paired_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    result = physical_error_metrics(pred, target)
    error = np.asarray(pred, float) - np.asarray(target, float)
    for name, sl in [("gyro", slice(0, 3)), ("accel", slice(3, 6))]:
        unit = "radps" if name == "gyro" else "mps2"
        e = error[:, sl]
        result[f"{name}_vector_mae_{unit}"] = float(np.mean(np.linalg.norm(e, axis=1)))
        result[f"{name}_bias_norm_{unit}"] = float(np.linalg.norm(e.mean(axis=0)))
        result[f"{name}_centered_std_norm_{unit}"] = float(np.linalg.norm(e.std(axis=0)))
        result[f"{name}_axis_mean_mae_{unit}"] = float(np.abs(e).mean())
        bias2 = float(np.sum(e.mean(axis=0) ** 2))
        centered2 = float(np.sum(e.std(axis=0) ** 2))
        rmse2 = float(np.mean(np.sum(e * e, axis=1)))
        assert_close(bias2 + centered2, rmse2, f"{name}/MSE decomposition")
        result[f"{name}_mean_residual_share_mse_percent"] = 100 * bias2 / max(rmse2, 1e-30)
    return result


def scalar_metrics(result: dict) -> dict:
    return {k: v for k, v in result.items() if isinstance(v, (int, float, np.number))}


def nav_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    result = navigation_error_metrics(pred, target)
    e = pred - target
    result.update(
        vector_mae_m=float(np.linalg.norm(e, axis=1).mean()),
        horizontal_mae_m=float(np.linalg.norm(e[:, :2], axis=1).mean()),
        vertical_mae_m=float(np.abs(e[:, 2]).mean()),
        bias_norm_m=float(np.linalg.norm(e.mean(axis=0))),
        centered_std_norm_m=float(np.linalg.norm(e.std(axis=0))),
    )
    return result


def summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    numeric = [c for c in frame.select_dtypes(include="number") if c not in ["seed", "parameters"]]
    # pandas 3.0's list aggregation can resolve a data column named ``axis``
    # through DataFrameGroupBy.__getattr__ as its removed axis attribute.
    # Explicit numeric reductions also make the ddof and sample counts clear.
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, values, strict=True))
        for metric in numeric:
            data = group[metric].to_numpy(float)
            data = data[np.isfinite(data)]
            row[f"{metric}_mean"] = float(data.mean()) if len(data) else None
            row[f"{metric}_sd"] = float(data.std(ddof=1)) if len(data) > 1 else None
            row[f"{metric}_n"] = len(data)
        rows.append(row)
    return pd.DataFrame(rows)


class Analysis:
    def __init__(self, root: Path, out: Path, old: Path):
        self.root, self.out, self.old = root.resolve(), out.resolve(), old.resolve()
        self.server = self.root.parents[1]
        self.old_server = self.old.parents[1]
        self.tables = self.out / "tables"
        self.tables.mkdir(parents=True, exist_ok=True)
        self.checks = {
            "artifact_hashes": 0,
            "checkpoint_configs": 0,
            "checkpoint_finite_tensors": 0,
            "json_recomputed_values": 0,
            "csv_recomputed_values": 0,
            "matched_sample_or_endpoint_sets": 0,
            "classical_baseline_equalities": 0,
        }
        (
            self.sources,
            self.entries,
            self.paired,
            self.navigation,
            self.paired_axes,
            self.nav_axes,
            self.paired_seq,
            self.nav_seq,
            self.dataset,
            self.registry,
        ) = {}, [], [], [], [], [], [], [], [], []
        self.sample_fingerprints, self.baselines, self.baseline_states, self.parameter_counts = (
            {},
            {},
            {},
            {},
        )
        self.table_paths = []

    def bind(self, path):
        path = Path(path).resolve()
        if path.as_posix() not in self.sources:
            self.sources[path.as_posix()] = {
                "path": path.as_posix(),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
        return path

    def map_remote(self, value):
        value = str(value).replace("\\", "/")
        for checkout, local in [
            ("server_bundle_benchmark_20261024_v1", self.server),
            ("server_bundle_formal_deterministic_20261023", self.old_server),
        ]:
            token = "/" + checkout + "/"
            if token in value:
                return local / value.split(token, 1)[1]
        raise ValueError(f"Unrecognized source identity: {value}")

    def csv(self, name, frame):
        path = self.tables / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        self.table_paths.append(
            {"path": path.as_posix(), "rows": len(frame), "columns": list(frame.columns)}
        )
        return frame

    def load(self):
        self.bind(Path(__file__).resolve())
        pipeline = read_json(self.bind(self.root / "pipeline.json"))
        self.bind(self.root / "matrix_config.yaml")
        sources = {
            task: yaml.safe_load(
                self.bind(self.root / "configs" / f"{task}_source.yaml").read_text(encoding="utf-8")
            )
            for task in ["paired", "navigation"]
        }
        assert (
            pipeline["status"] == "complete"
            and not pipeline["smoke"]
            and len(pipeline["runs"]) == 45
        )
        expected = {
            (c, t, s)
            for c in [
                "tcn_original",
                "plain_cnn",
                "gru",
                "transformer",
                "cnn_bilstm",
                "deepgyro_adapter",
                "tlio_resnet18_adapter",
            ]
            for t in ["paired", "navigation"]
            for s in SEEDS
        }
        expected.update(("tartanimu_window_adapter", "navigation", s) for s in SEEDS)
        assert {(e["candidate_id"], e["task"], e["seed"]) for e in pipeline["runs"]} == expected
        assert sum(bool(e.get("reused")) for e in pipeline["runs"]) == 6
        workflow = (self.server / "runs/workflow_status.log").read_text()
        assert "BENCHMARK_PASSED" in workflow and "WORKFLOW_EXIT=0" in workflow
        self.bind(self.server / "runs/workflow_status.log")
        for e in pipeline["runs"]:
            assert e["status"] == "evaluated"
            run = self.map_remote(e["run_dir"])
            for remote, expected_hash in e["artifacts"]["sha256"].items():
                assert digest(self.bind(self.map_remote(remote))) == expected_hash, remote
                self.checks["artifact_hashes"] += 1
            config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
            assert config["data"] == sources[e["task"]]["data"]
            assert config["training"] == sources[e["task"]]["training"]
            assert config["model"]["in_channels"] == (9 if e["task"] == "paired" else 6)
            assert config["model"]["out_channels"] == (6 if e["task"] == "paired" else 3)
            checkpoint = torch.load(run / "best.ckpt", map_location="cpu", weights_only=False)
            assert config == checkpoint["config"] and config["seed"] == e["seed"]
            self.checks["checkpoint_configs"] += 1
            for name, tensor in checkpoint["model_state_dict"].items():
                assert torch.isfinite(tensor).all(), (run, name)
                self.checks["checkpoint_finite_tensors"] += 1
            assert digest(run / "data_manifest.json") == MANIFEST
            source = read_json(run / "source_bundle.json")
            assert source["source_fingerprint"] == (SOURCE_OLD if e.get("reused") else SOURCE_NEW)
            history = pd.read_csv(run / "metrics.csv")
            selection = config["training"]["checkpoint_metric"]
            row = history.loc[history[selection].idxmin()]
            assert int(row.epoch) == checkpoint["best_epoch"]
            assert_close(
                float(row[selection]), checkpoint["best_metric"], "validation checkpoint selection"
            )
            model_key = json.dumps(config["model"], sort_keys=True)
            if model_key not in self.parameter_counts:
                torch.manual_seed(config["seed"])
                model = make_model(config["model"])
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                self.parameter_counts[model_key] = sum(p.numel() for p in model.parameters())
                del model
            params = self.parameter_counts[model_key]
            elapsed = None
            if e.get("started_at") and e.get("finished_at"):
                elapsed = (
                    datetime.fromisoformat(e["finished_at"])
                    - datetime.fromisoformat(e["started_at"])
                ).total_seconds()
            self.registry.append(
                {
                    "candidate_id": e["candidate_id"],
                    "candidate_label": LABEL[e["candidate_id"]],
                    "task": e["task"],
                    "seed": e["seed"],
                    "reused": bool(e.get("reused")),
                    "epochs": len(history),
                    "best_epoch": int(row.epoch),
                    "checkpoint_metric": selection,
                    "best_validation_value": float(row[selection]),
                    "stopped_before_max_epochs": len(history) < config["training"]["epochs"],
                    "max_epochs": config["training"]["epochs"],
                    "early_stopping_patience": config["training"]["early_stopping_patience"],
                    "parameters": params,
                    "completed_run_wall_seconds": elapsed,
                    "temporal_context": e["temporal_context"],
                    "fidelity": e["fidelity"],
                    "run_dir": run.as_posix(),
                    "remote_run_dir": e["run_dir"],
                    "config": (run / "config.yaml").as_posix(),
                    "config_sha256": digest(run / "config.yaml"),
                    "checkpoint": (run / "best.ckpt").as_posix(),
                    "checkpoint_sha256": digest(run / "best.ckpt"),
                    "source_fingerprint": source["source_fingerprint"],
                    "data_manifest_sha256": MANIFEST,
                    "data_manifest": (run / "data_manifest.json").as_posix(),
                    "source_bundle": (run / "source_bundle.json").as_posix(),
                    "evaluation": (run / "evaluation/metrics.json").as_posix(),
                    "evaluation_sha256": digest(run / "evaluation/metrics.json"),
                    "test_predictions": (run / "evaluation/test_predictions.parquet").as_posix(),
                    "test_predictions_sha256": digest(run / "evaluation/test_predictions.parquet"),
                }
            )
            for field in [
                "config",
                "checkpoint",
                "evaluation",
                "test_predictions",
                "data_manifest",
                "source_bundle",
            ]:
                self.registry[-1][f"{field}_relative_to_report"] = Path(
                    os.path.relpath(self.registry[-1][field], self.out)
                ).as_posix()
            self.entries.append(
                {
                    **e,
                    "run": run,
                    "metrics": read_json(run / "evaluation/metrics.json"),
                    "config_dict": config,
                    "parameters": params,
                }
            )
            del checkpoint
        self.state = pipeline

    def meta(self, e, split, method="model"):
        candidate = e["candidate_id"] if method == "model" else method
        return {
            "candidate_id": candidate,
            "candidate_label": LABEL[candidate],
            "method": "model" if method == "model" else method,
            "method_category": "learned" if method == "model" else "classical",
            "temporal_context": e["temporal_context"]
            if method == "model"
            else "fixed_classical_protocol",
            "fidelity": e["fidelity"] if method == "model" else "unchanged_train_fitted_classical",
            "split": split,
            "seed": e["seed"],
        }

    def sample_set(self, e, split, frame, columns):
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(frame[columns], index=False).values.tobytes()
        ).hexdigest()
        key = e["task"], split
        if key in self.sample_fingerprints:
            assert fingerprint == self.sample_fingerprints[key], (
                e["candidate_id"],
                key,
                "different physical samples",
            )
        else:
            self.sample_fingerprints[key] = fingerprint
        self.checks["matched_sample_or_endpoint_sets"] += 1

    def replay_pair(self, e, split, frame):
        expected_count = 96320 if split == "test" else 48840
        assert (
            len(frame) == expected_count
            and not frame.duplicated(["sequence_id", "timestamp_s"]).any()
        )
        self.sample_set(e, split, frame, ["sequence_id", "timestamp_s", *IDENTITY, *LOW, *HIGH])
        result = e["metrics"]["splits"][split]
        baseline_key = ("paired", split, e["seed"])
        include_classic = baseline_key not in self.baselines
        state_path = e["run"] / "evaluation/baseline_train_state.json"
        baseline_state = read_json(state_path)
        states = {tuple(s["identity"][k] for k in IDENTITY): s for s in baseline_state}
        state_hash = digest(state_path)
        if "paired" in self.baseline_states:
            assert self.baseline_states["paired"] == state_hash
            self.checks["classical_baseline_equalities"] += 1
        else:
            self.baseline_states["paired"] = state_hash
        actual_classic = {}
        for pair in result["pairs"]:
            identity = tuple(pair[k] for k in IDENTITY)
            mask = np.logical_and.reduce([frame[k] == pair[k] for k in IDENTITY])
            group = frame.loc[mask].copy()
            target = group[HIGH].to_numpy(float)
            methods = {"model": group[VIRTUAL].to_numpy(float)}
            if include_classic:
                methods.update(predict_pair_baselines(states[identity], group))
            for method, pred in methods.items():
                physical = paired_metrics(pred, target)
                self.checks["json_recomputed_values"] += compare_metrics(
                    physical,
                    pair["metrics"][method],
                    f"{e['candidate_id']}/{split}/{identity}/{method}",
                )
                meta = {**self.meta(e, split, method), **dict(zip(IDENTITY, identity, strict=True))}
                self.paired.append({**meta, **scalar_metrics(physical)})
                for index, axis in enumerate(AXES):
                    self.paired_axes.append(
                        {
                            **meta,
                            "axis": axis,
                            "unit": "rad/s" if index < 3 else "m/s^2",
                            "rmse": physical["axis_rmse"][index],
                            "mae": physical["axis_mae"][index],
                            "mean_error": physical["axis_mean_error"][index],
                            "centered_error_std_ddof0": physical["axis_error_std"][index],
                        }
                    )
                if method != "model":
                    actual_classic[(*identity, method)] = pair["metrics"][method]
                for sequence_id, segment in group.groupby("sequence_id", sort=True):
                    positions = group.index.get_indexer(segment.index)
                    y = pred[positions]
                    ref = segment[HIGH].to_numpy(float)
                    m = paired_metrics(y, ref)
                    times = segment.timestamp_s.to_numpy(float)
                    assert len(times) < 2 or np.allclose(
                        np.diff(times), 0.025, rtol=1e-5, atol=1e-6
                    )
                    error = y - ref
                    integrals = [
                        np.linalg.norm(np.trapz(error[start : start + 80, :3], dx=0.025, axis=0))
                        for start in range(0, len(error) - 79, 80)
                    ]
                    a_integrals = [
                        np.linalg.norm(np.trapz(error[start : start + 80, 3:], dx=0.025, axis=0))
                        for start in range(0, len(error) - 79, 80)
                    ]
                    self.paired_seq.append(
                        {
                            **meta,
                            "sequence_id": sequence_id,
                            **scalar_metrics(m),
                            "span_s": times[-1] - times[0],
                            "integral_windows": len(integrals),
                            "delta_gyro_integral_error_rad": float(np.mean(integrals)),
                            "delta_accel_integral_error_mps": float(np.mean(a_integrals)),
                        }
                    )
                    if method == "model":
                        original = next(
                            s for s in result["sequences"] if s["sequence_id"] == sequence_id
                        )
                        self.checks["json_recomputed_values"] += compare_metrics(
                            m, original["metrics"], sequence_id
                        )
                        assert_close(
                            float(np.mean(integrals)),
                            original["diagnostics"]["delta_gyro_integral_error_rad"],
                            "segment gyro integration",
                        )
                        assert_close(
                            float(np.mean(a_integrals)),
                            original["diagnostics"]["delta_accel_integral_error_mps"],
                            "segment accel integration",
                        )
                # Per-dataset pooling is computed separately after grouping stored errors.
        if include_classic:
            self.baselines[baseline_key] = actual_classic
        else:
            for pair in result["pairs"]:
                identity = tuple(pair[k] for k in IDENTITY)
                for method in CLASSICAL:
                    compare_metrics(
                        pair["metrics"][method],
                        self.baselines[baseline_key][(*identity, method)],
                        "classical equality",
                    )
                    self.checks["classical_baseline_equalities"] += 1

    def replay_navigation(self, e, split, frame):
        assert len(frame) == (1649 if split == "test" else 3637)
        target_cols = [f"true_delta_{axis}_local_m" for axis in "xyz"]
        self.sample_set(
            e,
            split,
            frame,
            ["sequence_id", "start_index", "end_index", "start_time_s", "end_time_s", *target_cols],
        )
        result = e["metrics"]["splits"][split]
        key = ("navigation", split, e["seed"])
        include_classic = key not in self.baselines
        methods = ["model", "zero", "train_mean", "ridge"] if include_classic else ["model"]
        actual_classic = {}
        for method in methods:
            prefix = "pred" if method == "model" else method
            pred_cols = [f"{prefix}_delta_{axis}_local_m" for axis in "xyz"]
            pred, target = frame[pred_cols].to_numpy(float), frame[target_cols].to_numpy(float)
            m = nav_metrics(pred, target)
            self.checks["json_recomputed_values"] += compare_metrics(
                m, result["metrics"][method], f"nav/{e['candidate_id']}/{split}/{method}"
            )
            meta = self.meta(e, split, method)
            self.navigation.append({**meta, **scalar_metrics(m)})
            for axis_index, axis in enumerate(NAV_AXES):
                self.nav_axes.append(
                    {
                        **meta,
                        "axis": axis,
                        "unit": "m",
                        "rmse": m["axis_rmse_m"][axis_index],
                        "mae": m["axis_mae_m"][axis_index],
                        "mean_error": m["axis_bias_m"][axis_index],
                        "centered_error_std_ddof0": float(
                            np.std(pred[:, axis_index] - target[:, axis_index])
                        ),
                    }
                )
            for sequence_id, segment in frame.groupby("sequence_id", sort=True):
                sm = nav_metrics(
                    segment[pred_cols].to_numpy(float), segment[target_cols].to_numpy(float)
                )
                self.nav_seq.append(
                    {
                        **meta,
                        "sequence_id": sequence_id,
                        "original_sequence_id": segment.original_sequence_id.iloc[0],
                        **scalar_metrics(sm),
                    }
                )
                original = next(s for s in result["sequences"] if s["sequence_id"] == sequence_id)
                self.checks["json_recomputed_values"] += compare_metrics(
                    sm, original["metrics"][method], f"nav/{sequence_id}/{method}"
                )
            if method != "model":
                actual_classic[method] = result["metrics"][method]
        if include_classic:
            self.baselines[key] = actual_classic
        else:
            for method in ["zero", "train_mean", "ridge"]:
                compare_metrics(
                    result["metrics"][method], self.baselines[key][method], "nav baseline equality"
                )
                self.checks["classical_baseline_equalities"] += 1

    def replay(self):
        for e in self.entries:
            for split in ["validation", "test"]:
                path = self.bind(e["run"] / "evaluation" / f"{split}_predictions.parquet")
                frame = pd.read_parquet(path)
                if e["task"] == "paired":
                    self.replay_pair(e, split, frame)
                else:
                    self.replay_navigation(e, split, frame)
            print(
                f"VERIFIED: {e['candidate_id']}/{e['task']}/{e['seed']}: finite checkpoint and independent validation/test metric replay",
                flush=True,
            )

    def output(self):
        pair = self.csv("paired_per_seed", pd.DataFrame(self.paired))
        nav = self.csv("navigation_per_seed", pd.DataFrame(self.navigation))
        pair_keys = [
            "candidate_id",
            "candidate_label",
            "method",
            "method_category",
            "temporal_context",
            "fidelity",
            "split",
            *IDENTITY,
        ]
        nav_keys = [
            "candidate_id",
            "candidate_label",
            "method",
            "method_category",
            "temporal_context",
            "fidelity",
            "split",
        ]
        pair_summary = self.csv("paired_comparison", summary(pair, pair_keys))
        nav_summary = self.csv("navigation_comparison", summary(nav, nav_keys))
        pair_axes = self.csv("paired_axis_metrics_per_seed", pd.DataFrame(self.paired_axes))
        nav_axes = self.csv("navigation_axis_metrics_per_seed", pd.DataFrame(self.nav_axes))
        self.csv("paired_axis_metrics_summary", summary(pair_axes, [*pair_keys, "axis", "unit"]))
        self.csv("navigation_axis_metrics_summary", summary(nav_axes, [*nav_keys, "axis", "unit"]))
        paired_seq = self.csv("paired_continuous_segments_per_seed", pd.DataFrame(self.paired_seq))
        nav_seq = self.csv("navigation_continuous_segments_per_seed", pd.DataFrame(self.nav_seq))
        self.csv(
            "paired_continuous_segments_summary", summary(paired_seq, [*pair_keys, "sequence_id"])
        )
        self.csv(
            "navigation_continuous_segments_summary",
            summary(nav_seq, [*nav_keys, "sequence_id", "original_sequence_id"]),
        )
        registry = self.csv("training_registry", pd.DataFrame(self.registry))
        self.csv(
            "model_parameters",
            registry[
                [
                    "candidate_id",
                    "candidate_label",
                    "task",
                    "parameters",
                    "temporal_context",
                    "fidelity",
                ]
            ].drop_duplicates(),
        )
        grouped_scores = []
        for entry in self.entries:
            if entry["task"] != "paired":
                continue
            for split in ["validation", "test"]:
                model_rows = pair[
                    (pair.candidate_id == entry["candidate_id"])
                    & (pair.seed == entry["seed"])
                    & (pair.split == split)
                ]
                ratios = []
                for _dataset_id, dataset_rows in model_rows.groupby("dataset_id"):
                    per_pair = []
                    for r in dataset_rows.to_dict("records"):
                        identity_rows = pair[
                            (pair.candidate_id == "identity")
                            & (pair.seed == r["seed"])
                            & (pair.split == r["split"])
                        ]
                        for key in IDENTITY:
                            identity_rows = identity_rows[identity_rows[key] == r[key]]
                        baseline = identity_rows.iloc[0]
                        per_pair.append(
                            np.mean(
                                [
                                    r[metric] / baseline[metric]
                                    for metric in [
                                        "gyro_vector_rmse_radps",
                                        "accel_vector_rmse_mps2",
                                    ]
                                ]
                            )
                        )
                    ratios.append(float(np.mean(per_pair)))
                score = float(np.mean(ratios))
                assert_close(
                    score,
                    entry["metrics"]["splits"][split]["grouped_relative_score"],
                    "dimensionless grouped score",
                )
                grouped_scores.append(
                    {
                        "candidate_id": entry["candidate_id"],
                        "candidate_label": LABEL[entry["candidate_id"]],
                        "seed": entry["seed"],
                        "split": split,
                        "grouped_relative_score": score,
                    }
                )
        score_table = self.csv("paired_grouped_scores_per_seed", pd.DataFrame(grouped_scores))
        self.csv(
            "paired_grouped_scores_summary",
            summary(score_table, ["candidate_id", "candidate_label", "split"]),
        )
        # Dataset-level RMSE pools squared errors weighted by evaluated samples;
        # dataset-level MAE pools absolute/norm errors with the same weights.
        dataset_rows = []
        for values, group in pair.groupby([*nav_keys, "seed", "dataset_id"], sort=False):
            meta = dict(zip([*nav_keys, "seed", "dataset_id"], values, strict=True))
            n = group.samples.to_numpy(float)
            row = {**meta, "samples": int(n.sum()), "pair_count": len(group)}
            for metric in ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"]:
                row[metric] = float(np.sqrt(np.average(group[metric] ** 2, weights=n)))
            for metric in ["gyro_vector_mae_radps", "accel_vector_mae_mps2"]:
                row[metric] = float(np.average(group[metric], weights=n))
            dataset_rows.append(row)
        dataset = self.csv("paired_dataset_metrics_per_seed", pd.DataFrame(dataset_rows))
        self.csv("paired_dataset_metrics_summary", summary(dataset, [*nav_keys, "dataset_id"]))
        improvements = []
        for task, frame, metrics, baselines, keys in [
            (
                "paired",
                pair,
                ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"],
                ["identity", "learned_fir_causal", "arx_linear"],
                ["split", "seed", *IDENTITY],
            ),
            (
                "navigation",
                nav,
                ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"],
                ["zero", "train_mean", "ridge"],
                ["split", "seed"],
            ),
        ]:
            for baseline in baselines:
                base = frame[frame.candidate_id == baseline]
                for _, row in frame[frame.method_category == "learned"].iterrows():
                    match = base
                    for k in keys:
                        match = match[match[k] == row[k]]
                    assert len(match) == 1
                    for metric in metrics:
                        improvements.append(
                            {
                                "task": task,
                                "candidate_id": row.candidate_id,
                                "candidate_label": row.candidate_label,
                                **{k: row[k] for k in keys},
                                "baseline": baseline,
                                "metric": metric,
                                "improvement_percent": 100
                                * (1 - row[metric] / max(match.iloc[0][metric], 1e-12)),
                            }
                        )
        improvement = self.csv("improvements", pd.DataFrame(improvements))
        imp_keys = [k for k in improvement if k not in ["seed", "improvement_percent"]]
        self.csv("improvements_summary", summary(improvement, imp_keys))
        # Cross-check the source pipeline's compact CSVs without pooling candidates.
        for task, frame, metrics in [
            ("paired", pair, ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"]),
            ("navigation", nav, ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"]),
        ]:
            source = pd.read_csv(self.bind(self.root / f"{task}_all_seeds.csv"))
            for row in source.to_dict("records"):
                candidate = row["candidate_id"] if row["method"] == "model" else row["method"]
                match = frame[
                    (frame.candidate_id == candidate)
                    & (frame.seed == row["seed"])
                    & (frame.split == row["split"])
                ]
                if task == "paired":
                    for k in IDENTITY:
                        match = match[match[k] == row[k]]
                assert len(match) == 1
                for metric in metrics:
                    assert_close(
                        float(match.iloc[0][metric]),
                        row[metric],
                        f"source CSV {task}/{candidate}/{metric}",
                    )
                    self.checks["csv_recomputed_values"] += 1
        findings = self.findings(pair_summary, nav_summary, improvement)
        self.markdown(pair_summary, nav_summary, registry)
        audit = {
            "status": "passed",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "expected_runs": 45,
            "new_runs": 39,
            "reused_runs": 6,
            "checks": self.checks,
            "checkpoint_validation_selection": "All best_epoch/best_metric agree with validation history, never test selection",
            "sample_coverage": {
                "paired_validation": 48840,
                "paired_test": 96320,
                "navigation_validation": 3637,
                "navigation_test": 1649,
            },
            "baseline_policy": "Each classic emitted once per task/split/seed after equality checks across candidates; saved train states replayed with no fitting; seed slots are not independent classic fits",
            "scope": "Retrospective common-protocol architecture/backbone adaptation comparison, not native-paper SOTA reproduction",
            "all_table_outputs": self.table_paths,
            "source_files": list(self.sources.values()),
        }
        write_json(self.out / "scientific_audit.json", audit)
        analysis = {
            "scope": audit["scope"],
            "counts": {
                "neural_runs": 45,
                "new_runs": 39,
                "reused_runs": 6,
                "paired_methods": 13,
                "navigation_methods": 11,
            },
            "findings": findings,
            "paired_summary": pair_summary.to_dict("records"),
            "navigation_summary": nav_summary.to_dict("records"),
            "training_runs": registry.to_dict("records"),
            "table_files": self.table_paths,
            "audit": {
                "path": (self.out / "scientific_audit.json").as_posix(),
                "status": "passed",
                "checks": self.checks,
            },
        }
        write_json(self.out / "analysis.json", analysis)
        print(
            "VERIFIED: 45 run identities; all checkpoint/artifact hashes; independent prediction metric replay and complete comparison tables",
            flush=True,
        )
        print(json.dumps(plain(findings), ensure_ascii=False, indent=2), flush=True)

    def findings(self, pair_summary, nav_summary, improvement):
        findings = {
            "paired_test_rankings_descriptive_only": [],
            "navigation_test_rows": [],
            "neural_vs_identity": [],
            "neural_vs_strong_baselines": [],
        }
        test = pair_summary[pair_summary.split == "test"]
        for identity, group in test.groupby(IDENTITY):
            for metric in ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"]:
                ordered = group.sort_values(f"{metric}_mean")
                findings["paired_test_rankings_descriptive_only"].append(
                    {
                        **dict(zip(IDENTITY, identity, strict=True)),
                        "metric": metric,
                        "ranking": [
                            {
                                "candidate_id": r.candidate_id,
                                "mean": r[f"{metric}_mean"],
                                "sample_sd": r[f"{metric}_sd"],
                            }
                            for _, r in ordered.iterrows()
                        ],
                    }
                )
        for _, row in nav_summary[nav_summary.split == "test"].iterrows():
            findings["navigation_test_rows"].append(
                {
                    "candidate_id": row.candidate_id,
                    **{
                        f"{m}_{stat}": row[f"{m}_{stat}"]
                        for m in ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"]
                        for stat in ["mean", "sd"]
                    },
                }
            )
        for cid, group in improvement[improvement.split == "test"].groupby("candidate_id"):
            identity = group[(group.task == "paired") & (group.baseline == "identity")]
            grouped = identity.groupby([*IDENTITY, "metric"]).improvement_percent.mean()
            findings["neural_vs_identity"].append(
                {
                    "candidate_id": cid,
                    "improved_mean_pairs_channels": int((grouped > 0).sum()),
                    "total_pairs_channels": len(grouped),
                    "all_seed_comparisons_improved": int((identity.improvement_percent > 0).sum()),
                    "all_seed_comparisons": len(identity),
                }
            )
            strong = group[
                (group.task == "paired") & group.baseline.isin(["learned_fir_causal", "arx_linear"])
            ]
            means = strong.groupby([*IDENTITY, "metric", "baseline"]).improvement_percent.mean()
            findings["neural_vs_strong_baselines"].append(
                {
                    "candidate_id": cid,
                    "improved_mean_comparisons": int((means > 0).sum()),
                    "total_mean_comparisons": len(means),
                    "all_seed_comparisons_improved": int((strong.improvement_percent > 0).sum()),
                    "all_seed_comparisons": len(strong),
                }
            )
        return findings

    def markdown(self, paired, nav, registry):
        lines = [
            "# 完整扩展模型比较表",
            "",
            "本轮为回顾性的共同任务架构/骨干适配比较，不是所有论文原方法的官方 SOTA 成绩。均值±样本标准差（ddof=1）来自三个优化种子，不是独立录制的置信区间。固定经典方法三次重复一致，表中只展示均值。",
            "",
            "gyro 和 accel 保持物理单位分开；SmartPNT 是 1.975 秒窗口端点位移，不是完整轨迹 ATE/RPE。",
            "",
        ]
        sensors = ["BMX160", "MPU9250", "LSM9DS1", "Tango Top IMU"]
        for split, split_cn in [("test", "测试集"), ("validation", "验证集")]:
            for metric, title in [
                ("gyro_vector_rmse_radps", "陀螺仪矢量 RMSE（rad/s）"),
                ("accel_vector_rmse_mps2", "加速度矢量 RMSE（m/s²）"),
                ("gyro_vector_mae_radps", "陀螺仪矢量 MAE（rad/s）"),
                ("accel_vector_mae_mps2", "加速度矢量 MAE（m/s²）"),
            ]:
                lines += [
                    f"## {split_cn}：{title}",
                    "",
                    "| 方法 | BMX160→VN100 | MPU9250→VN100 | LSM9DS1→VN100 | Tango→ADIS16488 |",
                    "|---|---:|---:|---:|---:|",
                ]
                data = paired[paired.split == split]
                for cid in ORDER:
                    if cid not in set(data.candidate_id):
                        continue
                    cells = [LABEL[cid]]
                    for sensor in sensors:
                        row = data[(data.candidate_id == cid) & (data.low_sensor == sensor)].iloc[0]
                        value = f"{row[f'{metric}_mean']:.6f}"
                        if row.method_category == "learned":
                            value += f" ± {row[f'{metric}_sd']:.6f}"
                        cells.append(value)
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")
            lines += [
                f"## {split_cn}：SmartPNT 窗口位移 RMSE（m）",
                "",
                "| 方法 | 三维 | 水平 | 垂直 |",
                "|---|---:|---:|---:|",
            ]
            data = nav[nav.split == split]
            for cid in ["zero", "train_mean", "ridge", *ORDER]:
                if cid not in set(data.candidate_id):
                    continue
                row = data[data.candidate_id == cid].iloc[0]
                cells = [LABEL[cid]]
                for m in ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"]:
                    value = f"{row[f'{m}_mean']:.4f}"
                    if row.method_category == "learned":
                        value += f" ± {row[f'{m}_sd']:.4f}"
                    cells.append(value)
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        lines += [
            "## 全部训练运行",
            "",
            "| 候选 | 任务 | 种子 | 训练轮数 | 最优轮 | 验证最优值 | 参数量 | 提前结束 |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in registry.to_dict("records"):
            lines.append(
                f"| {r['candidate_label']} | {r['task']} | {r['seed']} | {r['epochs']} | {r['best_epoch']} | {r['best_validation_value']:.6g} | {r['parameters']} | {'是' if r['stopped_before_max_epochs'] else '否'} |"
            )
        lines += [
            "",
            "全部分轴 RMSE/MAE/平均残差/中心化标准差、连续片段、数据集分组、改进百分比和来源哈希见同目录 CSV。向量 MAE 定义为每时刻误差向量范数的均值。中心化标准差使用总体定义 ddof=0，与种子样本 SD 的 ddof=1 不同。",
            "",
            "DeepGyro/TLIO/TartanIMU 均为已核对骨干的任务适配：输入/输出头和损失与原方法不同；RF511/80点、stride32仅3个粗特征、单窗口单token的限制均保留。",
        ]
        (self.tables / "complete_comparison.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--old-pipeline-root",
        type=Path,
        default=Path(
            "runs/remote_results_formal_20261024/server/runs/formal_training_20261023_141916_733429"
        ),
    )
    args = parser.parse_args()
    print(
        f"READS: explicit immutable benchmark {args.pipeline_root}, six referenced old runs, checkpoints/config/manifests and predictions",
        flush=True,
    )
    print(f"WRITES: {args.report_dir}/tables, analysis.json and scientific_audit.json", flush=True)
    print(
        "SKIPS: neural inference/training, baseline fitting, test-based selection, original artifact mutation and unit pooling",
        flush=True,
    )
    torch.set_num_threads(1)
    analysis = Analysis(args.pipeline_root, args.report_dir, args.old_pipeline_root)
    analysis.load()
    analysis.replay()
    analysis.output()


if __name__ == "__main__":
    main()
