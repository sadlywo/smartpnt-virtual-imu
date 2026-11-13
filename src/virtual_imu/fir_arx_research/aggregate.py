"""Aggregate the three FIR/ARX preprocessing branches under one evaluation scheme."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger(__name__)

VMD_PROTOCOLS = ("alimu_only", "penn_only", "penn_loso_af", "penn_loso_as", "penn_loso_bf", "pooled")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment table is missing: {path}")
    frame = pd.read_csv(path)
    if "protocol" not in frame:
        raise ValueError(f"{path} has no protocol column")
    return frame


def _raw_baseline(vmd: pd.DataFrame) -> pd.DataFrame:
    raw = vmd[vmd["condition"].eq("raw")].copy()
    if raw.empty:
        raise ValueError("VMD run must retain raw FIR/ARX controls")
    return raw


def _raw_by_method(vmd: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
    raw = _raw_baseline(vmd)
    return {
        (str(row.protocol), str(row.model)): {
            "test_gyro_vector_rmse": float(row.test_gyro_vector_rmse),
            "test_accel_vector_rmse": float(row.test_accel_vector_rmse),
        }
        for row in raw.itertuples(index=False)
    }


def _vmd_comparison(vmd: pd.DataFrame) -> pd.DataFrame:
    raw = _raw_by_method(vmd)
    rows: list[dict[str, Any]] = []
    for protocol, group in vmd.groupby("protocol", sort=False):
        for condition in group["condition"].unique():
            if condition == "raw":
                continue
            selected = group[(group["condition"] == condition) & group["model"].isin(["fir", "arx"])]
            for method in ["fir", "arx"]:
                candidate = selected[selected["model"].eq(method)]
                control = raw.get((str(protocol), method))
                if candidate.empty or control is None:
                    continue
                candidate_row = candidate.iloc[0]
                rows.append(
                    {
                        "branch": "vmd",
                        "protocol": protocol,
                        "condition": condition,
                        "model": method,
                        "feature_count": float(candidate_row["feature_count"]),
                        "test_gyro_vector_rmse": float(candidate_row["test_gyro_vector_rmse"]),
                        "test_accel_vector_rmse": float(candidate_row["test_accel_vector_rmse"]),
                        "raw_gyro_vector_rmse": float(control["test_gyro_vector_rmse"]),
                        "raw_accel_vector_rmse": float(control["test_accel_vector_rmse"]),
                    }
                )
    out = pd.DataFrame(rows)
    out["gyro_delta_pct"] = 100.0 * (out["test_gyro_vector_rmse"] / out["raw_gyro_vector_rmse"] - 1.0)
    out["accel_delta_pct"] = 100.0 * (out["test_accel_vector_rmse"] / out["raw_accel_vector_rmse"] - 1.0)
    out["joint_score_delta"] = 0.5 * (out["gyro_delta_pct"] + out["accel_delta_pct"])
    return out


def _tda_comparison(tda: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol, group in tda.groupby("protocol", sort=False):
        for method in ["fir", "arx"]:
            raw = group[(group["pipeline"] == "raw_windowed") & (group["model"] == method)]
            routed = group[(group["pipeline"] == "tda_routed") & (group["model"] == method)]
            if raw.empty or routed.empty:
                continue
            rows.append(
                {
                    "branch": "tda",
                    "protocol": protocol,
                    "condition": "tda_routed",
                    "model": method,
                    "test_gyro_vector_rmse": float(routed["test_gyro_vector_rmse"].iloc[0]),
                    "test_accel_vector_rmse": float(routed["test_accel_vector_rmse"].iloc[0]),
                    "raw_gyro_vector_rmse": float(raw["test_gyro_vector_rmse"].iloc[0]),
                    "raw_accel_vector_rmse": float(raw["test_accel_vector_rmse"].iloc[0]),
                }
            )
    out = pd.DataFrame(rows)
    out["gyro_delta_pct"] = 100.0 * (out["test_gyro_vector_rmse"] / out["raw_gyro_vector_rmse"] - 1.0)
    out["accel_delta_pct"] = 100.0 * (out["test_accel_vector_rmse"] / out["raw_accel_vector_rmse"] - 1.0)
    out["joint_score_delta"] = 0.5 * (out["gyro_delta_pct"] + out["accel_delta_pct"])
    return out


def _minimal_comparison(minimal: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for protocol, group in minimal.groupby("protocol", sort=False):
        raw = group[group["model"].eq("raw_arx")]
        chosen = group[group["model"].eq("minimal_fir_arx")]
        if raw.empty or chosen.empty:
            continue
        selected = selection[selection["protocol"].eq(protocol)]
        rows.append(
            {
                "branch": "minimal",
                "protocol": protocol,
                "condition": "scenario_minimal",
                "model": "fir_arx",
                "test_gyro_vector_rmse": float(chosen["test_gyro_vector_rmse"].iloc[0]),
                "test_accel_vector_rmse": float(chosen["test_accel_vector_rmse"].iloc[0]),
                "raw_gyro_vector_rmse": float(raw["test_gyro_vector_rmse"].iloc[0]),
                "raw_accel_vector_rmse": float(raw["test_accel_vector_rmse"].iloc[0]),
                "mean_component_count": float(selected["component_count"].mean()) if len(selected) else np.nan,
                "fallback_fraction": float(selected["fallback"].mean()) if len(selected) else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["gyro_delta_pct"] = 100.0 * (out["test_gyro_vector_rmse"] / out["raw_gyro_vector_rmse"] - 1.0)
    out["accel_delta_pct"] = 100.0 * (out["test_accel_vector_rmse"] / out["raw_accel_vector_rmse"] - 1.0)
    out["joint_score_delta"] = 0.5 * (out["gyro_delta_pct"] + out["accel_delta_pct"])
    return out


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fir_arx/comparison.yaml"))
    parser.add_argument("--run-id", default="fir_arx_comparison_20261105")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    inputs = {name: Path(value) for name, value in config["inputs"].items()}
    out_dir = Path(config["output"]["root"]) / args.run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    tables = out_dir / "tables"
    tables.mkdir()
    vmd = _load_csv(inputs["vmd_protocol"])
    tda = _load_csv(inputs["tda_protocol"])
    minimal = _load_csv(inputs["minimal_protocol"])
    selection = _load_csv(inputs["minimal_selection"])
    vmd_comparison = _vmd_comparison(vmd)
    tda_comparison = _tda_comparison(tda)
    minimal_comparison = _minimal_comparison(minimal, selection)
    all_comparison = pd.concat([vmd_comparison, tda_comparison, minimal_comparison], ignore_index=True)
    vmd_comparison.to_csv(tables / "vmd_vs_raw.csv", index=False)
    tda_comparison.to_csv(tables / "tda_vs_raw.csv", index=False)
    minimal_comparison.to_csv(tables / "minimal_vs_raw.csv", index=False)
    all_comparison.to_csv(tables / "all_branch_comparison.csv", index=False)
    summary = (
        all_comparison.groupby(["branch", "condition", "model"], as_index=False)
        .agg(
            protocols=("protocol", "nunique"),
            gyro_delta_pct_mean=("gyro_delta_pct", "mean"),
            gyro_delta_pct_median=("gyro_delta_pct", "median"),
            accel_delta_pct_mean=("accel_delta_pct", "mean"),
            accel_delta_pct_median=("accel_delta_pct", "median"),
            joint_score_delta_mean=("joint_score_delta", "mean"),
            joint_score_delta_median=("joint_score_delta", "median"),
            branch_win_protocols=("joint_score_delta", lambda values: int((values < 0).sum())),
        )
    )
    summary.to_csv(tables / "branch_summary.csv", index=False)
    manifest = {
        "inputs": {name: {"path": str(path), "sha256": _sha(path)} for name, path in inputs.items()},
        "outputs": {},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True).stdout.strip() or None,
    }
    _write_json(out_dir / "provenance.json", manifest)
    lines = [
        "# FIR/ARX preprocessing branch comparison",
        "",
        "All branch scores are relative to the matching raw FIR/ARX control on the same test support.",
        "Negative delta means lower RMSE than the raw control.",
        "",
        "| branch | condition | model | protocols | median gyro delta % | median accel delta % | median joint delta % | protocols improved |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.sort_values(["branch", "condition", "model"]).itertuples(index=False):
        lines.append(
            f"| {row.branch} | {row.condition} | {row.model} | {row.protocols} | "
            f"{row.gyro_delta_pct_median:.2f} | {row.accel_delta_pct_median:.2f} | "
            f"{row.joint_score_delta_median:.2f} | {row.branch_win_protocols} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RUN_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
