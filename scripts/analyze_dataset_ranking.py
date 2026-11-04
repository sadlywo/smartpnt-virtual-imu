"""Describe domain-dependent ranks in the completed pooled-training benchmark.

This audit never retrains models or interprets evaluation-domain slices as
dataset-specific training. Optimization seeds are repeated fits, not recordings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

METRICS = ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2")
SCOPE = "completed_pooled_alimu_penn_training; descriptive evaluation-domain ranks"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config_path: Path) -> Path:
    print(f"READS: ranking config {config_path}", flush=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tables = Path(config["tables_root"])
    inputs = {name: tables / name for name in ("paired_per_seed.csv", "paired_dataset_metrics_per_seed.csv")}
    data = {}
    for name, path in inputs.items():
        print(f"READS: immutable completed metrics {path}", flush=True)
        data[name] = pd.read_csv(path)
    output = Path(config["output_root"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace an existing ranking audit: {output}")
    output.mkdir(parents=True, exist_ok=True)
    print("SKIPS: training, target fitting, source-run mutation, inferential p-values", flush=True)
    result_tables = {}
    for level, filename, keys in (
        ("device", "paired_per_seed.csv", ["dataset_id", "low_sensor", "high_sensor"]),
        ("dataset", "paired_dataset_metrics_per_seed.csv", ["dataset_id"]),
    ):
        source = data[filename].loc[lambda frame: frame.split.eq("test")].copy()
        for set_name, selected in (
            ("learned", source.loc[source.method_category.eq("learned")]),
            ("all_methods", source),
        ):
            identity = [*keys, "candidate_id", "candidate_label", "method_category"]
            if selected.duplicated([*keys, "candidate_id", "seed"]).any():
                raise ValueError(f"Duplicate test method/seed in {filename}")
            seed_counts = selected.groupby([*keys, "candidate_id"]).seed.nunique()
            expected_seeds = set(config["optimization_seeds"])
            if set(selected.seed) != expected_seeds or not seed_counts.eq(len(expected_seeds)).all():
                raise ValueError(f"Incomplete declared optimization seeds in {filename}")
            long = selected.melt(
                id_vars=[*identity, "seed"], value_vars=list(METRICS),
                var_name="metric", value_name="value",
            )
            long["rank"] = long.groupby([*keys, "metric", "seed"]).value.rank(method="average")
            means = long.groupby([*identity, "metric"], as_index=False).agg(
                value_mean=("value", "mean"), value_seed_sd=("value", "std"),
                optimization_seed_count=("seed", "nunique"),
                seed_rank_min=("rank", "min"), seed_rank_max=("rank", "max"),
            )
            means["rank"] = means.groupby([*keys, "metric"]).value_mean.rank(method="average")
            means["training_regime"] = SCOPE
            means = means.sort_values([*keys, "metric", "rank"])
            result_tables[f"{level}_{set_name}_rankings.csv"] = means
            correlations = []
            for metric, group in means.groupby("metric"):
                pivot = group.pivot(index="candidate_id", columns=keys, values="value_mean")
                for left, right in combinations(pivot.columns, 2):
                    paired = pivot[[left, right]].dropna()
                    if len(paired) != len(pivot):
                        raise ValueError("Domain comparisons require identical candidate sets")
                    correlations.append({
                        "metric": metric, "left_domain": str(left), "right_domain": str(right),
                        "method_count": len(paired),
                        "spearman_rho": float(spearmanr(paired[left], paired[right]).statistic),
                        "kendall_tau_b": float(kendalltau(paired[left], paired[right]).statistic),
                        "scope": SCOPE,
                    })
            result_tables[f"{level}_{set_name}_rank_correlations.csv"] = pd.DataFrame(correlations)
    for name, frame in result_tables.items():
        destination = output / name
        frame.to_csv(destination, index=False)
        print(f"WRITES: descriptive table {destination} rows={len(frame)}", flush=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    provenance = {
        "created_utc": datetime.now(UTC).isoformat(), "scope": SCOPE, "git_commit": commit,
        "config": {"path": str(config_path), "sha256": digest(config_path)},
        "implementation": {"path": str(Path(__file__)), "sha256": digest(Path(__file__))},
        "inputs": {name: {"path": str(path), "sha256": digest(path)} for name, path in inputs.items()},
        "source_of_truth_pipeline": config["source_of_truth_pipeline"],
        "limitations": [
            "All learned models were trained on pooled ALiMU and PennCOSYVIO.",
            "These slices do not establish rankings after training on different datasets.",
            "Three optimization seeds are not independent recording replicates.",
            "Only two AL test recordings and one Penn test recording; no scene-population inference.",
            "AL aggregate metrics are sample-weighted physical RMSE, not a mixed-unit average.",
        ],
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    (output / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    lines = [
        "# Completed pooled-training dataset ranking audit", "",
        "The completed models' ordering changes across evaluation domains. All models were trained on the pooled ALiMU/Penn corpus; dataset-specific training and zero-shot generalization have not been measured here.", "",
        "## Learned model ranks by dataset", "",
        "| Dataset | Physical metric | Best mean model | Mean | Ordering |", "|---|---|---|---:|---|",
    ]
    ranks = result_tables["dataset_learned_rankings.csv"]
    for (dataset, metric), frame in ranks.groupby(["dataset_id", "metric"], sort=True):
        frame = frame.sort_values("rank")
        best = frame.iloc[0]
        lines.append(f"| {dataset} | {metric} | {best.candidate_label} | {best.value_mean:.6g} | {' > '.join(frame.candidate_label)} |")
    lines.extend(["", "## Descriptive rank agreement", "", "| Physical metric | Spearman rho | Kendall tau-b | Models |", "|---|---:|---:|---:|"])
    for row in result_tables["dataset_learned_rank_correlations.csv"].itertuples():
        lines.append(f"| {row.metric} | {row.spearman_rho:.6f} | {row.kendall_tau_b:.6f} | {row.method_count} |")
    lines.extend(["", "Seeds describe optimization variability; no p-values or independent scene confidence intervals are claimed. Gyroscope and accelerometer errors retain their physical units. Device-specific tables provide the eight device-by-metric slices and their pairwise rank agreement.", "", "Source pipeline: `" + config["source_of_truth_pipeline"] + "`.", "", "A proper follow-up must retrain each model on AL-only, Penn-only and pooled data under matched search budgets, choose checkpoints using each training scope's validation recordings, and keep target-domain labels out of zero-shot selection.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"WRITES: audit provenance and report {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/analyze_dataset_ranking.yaml"))
    arguments = parser.parse_args()
    run(arguments.config)
