"""Run the fixed paired/navigation candidate across configured optimization seeds."""

from __future__ import annotations

import copy
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from virtual_imu.eval.formal_evaluation import write_json
from virtual_imu.utils.paths import git_commit
from virtual_imu.viz.style import new_figure, save_figure


def _train(task: str, config: dict, config_path: Path) -> Path:
    if task == "paired":
        from virtual_imu.training.trainer import train_model

        return train_model(config, config_path=config_path)
    from virtual_imu.training.gt_trainer import train_gt_supervised_model

    return train_gt_supervised_model(config, config_path=config_path)


def _evaluate(task: str, run_dir: Path, device: str) -> dict:
    if task == "paired":
        from virtual_imu.eval.formal_evaluation import evaluate_paired_run

        return evaluate_paired_run(run_dir, device=device)
    from virtual_imu.eval.formal_navigation import evaluate_navigation_run

    return evaluate_navigation_run(run_dir, device=device)


def aggregate_results(root: Path, entries: list[dict]) -> dict:
    """Summarize optimization variability; recordings remain the experimental units."""
    frames = {"paired": [], "navigation": []}
    for entry in entries:
        task = entry["task"]
        path = Path(entry["run_dir"]) / "evaluation" / f"{task}_metrics.csv"
        frame = pd.read_csv(path)
        frame["seed"] = entry["seed"]
        frame["run_dir"] = entry["run_dir"]
        frames[task].append(frame)
    summaries = {}
    index = []
    for task, values in frames.items():
        if not values:
            continue
        frame = pd.concat(values, ignore_index=True)
        frame.to_csv(root / f"{task}_all_seeds.csv", index=False)
        groups = ["split", "method"] + (
            ["dataset_id", "low_sensor", "high_sensor"] if task == "paired" else []
        )
        columns = (
            ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"]
            if task == "paired"
            else ["vector_rmse_m", "horizontal_rmse_m", "vertical_rmse_m"]
        )
        summary = frame.groupby(groups, dropna=False, sort=True)[columns].agg(
            ["mean", "std", "count"]
        )
        summary.columns = [f"{column}_{statistic}" for column, statistic in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(root / f"{task}_summary.csv", index=False)
        summaries[task] = summary.to_dict("records")
        if task == "paired":
            model = frame[(frame.split == "test") & (frame.method == "model")]
            identity = frame[(frame.split == "test") & (frame.method == "identity")]
            paired = model.merge(
                identity,
                on=["seed", "dataset_id", "low_sensor", "high_sensor"],
                suffixes=("_model", "_identity"),
            )
            for column in columns:
                ratios = paired.copy()
                ratios["ratio"] = ratios[f"{column}_model"] / ratios[f"{column}_identity"].clip(
                    lower=1e-12
                )
                fig, ax = new_figure("double")
                for x, (_sensor, group) in enumerate(ratios.groupby("low_sensor", sort=True)):
                    ax.errorbar(
                        x,
                        group.ratio.mean(),
                        yerr=group.ratio.std(ddof=1) if len(group) > 1 else 0,
                        fmt="o",
                        capsize=4,
                    )
                sensors = sorted(ratios.low_sensor.unique())
                ax.set_xticks(np.arange(len(sensors)), sensors, rotation=15)
                ax.axhline(1, color="#1f2933", linestyle="--", linewidth=1)
                ax.set_ylabel("Model / identity vector RMSE [1]")
                ax.set_xlabel("Low-cost device")
                stem = f"test_{column}_relative"
                save_figure(fig, root / "figures", stem)
                index.append(
                    f"- {stem}: held-out test vector error relative to the original low-cost signal. Points show mean and sample SD across optimization seeds, not independent recordings; values below 1 indicate lower error."
                )
    if index:
        (root / "figures" / "figure_index.md").write_text(
            "# Figure index\n\n" + "\n".join(index) + "\n", encoding="utf-8"
        )
    write_json(
        root / "aggregate_metrics.json",
        {
            "scope": "Mean and sample SD over optimization seeds; no test-driven candidate selection",
            "tasks": summaries,
        },
    )
    lines = [
        "# Fixed candidate training results",
        "",
        "Mean/std describe optimization randomness, not independent experimental replication.",
        "Paired signal metrics preserve rad/s and m/s² separately. Navigation is fixed-local-ENU window displacement; no ATE/RPE or hardware-grade claim.",
        "",
        "| task | seed | source-of-truth run |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| {entry['task']} | {entry['seed']} | `{entry['run_dir']}` |" for entry in entries
    )
    lines.extend(
        [
            "",
            "See paired_summary.csv/navigation_summary.csv and each run's evaluation/metrics.json for all baselines, device mappings, per-sequence diagnostics and coverage.",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return summaries


def run_formal_pipeline(
    config: dict, *, config_path: Path | None = None, smoke: bool = False
) -> Path:
    seeds = config.get("seeds", [])
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list of distinct integers")
    tasks = config.get("tasks", {})
    if not tasks or set(tasks) - {"paired", "navigation"}:
        raise ValueError("tasks must declare paired and/or navigation YAML configs")
    device = config.get("device", "cuda")
    if config.get("deterministic", True):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Formal pipeline requires CUDA; no silent CPU fallback")
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    root = (
        Path(config.get("runs_dir", "runs")) / f"formal_{'smoke' if smoke else 'training'}_{stamp}"
    )
    root.mkdir(parents=True, exist_ok=False)
    (root / "configs").mkdir()
    (root / "matrix_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    state = {
        "status": "running",
        "smoke": smoke,
        "source_config": str(config_path) if config_path else None,
        "server_launch_run": os.environ.get("VIRTUAL_IMU_SERVER_LAUNCH_RUN"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "runs": [],
    }
    write_json(root / "pipeline.json", state)
    print(f"WRITES: pipeline source-of-truth {root}", flush=True)
    try:
        for task, path in tasks.items():
            source = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
            (root / "configs" / f"{task}_source.yaml").write_text(
                yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
            )
            for seed in seeds[:1] if smoke else seeds:
                actual = copy.deepcopy(source)
                actual["seed"] = seed
                actual["experiment_name"] = f"formal_{'smoke_' if smoke else ''}{task}_seed{seed}"
                actual.setdefault("output", {})["runs_dir"] = str(root / "experiments")
                actual.setdefault("training", {})["device"] = device
                if smoke:
                    actual["training"]["epochs"] = 1
                frozen = root / "configs" / f"{task}_seed{seed}.yaml"
                frozen.write_text(yaml.safe_dump(actual, sort_keys=False), encoding="utf-8")
                state.update(active_task=task, active_seed=seed)
                write_json(root / "pipeline.json", state)
                print(f"READS: frozen {task} seed={seed} config={frozen}", flush=True)
                run_dir = _train(task, actual, frozen)
                entry = {
                    "task": task,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "config": str(frozen),
                    "status": "trained",
                }
                state["runs"].append(entry)
                write_json(root / "pipeline.json", state)
                _evaluate(task, run_dir, device)
                entry["status"] = "evaluated"
                write_json(root / "pipeline.json", state)
                print(f"COMPLETED: task={task} seed={seed} run={run_dir}", flush=True)
        aggregate_results(root, state["runs"])
        state["status"] = "complete"
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(root / "pipeline.json", state)
        raise
    write_json(root / "pipeline.json", state)
    print(f"PIPELINE: {root} status={state['status']}", flush=True)
    return root
