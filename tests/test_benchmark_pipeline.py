"""Behavioral checks for isolated, model-aware and resumable comparisons."""

from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.io.training_bundle import sha256_file
from virtual_imu.training import benchmark_pipeline as pipeline


@pytest.fixture(autouse=True)
def restore_torch_runtime():
    flags = (
        torch.are_deterministic_algorithms_enabled(),
        torch.get_num_threads(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    cudnn_sdp = (
        torch.backends.cuda.cudnn_sdp_enabled()
        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
        else None
    )
    yield
    torch.use_deterministic_algorithms(flags[0])
    torch.set_num_threads(flags[1])
    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = flags[2:4]
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags[4:6]
    torch.backends.cuda.enable_flash_sdp(flags[6])
    torch.backends.cuda.enable_mem_efficient_sdp(flags[7])
    torch.backends.cuda.enable_math_sdp(flags[8])
    if cudnn_sdp is not None:
        torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp)
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def _config(tmp_path: Path, *, tasks=("paired", "navigation"), seeds=(11, 12)) -> dict:
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)
    manifests = {}
    for name in ("dataset_manifest", "split_manifest", "strict_manifest"):
        path = data_root / f"{name}.json"
        path.write_text(json.dumps({"kind": name}), encoding="utf-8")
        manifests[name] = str(path)
    paths = {}
    for task in tasks:
        source = {
            "experiment_name": f"formal_{task}",
            "seed": 11,
            "data": {**manifests, "window_size": 80, "stride": 40, "rate_hz": 40.0},
            "training": {"epochs": 60, "device": "cpu", "lr": 0.001, "early_stopping_patience": 10},
            "output": {"runs_dir": "runs"},
        }
        if task == "paired":
            source["data"]["feature_columns"] = [f"feature_{index}" for index in range(9)]
            source["model"] = {
                "type": "tcn",
                "target_mode": "residual",
                "in_channels": 9,
                "out_channels": 6,
                "levels": 4,
            }
            source["training"]["checkpoint_metric"] = "val_grouped_score"
        else:
            source["data"]["label_mode"] = "position_delta"
            source["model"] = {
                "type": "tcn",
                "prediction_head": "last",
                "target_layout": "window",
                "in_channels": 6,
                "out_channels": 3,
                "levels": 4,
            }
            source["training"]["checkpoint_metric"] = "val_rmse"
        path = tmp_path / f"{task}.yaml"
        path.write_text(yaml.safe_dump(source), encoding="utf-8")
        paths[task] = str(path)
    return {
        "seeds": list(seeds),
        "device": "cpu",
        "deterministic": True,
        "cpu_threads": 1,
        "tasks": paths,
        "runs_dir": str(tmp_path / "runs"),
        "data_manifest_sha256": sha256_file(Path(manifests["dataset_manifest"])),
        "candidates": [
            {
                "id": name,
                "label": name.upper(),
                "model": {"type": model},
                "tasks": list(tasks),
                "fidelity": "task_adapter",
                "temporal_context": "window",
                "references": ["https://example.org/paper"],
            }
            for name, model in (("gru", "gru"), ("resnet", "ronin_resnet"))
        ],
    }


def _fake_train(task: str, config: dict, frozen: Path) -> Path:
    run = Path(config["output"]["runs_dir"]) / f"{config['experiment_name']}_fixture"
    run.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(frozen, run / "config.yaml")
    inputs = {}
    names = {
        "dataset_manifest": "data_manifest.json",
        "split_manifest": "split_manifest.json",
        "strict_manifest": "strict_manifest.json",
    }
    for key, name in names.items():
        path = Path(config["data"][key])
        shutil.copyfile(path, run / name)
        inputs[key] = {"path": str(path), "sha256": sha256_file(path)}
    (run / "provenance.json").write_text(
        json.dumps({"unified_dataset": {"inputs": inputs}}), encoding="utf-8"
    )
    selected = config["training"]["checkpoint_metric"]
    pd.DataFrame({"epoch": [1], "train_loss": [0.4], "val_loss": [0.3], selected: [0.2]}).to_csv(
        run / "metrics.csv", index=False
    )
    torch.save(
        {
            "config": config,
            "model_config": config["model"],
            "model_state_dict": {"weight": torch.tensor([0.1])},
            "best_epoch": 1,
            "best_metric": 0.2,
        },
        run / "best.ckpt",
    )
    return run


def _write_evaluation(
    task: str, run: Path, *, model_value: float = 2.0, baseline_value: float = 1.0
) -> None:
    directory = run / "evaluation"
    directory.mkdir(exist_ok=False)
    rows = []
    report = {
        "selection": yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["training"][
            "checkpoint_metric"
        ],
        "splits": {},
    }
    for split in ("validation", "test"):
        reported_metrics = {}
        for method in ("model", *sorted(pipeline.BASELINES[task])):
            row = {
                "split": split,
                "method": method,
                **{
                    metric: model_value if method == "model" else baseline_value
                    for metric in pipeline.PRIMARY_METRICS[task]
                },
            }
            if task == "paired":
                row.update(dataset_id="alimu", low_sensor="BMX160", high_sensor="VN100")
            rows.append(row)
            reported_metrics[method] = {
                metric: row[metric] for metric in pipeline.PRIMARY_METRICS[task]
            }
        report["splits"][split] = (
            {
                "samples": 2,
                "pairs": [
                    {
                        "dataset_id": "alimu",
                        "low_sensor": "BMX160",
                        "high_sensor": "VN100",
                        "metrics": reported_metrics,
                    }
                ],
            }
            if task == "paired"
            else {"windows": 2, "metrics": reported_metrics}
        )
        if task == "paired":
            columns = [
                f"{axis}_{kind}"
                for kind in ("high", "virtual")
                for axis in ("gx", "gy", "gz", "ax", "ay", "az")
            ]
        else:
            columns = [
                f"{method}_delta_{axis}_local_m"
                for method in ("true", "pred", "zero", "train_mean", "ridge")
                for axis in ("x", "y", "z")
            ]
        pd.DataFrame({column: [0.1, 0.2] for column in columns}).to_parquet(
            directory / f"{split}_predictions.parquet", index=False
        )
    pd.DataFrame(rows).to_csv(directory / f"{task}_metrics.csv", index=False)
    (directory / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "baseline_train_state.json").write_text('{"fit_split":"train"}', encoding="utf-8")
    for name in (
        "provenance.json",
        "data_manifest.json",
        "split_manifest.json",
        "strict_manifest.json",
    ):
        shutil.copyfile(run / name, directory / name)


def _install_training_mocks(monkeypatch, *, seen=None) -> None:
    def train(task, config, frozen):
        if seen is not None:
            seen.append((task, copy.deepcopy(config)))
        return _fake_train(task, config, frozen)

    def evaluate(task, run, device):
        config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        value = 2.0 if config["model"]["type"] == "gru" else 8.0
        _write_evaluation(task, run, model_value=value)
        return {}

    monkeypatch.setattr(pipeline, "_train", train)
    monkeypatch.setattr(pipeline, "_evaluate", evaluate)


def test_full_matrix_separates_candidate_metrics_and_deduplicates_classical(tmp_path, monkeypatch):
    config = _config(tmp_path)
    seen = []
    _install_training_mocks(monkeypatch, seen=seen)
    root = pipeline.run_benchmark_pipeline(config)
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert len(seen) == len(state["runs"]) == 8
    assert {entry["status"] for entry in state["runs"]} == {"evaluated"}
    for task, actual in seen:
        base = yaml.safe_load(Path(config["tasks"][task]).read_text(encoding="utf-8"))
        assert actual["data"] == base["data"]
        assert actual["training"] == base["training"]
        assert actual["model"]["in_channels"] == (9 if task == "paired" else 6)
        assert actual["model"]["out_channels"] == (6 if task == "paired" else 3)
        assert "levels" not in actual["model"]
    frame = pd.read_csv(root / "paired_all_seeds.csv")
    assert len(frame[frame.method == "model"]) == 8
    assert len(frame[frame.method != "model"]) == 24  # 6 baselines * 2 splits * 2 seeds
    summary = pd.read_csv(root / "paired_summary.csv")
    model = summary[(summary.method == "model") & (summary.split == "test")].set_index(
        "candidate_id"
    )
    assert model.loc["gru", "gyro_vector_rmse_radps_mean"] == 2.0
    assert model.loc["resnet", "gyro_vector_rmse_radps_mean"] == 8.0
    assert model.gyro_vector_rmse_radps_count.tolist() == [2, 2]
    assert len(summary[summary.method == "identity"]) == 2
    assert "Retrospective" in (root / "comparison_table.md").read_text(encoding="utf-8")
    assert torch.backends.cuda.math_sdp_enabled()
    assert not torch.backends.cuda.flash_sdp_enabled()


def test_smoke_is_separate_one_epoch_first_seed_and_never_a_comparison(tmp_path, monkeypatch):
    config = _config(tmp_path)
    seen = []
    _install_training_mocks(monkeypatch, seen=seen)
    monkeypatch.setattr(
        pipeline, "_evaluate", lambda *args: pytest.fail("Smoke must skip standalone evaluator")
    )
    smoke = pipeline.run_benchmark_pipeline(config, smoke=True)
    assert smoke.name.startswith("benchmark_smoke_")
    assert len(seen) == 4
    assert {actual["seed"] for _, actual in seen} == {11}
    assert {actual["training"]["epochs"] for _, actual in seen} == {1}
    state = json.loads((smoke / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "smoke_complete"
    assert {entry["status"] for entry in state["runs"]} == {"smoke_trained"}
    assert all("built-in" in entry["smoke_evaluation_note"] for entry in state["runs"])
    assert not (smoke / "paired_summary.csv").exists()
    with pytest.raises(ValueError, match="mix smoke"):
        pipeline.run_benchmark_pipeline(config, resume_root=smoke)


def test_plan_only_does_not_open_missing_datasets_previous_runs_or_cuda(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["device"] = "cuda"
    config["previous_pipeline_root"] = str(tmp_path / "nonexistent_previous_run")
    config["previous_candidate"] = {
        "id": "original",
        "label": "TCN original",
        "fidelity": "original_fixed_candidate",
    }
    shutil.rmtree(tmp_path / "data")
    monkeypatch.setattr(pipeline, "_runtime", lambda *args: pytest.fail("Plan initialized runtime"))
    monkeypatch.setattr(
        pipeline, "_import_previous", lambda *args: pytest.fail("Plan opened previous results")
    )
    root = pipeline.run_benchmark_pipeline(config, plan_only=True)
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "planned"
    assert len(state["runs"]) == 8
    assert (root / "job_contract.json").is_file()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("seeds", [], "seeds"),
        ("seeds", [11, 11], "seeds"),
        ("seeds", [True], "seeds"),
        ("seeds", [-1], "seeds"),
        ("seeds", [2**32], "seeds"),
        ("deterministic", False, "deterministic"),
        ("cpu_threads", 0, "cpu_threads"),
        ("data_manifest_sha256", "wrong", "data_manifest_sha256"),
    ],
)
def test_invalid_matrix_rejected_without_starting(tmp_path, field, value, match):
    config = _config(tmp_path)
    config[field] = value
    with pytest.raises(ValueError, match=match):
        pipeline.run_benchmark_pipeline(config, plan_only=True)
    assert not (tmp_path / "runs").exists()


def test_duplicate_candidate_and_changed_task_interface_rejected(tmp_path):
    config = _config(tmp_path)
    config["candidates"][1]["id"] = config["candidates"][0]["id"]
    with pytest.raises(ValueError, match="unique"):
        pipeline.validate_matrix(config)
    config = _config(tmp_path)
    config["candidates"][0]["model"]["in_channels"] = 17
    with pytest.raises(ValueError, match="cannot change paired"):
        pipeline.run_benchmark_pipeline(config, plan_only=True)


def test_manifest_hash_and_requested_cuda_are_checked_before_training(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["data_manifest_sha256"] = "a" * 64
    monkeypatch.setattr(pipeline, "_train", lambda *args: pytest.fail("Training must not start"))
    with pytest.raises(ValueError, match="manifest SHA256"):
        pipeline.run_benchmark_pipeline(config)
    config = _config(tmp_path)
    config["device"] = "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no silent CPU fallback"):
        pipeline.run_benchmark_pipeline(config)


def test_baseline_difference_aborts_aggregation_without_output_table(tmp_path, monkeypatch):
    config = _config(tmp_path, tasks=("paired",), seeds=(11,))
    _install_training_mocks(monkeypatch)
    root = pipeline.run_benchmark_pipeline(config)
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    run = Path(state["runs"][1]["run_dir"])
    path = run / "evaluation" / "paired_metrics.csv"
    frame = pd.read_csv(path)
    frame.loc[frame.method == "identity", "gyro_vector_rmse_radps"] = 1.1
    frame.to_csv(path, index=False)
    output = tmp_path / "bad_aggregate"
    with pytest.raises(ValueError, match="baseline metrics differ"):
        pipeline.aggregate_benchmark_results(output, state["runs"])
    assert not output.exists()


def test_resume_rechecks_complete_artifacts_and_skips_training(tmp_path, monkeypatch):
    config = _config(tmp_path, seeds=(11,))
    _install_training_mocks(monkeypatch)
    root = pipeline.run_benchmark_pipeline(config)
    monkeypatch.setattr(
        pipeline, "_train", lambda *args: pytest.fail("Completed jobs must not retrain")
    )
    assert pipeline.run_benchmark_pipeline(config, resume_root=root) == root
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    prediction = Path(state["runs"][0]["run_dir"]) / "evaluation" / "test_predictions.parquet"
    prediction.unlink()
    with pytest.raises(FileNotFoundError):
        pipeline.run_benchmark_pipeline(config, resume_root=root)


@pytest.mark.parametrize(
    "changed", ["frozen", "checkpoint", "predictions", "history", "matrix", "source"]
)
def test_resume_rejects_tampering_and_nonfinite_artifacts(tmp_path, monkeypatch, changed):
    config = _config(tmp_path, tasks=("paired",), seeds=(11,))
    _install_training_mocks(monkeypatch)
    root = pipeline.run_benchmark_pipeline(config)
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    entry = state["runs"][0]
    run = Path(entry["run_dir"])
    if changed == "frozen":
        with Path(entry["config"]).open("a", encoding="utf-8") as handle:
            handle.write("\n# changed\n")
    elif changed == "checkpoint":
        path = run / "best.ckpt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["model_state_dict"]["weight"] = torch.tensor([float("nan")])
        torch.save(checkpoint, path)
    elif changed == "predictions":
        path = run / "evaluation" / "test_predictions.parquet"
        frame = pd.read_parquet(path)
        frame.iloc[0, 0] = np.inf
        frame.to_parquet(path, index=False)
    elif changed == "history":
        path = run / "metrics.csv"
        frame = pd.read_csv(path)
        frame["val_grouped_score"] = 0.1
        frame.to_csv(path, index=False)
    elif changed == "matrix":
        config["candidates"][0]["label"] = "Different label"
    else:
        monkeypatch.setattr(
            pipeline, "_source_contract", lambda: {"sha256": "changed", "files": {}}
        )
    with pytest.raises(ValueError):
        pipeline.run_benchmark_pipeline(config, resume_root=root)


def test_failure_records_partial_job_and_resume_does_not_restart_it(tmp_path, monkeypatch):
    config = _config(tmp_path, tasks=("paired",))
    calls = []

    def fail(task, actual, frozen):
        calls.append(actual["seed"])
        _fake_train(task, actual, frozen)
        raise RuntimeError("synthetic optimizer interruption")

    monkeypatch.setattr(pipeline, "_train", fail)
    with pytest.raises(RuntimeError, match="synthetic optimizer"):
        pipeline.run_benchmark_pipeline(config)
    root = next((tmp_path / "runs").iterdir())
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == state["runs"][0]["status"] == "failed"
    assert state["runs"][0]["failed_stage"] == "training"
    assert state["runs"][1]["status"] == "planned"
    assert calls == [11]
    with pytest.raises(RuntimeError, match="Partial run"):
        pipeline.run_benchmark_pipeline(config, resume_root=root)
    assert calls == [11]


def test_resume_can_finish_untouched_planned_jobs_after_a_clean_boundary(tmp_path, monkeypatch):
    config = _config(tmp_path, tasks=("paired",), seeds=(11,))
    _install_training_mocks(monkeypatch)
    root = pipeline.run_benchmark_pipeline(config)
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    removed = state["runs"][1]
    shutil.rmtree(removed["run_dir"])
    removed.update(status="planned", run_dir=None)
    removed.pop("artifacts")
    state["status"] = "running"
    (root / "pipeline.json").write_text(json.dumps(state), encoding="utf-8")
    seen = []
    _install_training_mocks(monkeypatch, seen=seen)
    pipeline.run_benchmark_pipeline(config, resume_root=root)
    assert len(seen) == 1
    assert seen[0][1]["model"]["type"] == "ronin_resnet"


def _make_previous(config: dict, root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    entries = []
    for task, source_path in config["tasks"].items():
        for seed in config["seeds"]:
            actual = yaml.safe_load(Path(source_path).read_text(encoding="utf-8"))
            actual.update(seed=seed, experiment_name=f"formal_{task}_seed{seed}")
            actual["output"]["runs_dir"] = str(root / "experiments")
            frozen = root / "configs" / f"{task}_seed{seed}.yaml"
            frozen.write_text(yaml.safe_dump(actual), encoding="utf-8")
            run = _fake_train(task, actual, frozen)
            _write_evaluation(task, run, model_value=4.0)
            entries.append(
                {
                    "task": task,
                    "seed": seed,
                    "run_dir": str(Path("runs") / root.name / "experiments" / run.name),
                    "config": str(Path("runs") / root.name / "configs" / frozen.name),
                    "status": "evaluated",
                }
            )
    (root / "pipeline.json").write_text(
        json.dumps({"status": "complete", "smoke": False, "runs": entries}), encoding="utf-8"
    )


def test_previous_original_candidate_is_imported_without_retraining(tmp_path, monkeypatch):
    config = _config(tmp_path)
    old = tmp_path / "old_checkout" / "runs" / "formal_training_old"
    _make_previous(config, old)
    config.update(
        previous_pipeline_root=str(old),
        previous_candidate={
            "id": "tcn_original",
            "label": "TCN original",
            "fidelity": "original_fixed_candidate",
        },
    )
    seen = []
    _install_training_mocks(monkeypatch, seen=seen)
    root = pipeline.run_benchmark_pipeline(config)
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    assert len(seen) == 8
    assert len(state["runs"]) == 12
    reused = [entry for entry in state["runs"] if entry["reused"]]
    assert len(reused) == 4
    assert all(Path(entry["run_dir"]).is_relative_to(old) for entry in reused)
    summary = pd.read_csv(root / "paired_summary.csv")
    original = summary[(summary.candidate_id == "tcn_original") & (summary.split == "test")]
    assert original.gyro_vector_rmse_radps_mean.tolist() == [4.0]
    assert original.gyro_vector_rmse_radps_count.tolist() == [2]


@pytest.mark.parametrize("invalid", ["incomplete", "seed", "data", "model", "training"])
def test_previous_candidate_cannot_be_reused_with_incompatible_contract(
    tmp_path, monkeypatch, invalid
):
    config = _config(tmp_path, seeds=(11,))
    old = tmp_path / "old" / "runs" / "formal_old"
    _make_previous(config, old)
    if invalid in {"incomplete", "seed"}:
        path = old / "pipeline.json"
        state = json.loads(path.read_text(encoding="utf-8"))
        if invalid == "incomplete":
            state["status"] = "running"
        else:
            state["runs"][0]["seed"] = 19
        path.write_text(json.dumps(state), encoding="utf-8")
    else:
        path = old / "configs" / "paired_seed11.yaml"
        actual = yaml.safe_load(path.read_text(encoding="utf-8"))
        actual[invalid]["changed"] = True
        path.write_text(yaml.safe_dump(actual), encoding="utf-8")
    config.update(
        previous_pipeline_root=str(old),
        previous_candidate={
            "id": "tcn_original",
            "label": "TCN original",
            "fidelity": "original_fixed_candidate",
        },
    )
    monkeypatch.setattr(
        pipeline, "_train", lambda *args: pytest.fail("Invalid reuse must fail before new training")
    )
    with pytest.raises(ValueError):
        pipeline.run_benchmark_pipeline(config)
