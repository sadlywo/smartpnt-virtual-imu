"""Validation isolation, fair budgets and restart-safe reviewer HPO."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

from virtual_imu.training.reviewer_hpo import (
    build_trial_config,
    deterministic_candidates,
    read_validation_score,
    run_reviewer_hpo,
)


def matrix(tmp_path: Path) -> dict:
    source = {
        "data": {"dataset_manifest": "data/never_read.json", "window_size": 80},
        "model": {"type": "tcn"}, "training": {"loss": "huber"},
    }
    (tmp_path / "paired.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")
    return {
        "seed": 100, "confirmation_seed": 101, "seeds": [3, 4],
        "device": "cpu", "deterministic": True, "tasks": {"paired": "paired.yaml"},
        "protocols": [{"id": "alimu_only", "task": "paired"}],
        "candidates": [{"id": "tcn", "tasks": ["paired"], "model": {"type": "tcn", "hidden_channels": 32}, "search_space": {"model.hidden_channels": {"choice": [16, 32, 64]}}}],
        "search": {
            "trials": 4, "rung_epochs": [1, 2, 3], "rung_keep": [4, 2, 1],
            "common_space": {"training.lr": {"loguniform": [1e-4, 3e-3]}, "loss_profile": {"choice": ["mse_only", "huber_only", "composite"]}},
            "paired_space": {"data.window_size": {"choice": [80, 160, 320]}},
        },
    }


class Backend:
    def __init__(self, fail_first: bool = False):
        self.trained = []
        self.evaluated = []
        self.fail_first = fail_first

    def train(self, task: str, config: dict, frozen: Path) -> Path:
        assert config["data"]["load_splits"] == ["train", "validation"]
        assert config["training"]["evaluate_test"] is False
        assert config["training"]["disable_test_evaluation"] is True
        self.trained.append(copy.deepcopy(config))
        if self.fail_first and len(self.trained) == 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        directory = Path(config["output"]["runs_dir"]) / "fit"
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (directory / "best.ckpt").write_bytes(b"fake checkpoint for injected validation backend")
        (directory / "metrics.csv").write_text("epoch,val_grouped_score\n1,0.2\n", encoding="utf-8")
        (directory / "provenance.json").write_text('{"fit_split":"train"}\n', encoding="utf-8")
        return directory

    def validate(self, directory: Path, config: dict) -> dict:
        assert yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8")) == config
        return {"score": config["model"]["hidden_channels"] / 100 + config["training"]["lr"], "best_epoch": 1, "metric": "val_grouped_score"}

    def evaluate(self, task: str, directory: Path, config: dict) -> dict:
        state = json.loads((directory.parents[2] / "pipeline.json").read_text(encoding="utf-8"))
        assert all("selected" in cell for cell in state["cells"].values())
        assert state["status"] == "finalizing"
        assert config["reviewer_protocol"]["phase"].startswith("final_")
        self.evaluated.append(config)
        evaluation = directory / "evaluation"
        evaluation.mkdir()
        (evaluation / "test_predictions.parquet").write_bytes(b"prediction fixture")
        (evaluation / "metrics.json").write_text('{"test_rmse":1.0}\n', encoding="utf-8")
        return {"test_rmse": 1.0}


def test_deterministic_search_has_frozen_default_and_unique_seeded_candidates(tmp_path: Path):
    config = matrix(tmp_path)
    candidate = config["candidates"][0]
    trials = deterministic_candidates(config, candidate, "paired")
    assert trials == deterministic_candidates(config, candidate, "paired")
    assert len(trials) == 4 and len({json.dumps(t["parameters"], sort_keys=True) for t in trials}) == 4
    default = trials[0]
    assert default["is_default"] and default["parameters"]["model.hidden_channels"] == 32
    assert default["parameters"]["training.lr"] == 0.001
    assert default["parameters"]["data.window_size"] == 80
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    assert deterministic_candidates(changed, candidate, "paired")[1:] != trials[1:]


def test_build_preserves_six_channel_target_and_validation_only_contract(tmp_path: Path):
    config = matrix(tmp_path)
    source = yaml.safe_load((tmp_path / "paired.yaml").read_text(encoding="utf-8"))
    actual = build_trial_config(config, source, config["protocols"][0], config["candidates"][0], {"loss_profile": "mse_only", "data.window_size": 320}, seed=3, epochs=80, phase="final", output_root=tmp_path / "fit")
    assert actual["model"]["target_mode"] == "residual" and actual["model"]["in_channels"] == 6
    assert len(actual["data"]["feature_columns"]) == 6
    assert actual["training"]["checkpoint_metric"] == "val_grouped_score"
    assert actual["training"]["gyro_signal_weight"] == 1
    assert actual["training"]["spectral_weight"] == 0
    assert actual["data"]["load_splits"] == ["train", "validation"]
    assert source["model"] == {"type": "tcn"}


def test_plan_only_never_imports_torch_or_reads_data(tmp_path: Path, monkeypatch):
    config = matrix(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", None)
    directory = run_reviewer_hpo(config, root=tmp_path, plan_only=True)
    state = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "planned"
    assert len(state["jobs"]) == 4
    assert all(job["status"] == "planned" for job in state["jobs"].values())
    assert len(list((directory / "configs").glob("*.yaml"))) == 4


def test_halving_confirmation_finalization_and_verified_resume(tmp_path: Path):
    config = matrix(tmp_path)
    backend = Backend()
    directory = run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate)
    state = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete" and state["counts"]["final_evaluated"] == 2
    assert len(backend.trained) == 10 and len(backend.evaluated) == 2
    search_epochs = [c["training"]["epochs"] for c in backend.trained if c["reviewer_protocol"]["phase"] == "search"]
    assert search_epochs == [1, 1, 1, 1, 2, 2, 3]
    run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate, resume_root=directory)
    assert len(backend.trained) == 10 and len(backend.evaluated) == 2
    first = next(iter(state["jobs"].values()))
    Path(first["run_dir"], "best.ckpt").write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="artifact checksum"):
        run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate, resume_root=directory)


def test_candidate_failure_visible_but_other_search_candidates_continue(tmp_path: Path):
    config = matrix(tmp_path)
    backend = Backend(fail_first=True)
    directory = run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate)
    state = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete" and state["counts"]["failed_training_jobs"] == 1
    failed = [job for job in state["jobs"].values() if job["status"] == "failed"]
    assert "out of memory" in failed[0]["error"]
    assert state["counts"]["final_evaluated"] == 2
    assert failed[0]["failure_recoverable"] is True
    assert "gc_collected" in failed[0]["resource_cleanup"]


@pytest.mark.parametrize("error", [OSError("disk full"), ValueError("checkpoint config mismatch"), RuntimeError("unexpected model shape mismatch")])
def test_io_integrity_and_unexpected_model_errors_abort_search(tmp_path: Path, error):
    config = matrix(tmp_path)
    calls = []

    def broken_train(*args):
        calls.append(args)
        raise error

    with pytest.raises(type(error), match=str(error)):
        run_reviewer_hpo(config, root=tmp_path, train_backend=broken_train)
    assert len(calls) == 1
    directory = next((tmp_path / "runs").glob("reviewer_hpo_*"))
    state = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    job = next(iter(state["jobs"].values()))
    assert job["status"] == "failed" and job["failure_recoverable"] is False


def test_checkpoint_validation_failure_is_not_an_hpo_candidate_failure(tmp_path: Path):
    config = matrix(tmp_path)
    backend = Backend()

    def corrupt_checkpoint(*args):
        raise ValueError("checkpoint history mismatch")

    with pytest.raises(ValueError, match="history mismatch"):
        run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=corrupt_checkpoint)
    assert len(backend.trained) == 1


@pytest.mark.parametrize("error_type", [MemoryError, FloatingPointError])
def test_known_resource_and_nonfinite_optimization_failures_continue(tmp_path: Path, error_type):
    config = matrix(tmp_path)
    backend = Backend()
    attempts = []

    def transient_training(*args):
        attempts.append(args)
        if len(attempts) == 1:
            raise error_type("recoverable optimization failure")
        return backend.train(*args)

    directory = run_reviewer_hpo(config, root=tmp_path, train_backend=transient_training, validation_backend=backend.validate, evaluate_backend=backend.evaluate)
    state = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete" and state["counts"]["failed_training_jobs"] == 1


@pytest.mark.parametrize("tamper", ["selection_file", "rehashed_selected_score"])
def test_resume_rejects_selection_file_or_validation_rank_tampering(tmp_path: Path, tamper):
    config = matrix(tmp_path)
    backend = Backend()
    directory = run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate)
    pipeline = directory / "pipeline.json"
    state = json.loads(pipeline.read_text(encoding="utf-8"))
    cell_id, cell = next(iter(state["cells"].items()))
    path = directory / "selections" / f"{cell_id.replace('/', '__')}.json"
    if tamper == "selection_file":
        path.write_text("{}\n", encoding="utf-8")
    else:
        cell["selected"]["mean_two_seed_validation_score"] += 1
        path.write_text(json.dumps(cell["selected"]), encoding="utf-8")
        cell["selection_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        pipeline.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match=r"selection|Selected winner"):
        run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate, resume_root=directory)


@pytest.mark.parametrize("artifact", ["metrics.csv", "evaluation/test_predictions.parquet"])
def test_resume_rejects_changed_history_or_actual_prediction_payload(tmp_path: Path, artifact):
    config = matrix(tmp_path)
    backend = Backend()
    directory = run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate)
    state = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
    job = next(job for job in state["jobs"].values() if job.get("evaluation_status") == "complete")
    path = Path(job["run_dir"]) / artifact
    path.write_bytes(b"tampered payload")
    with pytest.raises(ValueError, match="artifact checksum"):
        run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate, resume_root=directory)


def test_full_budget_defaults_are_evaluated_after_search_with_explicit_variant(tmp_path: Path):
    config = matrix(tmp_path)
    config["comparison"] = {"default_refits": True}
    backend = Backend()
    directory = run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=backend.evaluate)
    state = json.loads((directory / "pipeline.json").read_text(encoding="utf-8"))
    assert state["counts"]["expected_final_runs"] == 4
    assert state["counts"]["planned_total_training_jobs"] == 12
    assert state["counts"]["final_evaluated"] == 4
    assert [c["reviewer_protocol"]["variant"] for c in backend.evaluated] == ["default", "default", "tuned", "tuned"]
    assert all(c["training"]["epochs"] == 3 for c in backend.evaluated)
    assert all(c["training"]["lr"] == 0.001 for c in backend.evaluated[:2])


def test_no_pipeline_completion_when_final_evaluation_fails(tmp_path: Path):
    config = matrix(tmp_path)
    backend = Backend()

    def failed_evaluation(task: str, directory: Path, actual: dict) -> dict:
        raise RuntimeError("incomplete common test coverage")

    with pytest.raises(RuntimeError, match="common test coverage"):
        run_reviewer_hpo(config, root=tmp_path, train_backend=backend.train, validation_backend=backend.validate, evaluate_backend=failed_evaluation)
    directories = list((tmp_path / "runs").glob("reviewer_hpo_*"))
    state = json.loads((directories[0] / "pipeline.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["counts"]["final_evaluated"] == 0
    assert any(job.get("evaluation_status") == "failed" for job in state["jobs"].values())


def test_interrupted_partial_optimization_requires_explicit_replacement(tmp_path: Path):
    config = matrix(tmp_path)
    directory = run_reviewer_hpo(config, root=tmp_path, plan_only=True)
    path = directory / "pipeline.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    next(iter(state["jobs"].values()))["status"] = "running"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Interrupted partial optimization"):
        run_reviewer_hpo(config, root=tmp_path, resume_root=directory)


def test_validation_reader_rejects_checkpoint_history_disagreement(tmp_path: Path):
    import torch

    config = {"model": {"type": "tcn"}, "training": {"epochs": 3, "checkpoint_metric": "val_grouped_score"}}
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    checkpoint = {"config": config, "model_config": config["model"], "model_state_dict": {"weight": torch.ones(2)}, "best_epoch": 2, "best_metric": 0.2, "checkpoint_metric": "val_grouped_score"}
    torch.save(checkpoint, tmp_path / "best.ckpt")
    (tmp_path / "metrics.csv").write_text("epoch,val_grouped_score\n1,0.3\n2,0.2\n3,0.25\n", encoding="utf-8")
    assert read_validation_score(tmp_path, config)["score"] == 0.2
    (tmp_path / "metrics.csv").write_text("epoch,val_grouped_score\n1,0.1\n2,0.2\n3,0.25\n", encoding="utf-8")
    with pytest.raises(ValueError, match="minimum saved validation"):
        read_validation_score(tmp_path, config)
