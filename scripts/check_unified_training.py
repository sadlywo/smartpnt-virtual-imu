"""Validate real task labels/loaders and bounded optimizer steps before server training."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.io.training_split_manifest import load_training_split_manifest
from virtual_imu.io.unified_dataset import (
    file_sha256,
    source_state_fingerprint,
    verify_unified_dataset,
)
from virtual_imu.models import make_model
from virtual_imu.training.dataset import IMUWindowDataset
from virtual_imu.training.gt_supervised_dataset import make_smartpnt_gt_split_dataloaders
from virtual_imu.training.unified_data import (
    dataset_balanced_sampler,
    snapshot_unified_training_inputs,
    unified_navigation_frame,
)


def check(config: dict, root: Path, run: Path) -> dict:
    seed = int(config["seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = config.get("device", "auto")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    pair_path, nav_path = root / config["paired_config"], root / config["navigation_config"]
    paired, navigation = [
        yaml.safe_load(p.read_text(encoding="utf-8")) for p in (pair_path, nav_path)
    ]
    verification = verify_unified_dataset(root / paired["data"]["dataset_manifest"], root)
    bundle = load_training_split_manifest(
        Path(paired["data"]["split_manifest"]), Path(paired["data"]["strict_manifest"])
    )
    pair_cfg = paired["data"]
    pair_train = IMUWindowDataset(
        bundle.frames["train"],
        window_size=pair_cfg["window_size"],
        stride=pair_cfg["stride"],
        feature_columns=pair_cfg["feature_columns"],
        target_mode=paired["model"]["target_mode"],
    )
    pair_sets = {"train": pair_train}
    for name in ("validation", "test"):
        pair_sets[name] = IMUWindowDataset(
            bundle.frames[name],
            window_size=pair_cfg["window_size"],
            stride=pair_cfg["stride"],
            feature_columns=pair_cfg["feature_columns"],
            target_mode=paired["model"]["target_mode"],
            feature_stats=pair_train.feature_stats,
            target_stats=pair_train.target_stats,
        )
    sampler = dataset_balanced_sampler(
        bundle.frames["train"],
        window_size=pair_cfg["window_size"],
        stride=pair_cfg["stride"],
        seed=seed,
    )
    pair_loader = DataLoader(pair_train, batch_size=config["batch_size"], sampler=sampler)
    nav_cfg = navigation["data"]
    nav_frame = unified_navigation_frame(nav_cfg)
    nav_loaders = make_smartpnt_gt_split_dataloaders(
        nav_frame,
        {
            "train": nav_cfg["train_sequences"],
            "val": nav_cfg["val_sequences"],
            "test": nav_cfg["test_sequences"],
        },
        window_size=nav_cfg["window_size"],
        stride=nav_cfg["stride"],
        batch_size=config["batch_size"],
        label_mode=nav_cfg["label_mode"],
        normalize_features=True,
        normalize_targets=True,
    )
    results = {}
    for task, train_loader, datasets, model_cfg, data_cfg in (
        ("paired_signal", pair_loader, pair_sets, paired["model"], pair_cfg),
        (
            "navigation",
            nav_loaders["train"],
            {name: loader.dataset for name, loader in nav_loaders.items()},
            navigation["model"],
            nav_cfg,
        ),
    ):
        model = make_model(model_cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        losses = []
        for step, batch in enumerate(train_loader):
            if step >= int(config["optimizer_steps"]):
                break
            x, y = batch[:2]
            x, y = x.to(device), y.to(device)
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Nonfinite actual training tensor in {task}")
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            if task == "navigation":
                prediction = prediction[..., -1]
            if prediction.shape != y.shape:
                raise ValueError(
                    f"Model/label shape mismatch in {task}: {prediction.shape} vs {y.shape}"
                )
            loss = torch.nn.functional.smooth_l1_loss(prediction, y)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite optimization loss in {task}")
            loss.backward()
            if not all(
                torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
            ):
                raise ValueError(f"Nonfinite model gradient in {task}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if len(losses) != int(config["optimizer_steps"]) or not any(
            not torch.equal(before[name], p) for name, p in model.named_parameters()
        ):
            raise ValueError(f"No verified optimizer update in {task}")
        preview = {}
        for name, dataset in datasets.items():
            x, y = dataset[len(dataset) // 2][:2]
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Nonfinite validation/test tensor in {task}/{name}")
            preview[name] = {
                "windows": len(dataset),
                "input_shape": list(x.shape),
                "target_shape": list(y.shape),
            }
        task_run = run / task
        task_run.mkdir()
        provenance = snapshot_unified_training_inputs(data_cfg, task_run)
        (task_run / "normalization.json").write_text(
            json.dumps(
                {
                    "features": datasets["train"].feature_stats,
                    "targets": datasets["train"].target_stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        results[task] = {
            "splits": preview,
            "optimizer_steps": len(losses),
            "losses": losses,
            "finite_gradients": True,
            "parameters_updated": True,
            "provenance": provenance,
        }
    return {
        "passed": True,
        "purpose": "readiness check; optimizer steps do not establish model quality",
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "data_verification": verification,
        "tasks": results,
        "training_configs": {
            str(p.relative_to(root)): file_sha256(p) for p in (pair_path, nav_path)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/check_unified_training.yaml"))
    args = parser.parse_args()
    print(f"READS: {args.config}, declared task configs/manifests/real processed data")
    print("WRITES: seeded bounded readiness run, metrics/report/data/source snapshots")
    print("SKIPS: raw data, full training, model quality claims, checkpoint publication, SSH")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / args.config).read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")  # noqa: UP017
    run = root / config["output_root"] / f"{config['experiment_name']}_{stamp}"
    run.mkdir(parents=True)
    (run / "config.yaml").write_bytes(yaml.safe_dump(config, sort_keys=False).encode())
    results = check(config, root, run)
    source = source_state_fingerprint(root)
    for name, value in (("metrics.json", results), ("source_state.json", source)):
        (run / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    dataset_manifest = (
        root
        / yaml.safe_load((root / config["paired_config"]).read_text(encoding="utf-8"))["data"][
            "dataset_manifest"
        ]
    )
    (run / "data_manifest.json").write_bytes(dataset_manifest.read_bytes())
    lines = [
        "# Unified training readiness",
        "",
        "Actual exported data hashes, task labels and tensor shapes passed.",
        "Two seeded optimizer steps per task passed finite-loss/gradient/parameter-update checks; this does not measure trained-model accuracy.",
        "",
        f"Runtime: {results['runtime']}",
        f"Dataset build: {json.loads(dataset_manifest.read_text(encoding='utf-8'))['build_run']}",
        "",
    ]
    for task, value in results["tasks"].items():
        lines.append(f"- {task}: {value['splits']}")
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"READINESS_RUN: {run.relative_to(root).as_posix()}")
    print(f"PASSED: {results['runtime']} { {k: v['splits'] for k, v in results['tasks'].items()} }")


if __name__ == "__main__":
    main()
