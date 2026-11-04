"""Verify declared reviewer-HPO model/operator coverage on the actual CUDA host."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
import yaml
from torch.nn import functional as F

from virtual_imu.io.training_bundle import sha256_file
from virtual_imu.models import make_model
from virtual_imu.models.losses import combined_loss
from virtual_imu.training.reviewer_hpo import build_trial_config, deterministic_candidates


def fingerprint(state: dict) -> str:
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite state tensor: {name}")
        result.update(name.encode())
        result.update(str((str(value.dtype), tuple(value.shape))).encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def trial(actual: dict, task: str, seed: int) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(actual["model"]).cuda().train()
    batch, length = actual["data"]["batch_size"], actual["data"]["window_size"]
    inputs = torch.randn(batch, 6, length, device="cuda")
    channels = 6 if task == "paired" else 3
    target = torch.randn(batch, channels, length, device="cuda")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=actual["training"]["lr"],
        weight_decay=actual["training"]["weight_decay"],
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(inputs)
    if tuple(output.shape) != (batch, channels, length) or not torch.isfinite(output).all():
        raise ValueError("Invalid model output shape or values")
    if task == "paired":
        loss, _ = combined_loss(output, target, actual["training"], dt=0.025)
    else:
        loss = (
            F.mse_loss(output[:, :, -1], target[:, :, -1])
            if actual["training"]["loss"] == "mse"
            else F.smooth_l1_loss(output[:, :, -1], target[:, :, -1])
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite synthetic preflight loss")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and (
            parameter.grad is None or not torch.isfinite(parameter.grad).all()
        ):
            raise FloatingPointError(f"Missing or nonfinite gradient: {name}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        evaluation = model(inputs)
    result = {
        "loss": float(loss.detach().cpu()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "state_sha256": fingerprint(model.state_dict()),
        "training_output_sha256": fingerprint({"output": output}),
        "evaluation_output_sha256": fingerprint({"output": evaluation}),
        "synthetic_step_seconds": elapsed,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
    }
    del model, optimizer, inputs, target, output, evaluation, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"READS: {args.config}, source model/training YAMLs, synthetic tensors and CUDA hardware",
        flush=True,
    )
    print(f"WRITES: {args.report}; transient GPU state only", flush=True)
    print(
        "SKIPS: actual dataset payloads, validation/test scores, HPO ranking and production runs",
        flush=True,
    )
    if args.report.exists():
        raise FileExistsError(f"Preflight output already exists: {args.report}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("Reviewer preflight requires actual CUDA")
    torch.set_num_threads(config.get("cpu_threads", 4))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    report = {
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "matrix_sha256": sha256_file(args.config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "cases": [],
        "scope": "Synthetic operator/gradient and exact same-seed repeatability check; no scientific accuracy or inference-speed conclusion",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        for candidate in config["candidates"]:
            for task in candidate["tasks"]:
                source = yaml.safe_load(Path(config["tasks"][task]).read_text(encoding="utf-8"))
                protocol = {
                    "id": "penn_only" if task == "paired" else "navigation_rotation_0",
                    "task": task,
                }
                choices = deterministic_candidates(config, candidate, task)
                stress = max(
                    choices[1:] or choices,
                    key=lambda value: (
                        int(value["parameters"].get("data.window_size", 80))
                        * int(value["parameters"].get("data.batch_size", 128))
                        * int(
                            value["parameters"].get(
                                "model.d_model",
                                value["parameters"].get(
                                    "model.hidden_channels",
                                    value["parameters"].get("model.hidden_size", 64),
                                ),
                            )
                        )
                        * int(
                            value["parameters"].get(
                                "model.num_layers", value["parameters"].get("model.levels", 4)
                            )
                        )
                    ),
                )
                cases = [
                    ("default", choices[0]["parameters"]),
                    ("sampled_stress", stress["parameters"]),
                ]
                if task == "paired" and candidate["id"] in {
                    "transformer",
                    "informer",
                    "patchtst",
                    "itransformer",
                }:
                    for length in (160, 320):
                        cases.append(
                            (
                                f"context_{length}",
                                {**choices[0]["parameters"], "data.window_size": length},
                            )
                        )
                for name, parameters in cases:
                    actual = build_trial_config(
                        config,
                        source,
                        protocol,
                        candidate,
                        parameters,
                        seed=config["seed"],
                        epochs=8,
                        phase="operator_preflight",
                        output_root=Path("runs/synthetic_preflight"),
                    )
                    first, second = (
                        trial(actual, task, config["seed"]),
                        trial(actual, task, config["seed"]),
                    )
                    for key in (
                        "state_sha256",
                        "training_output_sha256",
                        "evaluation_output_sha256",
                        "loss",
                    ):
                        if first[key] != second[key]:
                            raise ValueError(
                                f"Strict same-seed repeatability failed: {candidate['id']}/{task}/{name}/{key}"
                            )
                    report["cases"].append(
                        {
                            "candidate_id": candidate["id"],
                            "task": task,
                            "case": name,
                            "model_config": actual["model"],
                            "batch_size": actual["data"]["batch_size"],
                            "context_points": actual["data"]["window_size"],
                            "first": first,
                            "second": second,
                            "status": "passed",
                        }
                    )
                    print(
                        f"PASSED: {candidate['id']}/{task}/{name} length={actual['data']['window_size']} batch={actual['data']['batch_size']}",
                        flush=True,
                    )
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        args.report.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(
        f"REVIEWER_PREFLIGHT_PASSED: {len(report['cases'])} repeated model/task/context configurations",
        flush=True,
    )


if __name__ == "__main__":
    main()
