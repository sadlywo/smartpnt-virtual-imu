"""Exercise every frozen benchmark architecture with strict CUDA forward/backward.

Synthetic inputs only; no held-out samples or metrics are read. This is an
operator, shape, finite-gradient and repeatability check, not a model comparison.
"""

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
from virtual_imu.training.benchmark_pipeline import validate_matrix


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        data = tensor.detach().cpu().contiguous()
        if not torch.isfinite(data).all():
            raise ValueError(f"Nonfinite state: {name}")
        digest.update(name.encode())
        digest.update(str((str(data.dtype), tuple(data.shape))).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def trial(model_cfg: dict, train_cfg: dict, task: str, seed: int, batch: int) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(model_cfg).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    inputs = torch.randn(batch, model_cfg["in_channels"], 80, device="cuda")
    target = (
        torch.randn(batch, 6, 80, device="cuda")
        if task == "paired"
        else torch.randn(batch, 3, device="cuda")
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = model(inputs)
    expected = (batch, model_cfg["out_channels"], 80)
    if tuple(output.shape) != expected or not torch.isfinite(output).all():
        raise ValueError(f"Invalid {task} output: {tuple(output.shape)}, expected {expected}")
    if task == "paired":
        loss, _parts = combined_loss(output, target, train_cfg, dt=1.0 / 40)
    else:
        loss = F.smooth_l1_loss(output[:, :, -1], target)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite synthetic loss")
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        raise ValueError(f"Trainable parameters without gradient: {missing}")
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError("Nonfinite synthetic gradient")
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    result = {
        "output_shape": list(output.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": float(loss.detach().cpu()),
        "post_step_state_sha256": state_hash(model.state_dict()),
        "output_sha256": state_hash({"output": output}),
        "elapsed_synthetic_step_seconds": elapsed,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
        "missing_gradients": missing,
    }
    model.eval()
    with torch.inference_mode():
        evaluation = model(inputs)
    if not torch.isfinite(evaluation).all():
        raise ValueError("Nonfinite synthetic inference")
    del model, optimizer, inputs, target, output, loss, evaluation
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(
        f"READS: {args.config}, task YAMLs, source models, CUDA runtime; synthetic tensors only",
        flush=True,
    )
    print(f"WRITES: {args.report}; transient GPU tensors only", flush=True)
    print(
        "SKIPS: dataset samples, validation/test metrics, production checkpoints, training runs and hyperparameter selection",
        flush=True,
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_matrix(config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    report = {
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "matrix_config_sha256": sha256_file(args.config),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "scope": "Synthetic operator/determinism preflight, not scientific results or throughput benchmark",
        "cases": [],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        raise FileExistsError(f"Preflight report already exists: {args.report}")
    try:
        for candidate in config["candidates"]:
            for task in candidate["tasks"]:
                source = yaml.safe_load(Path(config["tasks"][task]).read_text(encoding="utf-8"))
                model_cfg = dict(candidate["model"])
                model_cfg.update(
                    in_channels=9 if task == "paired" else 6,
                    out_channels=6 if task == "paired" else 3,
                )
                batch = int(source["data"]["batch_size"])
                first = trial(model_cfg, source["training"], task, config["seeds"][0], batch)
                second = trial(model_cfg, source["training"], task, config["seeds"][0], batch)
                if any(
                    first[key] != second[key]
                    for key in ("post_step_state_sha256", "output_sha256", "loss")
                ):
                    raise ValueError(
                        f"Exact same-seed repeatability failed: {candidate['id']}/{task}"
                    )
                report["cases"].append(
                    {
                        "candidate_id": candidate["id"],
                        "task": task,
                        "batch_size": batch,
                        "model_config": model_cfg,
                        "first": first,
                        "second": second,
                        "strict_repeatability": True,
                    }
                )
                print(
                    f"PASSED: {candidate['id']}/{task} batch={batch} parameters={first['parameters']} peakGPU={first['peak_allocated_gpu_bytes']} bytes",
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
    print(f"PREFLIGHT_PASSED: {len(report['cases'])} task/model cases repeated exactly", flush=True)


if __name__ == "__main__":
    main()
