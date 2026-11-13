"""Real TRAIN/VALID CUDA finite-gradient and throughput checks; never TEST."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from torch.nn import functional as F

from virtual_imu.ablations.data import balanced_order, grouped_score
from virtual_imu.ablations.training import write_json
from virtual_imu.replacements.training import configure, data_for_job, predict_normalized
from virtual_imu.utils.seed import seed_everything

LOG = logging.getLogger(__name__)


def preflight(root, cfg, run_dir):
    from virtual_imu.replacements.models import build_replacement_model

    device = configure(cfg)
    job = {"root": str(root), "protocol": "pooled", "config": cfg}
    source, train, val, (tx, ty, vx, _vy, _dense) = data_for_job(job, device)
    seed = cfg["seeds"][0]
    order = balanced_order(train, seed, 1)
    ix = torch.as_tensor(order[: cfg["data"]["batch_size"]], device=device)
    x, y = tx[ix], ty[ix]
    results = []
    for variant in cfg["variants"]:
        seed_everything(seed)
        model, meta = build_replacement_model(
            variant, seed, cfg["model"], cfg["replacement_models"]
        )
        model = model.to(device).train()
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
            betas=tuple(cfg["training"]["betas"]),
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        # Full precision first establishes a usable finite graph independently
        # of transient GradScaler overflow during the later real AMP training.
        opt.zero_grad(set_to_none=True)
        output = model(x)
        loss = F.smooth_l1_loss(output[..., 140:180], y)
        assert torch.isfinite(output).all() and torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        amp = cfg["training"]["amp"] and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp, init_scale=1024.0)
        durations = []
        valid_steps = 0
        for step in range(5):
            opt.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=amp):
                out = model(x)
                loss = F.smooth_l1_loss(out[..., 140:180], y)
            assert torch.isfinite(out).all() and torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=not amp)
            old = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            valid_steps += int(scaler.get_scale() >= old)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if step >= 2:
                durations.append(1000 * (time.perf_counter() - begin))
        assert valid_steps > 0
        normal = predict_normalized(
            model,
            vx,
            window=320,
            max_window=320,
            block_size=40,
            batch_size=cfg["evaluation"]["batch_size"],
            amp=amp,
        )
        score = grouped_score(val.target_frame, val.physical(normal))
        assert np.isfinite(score)
        results.append(
            dict(
                variant=variant,
                parameters=meta["num_params"],
                batch32_train_step_ms=float(np.median(durations)),
                peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                finite_gradient=True,
                amp_successful_steps=valid_steps,
                full_VALID_score_after_smoke=score,
                initialization=meta,
            )
        )
        LOG.info(
            "PREFLIGHT %s parameters=%d train_step_ms=%.2f VALIDfinite=%s",
            variant,
            meta["num_params"],
            np.median(durations),
            np.isfinite(score),
        )
        del model, opt, output, out, loss, normal, grads
    assert "test" not in source.loaded_roles
    receipt = dict(
        status="PASS",
        real_sources=True,
        loaded_roles=source.loaded_roles,
        TEST_loaded=False,
        scientific_fits_performed=0,
        warning="Smoke optimizer steps are not formal fits or result estimates",
        device=str(device),
        models=results,
        train_anchors=len(train),
        validation_anchors=len(val),
    )
    write_json(run_dir / "preflight.json", receipt)
    LOG.info("WRITES all11 model CUDA checks; SKIPS TEST and formal fit results")
    return receipt
