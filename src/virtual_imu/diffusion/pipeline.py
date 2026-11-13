"""Single-dataset, separate-pair HPO and locked held-out evaluation."""

from __future__ import annotations

import json
import logging
import platform
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from virtual_imu.config import load_config, save_config
from virtual_imu.diffusion.data import PairedSource, PairedWindows, sha256
from virtual_imu.diffusion.evaluation import comparison_table, markdown_table, physical_prediction
from virtual_imu.diffusion.training import choose_device, fit_trial, load_trained
from virtual_imu.models.baseline_affine import AffineCalibration
from virtual_imu.viz.plots_diffusion import plot_diffusion_aggregate, plot_diffusion_results

LOG = logging.getLogger(__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def capture_provenance(root: Path, run: Path, config_path: Path, device) -> dict:
    snapshot = run / "source_snapshot"
    hashes = {}
    for parent in ("src/virtual_imu", "scripts", "configs", "tests"):
        for path in (root / parent).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".yaml"}:
                relative = path.relative_to(root)
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                hashes[relative.as_posix()] = sha256(path)

    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True)
        return result.stdout.decode("utf-8", errors="replace").strip()

    context = {
        "run_dir": str(run.resolve()),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("branch", "--show-current"),
        "git_status": git("status", "--short"),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "source_sha256": hashes,
        "data_manifest_sha256": sha256(root / load_config(config_path)["data"]["manifest"]),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    write_json(run / "provenance.json", context)
    (run / "working_tree.patch").write_text(git("diff", "HEAD"), encoding="utf-8")
    return context


def evaluate_pair(config, pair_run, source, train, validation, selected, context):
    checkpoint_path = Path(selected["checkpoint"])
    device = choose_device(config["training"]["device"])
    model, checkpoint = load_trained(checkpoint_path, device)
    if checkpoint["stats"] != train.stats:
        raise ValueError("Checkpoint normalization differs from train-only data statistics")
    affine = AffineCalibration().fit(train.low, train.high)
    write_json(
        pair_run / "affine_state.json",
        {"matrix": affine.matrix_.tolist(), "bias": affine.bias_.tolist()},
    )
    reports = {}
    for role in ("validation", "test"):
        if role == "validation":
            windows = validation
        else:
            # First test signal read occurs after selection.json has been persisted.
            frame = source.load("test")
            windows = PairedWindows(
                frame, config["data"]["window_size"], config["data"]["eval_stride"], train.stats
            )
        LOG.info(
            "READS locked checkpoint=%s split=%s windows=%d; WRITES predictions and metrics",
            checkpoint_path,
            role,
            len(windows),
        )
        prediction, mask = physical_prediction(
            model,
            windows,
            device,
            config["data"]["batch_size"],
            int(config["seed"]) + 20000,
            int(config["evaluation"]["draws"]),
            bool(config["training"]["amp"]) and device.type == "cuda",
        )
        frame = windows.frame.loc[mask].copy()
        low, high, pred = windows.low[mask], windows.high[mask], prediction[mask]
        methods = {"raw": low, "affine": affine.predict(low), "ddpm": pred}
        for i, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
            frame[f"{axis}_virtual"] = pred[:, i]
            frame[f"{axis}_affine"] = methods["affine"][:, i]
        table = comparison_table(low, high, methods)
        table.insert(0, "pair_id", source.records[0]["pair_id"])
        table.insert(1, "split", role)
        destination = pair_run / role
        destination.mkdir(exist_ok=True)
        frame.to_parquet(destination / "predictions.parquet", index=False)
        table.to_csv(destination / "metrics.csv", index=False)
        (destination / "metrics.md").write_text(markdown_table(table), encoding="utf-8")
        per_record = []
        for sequence, group in frame.groupby("sequence_id", sort=False):
            indexes = frame.index.get_indexer(group.index)
            t = comparison_table(
                low[indexes], high[indexes], {k: v[indexes] for k, v in methods.items()}
            )
            t.insert(0, "sequence_id", sequence)
            per_record.append(t)
        pd.concat(per_record).to_csv(destination / "per_record_metrics.csv", index=False)
        metrics_context = {
            "run_dir": str(pair_run.resolve()),
            "provenance": "../provenance.json",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "data_manifest": "data_manifest.json",
            "config": "config.yaml",
            "report": "report.md",
            "split": role,
            "eligible_rows": int(mask.sum()),
            "excluded_short_sequence_rows": int((~mask).sum()),
            "draws": config["evaluation"]["draws"],
            "metrics": table.to_dict("records"),
        }
        write_json(destination / "metrics.json", metrics_context)
        LOG.info("SKIPS %d unsupported short-sequence rows for %s", (~mask).sum(), role)
        reports[role] = table
        if role == "test":
            search = json.loads((pair_run / "search_results.json").read_text(encoding="utf-8"))
            plot_diffusion_results(pair_run, frame, table, search, source.manifest["rate_hz"])
    test = reports["test"]
    subset = test[test.component.isin(["gyro", "accel"])][
        ["component", "method", "unit", "rmse", "mae", "rmse_improvement_pct", "snr_gain_db"]
    ]
    text = [
        f"# {source.records[0]['pair_id']} conditional DDPM experiment",
        "",
        f"Source of truth: `{pair_run.resolve()}`",
        "",
        f"Git commit: `{context['git_commit']}` (exact source snapshot and file hashes saved in parent).",
        f"Selected: `{selected['trial']}`, epoch {selected['best_epoch']}, "
        f"validation relative RMSE {selected['validation_score']:.6f}.",
        "",
        markdown_table(subset),
        "RMSE reduction = 100 * (1 - method RMSE / raw RMSE). Negative values mean degradation. "
        "Group RMSE averages squared component errors, without mixing gyro and acceleration units.",
        "",
        "Training uses Adam and train-only feature/residual statistics. Hyperparameters and early "
        "stopping use fixed-seed full DDPM sampling on the saved validation subset. Full validation "
        "and test results use the locked checkpoint with overlap-averaged predictions and "
        f"{config['evaluation']['draws']} Monte Carlo draws per window. "
        "Test data never selects a candidate or an epoch.",
        "",
        f"The sampler executes all {config['diffusion']['timesteps']} ancestral DDPM steps. "
        f"The training parameterization is `{config['diffusion']['prediction_type']}`. "
        "Bidirectional windows are an offline mapping, not a causal navigation model.",
        "",
        "VN100 is the measured reference, not an error-free ground truth. ALiMU uses upstream "
        "corresponding sensor axes; extrinsics have not been independently verified. This run "
        "contains one split and one seed, so results do not establish cross-recording robustness.",
        "",
        "Artifacts: [selection](selection.json), [config](config.yaml), "
        "[manifest](data_manifest.json), [test metrics](test/metrics.csv), "
        "[per-record metrics](test/per_record_metrics.csv), "
        "[predictions](test/predictions.parquet), [figures](figures/figure_index.md).",
        "",
    ]
    (pair_run / "report.md").write_text("\n".join(text), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test


def run_experiment(
    config_path: Path, root: Path, run_id: str | None = None, resume: bool = False
) -> Path:
    config_path = config_path.resolve()
    config = load_config(config_path)
    device = choose_device(config["training"]["device"])
    torch.set_num_threads(int(config["training"].get("cpu_threads", 4)))
    identifier = run_id or f"{config['experiment_id']}_{datetime.now():%Y%m%d_%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("run-id must contain only letters, digits, underscores or hyphens")
    run = root / config["output"]["runs_dir"] / identifier
    if run.exists() and not resume:
        raise FileExistsError(f"Run exists; use --resume to reuse completed trials: {run}")
    run.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(run / "experiment.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        LOG.info(
            "READS %s; WRITES %s; SKIPS unrelated datasets and outputs/latest", config_path, run
        )
        provenance_path = run / "provenance.json"
        if provenance_path.exists():
            context = json.loads(provenance_path.read_text(encoding="utf-8"))
            if context["config_sha256"] != sha256(config_path):
                raise ValueError("Resume config differs from original run")
            if context.get("data_manifest_sha256") != sha256(root / config["data"]["manifest"]):
                raise ValueError("Resume data manifest differs from original run")
            for relative, digest in context["source_sha256"].items():
                if sha256(root / relative) != digest:
                    raise ValueError(
                        f"Resume source differs from snapshot: {relative}; use a new run-id"
                    )
        else:
            context = capture_provenance(root, run, config_path, device)
            save_config(config, run / "config.yaml")
        if context["data_manifest_sha256"] != sha256(root / config["data"]["manifest"]):
            raise ValueError("Run data manifest differs from its frozen initial version")
        LOG.info("CUDA=%s torch=%s GPU=%s", device, torch.__version__, context["gpu"])
        all_tables = []
        for pair_id in config["data"]["pairs"]:
            if context["data_manifest_sha256"] != sha256(root / config["data"]["manifest"]):
                raise ValueError("Run data manifest differs before processing a pair")
            pair_run = run / pair_id
            pair_run.mkdir(exist_ok=True)
            if (pair_run / "complete.json").exists():
                LOG.info("SKIPS completed pair=%s", pair_id)
                all_tables.append(pd.read_csv(pair_run / "test" / "metrics.csv"))
                continue
            source = PairedSource(root, config["data"], pair_id)
            train_frame, val_frame = source.load("train"), source.load("validation")
            train = PairedWindows(
                train_frame, config["data"]["window_size"], config["data"]["stride"]
            )
            val = PairedWindows(
                val_frame, config["data"]["window_size"], config["data"]["eval_stride"], train.stats
            )
            save_config(config, pair_run / "config.yaml")
            write_json(pair_run / "normalization.json", train.stats)
            manifest_digest = sha256(source.path)
            manifest_path = pair_run / "data_manifest.json"
            if manifest_path.exists():
                old = json.loads(manifest_path.read_text(encoding="utf-8"))
                if old["manifest_sha256"] != manifest_digest:
                    raise ValueError("Data manifest changed since original run")
            write_json(
                manifest_path,
                {
                    "run_dir": str(pair_run.resolve()),
                    "provenance": "../provenance.json",
                    "report": "report.md",
                    "manifest": str(source.path),
                    "manifest_sha256": manifest_digest,
                    "records": source.records,
                    "train_windows": len(train),
                    "validation_windows": len(val),
                    "split_policy": "upstream physical-recording-disjoint",
                },
            )
            summaries = []
            for i, trial in enumerate(config["search"]["trials"]):
                trial_run = pair_run / "trials" / f"trial_{i:03d}"
                summary_path = trial_run / "summary.json"
                if summary_path.exists():
                    LOG.info("SKIPS completed trial %s", trial_run)
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    if trial_run.exists():
                        # Keep an interrupted attempt for audit; restart that trial from its seed.
                        archive = trial_run.with_name(
                            trial_run.name + f"_interrupted_{datetime.now():%H%M%S_%f}"
                        )
                        trial_run.rename(archive)
                        LOG.info(
                            "WRITES interrupted attempt archive=%s; restarts trial from seed",
                            archive,
                        )
                    summary = fit_trial(
                        config,
                        trial,
                        train,
                        val,
                        trial_run,
                        {**context, "pair_id": pair_id, "manifest_sha256": manifest_digest},
                    )
                summaries.append(summary)
                write_json(pair_run / "search_results.json", summaries)
                pd.json_normalize(summaries).to_csv(pair_run / "search_results.csv", index=False)
            selected = min(summaries, key=lambda row: row["validation_score"])
            write_json(
                pair_run / "selection.json",
                {
                    **selected,
                    "test_used_for_selection": False,
                    "selection_metric": "mean gyro/accel relative validation RMSE",
                },
            )
            table = evaluate_pair(config, pair_run, source, train, val, selected, context)
            all_tables.append(table)
            write_json(pair_run / "complete.json", {"status": "complete", "pair_id": pair_id})
        aggregate = pd.concat(all_tables, ignore_index=True)
        aggregate.to_csv(run / "comparison.csv", index=False)
        display = aggregate[aggregate.component.isin(["gyro", "accel"])][
            [
                "pair_id",
                "component",
                "method",
                "unit",
                "rmse",
                "rmse_improvement_pct",
                "snr_gain_db",
            ]
        ]
        (run / "comparison.md").write_text(markdown_table(display), encoding="utf-8")
        plot_diffusion_aggregate(aggregate, run / "figures")
        (run / "report.md").write_text(
            "# ALiMU conditional DDPM + Transformer\n\nThree independently tuned sensor-pair models.\n\n"
            + markdown_table(display)
            + "\nEach model uses the existing recording-disjoint split, train-only normalization, "
            "Adam, validation early stopping and full ancestral DDPM sampling. "
            "Negative improvement is reported without filtering.\n\n"
            + "\n".join(f"- [{p} report]({p}/report.md)" for p in config["data"]["pairs"])
            + "\n\n[Provenance](provenance.json) · [Config](config.yaml) · "
            "[All metrics](comparison.csv) · [Figures](figures/figure_index.md)\n",
            encoding="utf-8",
        )
        write_json(
            run / "complete.json",
            {
                "status": "complete",
                "pairs": config["data"]["pairs"],
                "provenance": "provenance.json",
                "report": "report.md",
            },
        )
        LOG.info("WRITES completed experiment report=%s", run / "report.md")
        return run
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
