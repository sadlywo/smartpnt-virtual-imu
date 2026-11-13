"""Run-local matched experiment reporting; no fitting or prediction is performed.

The pooled rows are physical per-seed metrics supplied by the runner. In
particular, this module never pools by averaging device-level RMSE values.
"""

# Chinese report prose intentionally uses Chinese punctuation.
# ruff: noqa: RUF001, B905

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from virtual_imu.viz.style import COLORS, apply_paper_style, save_figure

LOG = logging.getLogger(__name__)
POOLED = "__all_pairs__"
UNITS = {"gyro": "rad/s", "accel": "m/s^2"}
CONTROLS = {"raw", "affine"}
METHODS = (
    "raw",
    "affine",
    "ddpm",
    "cnn_bilstm",
    "informer",
    "transformer",
    "no_diffusion",
    "no_diffusion_matched",
)
LABELS = {
    "raw": "Raw",
    "affine": "Affine",
    "ddpm": "DDPM",
    "cnn_bilstm": "CNN-\nBiLSTM",
    "informer": "Informer",
    "transformer": "Transformer",
    "no_diffusion": "No diff.\n(own HPO)",
    "no_diffusion_matched": "No diff.\n(DDPM config)",
}
PALETTE = (
    COLORS["low"],
    COLORS["accent"],
    COLORS["virtual"],
    "#526777",
    "#697e99",
    "#658c83",
    "#9b7860",
    COLORS["high"],
)
KEYS = ["pair_id", "method", "component", "unit"]
SUMMARY_COLUMNS = [
    *KEYS,
    "rmse_mean",
    "rmse_sd",
    "vector_rmse_mean",
    "vector_rmse_sd",
    "n",
    "samples",
]


def _read_csv(run: Path, name: str, required: list[str], issues: list[str]) -> pd.DataFrame:
    path = run / name
    if not path.is_file():
        issues.append(f"{name}：缺失")
        LOG.warning("SKIPS missing report input %s", path)
        return pd.DataFrame(columns=required)
    LOG.info("READS %s", path)
    try:
        table = pd.read_csv(path)
        missing = sorted(set(required) - set(table))
        if missing:
            raise ValueError("缺少字段 " + ", ".join(missing))
        if table.empty:
            raise ValueError("无可用行")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        issues.append(f"{name}：{exc}")
        LOG.warning("SKIPS unusable report input %s: %s", path, exc)
        return pd.DataFrame(columns=required)
    return table


def _read_json(run: Path, name: str, issues: list[str]) -> dict:
    path = run / name
    if not path.is_file():
        issues.append(f"{name}：缺失")
        LOG.warning("SKIPS missing report input %s", path)
        return {}
    LOG.info("READS %s", path)
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("期望 JSON 对象")
        return result
    except (OSError, ValueError) as exc:
        issues.append(f"{name}：{exc}")
        LOG.warning("SKIPS unusable report input %s: %s", path, exc)
        return {}


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean/sample SD over unique seeds; deterministic controls have n=1.

    Only the two physical three-axis component groups enter this table.
    samples is the physical support of one seed, never a sum over seeds.
    """
    rows = []
    if metrics.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    frame = metrics[metrics.component.isin(UNITS)].copy()
    for field in ("rmse", "samples", "seed"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if (
        frame[KEYS].isna().any().any()
        or not np.isfinite(frame[["rmse", "samples", "seed"]]).all().all()
        or (frame.rmse < 0).any()
        or (frame.samples <= 0).any()
    ):
        raise ValueError("component 指标含缺失、非有限值、负 RMSE 或无效 samples")
    for key, group in frame.groupby(KEYS, sort=False):
        pair, method, component, unit = key
        if unit != UNITS[component]:
            raise ValueError(f"{pair}/{component} 单位不符：{unit}")
        # Identical duplicate records must not increase the effective seed count.
        group = group.drop_duplicates(["seed", "rmse", "samples"])
        if group.seed.duplicated().any():
            raise ValueError(f"{pair}/{method}/{component} 同 seed 存在冲突指标")
        if group.samples.nunique() != 1:
            raise ValueError(f"{pair}/{method}/{component} 不同 seed 的样本支持不一致")
        deterministic = method in CONTROLS
        if deterministic and not np.allclose(
            group.rmse, group.rmse.iloc[0], rtol=1e-10, atol=1e-12
        ):
            raise ValueError(f"{pair}/{method}/{component} 确定性基线跨 seed 不一致")
        mean = float(group.rmse.mean())
        sd = float(group.rmse.std(ddof=1)) if not deterministic and len(group) > 1 else np.nan
        rows.append(
            dict(
                zip(KEYS, key),
                rmse_mean=mean,
                rmse_sd=sd,
                vector_rmse_mean=np.sqrt(3) * mean,
                vector_rmse_sd=np.sqrt(3) * sd,
                n=1 if deterministic else len(group),
                samples=int(group.samples.iloc[0]),
            )
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _summary_fallback(summary: pd.DataFrame) -> pd.DataFrame:
    """Use a supplied summary only when per-seed data are unavailable."""
    frame = summary[summary.component.isin(UNITS)].copy()
    if frame.empty:
        return frame
    for field in SUMMARY_COLUMNS[4:]:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if (
        frame.duplicated(KEYS).any()
        or frame[KEYS].isna().any().any()
        or not np.isfinite(frame[["rmse_mean", "n", "samples"]]).all().all()
        or (frame.rmse_mean < 0).any()
        or (frame.n < 1).any()
        or (frame.samples <= 0).any()
        or any(row.unit != UNITS[row.component] for row in frame.itertuples())
    ):
        raise ValueError("summary 的分组、数值或单位无效")
    controls = frame.method.isin(CONTROLS)
    frame.loc[controls, ["rmse_sd", "vector_rmse_sd"]] = np.nan
    frame.loc[controls, "n"] = 1
    return frame


def _ordered(values) -> list[str]:
    observed = set(values)
    return [m for m in METHODS if m in observed] + sorted(observed - set(METHODS))


def _slug(value: str) -> str:
    # Encode unusual characters instead of allowing path separators or collisions.
    return quote(str(value), safe="_-").replace(".", "%2E")


def _link(run: Path, path: Path, label: str | None = None) -> str:
    return f"[{label or path.name}]({quote(path.relative_to(run).as_posix(), safe='/_.-')})"


def _markdown(frame: pd.DataFrame) -> str:
    """Render tables without the optional tabulate dependency."""
    if frame.empty:
        return "无可用数值。"

    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |",
            *(
                "| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None)
            ),
        ]
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    LOG.info("WRITES %s", path)


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    LOG.info("WRITES %s rows=%d", path, len(frame))


def _save(fig, out: Path, stem: str, caption: str, figures: list) -> None:
    fig.tight_layout()
    try:
        paths = save_figure(fig, out, stem)
        for path in paths:
            LOG.info("WRITES %s", path)
        figures.append((stem, caption, paths))
    finally:
        plt.close(fig)


def _plot_rmse(summary: pd.DataFrame, out: Path, figures: list) -> None:
    for pair, group in summary.groupby("pair_id", sort=False):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
        for ax, (component, unit) in zip(axes, UNITS.items()):
            data = group[group.component.eq(component)].set_index("method")
            methods = _ordered(data.index)
            for i, method in enumerate(methods):
                row = data.loc[method]
                color = (
                    PALETTE[METHODS.index(method) % len(PALETTE)]
                    if method in METHODS
                    else COLORS["high"]
                )
                ax.bar(i, row.rmse_mean, width=0.72, color=color)
                if method not in CONTROLS and pd.notna(row.rmse_sd) and row.rmse_sd >= 0:
                    ax.errorbar(
                        i,
                        row.rmse_mean,
                        yerr=row.rmse_sd,
                        fmt="none",
                        color=COLORS["high"],
                        capsize=2,
                        linewidth=1,
                    )
                ax.annotate(
                    "control" if method in CONTROLS else f"n={int(row.n)}",
                    (i, row.rmse_mean + (row.rmse_sd if pd.notna(row.rmse_sd) else 0)),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6,
                )
            ax.set_xticks(
                range(len(methods)),
                [LABELS.get(m, m) for m in methods],
                rotation=35,
                ha="right",
                fontsize=6,
            )
            ax.set_ylabel(f"{component.capitalize()} component RMSE [{unit}]")
            ax.set_ylim(bottom=0)
            ax.margins(y=0.17)
            ax.grid(axis="x", visible=False)
            if not methods:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
        stem = "rmse_pooled" if pair == POOLED else f"rmse_{_slug(pair)}"
        scope = "pooled（仅使用 __all_pairs__ 的物理逐 seed 指标）" if pair == POOLED else str(pair)
        _save(
            fig,
            out,
            stem,
            f"{scope} 相对实测 VN100 的 gyro/accel component RMSE，分面保持物理单位。"
            "学习方法为实际可用 seed 的均值 ± 样本标准差（ddof=1；n 见图）；"
            "raw/affine 为确定性控制，不画重复 seed 误差条。vector RMSE = √3 × component RMSE。",
            figures,
        )


def _prepare_deltas(deltas: pd.DataFrame, issues: list[str]) -> pd.DataFrame:
    frame = deltas[deltas.component.isin(UNITS)].copy()
    for field in ("ddpm_rmse", "no_diffusion_rmse", "ddpm_change_pct"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if frame.empty:
        return frame
    if (
        frame[["pair_id", "seed", "component"]].isna().any().any()
        or frame.duplicated(["pair_id", "seed", "component"]).any()
    ):
        issues.append("matched_seed_deltas.csv：分组标识缺失或重复，跳过差值图")
        return frame.iloc[:0]
    valid = np.isfinite(frame[["ddpm_rmse", "no_diffusion_rmse", "ddpm_change_pct"]]).all(axis=1)
    valid &= (frame.ddpm_rmse >= 0) & (frame.no_diffusion_rmse > 0)
    expected = 100 * (frame.ddpm_rmse / frame.no_diffusion_rmse - 1)
    valid &= np.isclose(frame.ddpm_change_pct, expected, rtol=1e-6, atol=1e-6)
    if not valid.all():
        issues.append(
            f"matched_seed_deltas.csv：{int((~valid).sum())} 行非有限或百分比不符，已跳过"
        )
    frame = frame[valid].copy()
    states = frame.initial_backbone_matches.astype(str).str.lower()
    frame["initial_backbone_matches"] = states.map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if states.isin(["false", "0"]).any():
        issues.append(
            "严格消融初始化不一致：initial_backbone_matches 包含 false，不能视为严格匹配证据"
        )
    if frame.initial_backbone_matches.isna().any():
        issues.append("严格消融初始化匹配状态未知，不能确认初始化相同")
    return frame


def _plot_deltas(frame: pd.DataFrame, out: Path, figures: list) -> None:
    if frame.empty:
        LOG.info("SKIPS matched-seed delta figure: no valid data")
        return
    pairs = list(dict.fromkeys(frame.pair_id))
    seeds = sorted(frame.seed.unique(), key=str)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, component in zip(axes, UNITS):
        for i, seed in enumerate(seeds):
            group = frame[frame.component.eq(component) & frame.seed.eq(seed)]
            offset = (i - (len(seeds) - 1) / 2) * min(0.16, 0.6 / len(seeds))
            ax.scatter(
                [pairs.index(p) + offset for p in group.pair_id],
                group.ddpm_change_pct,
                label=str(seed),
                s=23,
                color=PALETTE[(i + 2) % len(PALETTE)],
                marker=("o", "s", "^")[i % 3],
            )
        ax.axhline(0, color=COLORS["high"], linewidth=0.9)
        ax.set_xticks(range(len(pairs)), [p.replace("_to_", "\nto ") for p in pairs], fontsize=6)
        ax.set(xlim=(-0.5, len(pairs) - 0.5), title=component.capitalize())
        ax.margins(y=0.2)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("DDPM change vs. strict no-diffusion [%]\nPositive: DDPM worse")
    axes[1].legend(title="Seed", frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    _save(
        fig,
        out,
        "matched_seed_deltas",
        "逐设备、同 seed 严格消融差值：100 × (DDPM RMSE / no_diffusion_matched RMSE − 1)。"
        "正值表示 DDPM 更差；负值表示 DDPM 更好，保留全部负值。每点是一项优化 seed，"
        "不是独立录制。初始化匹配状态及其异常见报告和原始差值表。",
        figures,
    )


def _plot_runtime(
    latency: pd.DataFrame, out: Path, figures: list, issues: list[str]
) -> pd.DataFrame:
    if latency.empty:
        LOG.info("SKIPS runtime figure: no latency rows")
        return pd.DataFrame()
    frame = latency.copy()
    numeric = [
        "batch_size",
        "window_points",
        "draws",
        "reverse_steps",
        "mean_batch_ms",
        "mean_window_ms",
    ]
    for field in numeric:
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    valid = np.isfinite(frame[numeric]).all(axis=1)
    valid &= (frame[numeric[:-2]] >= 0).all(axis=1)
    valid &= (
        frame[["batch_size", "window_points", "draws", "mean_batch_ms", "mean_window_ms"]] > 0
    ).all(axis=1)
    if not valid.all():
        issues.append("latency.csv：无效或非正延迟行已跳过")
    frame = frame[valid]
    groups = ["pair_id", "method", "batch_size", "window_points", "draws", "reverse_steps"]
    result = (
        frame.groupby(groups, sort=False, dropna=False)
        .agg(
            mean_batch_ms=("mean_batch_ms", "mean"),
            mean_window_ms=("mean_window_ms", "mean"),
            measured_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    for (batch, window), group in result.groupby(["batch_size", "window_points"], sort=False):
        # Keep devices and inference settings visible; never silently average devices.
        fig, ax = plt.subplots(figsize=(7.2, max(3.6, 0.22 * len(group) + 0.9)))
        positions = np.arange(len(group))
        ax.barh(positions, group.mean_window_ms, color=COLORS["virtual"])
        labels = [
            f"{r.pair_id} / {r.method} (draws={int(r.draws)}, steps={int(r.reverse_steps)})"
            for r in group.itertuples()
        ]
        ax.set_yticks(positions, labels, fontsize=6)
        ax.invert_yaxis()
        log = group.mean_window_ms.max() / group.mean_window_ms.min() >= 30
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(f"Mean compute time per window [ms]{' (log scale)' if log else ''}")
        ax.grid(axis="y", visible=False)
        _save(
            fig,
            out,
            f"runtime_batch_{int(batch)}_window_{int(window)}",
            f"实测计算延迟，batch={int(batch)}、窗口={int(window)} 点；同一推理配置的可用计时 seed 均值。"
            "设备、draws 和逆扩散步数逐行列出；完整 mean/p50 batch 延迟与参数量见 latency.csv。"
            "DDPM 包括全部采样；直接模型为单次前向。窗口上下文时长不等于计算延迟。",
            figures,
        )
    return result


def _plot_histories(
    run: Path, out: Path, figures: list, selections: dict, issues: list[str]
) -> None:
    fits = run / "fits"
    if not fits.is_dir():
        LOG.info("SKIPS training curves: no fits directory")
        return
    selected = {
        (str(s.get("pair_id")), str(s.get("method"))): str(s.get("candidate"))
        for s in selections.get("selections", [])
        if isinstance(s, dict)
    }
    histories = {}
    for path in sorted(fits.glob("*/*/*/history.csv")):
        directory = path.parent.name
        # Incomplete attempts archived by the runner must not enter paper plots.
        if not re.fullmatch(r"(?:search_trial_\d+|seed_\d+|matched_ddpm_seed_\d+)", directory):
            LOG.info("SKIPS archived or unknown fit history %s", path)
            continue
        method, pair = path.parents[1].name, path.parents[2].name
        history = _read_csv(
            run,
            path.relative_to(run).as_posix(),
            ["epoch", "train_loss", "validation_score"],
            issues,
        )
        if history.empty:
            continue
        history = history[["epoch", "train_loss", "validation_score"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if not np.isfinite(history).all().all():
            issues.append(f"{path.relative_to(run).as_posix()}：曲线含非有限值，跳过")
            continue
        histories.setdefault((pair, method), []).append((directory, history))
    for (pair, method), series in histories.items():
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
        for i, (directory, history) in enumerate(series):
            trial = re.fullmatch(r"search_trial_(\d+)", directory)
            chosen = trial is not None and str(int(trial[1])) == selected.get((pair, method))
            label = directory + (" [selected]" if chosen else "")
            for ax, field in zip(axes, ("train_loss", "validation_score")):
                ax.plot(
                    history.epoch,
                    history[field],
                    label=label,
                    color=PALETTE[i % len(PALETTE)],
                    linestyle="-" if trial else "--",
                    linewidth=1.8 if chosen else 1.1,
                )
                ax.set_xlabel("Epoch")
        axes[0].set_ylabel("Training residual objective [standardized MSE]")
        axes[1].set_ylabel("Validation mean relative component RMSE [1]")
        axes[1].axhline(1, color=COLORS["low"], linewidth=0.8, linestyle=":")
        axes[1].legend(frameon=False, fontsize=5, loc="upper left", bbox_to_anchor=(1, 1))
        _save(
            fig,
            out,
            f"history_{_slug(pair)}_{_slug(method)}",
            f"{pair}/{method} 的候选和最终 seed 训练/固定验证曲线；每条线对应 fits 下同名目录的 history.csv。"
            "selected 来自 selections.lock.json；虚线为补充 seed 拟合。不同方法的训练目标含义不同，"
            "尤其 DDPM 使用前向加噪目标，因此训练 loss 不能直接作方法排名；验证曲线不使用 TEST。",
            figures,
        )


def _audit_coverage(metrics: pd.DataFrame, config: dict, issues: list[str]) -> None:
    if metrics.empty:
        return
    pairs = config.get("data", {}).get("pairs") or list(dict.fromkeys(metrics.pair_id))
    methods = list(config.get("methods") or _ordered(metrics.method))
    if "ddpm" in methods and "no_diffusion" in methods:
        methods = list(dict.fromkeys([*methods, "no_diffusion_matched"]))
    seeds = set(config.get("seeds", []))
    for pair in dict.fromkeys([*pairs, POOLED]):
        for method in methods:
            for component in UNITS:
                group = metrics[
                    metrics.pair_id.eq(pair)
                    & metrics.method.eq(method)
                    & metrics.component.eq(component)
                ]
                if group.empty:
                    issues.append(f"指标缺失：{pair}/{method}/{component}")
                elif method not in CONTROLS and seeds and set(group.seed) != seeds:
                    issues.append(
                        f"seed 覆盖不符：{pair}/{method}/{component}，实际 {sorted(set(group.seed))}"
                    )


def _protocol(config: dict) -> str:
    data, training = config.get("data", {}), config.get("training", {})
    search, evaluation = config.get("search", {}), config.get("evaluation", {})
    actual = pd.DataFrame(
        [
            ("seed", str(config.get("seeds", "未提供"))),
            ("方法", ", ".join(config.get("methods", [])) or "未提供"),
            ("设备对", ", ".join(data.get("pairs", [])) or "未提供"),
            (
                "窗口/训练 stride/评估 stride",
                f"{data.get('window_size', '未提供')}/{data.get('stride', '未提供')}/{data.get('eval_stride', '未提供')}",
            ),
            ("优化器", "Adam（runner 约定）"),
            (
                "epoch 上限/patience",
                f"{training.get('epochs', '未提供')}/{training.get('early_stopping_patience', '未提供')}",
            ),
            ("固定验证窗口数", training.get("validation_windows", "未提供")),
            ("每方法候选数", len(search["trials"]) if "trials" in search else "未提供"),
            (
                "DDPM 逆扩散步数/draws",
                f"{config.get('diffusion', {}).get('timesteps', '未提供')}/{evaluation.get('draws', '未提供')}",
            ),
        ],
        columns=["项目", "当前传入配置（执行证据见原始产物）"],
    )
    return (
        "正式方案：每设备分别拟合，使用 3 个优化 seed；五种分别调参的方法为 DDPM、CNN-BiLSTM、"
        "Informer、Transformer、no_diffusion。共同采用 80 点窗口、stride 40、Adam、最多 40 epochs、"
        "patience 7、同一组 32 个固定验证窗口与每方法 3 个候选；验证选择后固定配置。"
        "DDPM 为 100 步逆扩散、4 draws；直接模型仅一次前向。\n\n"
        "下表记录本次配置，允许较小的合成/集成测试配置；正式方案文字不代表这些条件已执行或已校验。\n\n"
        + _markdown(actual)
        + "\n\nno_diffusion 是独立 HPO 的直接残差回归消融；no_diffusion_matched 才是严格消融，"
        "使用精确 DDPM core、DDPM 选定的结构/学习率和同 seed 初始化：六个噪声通道置零、"
        "扩散时间固定为零、无加噪、一次前向。两者分别保留，不把独立调参结果当作严格扩散效应。"
    )


def render_matched_report(run: Path, config: dict) -> None:
    """Write Chinese report, tables and paper figures below this run only.

    Missing/malformed inputs produce a clearly incomplete report. Figures and
    tables use available groups, including small smoke-test method/seed subsets.
    No completion marker, checkpoint, source CSV, or latest snapshot is changed.
    """
    run = Path(run)
    out, tables = run / "figures", run / "tables"
    out.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    figures: list = []
    metrics = _read_csv(run, "metrics_per_seed.csv", [*KEYS, "seed", "samples", "rmse"], issues)
    supplied = _read_csv(run, "summary.csv", SUMMARY_COLUMNS, issues)
    deltas = _read_csv(
        run,
        "matched_seed_deltas.csv",
        [
            "pair_id",
            "seed",
            "component",
            "ddpm_rmse",
            "no_diffusion_rmse",
            "ddpm_change_pct",
            "initial_backbone_matches",
        ],
        issues,
    )
    tuning = _read_csv(
        run, "tuning_trials.csv", ["pair_id", "method", "candidate", "validation_score"], issues
    )
    latency = _read_csv(
        run,
        "latency.csv",
        [
            "pair_id",
            "method",
            "seed",
            "batch_size",
            "window_points",
            "draws",
            "reverse_steps",
            "mean_batch_ms",
            "mean_window_ms",
        ],
        issues,
    )
    selections = _read_json(run, "selections.lock.json", issues)
    provenance = _read_json(run, "provenance.json", issues)
    if selections and not isinstance(selections.get("selections"), list):
        issues.append("selections.lock.json：缺少 selections 列表")
        selections = {}
    if selections.get("test_used_for_selection") is True:
        issues.append("选择锁标记 TEST 参与了选择，不能视为无测试泄漏的比较")
    try:
        summary = _summarize_metrics(metrics) if not metrics.empty else _summary_fallback(supplied)
    except ValueError as exc:
        issues.append(f"指标汇总跳过：{exc}")
        LOG.warning("SKIPS RMSE summary: %s", exc)
        summary = pd.DataFrame(columns=SUMMARY_COLUMNS)
    if summary.empty:
        issues.append("无可用 gyro/accel 物理指标，不生成 RMSE 数值或图")
    _audit_coverage(metrics, config, issues)
    deltas = _prepare_deltas(deltas, issues)
    with matplotlib.rc_context():
        apply_paper_style()
        _plot_rmse(summary, out, figures)
        _plot_deltas(deltas, out, figures)
        runtime = _plot_runtime(latency, out, figures, issues)
        _plot_histories(run, out, figures, selections, issues)
    written_tables = []
    for name, frame in (
        ("matched_rmse_summary.csv", summary),
        ("matched_seed_deltas.csv", deltas),
        ("latency_summary.csv", runtime),
    ):
        if not frame.empty:
            _write_table(tables / name, frame)
            written_tables.append(tables / name)

    index = [
        "# Matched comparison figure index",
        "",
        "[运行报告](../report.md)",
        "",
        "图宽 7.2 inch；PNG 300 dpi、SVG、PDF；样式来自 src/virtual_imu/viz/style.py。",
        "",
    ]
    for stem, caption, paths in figures:
        links = [f"[{p.suffix[1:].upper()}]({quote(p.name, safe='_.-')})" for p in paths]
        index.extend([f"## {stem}", "", " · ".join(links), "", caption, ""])
    if not figures:
        index.append("无可用图；缺失输入与原因见运行报告。")
    _write_text(out / "figure_index.md", "\n".join(index))

    lines = [
        "# 匹配条件 IMU 比较报告",
        "",
        "本目录是本报告的运行产物来源；所有产物链接相对于本运行目录。",
        "",
        "## 输入检查",
        "",
        "输入不完整或存在一致性问题；本报告只展示可读取的产物。"
        if issues
        else "所需报告输入可读取；这项检查不构成全部拟合/评估完成或协议执行正确的声明。",
        "",
    ]
    if issues:
        lines.extend([*(f"- {issue}" for issue in dict.fromkeys(issues)), ""])
    lines.extend(
        [
            "## 方案与实际配置",
            "",
            _protocol(config),
            "",
            "## 物理误差与 seed 统计",
            "",
            "component RMSE = sqrt(mean(error²))，mean 同时跨样本与三轴；"
            "vector RMSE = sqrt(mean(sum_axis(error²))) = √3 × component RMSE。"
            "gyro 的 rad/s 与 accel 的 m/s² 分开比较，不制作跨单位综合排名。",
            "",
            "下表按 metrics_per_seed.csv 中的实际 seed 重算均值与样本标准差（ddof=1），"
            "原始 summary.csv 保留供核查；若逐 seed 表缺失，仅使用已有 summary 并标记输入不完整。"
            "单 seed 无 SD；raw/affine 的重复行不作为随机重复，n=1 且不画 seed 误差条。"
            "samples 是每 seed 的物理时间样本数，不乘以 seed 数。",
            "",
            "pooled 只读取 pair_id=__all_pairs__ 的物理逐 seed 指标，不对设备 RMSE 求均值；"
            "多个 LOW 设备共享 HIGH，pooled 的设备时间样本不是独立录制。",
            "",
        ]
    )
    for component, unit in UNITS.items():
        lines.extend(
            [
                f"### {component} [{unit}]",
                "",
                _markdown(summary[summary.component.eq(component)]),
                "",
            ]
        )
    lines.extend(
        [
            "## 同 seed 严格消融",
            "",
            "差值 = 100 × (ddpm_rmse / no_diffusion_rmse − 1)，"
            "此表 no_diffusion_rmse 指 no_diffusion_matched。正值表示 DDPM 更差，负值表示 DDPM 更好。"
            "不丢弃负值，不用独立 HPO 的 no_diffusion 替换严格匹配项。",
            "",
            _markdown(deltas),
            "",
            "## 调参选择",
            "",
        ]
    )
    tuning_fields = [
        c
        for c in (
            "pair_id",
            "method",
            "seed",
            "candidate",
            "best_epoch",
            "epochs_ran",
            "validation_score",
            "parameters",
            "seconds",
        )
        if c in tuning
    ]
    lines.extend(
        [
            _markdown(tuning[tuning_fields]),
            "",
            "选择内容以 selections.lock.json 为准；本报告不从 TEST 指标反推最优候选。",
            "",
            "## 实测运行时间",
            "",
            _markdown(runtime),
            "",
            "延迟按相同 batch_size 与 window_points 分图，设备与 draws/reverse_steps 保留。"
            "mean_window_ms 是 batch 计算均值除以 batch 大小，吞吐摊销值不等于单请求延迟；"
            "单请求见 batch=1。原表保留 p50_batch_ms、参数量和计时 seed；"
            "窗口点数描述上下文，不能代替计算耗时或端到端延迟。",
            "",
            "## 证据范围",
            "",
            "正式数据的物理 TEST 录制仅 2 条，样本规模小；多个 LOW 设备共享同一 HIGH。"
            "这里的 3 个优化 seed（小型配置以实际数量为准）只反映训练随机性，"
            "不代表 3 条独立录制，误差条不是独立录制置信区间。"
            "VN100 是实测参考，含测量误差，不能视为无噪声真值。"
            "信号回归误差不构成导航性能改善的因果证据，也不支持未经外部基准验证的 SOTA 声明。",
            "",
            "## 运行溯源与产物",
            "",
            "[图与图注](figures/figure_index.md)",
            "",
        ]
    )
    for name in (
        "config.yaml",
        "provenance.json",
        "selections.lock.json",
        "metrics_per_seed.csv",
        "summary.csv",
        "matched_seed_deltas.csv",
        "tuning_trials.csv",
        "latency.csv",
        "final_fits.csv",
        "training.complete.json",
        "source_snapshot",
        "working_tree.patch",
    ):
        path = run / name
        if path.exists():
            lines.append("- " + _link(run, path))
    for path in written_tables:
        lines.append("- " + _link(run, path))
    for path in sorted(run.glob("*/data_manifest.json")):
        lines.append("- " + _link(run, path, path.parent.name + "/data_manifest.json"))
    pins = {
        k: provenance[k]
        for k in (
            "git_commit",
            "git_branch",
            "config_sha256",
            "data_manifest_sha256",
            "device",
            "gpu",
            "created_at",
        )
        if k in provenance
    }
    if pins:
        lines.extend(
            [
                "",
                "provenance.json 中记录的标识（source_sha256 逐文件固定源码）：",
                "",
                "```json",
                json.dumps(pins, ensure_ascii=False, indent=2),
                "```",
            ]
        )
    _write_text(run / "report.md", "\n".join(lines))
