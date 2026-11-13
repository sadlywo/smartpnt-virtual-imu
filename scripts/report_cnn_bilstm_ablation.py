"""Report a completed shared CNN-BiLSTM run without fitting or loading checkpoints.

CLI: python scripts/report_cnn_bilstm_ablation.py --run-dir runs/<experiment_id>
Public numerical APIs accept/return pandas frames; summary.csv is tidy (one
metric per row). SD always means sample SD across optimization seeds (ddof=1).
Paired delta percentages are 100*(treatment-reference)/reference: positive is
worse. A zero reference has an undefined percentage, exported as an empty cell.
"""

# Chinese report prose intentionally uses Chinese punctuation.
# zip(strict=...) and datetime.UTC are unavailable in the pinned Python3.9 runtime.
# ruff: noqa: RUF001, B905, UP017
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOG = logging.getLogger(__name__)
ALL = "__all__"
FULL = "cnn_bilstm"
PRIMARY = (FULL, "bilstm_only", "cnn_only", "mlp_only", "cnn_lstm")
SUPPLEMENTAL = ("cnn_lstm_wide", "pointwise_bilstm", "cnn_bilstm_dense")
FIT_KEYS = ["protocol", "variant", "window", "seed"]
SUPPORT = ["aggregation", "dataset_id", "pair_id", "original_sequence_id"]
CONDITION = ["protocol", "split", "variant", "window", "seed"]
IDENTITY = CONDITION + SUPPORT
GROUP = [c for c in IDENTITY if c != "seed"]
PHYSICAL = [
    f"{g}_{m}" for g in ("gyro", "accel") for m in ("rmse", "vector_rmse", "mae", "bias_norm")
]
PHYSICAL += [f"{a}_rmse" for a in ("gx", "gy", "gz", "ax", "ay", "az")]
METRICS = [*PHYSICAL, "normalized_score"]
CELLS = (("alimu_only", "alimu"), ("pooled", "alimu"), ("pooled", "penncosyvio"))


def prepare_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Validate wide physical metrics and derive pair/dataset-balanced scores.

    The supplied pooled physical metrics are retained verbatim: RMSE values
    must never be pooled by averaging recording or device RMSEs. Scores follow
    grouped_score: mean(gyro/Raw, accel/Raw) per pair, then equal pair means
    within each dataset and equal dataset means. Recording scores are undefined.
    """
    missing = set([*IDENTITY, "samples", *PHYSICAL]) - set(metrics)
    if missing or metrics.empty:
        raise ValueError(f"Missing metric fields or empty input: {sorted(missing)}")
    frame = metrics.copy().reset_index(drop=True)
    if frame[IDENTITY].isna().any().any():
        raise ValueError("Missing metric identity")
    for name in ("window", "seed", "samples"):
        values = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Nonintegral/nonfinite " + name)
        frame[name] = values.astype("int64")
    frame[PHYSICAL] = frame[PHYSICAL].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[PHYSICAL]).all().all() or (frame[PHYSICAL] < 0).any().any():
        raise ValueError("Invalid physical metrics")
    if (frame.samples <= 0).any() or frame.duplicated(IDENTITY).any():
        raise ValueError("Duplicate metric identities or invalid support samples")
    raw = frame.variant.eq("raw")
    if (
        not raw.any()
        or (frame.loc[raw, "window"] != 0).any()
        or (frame.loc[raw, "seed"] != -1).any()
    ):
        raise ValueError("Raw must occur once per support at window0/seed-1")
    if (frame.loc[~raw, "window"] <= 0).any() or (frame.loc[~raw, "seed"] < 0).any():
        raise ValueError("Invalid learned seed/window")
    if not frame.aggregation.isin(("overall", "dataset", "pair", "recording")).all():
        raise ValueError("Unknown aggregation")
    for field, active in (
        ("dataset_id", ["dataset", "pair", "recording"]),
        ("pair_id", ["pair", "recording"]),
        ("original_sequence_id", ["recording"]),
    ):
        if not frame[field].ne(ALL).eq(frame.aggregation.isin(active)).all():
            raise ValueError("Inconsistent aggregation identity: " + field)
    support_keys = ["protocol", "split", *SUPPORT]
    if frame.groupby(support_keys).samples.nunique().gt(1).any():
        raise ValueError("Sample supports differ across seeds/treatments/windows")
    for _, split_frame in frame.groupby(["protocol", "split"], sort=False):
        baseline = split_frame[split_frame.variant.eq("raw")]
        if set(baseline.aggregation) != {"overall", "dataset", "pair", "recording"}:
            raise ValueError("Missing required aggregation rows")
        overall = baseline[baseline.aggregation.eq("overall")].samples.iloc[0]
        datasets = baseline[baseline.aggregation.eq("dataset")]
        pairs_raw = baseline[baseline.aggregation.eq("pair")]
        recordings = baseline[baseline.aggregation.eq("recording")]
        if overall != datasets.samples.sum():
            raise ValueError("Overall/dataset sample counts disagree")
        for parent, child, keys in (
            (datasets, pairs_raw, ["dataset_id"]),
            (pairs_raw, recordings, ["dataset_id", "pair_id"]),
        ):
            if (
                not parent.set_index(keys)
                .samples.sort_index()
                .equals(child.groupby(keys).samples.sum().sort_index())
            ):
                raise ValueError("Parent/child aggregation sample counts disagree")
        expected = set(map(tuple, baseline[SUPPORT].to_numpy()))
        for _, group in split_frame.groupby(["variant", "window", "seed"], sort=False):
            if set(map(tuple, group[SUPPORT].to_numpy())) != expected:
                raise ValueError("Missing or mismatched aggregation supports")
    if "job_id" not in frame:
        frame["job_id"] = ""
    frame["job_id"] = frame.job_id.fillna("")
    supplied = frame.get("normalized_score")
    frame["normalized_score"] = np.nan
    pairs = frame[frame.aggregation.eq("pair")]
    baseline_keys = ["protocol", "split", "dataset_id", "pair_id"]
    baseline = pairs[pairs.variant.eq("raw")][[*baseline_keys, "gyro_rmse", "accel_rmse"]]
    paired = pairs.reset_index(names="row_index").merge(
        baseline, on=baseline_keys, suffixes=("", "_raw"), validate="many_to_one"
    )
    if paired.empty or paired.gyro_rmse_raw.isna().any():
        raise ValueError("Pair-level Raw metrics are required for normalized scores")
    paired["score"] = 0.5 * sum(
        paired[g + "_rmse"] / np.maximum(paired[g + "_rmse_raw"], 1e-12) for g in ("gyro", "accel")
    )
    frame.loc[paired.row_index, "normalized_score"] = paired.score.to_numpy()
    dataset_scores = paired.groupby([*CONDITION, "dataset_id"], sort=False).score.mean()
    overall_scores = dataset_scores.groupby(level=CONDITION, sort=False).mean()
    for index, row in frame[frame.aggregation.isin(("dataset", "overall"))].iterrows():
        key = tuple(row[c] for c in CONDITION)
        if row.aggregation == "dataset":
            key += (row.dataset_id,)
        scores = dataset_scores if row.aggregation == "dataset" else overall_scores
        if key not in scores:
            raise ValueError("Missing normalized score support")
        frame.at[index, "normalized_score"] = scores.loc[key]
    if supplied is not None:
        known = supplied.notna()
        if not np.allclose(
            supplied[known].astype(float),
            frame.loc[known, "normalized_score"],
            rtol=1e-7,
            atol=1e-9,
        ):
            raise ValueError("Supplied normalized scores disagree with balanced Raw ratios")
    return frame


def _long(frame):
    return frame.melt(
        id_vars=[*IDENTITY, "samples", "job_id"],
        value_vars=METRICS,
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])


def summarize_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per condition/support/metric: mean, sample SD, n, samples."""
    long = _long(frame)
    return long.groupby([*GROUP, "metric"], sort=True, as_index=False).agg(
        mean=("value", "mean"),
        sd=("value", "std"),
        n=("seed", "nunique"),
        samples=("samples", "first"),
    )


def paired_deltas(frame: pd.DataFrame, comparison="module") -> pd.DataFrame:
    """Exact seed pairs against full at the same window, or own window320.

    Dense control rows remain labelled supplemental_loss_support: their delta
    is a loss-support sensitivity comparison, not a strict module intervention.
    """
    if comparison not in ("module", "window"):
        raise ValueError("comparison must be module or window")
    long = _long(frame[~frame.variant.eq("raw")])
    fixed = "window" if comparison == "module" else "variant"
    changed = "variant" if comparison == "module" else "window"
    ref = long[long.variant.eq(FULL) if comparison == "module" else long.window.eq(320)]
    other = long[~long.variant.eq(FULL) if comparison == "module" else ~long.window.eq(320)]
    keys = ["protocol", "split", "seed", fixed, *SUPPORT, "metric"]
    ref = ref[[*keys, changed, "value", "samples", "job_id"]].rename(
        columns={c: "reference_" + c for c in (changed, "value", "samples", "job_id")}
    )
    result = other.merge(ref, on=keys, how="left", validate="many_to_one")
    if result.reference_value.isna().any() or not result.samples.eq(result.reference_samples).all():
        raise ValueError("Missing reference or mismatched paired support")
    result["reference_" + fixed] = result[fixed]
    result["reference_seed"] = result.seed
    result["delta"] = result.value - result.reference_value
    result["relative_delta_pct"] = 100 * result.delta / result.reference_value.replace(0, np.nan)
    result["comparison"] = comparison
    result["control_type"] = np.select(
        [result.variant.eq("cnn_bilstm_dense"), result.variant.isin(SUPPLEMENTAL)],
        ["supplemental_loss_support", "supplemental_capacity"],
        default="primary",
    )
    return result.sort_values([*IDENTITY, "metric"]).reset_index(drop=True)


def factorial_interactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-seed error-scale full - BiLSTM-only - CNN-only + MLP-only.

    Negative values suggest potential complementarity on that metric's additive
    scale; this is descriptive and provides no significance or recording CI.
    """
    variants = [FULL, "bilstm_only", "cnn_only", "mlp_only"]
    long = _long(frame[frame.variant.isin(variants)])
    keys = [c for c in IDENTITY if c != "variant"] + ["samples", "metric"]
    wide = long.pivot(index=keys, columns="variant", values="value").reindex(columns=variants)
    if wide.isna().any().any():
        raise ValueError("Incomplete four-cell factorial")
    wide["interaction"] = wide[FULL] - wide.bilstm_only - wide.cnn_only + wide.mlp_only
    return wide.rename_axis(columns=None).reset_index()


def select_validation_windows(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Minimize VALID balanced score averaged across configured seeds only.

    One choice per protocol/primary variant (pooled uses both datasets).
    Exact mean-score ties choose the shorter context. TEST never enters here.
    """
    val = frame[frame.split.eq("validation") & frame.aggregation.eq("overall")]
    rows = []
    for protocol, variant in product(config["protocols"], config["variants"]):
        group = val[val.protocol.eq(protocol) & val.variant.eq(variant)]
        expected = set(product(config["windows"], config["seeds"]))
        if set(zip(group.window, group.seed)) != expected:
            raise ValueError("Incomplete validation window/seed grid")
        scores = (
            group.groupby("window").normalized_score.agg(["mean", "std", "count"]).reset_index()
        )
        best = scores.sort_values(["mean", "window"], kind="stable").iloc[0]
        rows.append(
            dict(
                protocol=protocol,
                variant=variant,
                selected_window=int(best.window),
                score_mean=best["mean"],
                score_sd=best["std"],
                n=int(best["count"]),
                selection_split="validation",
                candidate_windows=",".join(map(str, sorted(config["windows"]))),
                score_source="equal datasets / equal pairs / mean gyro+accel Raw ratios",
                tie_break="shorter window for exactly equal seed means",
            )
        )
    return pd.DataFrame(rows)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(run, name, kind, hashes):
    path = run / name
    LOG.info("READS %s", path)
    before = _sha(path)
    if kind == "csv":
        value = pd.read_csv(path)
    elif kind == "yaml":
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if _sha(path) != before:
        raise ValueError("Input changed while reading: " + name)
    hashes[name] = before
    return value


def _require_columns(frame, columns, label):
    missing = set(columns) - set(frame)
    if missing or frame.empty:
        raise ValueError(f"{label}: missing fields/rows {sorted(missing)}")


def _validate_complete(config, planned, provenance, complete, lock, registry, cost, metrics):
    expected = set(
        (p, v, w, s)
        for p in config["protocols"]
        for v in config["variants"]
        for w in config["windows"]
        for s in config["seeds"]
    )
    expected |= set(
        (p, v, 320, s)
        for p in config["protocols"]
        for v in config.get("supplemental", [])
        for s in config["seeds"]
    )
    jobs = pd.DataFrame(planned).rename(columns={"id": "job_id"})
    _require_columns(jobs, [*FIT_KEYS, "job_id"], "planned jobs")
    for table in (jobs, registry, cost):
        for field in ("job_id", "directory"):
            if (
                field in table
                and table[field]
                .fillna("")
                .astype(str)
                .str.contains("_guard_diagnostic", regex=False)
                .any()
            ):
                raise ValueError("Excluded _guard_diagnostic job identity/directory")
    if (
        len(jobs) != len(expected)
        or set(map(tuple, jobs[FIT_KEYS].to_numpy())) != expected
        or jobs.job_id.duplicated().any()
    ):
        raise ValueError("Planned job grid differs from config")
    if (
        provenance.get("test_used_for_selection") is not False
        or lock.get("test_loaded") is not False
    ):
        raise ValueError("Missing validation-only checkpoint selection declaration")
    if (
        provenance.get("planned_fits") != len(jobs)
        or complete.get("status") != "complete"
        or complete.get("planned") != len(jobs)
        or complete.get("completed") != len(jobs)
    ):
        raise ValueError("Training is not complete")
    _require_columns(
        registry,
        [
            *FIT_KEYS,
            "job_id",
            "checkpoint_sha256",
            "parameters",
            "actual_epochs",
            "best_epoch",
            "elapsed_seconds",
            "amp_skipped_steps",
        ],
        "fit registry",
    )
    _require_columns(
        cost,
        [*FIT_KEYS, "job_id", "parameters", "batch1_ms_mean", "batch1_ms_median"],
        "compute cost",
    )
    for label, table in (("registry", registry), ("compute", cost)):
        if table.job_id.duplicated().any() or set(table.job_id) != set(jobs.job_id):
            raise ValueError("Incomplete/duplicate jobs in " + label)
        joined = table.merge(
            jobs[[*FIT_KEYS, "job_id"]], on="job_id", suffixes=("", "_plan"), validate="one_to_one"
        )
        if any(not joined[c].eq(joined[c + "_plan"]).all() for c in FIT_KEYS):
            raise ValueError("Job metadata differs from plan: " + label)
    if dict(zip(registry.job_id, registry.checkpoint_sha256)) != lock.get("checkpoints_sha256"):
        raise ValueError("Registry checkpoints differ from pre-TEST lock")
    cost_fields = ["parameters", "batch1_ms_mean", "batch1_ms_median"]
    fit_fields = [
        "parameters",
        "actual_epochs",
        "best_epoch",
        "elapsed_seconds",
        "amp_skipped_steps",
    ]
    for table, cols in ((cost, cost_fields), (registry, fit_fields)):
        if not np.isfinite(table[cols].to_numpy(float)).all() or (table[cols] < 0).any().any():
            raise ValueError("Invalid compute/fit measurements")
    if (
        (cost.parameters <= 0).any()
        or (registry.best_epoch < 1).any()
        or (registry.best_epoch > registry.actual_epochs).any()
    ):
        raise ValueError("Invalid parameter counts or epoch bounds")
    if (
        not cost.set_index("job_id")
        .parameters.sort_index()
        .equals(registry.set_index("job_id").parameters.sort_index())
    ):
        raise ValueError("Compute/registry parameter counts disagree")
    learned = metrics[~metrics.variant.eq("raw")]
    observed = learned[[*CONDITION, "job_id"]].drop_duplicates()
    expected_evaluation = jobs[[*FIT_KEYS, "job_id"]].merge(
        pd.DataFrame({"split": ["validation", "test"]}), how="cross"
    )
    cols = [*CONDITION, "job_id"]
    if set(map(tuple, observed[cols].to_numpy())) != set(
        map(tuple, expected_evaluation[cols].to_numpy())
    ):
        raise ValueError("Evaluation is incomplete or differs from planned job identities")
    if set(map(tuple, metrics[["protocol", "split"]].drop_duplicates().to_numpy())) != set(
        product(config["protocols"], ["validation", "test"])
    ):
        raise ValueError("Unexpected protocol/split coverage")
    for protocol, dataset in CELLS:
        cell = metrics[
            metrics.aggregation.eq("dataset")
            & metrics.protocol.eq(protocol)
            & metrics.dataset_id.eq(dataset)
        ]
        if protocol in config["protocols"] and cell.empty:
            raise ValueError(f"Missing reporting cell {protocol}/{dataset}")


def summarize_compute(registry, cost):
    """Tidy seed summaries of measured timing, parameters and fit costs."""
    fields = ["best_epoch", "actual_epochs", "elapsed_seconds", "amp_skipped_steps"]
    merged = cost.merge(registry[["job_id", *fields]], on="job_id", validate="one_to_one")
    long = merged.melt(
        id_vars=[*FIT_KEYS, "job_id"],
        value_vars=["parameters", "batch1_ms_mean", "batch1_ms_median", *fields],
        var_name="metric",
        value_name="value",
    )
    return long.groupby(["protocol", "variant", "window", "metric"], as_index=False).agg(
        mean=("value", "mean"),
        sd=("value", "std"),
        n=("seed", "nunique"),
        minimum=("value", "min"),
        maximum=("value", "max"),
    )


def _number(value):
    return "—" if pd.isna(value) else f"{value:.6g}"


def _estimate(mean, sd, n):
    if n == 0:
        return "—（n=0）"
    if n == 1:
        return _number(mean) + "（n=1）"
    return f"{_number(mean)} ± {_number(sd)}"


def _table(headers, rows):
    def escape(x):
        return str(x).replace("|", "\\|").replace("\n", " ")

    return (
        "\n".join(
            [
                "| " + " | ".join(map(escape, headers)) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            + ["| " + " | ".join(map(escape, row)) + " |" for row in rows]
        )
        + "\n"
    )


def _metric_table(summary, identity, metrics):
    rows = []
    for key, group in summary.groupby(identity, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        lookup = group.set_index("metric")
        cells = [
            _estimate(lookup.loc[m, "mean"], lookup.loc[m, "sd"], lookup.loc[m, "n"])
            if m in lookup.index
            else "—"
            for m in metrics
        ]
        rows.append([*key, int(group.n.iloc[0]), int(group.samples.iloc[0]), *cells])
    return _table([*identity, "n种子", "每种子点数", *metrics], rows)


def _paired_table(deltas, identities, verdict=False):
    rows = []
    for key, group in deltas.groupby(identities, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        n = group.seed.nunique()
        worse, better, ties = (
            int((group.delta > 0).sum()),
            int((group.delta < 0).sum()),
            int((group.delta == 0).sum()),
        )
        sign = f"+{worse}/−{better}/0:{ties}"
        conclusion = (
            "参照在全部种子更好"
            if worse == n
            else "处理在全部种子更好"
            if better == n
            else "无跨种子一致收益"
        )
        row = [
            *key,
            n,
            _estimate(group.delta.mean(), group.delta.std(ddof=1), n),
            _estimate(
                group.relative_delta_pct.mean(),
                group.relative_delta_pct.std(ddof=1),
                int(group.relative_delta_pct.notna().sum()),
            ),
            sign,
        ]
        rows.append(row + ([conclusion] if verdict else []))
    return _table(
        identities
        + ["n配对种子", "绝对差均值±SD", "相对差%均值±SD", "+/−/相等种子数"]
        + (["描述性判断"] if verdict else []),
        rows,
    )


def _report_text(
    run,
    cfg,
    provenance,
    summary,
    module,
    windows,
    interactions,
    selected,
    compute,
    hashes,
    historical,
):
    lines = ["# 共享 CNN-BiLSTM 窗口与模块消融报告", ""]
    if provenance.get("synthetic_fixture"):
        lines += ["**合成测试产物：本报告仅验证报告代码，不能作为真实实验结论。**", ""]
    lines += [
        f"运行目录：`{run.name}`；实验：`{cfg.get('experiment_id', run.name)}`；训练源码 git：`{provenance.get('git_commit', '未知')}`。",
        f"已核对计划、fit_registry、checkpoint lock 与 VALID/TEST 指标作业覆盖，共 {provenance['planned_fits']} 个已登记作业。该计数表示输入产物覆盖，不等于独立预测回放验证。",
        "",
        "## 如何解释结果",
        "",
        "下述增益判断逐单元、逐指标给出。相同窗口/种子的处理值减去完整模型，正值表示处理更差、参照更好；负值表示处理更好。全部种子同号只代表优化重复的一致性，不能据此宣称统计显著或对新录制必然有效。",
        "物理指标保持各次运行中先池化样本误差再开方的结果；各设备/录制 RMSE 不再取算术平均来重建总体 RMSE。gyro 单位 rad/s，accel 单位 m/s²；vector RMSE 为三轴误差向量范数的 RMSE，component RMSE 为三轴分量均方误差的平方根。MAE 为分量平均绝对误差，bias_norm 为三轴均值误差向量范数。",
        f"配置种子：{cfg['seeds']}。均值±样本标准差跨种子计算（ddof=1），n 为实际种子数，单种子 SD 留空。Raw 是 window=0、seed=-1 的确定性基线，只记一次。种子数不是独立录制数，不生成录制置信区间。",
        "",
        "## 设计与受控范围",
        "",
        f"协议 {cfg['protocols']}，主模型 {cfg['variants']}，窗口 {cfg['windows']}，补充控制 {cfg.get('supplemental', [])}。",
        f"所有主处理使用相同 {cfg['data']['max_window']} 点锚窗（stride={cfg['data']['stride']}）中的中心 {cfg['data']['block_size']} 点监督及评价；裁剪窗口只改变上下文。采样率 {cfg['data'].get('rate_hz', 40)} Hz，名义时长=点数/采样率，首尾采样跨度=(点数−1)/采样率。",
        "共享架构：6 LOW 输入，TRAIN 标准化 HIGH−LOW 残差目标；CNN64、depth3、kernel5，BiLSTM hidden32，64维读出头、dropout0；Huber/SmoothL1 与 AdamW，最多80 epoch、patience15。准确参数见本运行配置。retained block 的共同初始化、采样序列和监督支持由训练产物保存，独立核验器负责回放。",
        "bilstm_only 用逐点投影移除时间卷积；cnn_only 移除循环块；mlp_only 同时移除二者；cnn_lstm 改为单向 hidden32。cnn_lstm_wide 单向 hidden64 控制输出宽度/容量；pointwise_bilstm 使用 kernel1 保留深度/归一化/非线性。cnn_bilstm_dense 在全部320点监督、在相同中心40点评价，属于损失支持敏感性控制，不纳入严格模块因子交互。",
        "模型使用窗口未来信息，包括对称上下文；结果不构成在线因果滤波性能声明。单向LSTM对照仍保留对称CNN，不能据此称为完全因果模型。",
        "本实验由已看到的历史结果提出，复用原数据集与TEST划分，属于后续机制消融；本轮所有处理与检查点在新TEST评估前冻结，但不是新增独立确认数据。",
        "",
        "## 历史证据",
        "",
        historical,
        "历史 default/tuned、联合 HPO、不同完整网络或 TCN 消融不能替代本次严格消融。本次中心块监督/评价约定不同于历史整窗及拼接结果，不能宣称历史绝对 RMSE 已复现。",
        "",
    ]
    main = summary[(summary.split == "test") & summary.variant.isin(["raw", *cfg["variants"]])]
    lines += ["## 三个协议×评估域单元：完整窗口主模型表", ""]
    for protocol, dataset in CELLS:
        cell = main[
            main.protocol.eq(protocol)
            & main.dataset_id.eq(dataset)
            & main.aggregation.eq("dataset")
        ]
        lines += [
            f"### {protocol} / {dataset}",
            "",
            _metric_table(
                cell,
                ["variant", "window"],
                ["gyro_rmse", "accel_rmse", "gyro_vector_rmse", "accel_vector_rmse"],
            ),
            _metric_table(
                cell,
                ["variant", "window"],
                ["gyro_mae", "accel_mae", "gyro_bias_norm", "accel_bias_norm"],
            ),
        ]
    cell_delta = module[
        (module.split == "test")
        & module.aggregation.eq("dataset")
        & module.metric.isin(["gyro_rmse", "accel_rmse"])
    ]
    cell_mask = pd.Series(
        list(zip(cell_delta.protocol, cell_delta.dataset_id)), index=cell_delta.index
    ).isin(CELLS)
    cell_delta = cell_delta[cell_mask]
    lines += [
        "## 320点模块贡献：有收益还是无一致收益",
        "",
        "参照为相同单元、相同种子、相同320点上下文的完整 CNN-BiLSTM。相对差先逐种子计算，再求均值±样本标准差，绝非用两个种子均值相除。",
        _paired_table(
            cell_delta[cell_delta.window.eq(320) & cell_delta.variant.isin(cfg["variants"])],
            ["protocol", "dataset_id", "variant", "metric"],
            verdict=True,
        ),
        "## 320点容量及密集监督控制",
        "",
        "以下包含容量/归一化/损失支持的变化；结合参数量表解释，不能把每项差异都归因为单个时间模块。",
        _metric_table(
            summary[
                (summary.split == "test")
                & summary.aggregation.eq("dataset")
                & summary.window.eq(320)
                & summary.variant.isin([FULL, "cnn_lstm", *cfg.get("supplemental", [])])
            ],
            ["protocol", "dataset_id", "variant"],
            ["gyro_rmse", "accel_rmse", "gyro_vector_rmse", "accel_vector_rmse"],
        ),
        _paired_table(
            cell_delta[cell_delta.variant.isin(cfg.get("supplemental", []))],
            ["protocol", "dataset_id", "variant", "metric"],
            verdict=True,
        ),
        "## 窗口配对差：处理窗口减去同模型320点",
        "",
        "正值表示320点参照更好，负值表示较短窗口更好；不因TEST结果改变验证选窗。",
        _paired_table(
            windows[
                (windows.split == "test")
                & windows.aggregation.eq("dataset")
                & windows.metric.isin(["gyro_rmse", "accel_rmse"])
            ],
            ["protocol", "dataset_id", "variant", "window", "metric"],
            verdict=True,
        ),
        "## CNN×循环块交互（描述性）",
        "",
        "逐种子、相同窗口与支持计算 full − bilstm_only − cnn_only + mlp_only。负值提示此指标加性误差尺度上的潜在互补性；该交互不是相对百分比，不能作为独立录制上的显著性结论。",
    ]
    interaction_rows = []
    inter = interactions[
        (interactions.split == "test")
        & interactions.aggregation.eq("dataset")
        & interactions.metric.isin(["gyro_rmse", "accel_rmse"])
    ]
    for key, group in inter.groupby(["protocol", "dataset_id", "window", "metric"], sort=True):
        interaction_rows.append(
            [
                *key,
                group.seed.nunique(),
                _estimate(
                    group.interaction.mean(), group.interaction.std(ddof=1), group.seed.nunique()
                ),
            ]
        )
    lines += [
        _table(
            ["protocol", "dataset_id", "window", "metric", "n", "交互均值±SD"], interaction_rows
        ),
        "## 仅由VALID选出的窗口及固定TEST读数",
        "",
        "VALID 先对每个设备对取 gyro/Raw 与 accel/Raw 分量 RMSE 比值的均值，再在数据集内等权设备对平均，最后等权数据集平均。按种子组均值最小选窗；完全相同的组均值选择较短窗口。pooled 的 ALiMU/Penn 共享一个选窗，不能各自用TEST选。",
        _table(
            ["protocol", "variant", "VALID选窗", "VALID score均值±SD", "n", "候选窗口"],
            [
                [
                    r.protocol,
                    r.variant,
                    r.selected_window,
                    _estimate(r.score_mean, r.score_sd, r.n),
                    r.n,
                    r.candidate_windows,
                ]
                for r in selected.itertuples()
            ],
        ),
    ]
    selected_test = summary.merge(
        selected[["protocol", "variant", "selected_window"]], on=["protocol", "variant"]
    )
    selected_test = selected_test[
        selected_test.split.eq("test")
        & selected_test.aggregation.eq("dataset")
        & selected_test.window.eq(selected_test.selected_window)
    ]
    lines += [
        _metric_table(
            selected_test,
            ["protocol", "dataset_id", "variant", "window"],
            ["gyro_rmse", "accel_rmse", "normalized_score"],
        ),
        "## 每设备对与每原始录制",
        "",
        "设备对及 original_sequence_id 按产物原值列出。不同设备的序列名称可能对应同一次物理录制；这些行不能视为独立物理重复。下表包括全部主模型窗口及320点补充控制；更多指标及所有VALID结果见 summary.csv。",
    ]
    for aggregation in ("pair", "recording"):
        breakdown = summary[summary.split.eq("test") & summary.aggregation.eq(aggregation)]
        ids = (
            ["protocol", "dataset_id", "pair_id"]
            + (["original_sequence_id"] if aggregation == "recording" else [])
            + ["variant", "window"]
        )
        lines += [
            f"### {aggregation}",
            "",
            _metric_table(
                breakdown, ids, ["gyro_rmse", "accel_rmse", "gyro_vector_rmse", "accel_vector_rmse"]
            ),
        ]
    lines += [
        "### 各轴RMSE（全部模型与窗口，按评估域）",
        "",
        _metric_table(
            summary[summary.split.eq("test") & summary.aggregation.eq("dataset")],
            ["protocol", "dataset_id", "variant", "window"],
            [a + "_rmse" for a in ("gx", "gy", "gz", "ax", "ay", "az")],
        ),
        "## 参数量、训练成本与延迟",
        "",
        f"延迟记录 batch1，warmup={cfg.get('evaluation', {}).get('latency_warmup', '未登记')}、repeats={cfg.get('evaluation', {}).get('latency_repeats', '未登记')}。主runner使用CUDA同步计时。表中 batch1_ms_median 是每种子的重复计时中位数，再跨种子求均值±SD；batch1_ms_mean 同理。",
        "elapsed_seconds 包含训练与VALID计算；多作业并发时各作业墙钟之和不是整个实验墙钟。CUDA计时受机器负载影响，不能直接作不同机器或负载下的速度排名。每窗口输出中心40点评价，参数量不代表标签或上下文量相同。",
    ]
    cost_rows = []
    cost_metrics = [
        "parameters",
        "best_epoch",
        "actual_epochs",
        "amp_skipped_steps",
        "elapsed_seconds",
        "batch1_ms_mean",
        "batch1_ms_median",
    ]
    for key, group in compute.groupby(["protocol", "variant", "window"], sort=True):
        lookup = group.set_index("metric")
        cost_rows.append(
            [
                *key,
                int(group.n.iloc[0]),
                *[
                    _estimate(lookup.loc[m, "mean"], lookup.loc[m, "sd"], lookup.loc[m, "n"])
                    for m in cost_metrics
                ],
            ]
        )
    lines += [
        _table(["protocol", "variant", "window", "n", *cost_metrics], cost_rows),
        "## 解释边界与追溯",
        "",
        "ALiMU TEST 仅2次物理录制、由3设备共享；Penn bs TEST 仅1次，限制对新录制/场景的外推。实际支持身份以 data_manifest.json 及 supports/ 中的 records/test_support 为准，不能用序列行数或3个训练种子增加物理录制计数。",
        "同号的种子差、负交互和较低验证分数均属于当前协议与支持上的描述性证据。没有跨种子一致改善时，明确报告无一致收益；不作普遍的模块必要性、绝对历史复现或部署因果性结论。",
        "本脚本读取已有指标、配置与登记产物，不加载检查点、不训练、不重新预测。输入覆盖/元数据校验完成；检查点文件哈希、共享初始化、采样/支持键、best epoch最小值及逐预测物理指标回放需由独立核验器完成，本报告不冒充该核验。",
        "[图及图注索引](figures/figure_index.md)；[运行来源](provenance.json)；[配置](config.yaml)；[数据清单](data_manifest.json)；[作业计划](planned_jobs.json)；[训练登记](fit_registry.csv)；[TEST前锁定](selections.lock.json)；[报告追溯及文件哈希](reporting_provenance.json)。",
        "[summary.csv](summary.csv) 每行一个指标，mean/sd/n/samples；[module_paired_deltas.csv](module_paired_deltas.csv) 与 [window_paired_deltas.csv](window_paired_deltas.csv) 保留逐种子 value/reference_value/delta/relative_delta_pct 与作业身份；[factorial_interactions.csv](factorial_interactions.csv) 保留四项原值与逐种子交互；[validation_selected_windows.csv](validation_selected_windows.csv) 记录仅VALID选窗；[compute_summary.csv](compute_summary.csv) 为成本种子汇总。相对参照为零时百分比为空，不写成零增益。",
        "",
        _table(["报告输入", "SHA256"], sorted(hashes.items())),
    ]
    return "\n\n".join(lines) + "\n"


def render_report(run_dir, make_figures=True):
    """Write derived CSVs, Chinese report and figures for a COMPLETE run only.

    Raises before writing on missing/malformed/incomplete inputs. No fallback to
    outputs/latest, prior results, partial fits or training is permitted.
    Returns the JSON-compatible receipt also saved as reporting_provenance.json.
    """
    run = Path(run_dir).resolve()
    if any("_guard_diagnostic" in part for part in run.parts):
        raise ValueError("Excluded _guard_diagnostic run; use the formal source-of-truth run")
    LOG.info(
        "SKIPS *_guard_diagnostic and unplanned fit directories; READS only explicit run-local tables"
    )
    hashes = {}
    cfg = _read(run, "config.yaml", "yaml", hashes)
    planned = _read(run, "planned_jobs.json", "json", hashes)
    provenance = _read(run, "provenance.json", "json", hashes)
    complete = _read(run, "training.complete.json", "json", hashes)
    lock = _read(run, "selections.lock.json", "json", hashes)
    registry = _read(run, "fit_registry.csv", "csv", hashes)
    costs = _read(run, "compute_cost.csv", "csv", hashes)
    raw_metrics = _read(run, "metrics_per_seed.csv", "csv", hashes)
    _read(run, "data_manifest.json", "json", hashes)
    if hashes["config.yaml"] != provenance.get("config_sha256") or hashes[
        "data_manifest.json"
    ] != provenance.get("manifest_sha256"):
        raise ValueError("Frozen config/manifest hashes disagree with provenance")
    metrics = prepare_metrics(raw_metrics)
    _validate_complete(cfg, planned, provenance, complete, lock, registry, costs, metrics)
    tables = {
        "summary.csv": summarize_metrics(metrics),
        "module_paired_deltas.csv": paired_deltas(metrics, "module"),
        "window_paired_deltas.csv": paired_deltas(metrics, "window"),
        "factorial_interactions.csv": factorial_interactions(metrics),
        "validation_selected_windows.csv": select_validation_windows(metrics, cfg),
        "compute_summary.csv": summarize_compute(registry, costs),
    }
    historical = "本次准备阶段对已查阅本地历史材料的审计未找到严格共享CNN-BiLSTM窗口/模块消融；此结论不扩展到未查阅或未登记实验。"
    audit = ROOT / "runs/cnn_bilstm_ablation_20261106_preparation/prior_ablation_audit.md"
    if audit.is_file():
        LOG.info("READS historical audit %s", audit)
        audit_text = audit.read_text(encoding="utf-8")
        historical += f" 历史审计来源 `{audit.parent.name}/{audit.name}`，SHA256 `{_sha(audit)}`。"
        if "未找到严格" not in audit_text:
            raise ValueError("Historical audit conclusion changed; review report wording")
        hashes["historical_audit:" + str(audit)] = _sha(audit)
    else:
        LOG.info("SKIPS unavailable preparation audit %s; historical scope remains limited", audit)
    # Concurrent evaluation writes a growing CSV; do not publish a moving read.
    for name, digest in hashes.items():
        path = (
            Path(name.split("historical_audit:", 1)[1])
            if name.startswith("historical_audit:")
            else run / name
        )
        if _sha(path) != digest:
            raise ValueError("Input changed during reporting: " + name)
    for name, table in tables.items():
        LOG.info("WRITES %s rows=%d", run / name, len(table))
        table.to_csv(run / name, index=False, float_format="%.17g")
    figures = []
    if make_figures:
        from virtual_imu.viz.plots_cnn_ablation import render_ablation_figures

        figures = render_ablation_figures(
            tables["summary.csv"],
            tables["module_paired_deltas.csv"],
            tables["compute_summary.csv"],
            run / "figures",
            cfg,
        )
    else:
        LOG.info("SKIPS figures: make_figures=False")
    text = _report_text(
        run,
        cfg,
        provenance,
        tables["summary.csv"],
        tables["module_paired_deltas.csv"],
        tables["window_paired_deltas.csv"],
        tables["factorial_interactions.csv"],
        tables["validation_selected_windows.csv"],
        tables["compute_summary.csv"],
        hashes,
        historical,
    )
    (run / "report.md").write_text(text, encoding="utf-8")
    LOG.info("WRITES %s", run / "report.md")
    paths = [run / n for n in tables] + [run / "report.md"]
    if make_figures:
        paths += [run / "figures/figure_index.md"] + [
            Path(p) for entry in figures for p in entry["paths"]
        ]
    receipt = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        run_directory=str(run),
        git_commit=provenance.get("git_commit"),
        training_performed=False,
        independent_prediction_replay_performed=False,
        inputs=hashes,
        reporting_source_sha256={
            "scripts/report_cnn_bilstm_ablation.py": _sha(__file__),
            "src/virtual_imu/viz/plots_cnn_ablation.py": _sha(
                ROOT / "src/virtual_imu/viz/plots_cnn_ablation.py"
            ),
        },
        outputs={p.relative_to(run).as_posix(): _sha(p) for p in paths},
    )
    (run / "reporting_provenance.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    LOG.info(
        "WRITES %s; SKIPS training, checkpoint loading, prior runs and outputs/latest",
        run / "reporting_provenance.json",
    )
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Source-of-truth run containing completed fits and evaluation",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        render_report(args.run_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOG.error("SKIPS formal reporting: inputs incomplete/inconsistent: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
