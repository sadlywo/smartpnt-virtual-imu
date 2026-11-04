"""Conditional temporal labels for the frozen reviewer comparison's displays.

This module interprets paired *per-sample* outputs given canonical prepared
40 Hz inputs, eval mode and frozen normalization/BatchNorm statistics. It
does not establish causality of preprocessing, window scheduling, streaming
state management or the whole system. Navigation describes displacement at
the supplied window's endpoint, rather than internal per-sample features.

Audited source references (repository-relative; no imports/reads performed):
* configs/reviewer_hpo.yaml: frozen family IDs, task membership and model types.
* src/virtual_imu/models/tcn.py: TemporalBlock's right chomp/pointwise head.
* src/virtual_imu/models/gru.py: unidirectional GRU and per-step linear head.
* src/virtual_imu/models/published_backbones.py: DeepGyro left padding and
  eval-frozen BatchNorm; TLIO centered convolutions/pooling and dense expansion.
* src/virtual_imu/models/advanced.py: centered PlainCNN and CNN-BiLSTM blocks.
* src/virtual_imu/models/transformer.py: unmasked full-window encoder.
* src/virtual_imu/models/time_series_attention.py: Informer/PatchTST/iTransformer
  encoder adapters' window context, including Informer query selection.
* src/virtual_imu/eval/baselines.py:_lag_features: current/past LOW lags and
  backward LOW difference; no HIGH autoregressive feedback.
* src/virtual_imu/training/gt_trainer.py:_project_prediction: window last head.

Labels are independent of device-information scope, variant, measured values
and ranks. This is display metadata, with no model/data dependency or I/O.
"""

from __future__ import annotations

_MODEL_TYPES = {
    "tcn": "tcn",
    "plain_cnn": "plain_cnn",
    "gru": "gru",
    "cnn_bilstm": "cnn_bilstm",
    "transformer": "transformer",
    "informer": "informer",
    "patchtst": "patchtst",
    "itransformer": "itransformer",
    "deepgyro_adapter": "deepgyro_backbone",
    "tlio_resnet18_adapter": "tlio_resnet18",
    "tartanimu_window_adapter": "tartanimu_window",
}
_PAIRED_CAUSAL_FAMILIES = frozenset({"tcn", "gru", "deepgyro_adapter"})
_PAIRED_CLASSICS = frozenset(
    {
        "identity",
        "bias_mean",
        "affine",
        "ridge_affine",
        "learned_fir_causal",
        "arx_linear",
        "ridge_tuned",
        "fir_tuned",
        "arx_tuned",
    }
)
_NAVIGATION_BASELINES = frozenset({"zero", "train_mean", "ridge", "ridge_tuned"})
_CONDITIONS = (
    "Conditional on canonical prepared 40 Hz inputs, eval mode, and frozen "
    "normalization/BatchNorm statistics; not an end-to-end online guarantee."
)


def classify_temporal_support(task: str, method: str, model_type: str | None = None) -> dict:
    """Return new display fields for an exact frozen family/canonical method.

    Optional ``model_type`` checks the neural factory type, including the
    explicit DeepGyro/TLIO/Tartan family-to-factory aliases. It is invalid for
    classical methods, which do not have a neural factory model type.
    Unknown/task-incompatible identities fail closed; source factory aliases
    are not accepted as replacement method IDs.
    """
    if not isinstance(task, str) or task not in {"paired", "navigation"}:
        raise ValueError(f"Unknown temporal task: {task!r}")
    if not isinstance(method, str):
        raise ValueError(f"Unknown temporal method: {method!r}")
    neural = method in _MODEL_TYPES
    if task == "paired":
        supported = (neural and method != "tartanimu_window_adapter") or method in _PAIRED_CLASSICS
    else:
        supported = neural or method in _NAVIGATION_BASELINES
    if not supported:
        raise ValueError(f"Unknown or task-incompatible temporal method: {task}/{method}")
    if model_type is not None and (
        not isinstance(model_type, str) or not neural or model_type != _MODEL_TYPES[method]
    ):
        raise ValueError(f"Incompatible model_type {model_type!r} for temporal method {task}/{method}")

    if task == "navigation":
        category, label, label_zh, order = "window_endpoint", "Window endpoint", "窗口终点", 2
        interpretation = (
            "Window displacement is predicted at its endpoint from the supplied "
            "window; this label does not classify internal per-sample features. "
        )
    elif method in _PAIRED_CAUSAL_FAMILIES or method in _PAIRED_CLASSICS:
        category, label, label_zh, order = "sample_causal_eval", "Causal", "逐点因果", 0
        interpretation = (
            "A paired output at sample t uses only current/past prepared LOW "
            "samples, conditional on fixed fitted parameters/statistics. "
        )
    else:
        category, label, label_zh, order = "window_lookahead_eval", "Window", "窗口未来上下文", 1
        interpretation = (
            "A paired output at sample t may use later prepared LOW samples "
            "inside its supplied window; lookahead depends on the architecture/context. "
        )
    return {
        "temporal_class": category,
        "temporal_label": label,
        "temporal_label_zh": label_zh,
        "temporal_note": interpretation + _CONDITIONS,
        "temporal_order": order,
    }
