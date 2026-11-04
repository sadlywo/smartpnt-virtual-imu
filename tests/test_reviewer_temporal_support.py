"""Display semantics guards; no model execution or numeric research data."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/reviewer_temporal_support.py"


@lru_cache(maxsize=1)
def classifier():
    assert SCRIPT.is_file(), "The required temporal display classifier is missing"
    spec = importlib.util.spec_from_file_location("reviewer_temporal_support_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.classify_temporal_support


def candidates():
    return yaml.safe_load((ROOT / "configs/reviewer_hpo.yaml").read_text(encoding="utf-8"))[
        "candidates"
    ]


def test_paired_causality_distinguishes_future_context():
    classify = classifier()
    causal = {"tcn", "gru", "deepgyro_adapter"}
    for candidate in candidates():
        if "paired" not in candidate["tasks"]:
            continue
        row = classify("paired", candidate["id"], candidate["model"]["type"])
        expected = "sample_causal_eval" if candidate["id"] in causal else "window_lookahead_eval"
        assert row["temporal_class"] == expected
        assert row["temporal_label"] == ("Causal" if candidate["id"] in causal else "Window")
        assert row["temporal_order"] == (0 if candidate["id"] in causal else 1)


def test_navigation_describes_endpoint_not_internal_per_sample_lookahead():
    classify = classifier()
    methods = [candidate["id"] for candidate in candidates()]
    methods.extend(["zero", "train_mean", "ridge", "ridge_tuned"])
    for method in methods:
        row = classify("navigation", method)
        assert row["temporal_class"] == "window_endpoint"
        assert row["temporal_label"] == "Window endpoint"
        assert row["temporal_order"] == 2
        assert "endpoint" in row["temporal_note"].lower()
        assert "noncausal" not in row["temporal_note"].lower()


def test_frozen_source_factory_aliases_are_checked_in_both_tasks():
    classify = classifier()
    for candidate in candidates():
        for task in candidate["tasks"]:
            actual = classify(task, candidate["id"], candidate["model"]["type"])
            assert actual == classify(task, candidate["id"])
            wrong_type = "gru" if candidate["model"]["type"] != "gru" else "tcn"
            with pytest.raises(ValueError, match="model_type"):
                classify(task, candidate["id"], wrong_type)


@pytest.mark.parametrize(
    "method",
    [
        "identity",
        "bias_mean",
        "affine",
        "ridge_affine",
        "learned_fir_causal",
        "arx_linear",
        "ridge_tuned",
        "fir_tuned",
        "arx_tuned",
    ],
)
def test_paired_classics_have_point_or_past_low_support(method):
    row = classifier()("paired", method)
    assert row["temporal_class"] == "sample_causal_eval"
    assert row["temporal_label"] == "Causal"
    assert row["temporal_order"] == 0
    with pytest.raises(ValueError, match="model_type"):
        classifier()("paired", method, "tcn")


@pytest.mark.parametrize(
    "task,method",
    [
        ("paired", "tartanimu_window_adapter"),
        ("paired", "zero"),
        ("paired", "train_mean"),
        ("paired", "ridge"),
        ("navigation", "identity"),
        ("navigation", "fir_tuned"),
        ("paired", "deepgyro_backbone"),
        ("navigation", "tlio_resnet18"),
        ("navigation", "tartanimu_window"),
        ("paired", "model"),
        ("paired", "HIGH"),
        ("paired", "new_model"),
        ("paired", "tcn/tuned"),
        ("unknown", "tcn"),
        (None, "tcn"),
        ([], "tcn"),
        ("paired", None),
    ],
)
def test_unknown_or_task_incompatible_identity_fails_closed(task, method):
    with pytest.raises(ValueError):
        classifier()(task, method)


def test_interpretation_is_conditional_and_returned_values_are_independent():
    classify = classifier()
    for task, method in [("paired", "deepgyro_adapter"), ("paired", "transformer"), ("navigation", "gru")]:
        row = classify(task, method)
        assert set(row) == {
            "temporal_class",
            "temporal_label",
            "temporal_label_zh",
            "temporal_note",
            "temporal_order",
        }
        assert row["temporal_label_zh"]
        note = row["temporal_note"].lower()
        assert "prepared" in note and "eval" in note and "frozen" in note
        assert "not an end-to-end online guarantee" in note
        row["temporal_class"] = "tampered_by_consumer"
        assert classify(task, method)["temporal_class"] != "tampered_by_consumer"


def test_fresh_interpreter_import_has_no_scientific_or_model_dependencies():
    assert SCRIPT.is_file(), "The required temporal display classifier is missing"
    code = (
        "import importlib.util,sys,json; "
        "spec=importlib.util.spec_from_file_location('temporal',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "row=module.classify_temporal_support('paired','deepgyro_adapter','deepgyro_backbone'); "
        "assert not any(n.split('.')[0] in {'torch','numpy','pandas','yaml','virtual_imu'} for n in sys.modules); "
        "print(json.dumps(row))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert json.loads(result.stdout)["temporal_class"] == "sample_causal_eval"
