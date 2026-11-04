from __future__ import annotations

import importlib.util
from pathlib import Path


def test_evaluate_baselines_script_imports_without_training_stack():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_baselines.py"
    spec = importlib.util.spec_from_file_location("evaluate_baselines_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)
