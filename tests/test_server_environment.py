from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import pytest

from virtual_imu.io import training_bundle as bundle


def _profile():
    return {
        "profile_id": "ubuntu2204_py312_torch240_cu121",
        "python": "3.12",
        "torch": "2.4.0",
        "numpy": "1.26.4",
        "cuda": "12.1",
        "platform": "linux",
        "machine": "x86_64",
    }


def _runtime():
    return {
        "python_version": "3.12.10",
        "torch": "2.4.0+cu121",
        "numpy": "1.26.4",
        "cuda_runtime": "12.1",
        "cuda_available": True,
        "platform": "linux",
        "machine": "x86_64",
    }


def _validate(report, *, require_gpu=True):
    assert hasattr(bundle, "validate_server_environment"), "Image-specific environment validation is required"
    return bundle.validate_server_environment(report, _profile(), require_gpu=require_gpu)


def test_selected_image_runtime_is_accepted_with_cuda_build_suffix():
    original = _runtime()
    _validate(original)
    assert original == _runtime()


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("python_version", "3.11.9"),
        ("torch", "2.12.1+cu130"),
        ("torch", "2.4.1+cu121"),
        ("numpy", "2.0.0"),
        ("cuda_runtime", "13.0"),
        ("platform", "win32"),
        ("machine", "aarch64"),
    ],
)
def test_runtime_drift_from_the_selected_image_is_rejected(field, wrong):
    report = copy.deepcopy(_runtime())
    report[field] = wrong
    with pytest.raises(RuntimeError, match=r"profile|expected|requires"):
        _validate(report)


def test_cpu_diagnostics_are_explicit_and_cannot_satisfy_gpu_mode():
    report = _runtime()
    report.update(cuda_available=False, cuda_runtime=None, torch="2.4.0+cpu")
    _validate(report, require_gpu=False)
    with pytest.raises(RuntimeError, match=r"GPU|CUDA"):
        _validate(report)


def test_missing_profile_fields_fail_clearly():
    assert hasattr(bundle, "validate_server_environment")
    with pytest.raises(ValueError, match=r"profile|Missing"):
        bundle.validate_server_environment(_runtime(), {"profile_id": "broken"})


def test_gpu_diagnostics_cannot_skip_the_compiled_cuda_version_check():
    report = _runtime()
    report["cuda_runtime"] = None
    with pytest.raises(RuntimeError, match="expected CUDA"):
        _validate(report, require_gpu=False)


def test_numpy_torch_abi_failure_is_not_mistaken_for_a_healthy_runtime(monkeypatch):
    def broken_bridge(_value):
        raise RuntimeError("Numpy is not available")

    fake_torch = SimpleNamespace(
        __version__="2.4.0+cpu",
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None),
        from_numpy=broken_bridge,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    with pytest.raises(RuntimeError, match="bridge failed"):
        bundle.check_training_environment()
