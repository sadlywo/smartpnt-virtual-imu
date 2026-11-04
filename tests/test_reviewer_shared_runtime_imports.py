"""Exercise the deployment import path in a clean process before payload reads."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def test_clean_validator_process_resolves_aux_scripts_before_loading_contract(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = project / "scripts/server/run_reviewer_shared_controls.py"
    code = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('isolated_shared_validator', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.verify_contract_chain(pathlib.Path(sys.argv[2]), {}, {}, {}, {})
except FileNotFoundError:
    print('HELPERS_RESOLVED_BEFORE_PAYLOAD')
else:
    raise AssertionError('The empty fixture must stop before loading a contract')
"""
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=str(project / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONIOENCODING="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code, str(script), str(tmp_path)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "HELPERS_RESOLVED_BEFORE_PAYLOAD" in result.stdout


def test_pinned_launch_script_is_source_and_numeric_payload_is_still_rejected(tmp_path):
    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "shared_source_manifest_test", project / "scripts/server/run_reviewer_shared_controls.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launcher = tmp_path / "scripts/server/launch_reviewer_shared_controls.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
    relative = launcher.relative_to(tmp_path).as_posix()
    manifest = tmp_path / "aux_manifest.json"
    manifest.write_text(json.dumps({"files": {relative: module.sha(launcher)}}))
    expected = module.sha(manifest)
    assert module.verify_source_manifest(tmp_path, expected)["files"][relative] == module.sha(
        launcher
    )
    launcher.write_bytes(b"#!/usr/bin/env bash\nexit 1\n")
    import pytest

    with pytest.raises(ValueError, match="source checksum"):
        module.verify_source_manifest(tmp_path, expected)
    numeric = tmp_path / "sequences.parquet"
    numeric.write_bytes(b"SYNTHETIC opaque data bytes")
    manifest.write_text(
        json.dumps({"files": {numeric.name: hashlib.sha256(numeric.read_bytes()).hexdigest()}})
    )
    with pytest.raises(ValueError, match="non-source payload"):
        module.verify_source_manifest(tmp_path, module.sha(manifest))
