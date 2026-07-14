from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad" / "freecad" / "tests"))

from verify_build import digest, verify_run  # noqa: E402


def test_verify_run_accepts_hashes_and_rejects_corruption(tmp_path: Path) -> None:
    artifact = tmp_path / "part.stl"
    artifact.write_bytes(b"solid printable\n")
    manifest = {
        "status": "complete",
        "artifacts": [
            {
                "relative_path": artifact.name,
                "sha256": digest(artifact),
                "size_bytes": artifact.stat().st_size,
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_run(tmp_path) == 1
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_run(tmp_path)


def test_powershell_wrapper_reports_inputs_outputs_and_skips() -> None:
    text = (ROOT / "scripts" / "build_xsens_dot2_watch.ps1").read_text(encoding="utf-8")
    for token in (
        "READ freecad=",
        "READ config=",
        "READ macro=",
        "WRITE output_root=",
        "WRITE run_dir=",
        "SKIP prior_run_count=",
        "verify_build.py",
    ):
        assert token in text
