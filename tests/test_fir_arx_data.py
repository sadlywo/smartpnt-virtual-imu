from __future__ import annotations

from pathlib import Path

import pytest

from virtual_imu.fir_arx_research.data import UnifiedDataset


@pytest.mark.integration
def test_frozen_protocol_is_recording_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = root / "data/processed/unified_imu_v2/manifest.json"
    if not manifest.is_file():
        pytest.skip("frozen unified_imu_v2 data is unavailable")
    dataset = UnifiedDataset(root, manifest, verify_hashes=False)
    protocol = dataset.protocol("alimu_only")
    assert set(protocol.frames) == {"train", "validation", "test"}
    for role in protocol.frames:
        assert len(protocol.frames[role]) > 0
        assert protocol.frames[role]["sequence_id"].nunique() > 0
    assert protocol.audit["rows"]["train"] > protocol.audit["rows"]["test"]
