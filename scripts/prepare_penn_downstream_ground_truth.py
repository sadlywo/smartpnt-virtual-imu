"""Acquire pinned official independent Penn optical poses and audit alignment.

This command never fits a correction model, clock lag, or downstream algorithm.
All downloaded/generated data remain outside version control under data/ and runs/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from scipy.spatial.transform import Rotation

from virtual_imu.eval.downstream_attitude import (
    ROTATION_COLUMNS,
    align_pose_ground_truth,
    read_penn_pose,
    source_clock_origin_s,
)

OFFICIAL_REPOSITORY = "https://github.com/daniilidis-group/penncosyvio"
FORMAT_SOURCE = "https://daniilidis-group.github.io/penncosyvio/file_format/"
OVERVIEW_SOURCE = "https://daniilidis-group.github.io/penncosyvio/"
CALIBRATION_SOURCE = "https://daniilidis-group.github.io/penncosyvio/extrinsic_calib/"
MAX_FILE_BYTES = 8 * 1024 * 1024


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    logger.info("WRITES {}", path)


def _clock_audit(frame: pd.DataFrame, truth: pd.DataFrame, max_gap_s: float) -> dict[str, Any]:
    times = frame["timestamp_s"].to_numpy(dtype=float)
    origin_s = source_clock_origin_s(frame)
    aligned = align_pose_ground_truth(times + origin_s, truth, max_gap_s=max_gap_s)
    valid = aligned["gt_valid"].to_numpy()
    rotations = aligned[ROTATION_COLUMNS].to_numpy(dtype=float).reshape(-1, 3, 3)
    dt = np.diff(times)
    supported = valid[:-1] & valid[1:] & (dt > 0) & (dt <= 1.5 / 40)
    index = np.flatnonzero(supported)
    relative = np.swapaxes(rotations[index], 1, 2) @ rotations[index + 1]
    angular_rate = Rotation.from_matrix(relative).as_rotvec() / dt[index, None]
    agreement = {}
    for prefix in ("low", "high"):
        omega = frame[[f"{axis}_{prefix}" for axis in ("gx", "gy", "gz")]].to_numpy(dtype=float)[
            index
        ]
        correlations = []
        for axis in range(3):
            correlation = np.corrcoef(omega[:, axis], angular_rate[:, axis])[0, 1]
            correlations.append(float(correlation) if np.isfinite(correlation) else None)
        errors = np.linalg.norm(omega - angular_rate, axis=1)
        agreement[prefix] = {
            "axis_correlation": correlations,
            "angular_rate_vector_rmse_radps": float(np.sqrt(np.mean(errors**2))),
            "angular_rate_vector_median_radps": float(np.median(errors)),
        }
    force = frame[[f"{axis}_high" for axis in ("ax", "ay", "az")]].to_numpy(dtype=float)
    world_force = np.einsum("nij,nj->ni", rotations[valid], force[valid])
    truth_times = truth["timestamp_s"].to_numpy(dtype=float)
    pose_gaps = np.diff(truth_times)
    return {
        "timestamp_policy": "official GT time = timestamp_s + constant(source_timestamp_s - timestamp_s); available_timestamp_s never used",
        "gt_timestamp_origin_s": origin_s,
        "clock_fit": "none; published common relative clocks used without fitted lag or axis alignment",
        "imu_rows": len(frame),
        "gt_rows": len(truth),
        "imu_start_s": float(times[0]),
        "imu_end_s": float(times[-1]),
        "gt_start_s": float(truth_times[0]),
        "gt_end_s": float(truth_times[-1]),
        "gt_median_rate_hz": float(1 / np.median(pose_gaps)),
        "gt_max_gap_s": float(pose_gaps.max()),
        "gt_gaps_over_policy": int((pose_gaps > max_gap_s).sum()),
        "supported_imu_rows": int(valid.sum()),
        "coverage_fraction": float(valid.mean()),
        "supported_consecutive_intervals": len(index),
        "zero_lag_pose_gyro_agreement": agreement,
        "high_world_specific_force_component_median_mps2": np.median(world_force, axis=0).tolist(),
        "high_world_specific_force_norm_median_mps2": float(
            np.median(np.linalg.norm(world_force, axis=1))
        ),
        "warning": "Pose differentiation and filtered 40Hz IMU differ in bandwidth. Agreement is a schema/clock diagnostic, not independent clock calibration or a fitted correction.",
    }


def prepare(config_path: Path) -> dict[str, Any]:
    logger.info("READS preparation YAML {}", config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    revision = str(config["source_revision"])
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("source_revision must pin a full official Git commit hash")
    sequences = list(config.get("sequences", ["af", "as", "bf", "bs"]))
    if (
        not sequences
        or len(set(sequences)) != len(sequences)
        or not set(sequences) <= {"af", "as", "bf", "bs"}
    ):
        raise ValueError("sequences must be unique official Penn recording names")
    raw_root = Path(config["raw_root"])
    output_root = Path(config["output_root"])
    paired_root = Path(config["paired_root"])
    audit_root = Path(config["audit_root"])
    audit_root.mkdir(parents=True, exist_ok=True)
    logger.add(audit_root / "preparation.log", encoding="utf-8")
    previous_path = raw_root / "manifest.json"
    previous = {}
    if previous_path.exists():
        logger.info("READS existing raw GT provenance {}", previous_path)
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous.get("source_revision") != revision:
            raise ValueError("Existing raw GT has a different revision; use a new raw_root")
    prior_records = {record["sequence"]: record for record in previous.get("records", [])}
    records = []
    for sequence in sequences:
        url = f"https://raw.githubusercontent.com/daniilidis-group/penncosyvio/{revision}/data/ground_truth/{sequence}/pose.txt"
        path = raw_root / sequence / "pose.txt"
        if path.exists():
            if sequence not in prior_records or _sha(path) != prior_records[sequence]["sha256"]:
                raise ValueError(f"Unverified existing GT file; preserve and inspect it: {path}")
            logger.info("SKIPS verified downloaded pose {}", path)
        else:
            logger.info("READS bounded official GT URL {}", url)
            request = urllib.request.Request(
                url, headers={"User-Agent": "virtual-imu-downstream-audit"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(MAX_FILE_BYTES + 1)
            if len(payload) > MAX_FILE_BYTES:
                raise ValueError(f"Official pose exceeds bounded text size: {url}")
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".txt.part")
            partial.write_bytes(payload)
            read_penn_pose(partial)
            partial.replace(path)
            logger.info("WRITES downloaded raw pose {} ({} bytes)", path, len(payload))
        truth = read_penn_pose(path)
        truth["original_sequence_id"] = sequence
        output_path = output_root / f"{sequence}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prior = prior_records.get(sequence, {})
        if output_path.exists() and _sha(output_path) == prior.get("processed_sha256"):
            logger.info("SKIPS verified processed GT {}", output_path)
        elif output_path.exists():
            raise ValueError(f"Unverified processed GT exists; use new output_root: {output_path}")
        else:
            truth.to_parquet(output_path, index=False)
            logger.info("WRITES processed independent optical GT {}", output_path)
        matched = sorted(paired_root.glob(f"penncosyvio__{sequence}__seg*.parquet"))
        if not matched:
            raise ValueError(f"No immutable unified paired sequence found for {sequence}")
        audits = []
        for paired_path in matched:
            logger.info("READS paired stream for schema and clock audit {}", paired_path)
            paired = pd.read_parquet(paired_path)
            if (
                not paired["calibration_id"].eq("penncosyvio_published_body_extrinsics_v1").all()
                or not paired["lever_arm_compensated"].all()
            ):
                raise ValueError(
                    "Expected existing calibrated rig-body streams; do not apply ADIS transform twice"
                )
            audits.append(
                {
                    "paired_path": paired_path.as_posix(),
                    "paired_sha256": _sha(paired_path),
                    **_clock_audit(paired, truth, float(config.get("max_gt_gap_s", 0.1))),
                }
            )
        records.append(
            {
                "sequence": sequence,
                "url": url,
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha(path),
                "processed_path": output_path.as_posix(),
                "processed_sha256": _sha(output_path),
                "rows": len(truth),
                "audits": audits,
            }
        )
        # Save after every sequence so interruption never strands unverified files.
        prior_records[sequence] = records[-1]
        _write_json(
            previous_path, {"source_revision": revision, "records": list(prior_records.values())}
        )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": config_path.as_posix(),
        "config_sha256": _sha(config_path),
        "source_revision": revision,
        "official_repository": OFFICIAL_REPOSITORY,
        "format_source": FORMAT_SOURCE,
        "overview_source": OVERVIEW_SOURCE,
        "calibration_source": CALIBRATION_SOURCE,
        "ground_truth_provenance": "optical AprilTag-based rig poses; independent of LOW/HIGH inertial six-axis labels; not Vicon/MoCap",
        "frame": "world_from_rig_body (3x4 row-major); paired LOW/HIGH already rotate to rig body and compensate to rig origin",
        "position_accuracy_note": "official overview reports localization about 10cm; no numeric orientation accuracy assumed",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "implementation_sha256": {
            str(path): _sha(path)
            for path in [Path(__file__), Path("src/virtual_imu/eval/downstream_attitude.py")]
        },
        "records": records,
    }
    _write_json(output_root / "manifest.json", result)
    _write_json(audit_root / "ground_truth_audit.json", result)
    snapshot = audit_root / "config_snapshot.yaml"
    snapshot.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    logger.info("WRITES config snapshot {}", snapshot)
    logger.info("SKIPS fitting clock/axis/model parameters and skips downstream model comparisons")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.config)
    print(
        json.dumps(
            {
                "records": len(result["records"]),
                "downloaded_bytes": sum(record["bytes"] for record in result["records"]),
                "audit": str(
                    Path(yaml.safe_load(args.config.read_text(encoding="utf-8"))["audit_root"])
                    / "ground_truth_audit.json"
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
