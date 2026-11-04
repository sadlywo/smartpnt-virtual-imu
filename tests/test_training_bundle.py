from __future__ import annotations

import hashlib
import importlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

DATASET_ROOT = Path("data/processed/unified_imu_v2")
INPUT_COLUMNS = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
TARGET_COLUMNS = [column.replace("_low", "_high") for column in INPUT_COLUMNS]


def _module():
    assert importlib.util.find_spec("virtual_imu.io.training_bundle") is not None, (
        "A training bundle implementation is required"
    )
    return importlib.import_module("virtual_imu.io.training_bundle")


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _rewrite_inventory(root: Path, inventory: dict) -> None:
    _json(root / "bundle_manifest.json", inventory)
    checksums = {entry["path"]: entry["sha256"] for entry in inventory["entries"]}
    checksums["bundle_manifest.json"] = hashlib.sha256(
        (root / "bundle_manifest.json").read_bytes()
    ).hexdigest()
    (root / "SHA256SUMS").write_text(
        "".join(f"{checksums[name]}  {name}\n" for name in sorted(checksums)), encoding="utf-8"
    )


def _project(tmp_path: Path, dataset_root: Path = DATASET_ROOT) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    for name, contents in {
        "pyproject.toml": '[project]\nname="fixture"\nversion="0.1"\nrequires-python=">=3.11"\n',
        "uv.lock": 'version=1\nrequires-python=">=3.11"\n',
        "README.md": "a current README\n",
        "AGENTS.md": "rules\n",
        "src/virtual_imu/new_untracked.py": "VALUE = 'current dirty source'\n",
        "scripts/train.py": "print('paired')\n",
        "scripts/train_gt_supervised.py": "print('navigation')\n",
        "scripts/server/preflight.py": "print('preflight')\n",
        "scripts/server/bootstrap.sh": "#!/usr/bin/env bash\n",
        "scripts/server/run.sh": "#!/usr/bin/env bash\n",
        "configs/train_unified_paired.yaml": "seed: 7\n",
        "configs/train_unified_navigation.yaml": "seed: 7\n",
        "docs/SERVER_TRAINING.md": "交接\n",
        "tests/test_current.py": "def test_current(): pass\n",
    }.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    for name in (
        ".git/config",
        ".venv/site-packages/leak.py",
        "data/raw/bag.bin",
        "data/processed/old/train.parquet",
        "runs/old/best.ckpt",
        "outputs/old/figure.svg",
        "src/virtual_imu/__pycache__/cached.pyc",
        "src/virtual_imu/leak.parquet",
        "scripts/.env",
        "configs/credentials.yaml",
        "configs/private_key.pem",
        "docs/private_key.md",
        "tests/cache/leak.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must not ship")
    dataset = root / dataset_root
    records = []
    strict_records = []
    label_schema = {"tasks": {}}
    for task in ("paired_signal", "navigation"):
        targets = (
            TARGET_COLUMNS
            if task == "paired_signal"
            else ["gt_x_local_m", "gt_v_east_mps", "gt_heading_deg"]
        )
        label_schema["tasks"][task] = {"input_columns": INPUT_COLUMNS, "target_columns": targets}
        splits = {name: [] for name in ("train", "validation", "test")}
        for index, split in enumerate(splits):
            sequence = f"fixture_{task}_{split}"
            path = dataset_root / "sequences" / task / "fixture" / f"{sequence}.parquet"
            table = pd.DataFrame(
                {column: [0.1, 0.2, 0.3, 0.4] for column in INPUT_COLUMNS + targets}
            )
            table["timestamp_s"] = [0.0, 0.025, 0.05, 0.075]
            table["sequence_id"] = sequence
            table["training_eligible"] = True
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(root / path, index=False)
            digest = hashlib.sha256((root / path).read_bytes()).hexdigest()
            record = {
                "task_view": task,
                "dataset_id": "fixture",
                "pair_id": f"pair_{index}",
                "sequence_id": sequence,
                "path": path.as_posix(),
                "sha256": digest,
                "fingerprint": f"fingerprint_{task}_{index}",
                "training_eligible": True,
            }
            records.append(record)
            splits[split].append(dict(record))
            if task == "paired_signal":
                strict_records.append({**record, "strict_path": path.as_posix()})
        filename = "paired_default.json" if task == "paired_signal" else "navigation_default.json"
        _json(dataset / "splits" / filename, {"task_view": task, "splits": splits})
    _json(dataset / "manifest.json", {"records": records, "build_run": "runs/unified_fixture"})
    _json(dataset / "strict_manifest.json", {"records": strict_records})
    _json(dataset / "label_schema.json", label_schema)
    _json(dataset / "normalization.json", {"fit_split": "train"})
    _json(
        dataset / "archive" / "duplicates.json",
        {"records": [{"sequence_id": "duplicate", "duplicate_of": "fixture"}]},
    )
    (dataset / "config_snapshot.yaml").write_text("seed: 7\nrate_hz: 40.0\n", encoding="utf-8")
    build_run = root / "runs/unified_fixture"
    build_run.mkdir(parents=True)
    (build_run / "config.yaml").write_text("seed: 7\nrate_hz: 40.0\n", encoding="utf-8")
    (build_run / "data_manifest.json").write_bytes((dataset / "manifest.json").read_bytes())
    (build_run / "report.md").write_text("fixture data build verified\n", encoding="utf-8")
    _json(build_run / "metrics.json", {"records": 6})
    _json(build_run / "source_state.json", {"sha256": "fixture_source_state", "files": {}})
    _json(
        build_run / "verification.json",
        {
            "passed": True,
            "checked_records": 6,
            "sha256": hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest(),
        },
    )
    (build_run / "do_not_ship.ckpt").write_bytes(
        b"must not export an unallowlisted build run artifact"
    )
    return root


def test_roundtrip_exports_current_untracked_source_and_only_curated_data(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    verified = bundle.verify_training_bundle(destination)
    assert verified["file_count"] >= 20
    assert verified["dataset_records"] == 6
    assert verified["task_records"] == {"paired_signal": 3, "navigation": 3}
    assert (
        destination / "src/virtual_imu/new_untracked.py"
    ).read_text() == "VALUE = 'current dirty source'\n"
    with tarfile.open(result.archive_path, "r:gz") as archive:
        names = archive.getnames()
        assert names == sorted(names)
        assert all(member.mtime == 0 for member in archive.getmembers())
    for excluded in (
        ".git",
        ".venv",
        "data/raw",
        "data/processed/old",
        "runs/old",
        "outputs/old",
        "tests/cache",
    ):
        assert not (destination / excluded).exists()
    assert not (destination / "configs/credentials.yaml").exists()
    assert not (destination / "docs/private_key.md").exists()
    assert not (destination / "src/virtual_imu/leak.parquet").exists()
    provenance = json.loads((result.run_dir / "source_snapshot.json").read_text())
    assert "src/virtual_imu/new_untracked.py" in provenance["sha256_by_path"]
    assert (result.run_dir / "data_manifest.json").read_bytes() == (
        root / DATASET_ROOT / "manifest.json"
    ).read_bytes()
    assert (
        json.loads((result.run_dir / "verification.json").read_text())["roundtrip_verified"] is True
    )
    assert result.archive_sha256 == hashlib.sha256(result.archive_path.read_bytes()).hexdigest()


def test_verify_rejects_a_changed_exported_file(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    (destination / "src/virtual_imu/new_untracked.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        bundle.verify_training_bundle(destination)


def test_build_rejects_explicit_superseded_v1_without_changing_manifest(tmp_path):
    bundle = _module()
    legacy_root = Path("data/processed/unified_imu_v1")
    root = _project(tmp_path, legacy_root)
    manifest_path = root / legacy_root / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    _json(
        root / legacy_root / "dataset_status.json",
        {
            "status": "superseded_diagnostic",
            "replacement": DATASET_ROOT.as_posix(),
            "reason": "reference_recording_overlap_and_partial_reuse",
        },
    )
    with pytest.raises(ValueError, match="superseded_diagnostic"):
        bundle.build_training_bundle(root, legacy_root, bundle_id="server_bundle_legacy")
    assert manifest_path.read_bytes() == original_manifest
    assert not (root / "outputs/server_bundles").exists()


def test_preflight_rejects_explicit_superseded_v1_sidecar(tmp_path):
    bundle = _module()
    legacy_root = Path("data/processed/unified_imu_v1")
    root = _project(tmp_path, legacy_root)
    result = bundle.build_training_bundle(root, legacy_root, bundle_id="server_bundle_legacy")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    _json(
        destination / legacy_root / "dataset_status.json",
        {"status": "superseded_diagnostic", "replacement": DATASET_ROOT.as_posix()},
    )
    with pytest.raises(ValueError, match="superseded_diagnostic"):
        bundle.verify_training_bundle(destination)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py",
        "/absolute.py",
        "C:/absolute.py",
        "src\\escape.py",
        "src/../escape.py",
        "src//escape.py",
        "src/./escape.py",
        "src/NUL.py",
        "src/file.",
        "src/file ",
        "src/CON.json",
        "src/file?.py",
    ],
)
def test_archive_extraction_rejects_nonportable_paths_before_writing(tmp_path, path):
    bundle = _module()
    archive_path = tmp_path / "malicious.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        safe = tarfile.TarInfo("safe.py")
        safe.size = 1
        archive.addfile(safe, io.BytesIO(b"x"))
        unsafe = tarfile.TarInfo(path)
        unsafe.size = 1
        archive.addfile(unsafe, io.BytesIO(b"x"))
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match=r"portable|unsafe|relative"):
        bundle.extract_training_bundle(archive_path, destination)
    assert not (destination / "safe.py").exists()


def test_archive_extraction_rejects_links(tmp_path):
    bundle = _module()
    archive_path = tmp_path / "symlink.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        member = tarfile.TarInfo("src/link.py")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../secret"
        archive.addfile(member)
    with pytest.raises(ValueError, match="regular"):
        bundle.extract_training_bundle(archive_path, tmp_path / "extracted")


def test_build_rejects_a_manifest_escape_path(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    manifest_path = root / DATASET_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["path"] = "../outside.parquet"
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match=r"portable|unsafe|relative"):
        bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    assert not (root / "outputs/server_bundles/server_bundle_fixture.tar.gz").exists()


def test_preflight_rejects_cross_split_sequence_leakage(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    path = root / DATASET_ROOT / "splits/paired_default.json"
    split = json.loads(path.read_text())
    split["splits"]["test"].append(dict(split["splits"]["train"][0]))
    _json(path, split)
    with pytest.raises(ValueError, match=r"split|leakage"):
        bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")


def test_preflight_rejects_missing_required_parquet_labels(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    manifest_path = root / DATASET_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest["records"][0]
    table = pd.read_parquet(root / record["path"]).drop(columns=["gx_high"])
    table.to_parquet(root / record["path"], index=False)
    record["sha256"] = hashlib.sha256((root / record["path"]).read_bytes()).hexdigest()
    _json(manifest_path, manifest)
    with pytest.raises(ValueError, match=r"schema|required|gx_high"):
        bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")


def test_builder_refuses_overwriting_an_existing_bundle(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    first = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    original = first.archive_path.read_bytes()
    with pytest.raises(FileExistsError):
        bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    assert first.archive_path.read_bytes() == original


def test_builder_rejects_an_output_directory_outside_the_exact_bundle_namespace(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    with pytest.raises(ValueError, match="outputs/server_bundles"):
        bundle.build_training_bundle(
            root,
            DATASET_ROOT,
            output_dir="outputs/server_bundles_other",
            bundle_id="server_bundle_fixture",
        )


def test_preflight_can_verify_without_site_packages(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    script = Path(__file__).resolve().parents[1] / "scripts/server/preflight.py"
    assert script.is_file(), "A standalone server preflight CLI is required"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-S",
            str(script),
            "--project-root",
            str(destination),
            "--checksums-only",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        "READS:" in completed.stdout
        and "WRITES:" in completed.stdout
        and "SKIPS:" in completed.stdout
    )
    assert "checksums_verified" in completed.stdout


def test_require_gpu_rejects_a_cpu_only_runtime(monkeypatch):
    bundle = _module()
    assert hasattr(bundle, "check_training_environment"), "A GPU runtime preflight is required"
    cpu_only = SimpleNamespace(
        __version__="fixture",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
        from_numpy=lambda value: SimpleNamespace(numpy=lambda: value),
    )
    monkeypatch.setitem(sys.modules, "torch", cpu_only)
    assert bundle.check_training_environment()["cuda_available"] is False
    with pytest.raises(RuntimeError, match=r"CUDA|GPU"):
        bundle.check_training_environment(require_gpu=True)


def test_package_cli_logs_inputs_outputs_and_exclusions(tmp_path):
    root = _project(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/package_training_bundle.py"
    assert script.is_file(), "A package CLI is required"
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(script),
            "--project-root",
            str(root),
            "--bundle-id",
            "server_bundle_cli_fixture",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (
        "READS:" in completed.stdout
        and "WRITES:" in completed.stdout
        and "SKIPS:" in completed.stdout
    )
    assert (root / "outputs/server_bundles/server_bundle_cli_fixture.tar.gz").is_file()


def test_verify_requires_a_checksum_for_every_curated_file(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    inventory = json.loads((destination / "bundle_manifest.json").read_text())
    inventory["entries"] = [
        entry
        for entry in inventory["entries"]
        if not entry["path"].endswith("fixture_paired_signal_train.parquet")
    ]
    _rewrite_inventory(destination, inventory)
    with pytest.raises(ValueError, match=r"checksum.*curated|curated.*checksum"):
        bundle.verify_training_bundle(destination)


def test_verify_rejects_an_inventory_that_exports_git_internals(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    inventory = json.loads((destination / "bundle_manifest.json").read_text())
    path = destination / ".git/config"
    path.parent.mkdir()
    path.write_bytes(b"private git metadata")
    inventory["entries"].append(
        {
            "path": ".git/config",
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kind": "source",
        }
    )
    _rewrite_inventory(destination, inventory)
    with pytest.raises(ValueError, match=r"allowlist|excluded|source"):
        bundle.verify_training_bundle(destination)


def test_navigation_fragments_of_one_original_sequence_cannot_cross_splits(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    path = root / DATASET_ROOT / "manifest.json"
    manifest = json.loads(path.read_text())
    for record in manifest["records"]:
        if record["task_view"] == "navigation":
            record["original_sequence_id"] = "same_raw_trajectory"
    _json(path, manifest)
    with pytest.raises(ValueError, match=r"leakage|original sequence"):
        bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")


def test_preflight_rejects_a_required_imu_column_with_text_values(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    path = root / DATASET_ROOT / "manifest.json"
    manifest = json.loads(path.read_text())
    record = manifest["records"][0]
    table = pd.read_parquet(root / record["path"])
    table["gx_low"] = "not a numeric signal"
    table.to_parquet(root / record["path"], index=False)
    record["sha256"] = hashlib.sha256((root / record["path"]).read_bytes()).hexdigest()
    _json(path, manifest)
    with pytest.raises(ValueError, match=r"numeric|type|schema"):
        bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")


def test_only_the_selected_data_build_run_audit_is_exported(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    for name in (
        "config.yaml",
        "data_manifest.json",
        "metrics.json",
        "report.md",
        "verification.json",
        "source_state.json",
    ):
        assert (destination / "runs/unified_fixture" / name).is_file()
        assert (destination / "runs/unified_fixture" / name).read_bytes() == (
            root / "runs/unified_fixture" / name
        ).read_bytes()
    assert not (destination / "runs/unified_fixture/do_not_ship.ckpt").exists()
    assert not (destination / "runs/old").exists()
    verified = bundle.verify_training_bundle(destination)
    assert verified["build_provenance_files"] == 6


def test_builder_requires_the_selected_data_build_to_have_passed_verification(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    path = root / "runs/unified_fixture/verification.json"
    verification = json.loads(path.read_text())
    verification["passed"] = False
    _json(path, verification)
    with pytest.raises(ValueError, match="verification"):
        bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")


def test_builder_rejects_a_missing_selected_data_build_report(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    (root / "runs/unified_fixture/report.md").unlink()
    with pytest.raises(FileNotFoundError, match=r"report\.md"):
        bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")


def test_verify_rejects_a_wrong_dataset_build_run_identity(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    inventory = json.loads((destination / "bundle_manifest.json").read_text())
    inventory["data_build_run"] = "runs/wrong_build"
    _rewrite_inventory(destination, inventory)
    with pytest.raises(ValueError, match=r"build_run|build run"):
        bundle.verify_training_bundle(destination)


def test_equivalent_live_snapshots_produce_identical_archives(tmp_path):
    bundle = _module()
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    first_root = _project(left)
    second_root = _project(right)
    first = bundle.build_training_bundle(
        first_root, DATASET_ROOT, bundle_id="server_bundle_fixture"
    )
    second = bundle.build_training_bundle(
        second_root, DATASET_ROOT, bundle_id="server_bundle_fixture"
    )
    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.archive_sha256 == second.archive_sha256


@pytest.mark.parametrize(
    ("field", "shared_value"),
    [("physical_group_id", "one_physical_recording"), ("reference_measurement_sha256", "1" * 64)],
)
def test_records_sharing_physical_or_reference_groups_cannot_cross_splits(
    tmp_path, field, shared_value
):
    bundle = _module()
    root = _project(tmp_path)
    path = root / DATASET_ROOT / "manifest.json"
    manifest = json.loads(path.read_text())
    for record in manifest["records"]:
        if record["task_view"] == "paired_signal":
            record[field] = shared_value
    _json(path, manifest)
    with pytest.raises(ValueError, match=r"group|reference|leakage"):
        bundle.validate_curated_dataset(root, DATASET_ROOT.as_posix())


def test_environment_profile_is_exported_and_bound_to_the_inventory(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    profile = root / "configs/server_environment.yaml"
    profile.write_text("profile_id: fixture_torch240\n", encoding="utf-8")
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    destination = tmp_path / "extracted"
    bundle.extract_training_bundle(result.archive_path, destination)
    expected = {
        "path": "configs/server_environment.yaml",
        "sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
    }
    assert bundle.verify_training_bundle(destination)["environment_profile"] == expected
    assert (destination / expected["path"]).read_bytes() == profile.read_bytes()
    inventory = json.loads((destination / "bundle_manifest.json").read_text())
    inventory["environment_profile"]["sha256"] = "0" * 64
    _rewrite_inventory(destination, inventory)
    with pytest.raises(ValueError, match="environment profile"):
        bundle.verify_training_bundle(destination)


def test_checksum_sidecar_is_portable_to_linux_sha256sum(tmp_path):
    bundle = _module()
    root = _project(tmp_path)
    result = bundle.build_training_bundle(root, DATASET_ROOT, bundle_id="server_bundle_fixture")
    contents = result.checksum_path.read_bytes()
    assert b"\r" not in contents
    assert contents == f"{result.archive_sha256}  {result.archive_path.name}\n".encode()
