# Multi-Dataset Pilot Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download, verify, and organize a small NavINST, PennCOSYVIO, and FusionPortable IMU pilot corpus under reproducible dataset manifests.

**Architecture:** A YAML catalog declares fixed HTTP artifacts and Google Drive folder matches. A focused orchestration module creates the standard raw-data layout, invokes resumable external download tools, validates files, extracts supported archives, and writes one manifest per dataset plus a pilot summary. The CLI is intentionally independent from training and preprocessing.

**Tech Stack:** Python 3.11 project code (also exercised in the existing Python 3.9 `pinn_imu` environment), Typer, PyYAML, Loguru, curl, isolated `uvx --from gdown==6.1.0`, and standard-library tarfile/hashlib/json/shutil/subprocess.

---

## File Map

- Create `configs/multi_dataset_pilot_download.yaml`: authoritative pilot artifact catalog.
- Create `src/virtual_imu/io/pilot_download.py`: config validation, layout, download, extraction, and manifest orchestration.
- Create `scripts/download_pilot_datasets.py`: logging CLI wrapper.
- Create `tests/test_pilot_download.py`: offline tests using temporary files and injected command results.
- Keep `pyproject.toml` unchanged: invoke `gdown==6.1.0` through `uvx` because current gdown releases do not support the existing Python 3.9 environment.
- Modify `README.md`: document the pilot command and source-of-truth paths.

No Git commit is included because the user did not request one.

### Task 1: Catalog And Layout Contract

**Files:**
- Create: `configs/multi_dataset_pilot_download.yaml`
- Create: `tests/test_pilot_download.py`
- Create: `src/virtual_imu/io/pilot_download.py`

- [x] **Step 1: Write failing catalog/layout tests**

```python
def test_load_catalog_and_create_layout(tmp_path):
    catalog = load_pilot_catalog(Path("configs/multi_dataset_pilot_download.yaml"))
    assert {item["dataset"] for item in catalog["artifacts"]} == {
        "navinst_pilot", "penncosyvio_pilot", "fusionportable_pilot"
    }
    roots = ensure_dataset_layout(tmp_path, catalog)
    assert (roots["navinst_pilot"] / "00_archives").is_dir()
    assert (roots["navinst_pilot"] / "01_extracted").is_dir()
    assert (roots["navinst_pilot"] / "_manifests").is_dir()

def test_catalog_rejects_destination_escape(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("artifacts:\n  - dataset: navinst_pilot\n    destination: ../escape\n", encoding="utf-8")
    with pytest.raises(ValueError, match="destination"):
        load_pilot_catalog(config)
```

- [x] **Step 2: Run tests and confirm they fail**

Run: `conda run -n pinn_imu pytest tests/test_pilot_download.py -q -p no:cacheprovider --basetemp .pytest-tmp`

Expected: import failure because `virtual_imu.io.pilot_download` does not exist.

- [x] **Step 3: Implement catalog parsing and standard directories**

Implement `load_pilot_catalog(path: Path) -> dict` with required keys, supported source types (`http`, `gdrive_folder_match`), unique artifact IDs, and relative destination validation. Implement `ensure_dataset_layout(raw_root: Path, catalog: dict) -> dict[str, Path]` for `00_archives`, `01_extracted`, and `_manifests`.

- [x] **Step 4: Add the concrete catalog**

The catalog contains fixed NavINST and PennCOSYVIO files:

```yaml
raw_root: data/raw
min_free_space_gb: 50
artifacts:
  - id: navinst_urban03_day_imus
    dataset: navinst_pilot
    sequence: Urban03/2023-11-03-09-42-41-Day
    source_type: http
    url: https://www.frdr-dfdr.ca/repo/files/1/published/publication_1084/submitted_data/Urban03/2023-11-03-09-42-41-Day/imus.bag
    destination: 00_archives/Urban03_2023-11-03_day_imus.bag
    expected_size: 426426786
    extraction: none
  - id: navinst_urban03_day_reference
    dataset: navinst_pilot
    sequence: Urban03/2023-11-03-09-42-41-Day
    source_type: http
    url: https://www.frdr-dfdr.ca/repo/files/1/published/publication_1084/submitted_data/Urban03/2023-11-03-09-42-41-Day/reference.bag
    destination: 00_archives/Urban03_2023-11-03_day_reference.bag
    expected_size: 12612548
    extraction: none
  - id: penncosyvio_tango_top
    dataset: penncosyvio_pilot
    sequence: all
    source_type: http
    url: https://visiondata.cis.upenn.edu/penncosyvio/tarfiles/tango_top.tar
    destination: 00_archives/tango_top.tar
    expected_size: 267509760
    extraction: tar
  - id: penncosyvio_visensor
    dataset: penncosyvio_pilot
    sequence: all
    source_type: http
    url: https://visiondata.cis.upenn.edu/penncosyvio/tarfiles/visensor.tar
    destination: 00_archives/visensor.tar
    expected_size: 10507970560
    extraction: tar
```

Add two `gdrive_folder_match` entries for `20220209_Static_IMUs_3h20mins` and `20220216_MCR_slow`, both rooted at the official FusionPortable Google Drive folder and restricted to `.7z` or `.bag` paths.

- [x] **Step 5: Run focused tests**

Expected: catalog/layout tests pass.

### Task 2: Resumable Download And Status Manifests

**Files:**
- Modify: `src/virtual_imu/io/pilot_download.py`
- Modify: `tests/test_pilot_download.py`
- Modify: `pyproject.toml`

- [x] **Step 1: Write failing tests for verified skip, blocked status, and command construction**

Use an injected `command_runner` so tests never access the network. Assert that an existing file with matching expected size returns `verified_skip`, HTTP uses `curl -L --fail --retry 5 --continue-at -`, and a failed command returns `blocked` with its return code and error text.

- [x] **Step 2: Run tests and confirm behavior is missing**

Expected: failures for undefined `download_artifact` and `resolve_gdrive_matches`.

- [x] **Step 3: Implement download orchestration**

Implement:

```python
def download_artifact(artifact, dataset_root, command_runner=subprocess.run) -> dict:
    # validate existing size; guard disk space; invoke curl or gdown; return status record

def resolve_gdrive_matches(artifact, command_runner=subprocess.run) -> list[dict]:
    # run: gdown <folder-url> --folder --json --quiet
    # parse JSON and retain paths matching configured regex
```

Use `.part` files for HTTP downloads and atomically rename only after a successful exit and size validation. Add `gdown>=6.1,<7` to project dependencies.

- [x] **Step 4: Write per-artifact and per-dataset manifests**

Every status record contains artifact ID, source page, resolved URL, UTC timestamp, local path, size, SHA256, status, sequence, sensor metadata, synchronization notes, license notes, and extraction status.

- [x] **Step 5: Run focused tests and lint**

Run pytest for `tests/test_pilot_download.py` and Ruff on the new module/test.

Expected: all pass.

### Task 3: Safe Extraction And Inventory

**Files:**
- Modify: `src/virtual_imu/io/pilot_download.py`
- Modify: `tests/test_pilot_download.py`

- [x] **Step 1: Write failing safe-extraction tests**

Create a small TAR in `tmp_path`, assert normal members extract under `01_extracted/<artifact-id>`, and assert a member named `../escape.txt` raises `ValueError` without writing outside the target.

- [x] **Step 2: Run tests and confirm extraction is missing**

Expected: failure for undefined `extract_artifact`.

- [x] **Step 3: Implement extraction and inventory**

Use `tarfile` with resolved-path containment checks. For 7z, invoke `7z x -y -o<target> <archive>` only when `7z` exists; otherwise record `downloaded_not_extracted`. Bags use `extraction: none`. Build a sorted inventory of relative path and byte size.

- [x] **Step 4: Run focused tests**

Expected: safe extraction and traversal rejection tests pass.

### Task 4: CLI, Documentation, And Offline Verification

**Files:**
- Create: `scripts/download_pilot_datasets.py`
- Modify: `README.md`
- Modify: `tests/test_pilot_download.py`

- [x] **Step 1: Write a CLI import/smoke test**

Assert `python scripts/download_pilot_datasets.py --help` exits zero and includes `--config`, `--dataset`, and `--dry-run`.

- [x] **Step 2: Implement Typer CLI**

The command logs the config it reads, selected datasets, destination roots, every write/skip/block, and summary path. `--dry-run` resolves and reports without downloading.

- [x] **Step 3: Document the command**

Add:

```powershell
conda run -n pinn_imu python scripts/download_pilot_datasets.py `
  --config configs/multi_dataset_pilot_download.yaml
```

Document `data/raw/*_pilot/_manifests/pilot_manifest.json` as the raw-data source of truth.

- [x] **Step 4: Run offline verification**

Run targeted tests, Ruff, and CLI `--help`. Expected: all pass without network access.

### Task 5: Resolve Sources, Download Pilot, And Verify Idempotency

**Files:**
- Generated only: `data/raw/navinst_pilot`, `data/raw/penncosyvio_pilot`, `data/raw/fusionportable_pilot`

- [x] **Step 1: Verify isolated Drive tooling and the existing environment**

Run the CLI with `PYTHONPATH=src` using `D:\Anaconda3\envs\pinn_imu\python.exe`; Drive operations invoke `uvx --from gdown==6.1.0` without changing that environment.

- [x] **Step 2: Dry-run source resolution**

Run the pilot CLI with `--dry-run`. Expected: fixed files report expected destinations; FusionPortable entries resolve matching Drive paths or are explicitly blocked.

- [x] **Step 3: Download NavINST and PennCOSYVIO**

Run with `--dataset navinst_pilot --dataset penncosyvio_pilot`. Expected total archive bytes are approximately 10.7 GB; manifests include SHA256 for four files.

- [x] **Step 4: Download FusionPortable matches**

Run with `--dataset fusionportable_pilot`. Download only the resolved static and MCR-slow archives. If Drive throttles or no unique match exists, preserve a blocked manifest and do not broaden the download scope.

- [x] **Step 5: Repeat the complete command**

Expected: downloaded files produce `verified_skip`, blocked files remain explicit, and no complete file is downloaded again.

- [x] **Step 6: Final integrity verification**

Check manifest JSON parsing, SHA256 presence, byte totals, extracted inventories, free disk space above 50 GB, and `git status --short` to confirm raw data remains ignored.
