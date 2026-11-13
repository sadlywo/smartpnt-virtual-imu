# Artifact and Publication Policy

## What belongs in Git

The public repository contains:

- Python source, tests, and command-line entry points;
- experiment, split, model, figure, and report YAML files;
- protocol documents, schemas, and reproducibility instructions;
- the shared visualization style and the local IMU figure skill;
- small CAD source/config files for the sensor mount.

## What stays outside Git

The following are reproducible artifacts or third-party data and must not be committed:

- `data/raw/`, `data/processed/`, `data/interim/`, and `data/external/`;
- `runs/`, `outputs/`, `checkpoints/`, and `artifacts/`;
- checkpoints and exported models such as `.ckpt`, `.pt`, `.pth`, `.onnx`, `.engine`, and `.safetensors`;
- Parquet prediction matrices, large CSV/Markdown table exports, and figure image files;
- archives, checksum manifests attached to remote packages, and local transfer logs;
- server credentials, tokens, SSH keys, and machine-local environment files.

## Run-directory contract

Every scientific run is created under a unique `runs/<experiment_id>/` directory. A complete run should contain, as applicable:

```text
runs/<experiment_id>/
  config.yaml or config_snapshot.yaml
  data_manifest.json
  git_commit.txt
  metrics.csv
  best.ckpt
  validation/
  evaluation/
  predictions/
  baselines/
  figures/
  figure_index.json
  report.md
  report.html
  report.pdf
  artifact_manifest.json
```

`outputs/latest/` may mirror a completed run for convenience. Its `latest_run.txt` pointer must be checked before attributing any metric or figure to a run.

## Immutable result packages

For multi-machine or long-running studies, retain an immutable package with:

1. a source archive and source fingerprint;
2. configuration and dataset-manifest hashes;
3. every selected checkpoint and evaluation prediction;
4. aggregate metrics, baselines, and downstream results;
5. source CSV files for every figure;
6. a figure index with captions and QA status;
7. a report manifest and page-level QA receipt;
8. per-file SHA256 values and an outer package hash.

The package may be stored on a cloud drive, institutional storage, or another artifact store. Git should contain only the instructions and checksums needed to identify it.

## Published Results

[The 2026-11-13 results index](results/2026-11-13/README.md) contains concise findings and source fingerprints. The associated GitHub Release stores reports, aggregate CSV tables, paper figures and frozen experiment sources. Dataset measurements, prepared tensors, sample predictions and fitted checkpoints remain in the complete local source runs. The Release reading package is a selected publication export, with member hashes recorded in its manifest.

The export is defined by `configs/analysis/github_results_20261113.yaml` and built by `scripts/package_research_results.py`. It refuses per-sample measurement CSVs and checks available source seals before including a file. Generated archives and preview images stay under ignored `outputs/` and are uploaded as Release assets.

## Release checklist

```bash
uv run python scripts/check_repository.py
uv run pytest -m "not integration"
git status --short
git ls-files | sort
```

Before publishing, verify that no data, checkpoint, run directory, output image, archive, credential, or machine-local environment file appears in `git status` or `git ls-files`.
