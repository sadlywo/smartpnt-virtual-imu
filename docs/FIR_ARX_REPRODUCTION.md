# FIR/ARX Reproduction Runbook

## Local verification

```powershell
$env:PYTHONPATH='src'
D:\Anaconda3\python.exe -B -m pytest -q tests/test_fir_arx_data.py tests/test_fir_arx_features.py tests/test_fir_arx_models.py
D:\Anaconda3\python.exe -B scripts/run_fir_arx_research.py --config configs/fir_arx/base.yaml --run-id fir_arx_base_smoke
```

## Server layout

```text
/root/fir_arx/
  data/processed/unified_imu_v2/
  bundles/
  project/
  runs/
```

Install dependencies once:

```bash
/root/miniconda3/envs/myconda/bin/pip install -r requirements-fir-arx.txt
```

Run:

```bash
cd /root/fir_arx/project
PYTHONPATH=src /root/miniconda3/envs/myconda/bin/python -B scripts/run_fir_arx_research.py \
  --config configs/fir_arx/base.yaml \
  --run-id fir_arx_base_20261105
```

## Artifact contract

Each run writes:

```text
config_snapshot.yaml
provenance.json
data_manifest.json
split_audit.json
artifact_manifest.json
tables/metrics_by_pair.csv
tables/metrics_by_sequence.csv
tables/metrics_by_dataset.csv
tables/metrics_by_protocol.csv
tables/trials.csv
states/<protocol>/<pair>__<model>.json
figures/*.{svg,pdf,png,tiff}
report.md
```

Large raw/processed data and generated run directories are ignored by Git. Only the code, YAML, tests, and documentation are committed to each experiment branch.
