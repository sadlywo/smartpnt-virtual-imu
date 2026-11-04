# Data Directory

This directory is a local mount point for external datasets. Raw data and prepared tensors are intentionally excluded from Git.

Expected layout:

```text
data/
  raw/
    smartpnt_msf/
    alimu/
    penncosyvio_pilot/
    penncosyvio_ground_truth/
    fusionportable_pilot/
  processed/
    unified_imu_v2/
    smartpnt_msf/
    alimu_strict/
    multi_imu_strict/
  external/
  interim/
```

Each dataset remains subject to its upstream license and citation requirements. Before assuming a schema, run the inspection tool and retain the resulting manifest. Do not commit downloaded archives or generated Parquet files.

See [Data Notes](../docs/DATA.md) and [Artifact Policy](../docs/ARTIFACTS.md).
