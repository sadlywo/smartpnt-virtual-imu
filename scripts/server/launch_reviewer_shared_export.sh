#!/usr/bin/env bash
set -euo pipefail

printf 'READS: complete core and all three auxiliary states; WRITES: isolated four-project export queue; SKIPS: premature export, original training changes and authentication\n'
if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'Expected one reviewed export-tools manifest SHA256 argument.\n' >&2
  exit 64
fi
imu_export_core=/root/imu-work/server_bundle_reviewer_hpo_20261024_v1
imu_export_aux1=/root/imu-work/reviewer_postprocessing_20261024_v1
imu_export_aux2=/root/imu-work/reviewer_final_analysis_20261024_v1
imu_export_aux3=/root/imu-work/reviewer_shared_controls_20261025_v1
imu_export_tools=/root/imu-work/reviewer_export_tools_20261025_v3
cd "$imu_export_core"
export PYTHONPATH="$imu_export_core/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20261024
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$imu_export_core/.venv/bin/python" -B "$imu_export_tools/scripts/server/queue_reviewer_shared_export.py" \
  --project-root "$imu_export_core" \
  --pipeline-root "$imu_export_core/runs/reviewer_hpo_20261024_122113_893148" \
  --aux1-root "$imu_export_aux1" \
  --aux2-root "$imu_export_aux2" \
  --aux3-root "$imu_export_aux3" \
  --analysis-state "$imu_export_aux2/runs/reviewer_final_analysis_20261024_core_122113_893148/final_analysis_state.json" \
  --shared-state "$imu_export_aux3/runs/reviewer_shared_controls_20261025_core_122113_893148/shared_controls_state.json" \
  --tools-root "$imu_export_tools" \
  --tools-manifest-sha256 "$1" \
  --config "$imu_export_tools/configs/reviewer_export_shared.yaml" \
  --out "$imu_export_tools/runs/reviewer_export_20261025_core_122113_893148"
