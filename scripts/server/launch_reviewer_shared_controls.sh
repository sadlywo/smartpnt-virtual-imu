#!/usr/bin/env bash
set -euo pipefail

printf 'READS: frozen reviewer source/status/manifests; WRITES: independent shared-controls run; SKIPS: fitting until aux2 succeeds, original training changes and authentication\n'
if [[ $# -ne 1 || ! "$1" =~ ^[0-9a-f]{64}$ ]]; then
  printf 'Expected one reviewed auxiliary manifest SHA256 argument.\n' >&2
  exit 64
fi
imu_shared_core=/root/imu-work/server_bundle_reviewer_hpo_20261024_v1
imu_shared_aux=/root/imu-work/reviewer_shared_controls_20261025_v1
imu_shared_prior=/root/imu-work/reviewer_postprocessing_20261024_v1/runs/reviewer_postprocess_20261024_core_122113_893148/postprocessing_state.json
imu_shared_preceding=/root/imu-work/reviewer_final_analysis_20261024_v1/runs/reviewer_final_analysis_20261024_core_122113_893148/final_analysis_state.json
imu_shared_dependencies=/root/imu-work/reviewer_final_analysis_20261024_v1/deps
cd "$imu_shared_core"
export PYTHONPATH="$imu_shared_core/src"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=20261024
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec "$imu_shared_core/.venv/bin/python" -B "$imu_shared_aux/scripts/server/run_reviewer_shared_controls.py" \
  --project-root "$imu_shared_core" \
  --pipeline-root "$imu_shared_core/runs/reviewer_hpo_20261024_122113_893148" \
  --aux-root "$imu_shared_aux" \
  --aux1-state "$imu_shared_prior" \
  --preceding-state "$imu_shared_preceding" \
  --out "$imu_shared_aux/runs/reviewer_shared_controls_20261025_core_122113_893148" \
  --aux-manifest-sha256 "$1" \
  --dependency-root "$imu_shared_dependencies"
