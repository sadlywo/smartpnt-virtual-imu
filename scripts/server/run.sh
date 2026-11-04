#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$BUNDLE_PROJECT_ROOT"
BUNDLE_TASK="${1:-}"
if [[ "$BUNDLE_TASK" == "paired" ]]; then
  BUNDLE_TRAIN_SCRIPT="scripts/train.py"
  BUNDLE_TRAIN_CONFIG="configs/train_unified_paired.yaml"
elif [[ "$BUNDLE_TASK" == "navigation" ]]; then
  BUNDLE_TRAIN_SCRIPT="scripts/train_gt_supervised.py"
  BUNDLE_TRAIN_CONFIG="configs/train_unified_navigation.yaml"
elif [[ "$BUNDLE_TASK" == "formal" ]]; then
  BUNDLE_TRAIN_SCRIPT="scripts/run_formal_training.py"
  BUNDLE_TRAIN_CONFIG="configs/train_formal_matrix.yaml"
elif [[ "$BUNDLE_TASK" == "benchmark" ]]; then
  BUNDLE_TRAIN_SCRIPT="scripts/run_benchmark_training.py"
  BUNDLE_TRAIN_CONFIG="configs/train_benchmark_matrix.yaml"
elif [[ "$BUNDLE_TASK" == "reviewer" ]]; then
  BUNDLE_TRAIN_SCRIPT="scripts/run_reviewer_hpo.py"
  BUNDLE_TRAIN_CONFIG="configs/reviewer_hpo.yaml"
else
  printf 'Usage: bash scripts/server/run.sh paired|navigation|formal|benchmark [--dry-run] [--smoke] [--plan-only] [--resume-root runs/benchmark_training_ID] [--allow-cpu] [--config configs/custom.yaml]\n' >&2
  exit 2
fi
shift
BUNDLE_DRY_RUN=0
BUNDLE_TRAIN_FLAGS=()
BUNDLE_PREFLIGHT_FLAGS=(--require-gpu)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) BUNDLE_DRY_RUN=1; shift ;;
    --smoke)
      [[ "$BUNDLE_TASK" == "formal" || "$BUNDLE_TASK" == "benchmark" || "$BUNDLE_TASK" == "reviewer" ]] || { printf 'FAILED: --smoke requires a matrix task\n' >&2; exit 2; }
      BUNDLE_TRAIN_FLAGS+=(--smoke); shift ;;
    --plan-only)
      [[ "$BUNDLE_TASK" == "benchmark" || "$BUNDLE_TASK" == "reviewer" ]] || { printf 'FAILED: --plan-only requires benchmark or reviewer task\n' >&2; exit 2; }
      BUNDLE_TRAIN_FLAGS+=(--plan-only); shift ;;
    --resume-root)
      [[ ( "$BUNDLE_TASK" == "benchmark" || "$BUNDLE_TASK" == "reviewer" ) && $# -ge 2 ]] || { printf 'FAILED: --resume-root requires benchmark/reviewer task and a run path\n' >&2; exit 2; }
      BUNDLE_TRAIN_FLAGS+=(--resume-root "$2"); shift 2 ;;
    --allow-cpu) BUNDLE_PREFLIGHT_FLAGS=(); shift ;;
    --config)
      [[ $# -ge 2 ]] || { printf 'FAILED: --config requires a YAML path\n' >&2; exit 2; }
      BUNDLE_TRAIN_CONFIG="$2"
      shift 2
      ;;
    *) printf 'FAILED: unknown argument %s\n' "$1" >&2; exit 2 ;;
  esac
done
printf 'READS: bundle inventory, curated data/schema/splits, %s\n' "$BUNDLE_TRAIN_CONFIG"
printf 'WRITES: runs/server_launch_<time>/ provenance/config/preflight; trainer writes its own runs/<experiment>_<time>/\n'
printf 'SKIPS: raw/old data, duplicate sequences, remote operations; dry-run=%s\n' "$BUNDLE_DRY_RUN"
[[ -x .venv/bin/python ]] || { printf 'FAILED: run scripts/server/bootstrap.sh first\n' >&2; exit 1; }
[[ -f "$BUNDLE_TRAIN_CONFIG" ]] || { printf 'FAILED: missing training config %s\n' "$BUNDLE_TRAIN_CONFIG" >&2; exit 1; }
BUNDLE_LAUNCH_RUN="runs/server_launch_$(date -u +%Y%m%d_%H%M%S_%N)"
mkdir -p -- "$BUNDLE_LAUNCH_RUN"
cp -- "$BUNDLE_TRAIN_CONFIG" "$BUNDLE_LAUNCH_RUN/config_snapshot.yaml"
cp -- bundle_manifest.json "$BUNDLE_LAUNCH_RUN/source_bundle.json"
.venv/bin/python -B scripts/server/preflight.py "${BUNDLE_PREFLIGHT_FLAGS[@]}" --report "$BUNDLE_LAUNCH_RUN/preflight.json"
printf '%q ' .venv/bin/python -B "$BUNDLE_TRAIN_SCRIPT" --config "$BUNDLE_TRAIN_CONFIG" "${BUNDLE_TRAIN_FLAGS[@]}" > "$BUNDLE_LAUNCH_RUN/command.sh"
printf '\n' >> "$BUNDLE_LAUNCH_RUN/command.sh"
printf 'COMMAND: .venv/bin/python -B %s --config %s\n' "$BUNDLE_TRAIN_SCRIPT" "$BUNDLE_TRAIN_CONFIG"
printf 'PROVENANCE: %s\n' "$BUNDLE_LAUNCH_RUN"
if [[ "$BUNDLE_DRY_RUN" == 1 ]]; then
  printf 'SKIPS: trainer invocation (--dry-run); data and environment preflight passed\n'
  exit 0
fi
export VIRTUAL_IMU_SERVER_LAUNCH_RUN="$BUNDLE_LAUNCH_RUN"
export MPLBACKEND="${MPLBACKEND:-Agg}"
exec .venv/bin/python -B "$BUNDLE_TRAIN_SCRIPT" --config "$BUNDLE_TRAIN_CONFIG" "${BUNDLE_TRAIN_FLAGS[@]}"
