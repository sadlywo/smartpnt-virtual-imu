# Agent Rules

- Never commit raw datasets, generated processed data, checkpoints, run directories, or output figures.
- Put large local files under `data/raw`, `data/processed`, `runs`, `outputs`, `checkpoints`, or an external DVC/cloud location.
- Manage repeatable experiments through YAML files in `configs/`.
- Every CLI command should log what it reads, writes, and skips.
- Inspect SmartPNT-MSF file structure and sample fields before assuming schemas.
- Keep random processes seeded through config.
- Generate experiment figures through `.agents/skills/imu-research-visualization` or `src/virtual_imu/viz` so paper styling remains consistent.
- Preserve traceability: config snapshot, data manifest, git commit, metrics, and report should point back to each other.
- Treat `runs/<experiment_id>/` as the source-of-truth artifact directory. `outputs/latest/` is only a convenience snapshot and must not be used to infer which run produced a result without checking `outputs/latest_run.txt`.
