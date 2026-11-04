# Server image adaptation: PyTorch 2.4.0 / CUDA 12.1

The user selected the screenshot's Ubuntu 22.04, Python 3.12, PyTorch 2.4.0, CUDA 12.1 image and explicitly requested adapting the training bundle. This authorizes implementation and a new local export; no remote connection or long training is part of this turn.

Use an isolated frozen environment matching the selected Python/PyTorch/CUDA versions. Pin NumPy 1.26.4 for the NumPy/PyTorch array bridge. Route Linux x86_64 and Windows AMD64 PyTorch resolution to the official cu121 index; keep supported Python versions at 3.11–3.12 because this pinned Torch has no newer interpreter wheels. Generate a complete new uv lock instead of editing package hashes. Independent profiles/reusing arbitrary base-image packages were considered, but a single lock and isolated install preserve reproducibility with the existing bootstrap/export flow.

Record the target in configs/server_environment.yaml, include it in the bundle inventory, and validate Python, Torch, NumPy, compiled CUDA and platform before training. The full runtime check must exercise NumPy↔Torch conversion and CUDA matrix multiplication plus Conv1d forward/backward, with finite gradient checks. Record the actual cuDNN version: the screenshot's system cuDNN label does not prescribe the official wheel's bundled cuDNN dependencies. GPU hardware and driver are unknown until SSH is supplied, so local Linux lock/export checks do not establish remote GPU success.

Preserve the immutable unified_imu_v2 data, earlier bundle and existing experiment configs. Update server instructions and package provenance for this new environment. Do not incorporate unrelated model/evaluation changes; TRAINING_PROTOCOL_V2.md retains the outstanding formal evaluation work.

Acceptance: regression tests cover runtime version drift/CPU diagnostics; old bundle/tamper tests still pass; official Linux cp312/cu121 wheels and lock metadata are checked; Bash syntax passes; a bounded real Torch 2.4 smoke check is attempted in an isolated local environment; a new full dataset archive passes independent extraction/checksums/schema verification. Report OS/device limitations explicitly.
