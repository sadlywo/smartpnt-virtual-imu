# 统一 IMU 数据的服务器训练交接

本交接面向 Linux 训练服务器：传输当前代码快照、整理后的数据、固定依赖和验证命令。2026-10-23 已按用户提供的连接登录 RTX 4090 服务器，独立环境和真实数据批次 GPU 检查通过。正式数据主目录是 `data/processed/unified_imu_v2/`；每次打包的真实记录放在 `runs/server_bundle_<id>/`。旧版本 `unified_imu_v1` 保留为已被替代的诊断数据，正式训练使用 v2。打包与预检会读取 `dataset_status.json`，拒绝状态为 `superseded_diagnostic` 的任何数据根目录；诊断状态单独记录，不改写旧 manifest 或其哈希。

## 2026-10-23 选定的服务器镜像

用户选定 Ubuntu 22.04、Python 3.12、PyTorch 2.4.0、CUDA 12.1 镜像，目标架构为 Linux x86_64。环境契约位于 `configs/server_environment.yaml`，随包校验并在预检时实际执行。新锁文件将 PyTorch 固定为 `2.4.0+cu121`（Linux x86_64 / Windows AMD64 从官方 cu121 源解析），NumPy 固定为 `1.26.4`，CUDA runtime / nvJitLink 固定为 `12.1.105`。项目 Python 范围为 `>=3.11,<3.13`；服务器固定选择 3.12。此适配使用新的源码包，2026-10-10 包保留为历史快照。

bootstrap 在项目独立 `.venv` 中安装 frozen 依赖。镜像自带的全局 Torch 不作为训练来源，不需要在全局环境升级或卸载它。尽可能复用镜像中的 Python 3.12；缺少时才安装托管解释器。PyTorch 的官方安装组合见 [PyTorch 2.4.0 历史版本说明](https://pytorch.org/get-started/previous-versions/#v240)。

镜像标注 cuDNN 8，但官方 Linux Torch 2.4.0/cu121 wheel 声明的 cuDNN 依赖是 `nvidia-cudnn-cu12==9.1.0.70`。独立环境使用 wheel 自身依赖，预检记录实际 `cudnn_version`，不强行链接系统 cuDNN 8，也不依据镜像卡片推断实际加载的库。镜像中的 CUDA Toolkit/NVCC 12.1 和 PyTorch 编译时 runtime 是不同信息。

没有 GPU 型号与实际 NVIDIA 驱动信息时，本地只能确认依赖解析与训练代码适配。SSH 连接后必须通过 CUDA 矩阵计算和 Conv1d 前后向及有限梯度检查；仅 `torch.cuda.is_available()` 为真不足以通过预检。与该旧构建不兼容的 GPU 会停止启动并报告实际错误。

本地适配验证记录在 `runs/server_image_20261023/`：Python 3.12.4、Torch 2.4.0+cpu、NumPy 1.26.4 和其余 frozen 依赖的真实数据检查通过，两类任务各完成两步优化且参数更新。Windows CPU 验证仅将 Torch 换为同版本 CPU wheel；正式服务器锁仍为 cu121。Ubuntu 22.04 对应 glibc 2.35 / Linux x86_64 / Python 3.12 的 frozen 安装预演通过，uv 0.6.1 和 0.13.5 的离线 lock 检查均通过。受影响的训练/数据回归 42 项和最终环境/打包回归 49 项通过。这里不宣称 Linux/CUDA 已实机运行，完整检查在 SSH 连接后执行。

此包只完成所选镜像的环境适配。`docs/TRAINING_PROTOCOL_V2.md` 列出的正式计分与评估流程改造仍单独跟踪，不能将环境检查通过等同于已完成正式实验评估。

## 交接包包含什么

`paired_signal` 使用真实的低等级/参考 IMU 配对信号；`navigation` 使用真实位置、速度和姿态标签。导航真值没有被转换成声称“实测高等级 IMU”的标签。两类训练分别使用 `configs/train_unified_paired.yaml` 和 `configs/train_unified_navigation.yaml`，参考设备、坐标系、字段单位和预处理语义以数据目录的 `manifest.json`、`label_schema.json` 为准。

统一采样率为 40 Hz，角速度单位为 rad/s，加速度单位为 m/s²。ALiMU 正式拆分按物理记录分组：同一次记录对应的三个低等级传感器和共享 VN100 参考只进入同一个拆分。原有 leave-MPU9250-out 分配中的 33 个传感器文件共享 11 份 VN100 参考记录，不能作为独立物理记录划分。预检在 manifest 提供 `physical_group_id` 或 `reference_measurement_sha256` 时拒绝同组跨拆分。PennCOSYVIO 保留 af/as 训练、bf 验证、bs 测试。SmartPNT 使用 Traj3 训练、Traj4 验证、Traj1 测试；连续的有效导航片段保留原始轨迹身份，不能跨拆分。`archive/duplicates.json` 记录 6 条 ALiMU 部分复用变体和 SmartPNT Traj2 的归档关系，这些数据不导出用于训练。FusionPortable 的纳入与否以本次构建的实际审计结果为准。

包内包含当前磁盘中的 `src/`、`scripts/`、`configs/`、`docs/`、`tests/` 白名单文件，以及 `pyproject.toml`、`uv.lock`、`README.md` 和 `AGENTS.md`。这包括训练需要的未提交、未跟踪代码。源代码只导出允许的文本扩展名，并排除凭据、密钥、缓存和环境目录。数据只导出已验证 manifest 所引用的 Parquet 和规定的元数据。此外，包内只允许本次 `manifest.build_run` 的六个审计文件：`config.yaml`、`data_manifest.json`、`metrics.json`、`report.md`、`verification.json` 和 `source_state.json`；验证必须通过，且绑定同一个数据 manifest。原始相机/LiDAR/ROS 数据、诊断缓存、旧数据集、其他 run、已有训练输出、checkpoint、`.git` 和虚拟环境均不包含在包内。

`bundle_manifest.json` 保存每个导出文件的 SHA256、字节数、当前 Git 提交和当前源代码总指纹；`SHA256SUMS` 同时覆盖导出文件和该 manifest。归档的文件顺序、权限、时间戳和 gzip 时间戳固定；给定同一 bundle id 和相同输入快照时，归档内容可重复。

## 本地生成与核对

在项目根目录的 PowerShell 中执行：

```powershell
.\.venv312\Scripts\python.exe -B scripts\package_training_bundle.py `
  --dataset-root data/processed/unified_imu_v2
```

脚本输出 `ARCHIVE`、`SHA256` 和 `RUN`。默认名称包含 UTC 时间；需要指定名称时，可加 `--bundle-id server_bundle_<your_id>`。同名归档或 run 已存在时会拒绝覆盖。生成前检查数据路径、Parquet SHA256、字段和数值类型、严格配对来源、完整的 train/validation/test 分配和原始导航轨迹泄漏；生成后在独立临时目录中完整解包并再次验证。

每次打包 run 包含以下可互相核对的文件：

| 文件 | 用途 |
| --- | --- |
| `config_snapshot.yaml` | 打包配置、数据根目录和导出策略 |
| `data_manifest.json` | 本次使用的数据 manifest 原样副本，包含数据构建 run |
| `source_snapshot.json` | Git 提交、dirty 标志、每个当前源文件的 SHA256 和总指纹 |
| `bundle_manifest.json` | 与归档内相同的文件清单和指纹 |
| `metrics.json` | `data_bytes`、`payload_bytes`、`archive_bytes` 和记录数量 |
| `verification.json` | 独立解包、checksum、数据契约验证结果 |
| `archive_sha256.txt`、`report.md` | 归档定位、SHA256 和结果说明 |

传输大小使用 `metrics.json` 中实测的 `archive_bytes`。`data_bytes` 是本次整理后的 Parquet 字节数，不能当作原始下载量或训练 RAM 占用。两个 loader 会读取并合并对应任务的训练数据；Parquet 的压缩体积也不能直接代表展开后的内存占用。磁盘还需容纳解包数据、依赖缓存和之后生成的训练 run。

手动核对最终归档时，把实际包名代入：

```powershell
Get-FileHash outputs\server_bundles\<BUNDLE>.tar.gz -Algorithm SHA256
```

不要根据 `outputs/latest/` 推断本次结果来源；定位构建或打包结果应使用相应的 `runs/<id>/`。数据和导出的代码都有哈希，修改任何已记录的文件后，需要新建一次打包。

## 传输与解包

获得服务器账号和目录后，只传输此次归档和它的 `.sha256` 文件。以下是待替换的命令示例，不会在本地自动执行：

```powershell
scp outputs/server_bundles/<BUNDLE>.tar.gz user@host:/srv/imu_upload/
scp outputs/server_bundles/<BUNDLE>.tar.gz.sha256 user@host:/srv/imu_upload/
```

服务器先校验归档，再解包到一个新目录。`mkdir` 使用新的 bundle 名称，避免覆盖已有实验：

```bash
cd /srv/imu_upload
sha256sum -c <BUNDLE>.tar.gz.sha256
mkdir -p /srv/imu_train
mkdir /srv/imu_train/<BUNDLE>
cd /srv/imu_train/<BUNDLE>
tar -xzf /srv/imu_upload/<BUNDLE>.tar.gz
```

包内所有用于加载数据的路径是项目根目录下的 POSIX 相对路径。不要只把 `sequences/` 拷到另一个目录，也不要改 manifest 中的根目录。当前数据构建 run 的六个审计文件随包导出，因此 `manifest.build_run` 在服务器上可直接定位；其他历史 run 和原始数据留在本地。

## 安装依赖和检查 GPU

服务器需要 Bash 和 uv。uv 的安装方式见 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)。进入解包目录后执行：

```bash
bash scripts/server/bootstrap.sh --require-gpu
```

bootstrap 先定位 Python 3.12（缺少时才安装），用标准库验证全部 checksum 和数据路径，再执行 `uv lock --check --offline --python 3.12` 检查项目与 lock 的一致性，最后执行 `uv sync --frozen --python 3.12 --no-dev`。依赖不重新解析、不自动升级。项目声明要求 Python >=3.11,<3.13；本交接统一选 Python 3.12。uv 的 `--frozen` 本身不检查 lock 是否过期，因此保留前一步 `--check`。[uv 官方锁定与同步说明](https://docs.astral.sh/uv/concepts/projects/sync/)、[Python 管理说明](https://docs.astral.sh/uv/guides/install-python/)。

当前 Linux lock 的 PyTorch 条目是 `torch==2.4.0+cu121`。实际服务器还要满足该构建、GPU 和驱动的兼容性；本地锁文件可解析不等于已验证远程 CUDA。完整预检按校验过的 YAML 拒绝 Python/PyTorch/NumPy/CUDA/平台偏离，记录依赖版本、cuDNN、GPU 型号和显存，验证 NumPy↔Torch 转换，并在 GPU 上实际运行矩阵计算和 Conv1d 前后向。`--require-gpu` 在 CUDA 不可用、运算失败或梯度非有限时以非零状态退出。CPU 诊断模式只能跳过 GPU 要求，仍检查软件版本和目标平台。

需要确认 CUDA wheel 或驱动时，只使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 和 [NVIDIA 官方兼容性文档](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。选择器应按服务器操作系统、GPU 和合适的 CUDA 构建使用；`nvidia-smi` 的 CUDA 显示与 `torch.version.cuda` 的含义不同，最终以 PyTorch 的实际运算验证为准。若必须改依赖版本，应生成新的配置/lock 和新交接包，保留版本变化的来源；不要在一个固定包中悄悄替换依赖后继续宣称原环境完全复现。

预检也可单独运行：

```bash
# 依赖安装前：Python >=3.9，仅用标准库校验文件、路径和拆分结构
python3 -B -S scripts/server/preflight.py --checksums-only

# 只检查数据与 schema，跳过 PyTorch 导入
.venv/bin/python -B scripts/server/preflight.py --data-only

# 正式 GPU 预检，把结果写到 run 目录
.venv/bin/python -B scripts/server/preflight.py --require-gpu \
  --report runs/server_preflight_manual/preflight.json
```

`--checksums-only` 不验证 Parquet 字段和 GPU，也不能与 `--require-gpu` 同用。`--data-only` 验证数据字段/类型和拆分，但不检查训练环境。正式训练使用完整预检。

## 验证启动命令和正式训练

以下命令完整验证交接包、数据和训练环境，输出实际训练命令，并创建 `runs/server_launch_<time>/`，但不会调用 trainer：

```bash
bash scripts/server/run.sh paired --dry-run
bash scripts/server/run.sh navigation --dry-run
```

仅在明确接受 CPU 环境时加 `--allow-cpu`；它允许 CPU，但不会改写 YAML 中的设备设置。正式训练默认要求 GPU。两个默认配置对应的原始入口分别为：

```bash
.venv/bin/python -B scripts/train.py --config configs/train_unified_paired.yaml
.venv/bin/python -B scripts/train_gt_supervised.py --config configs/train_unified_navigation.yaml
```

使用包装后的前台启动命令可以保留源包、预检和启动配置的关联。建议在 tmux 会话中执行，终端断开后可重新连接：

```bash
tmux new-session -s imu_paired
```

进入该会话后：

```bash
set -o pipefail
BUNDLE_CONSOLE_RUN="runs/server_console_$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$BUNDLE_CONSOLE_RUN"
bash scripts/server/run.sh paired 2>&1 | tee "$BUNDLE_CONSOLE_RUN/train.log"
```

导航任务把 `paired` 替换为 `navigation`。脚本保持前台运行；按 `Ctrl-b` 再按 `d` 可以退出 tmux 界面而保留训练。重新进入使用 `tmux attach-session -t imu_paired`。`Ctrl-c` 会中断训练。

调整超参数时，从默认配置复制出新的文件，例如 `configs/server_custom.yaml`，再运行 `bash scripts/server/run.sh paired --config configs/server_custom.yaml`。不要修改归档中已经记录 checksum 的文件。新配置保存在 launcher 的 `config_snapshot.yaml` 中；trainer 还会在它自己的 run 中保存实际配置。重现该实验时也要保留新增配置。

## checkpoint 和实验来源

当前两个训练器在验证指标改善时保存 `best.ckpt`，包含模型、配置、归一化和最佳 epoch/指标等内容。配对训练同时保存损失曲线和 metrics；导航训练还会使用最佳模型输出 test 指标与 summary。具体结果以各自实际 run 中的文件为准。

**当前没有完整断点续训接口。** 两个训练入口都没有 `--resume`；checkpoint 没有保存优化器、调度器和完整 RNG 状态。网络连接断开可通过 tmux 接续查看，但训练进程终止后不能把 `best.ckpt` 当作无损恢复的训练状态；重跑会创建一个新 run，已有最佳模型仍可用于后续评估/推理。

每个 `runs/server_launch_<time>/` 保存 `config_snapshot.yaml`、`source_bundle.json`、`preflight.json` 和 `command.sh`，启动脚本向 trainer 传入 `VIRTUAL_IMU_SERVER_LAUNCH_RUN` 关联信息。模型结果的来源应沿着“训练 run → 启动 run → bundle id/source fingerprint → 数据 build_run”核对。服务器包没有 `.git` 目录，源提交与 dirty 源码哈希以 `source_bundle.json` 为准。所有训练 run、日志和 checkpoint 留在 `runs/`；需要回传结果时传对应的 run，不能只回传一个 `outputs/latest/` 快照。
# 固定协议的正式运行入口（2026-10-23）

新包增加 `configs/train_formal_matrix.yaml`：按固定种子 20261010/11/12，先训练 ALiMU/Penn 六轴残差补偿，再独立训练 SmartPNT 的窗口位移。两个任务最多各 60 epoch，早停 10；不依据测试集调整候选模型。配对使用 `val_grouped_score` 同时驱动 checkpoint、降学习率和早停；SmartPNT 使用 `val_rmse`（米的分量 RMSE）。

```bash
bash scripts/server/bootstrap.sh --require-gpu
bash scripts/server/run.sh formal --smoke
tmux new-session -d -s imu_formal 'bash scripts/server/run.sh formal > runs/formal_console.log 2>&1'
```

`--smoke` 只跑首个种子、一个 epoch，输出独立 `runs/formal_smoke_*/`，用于执行链检查。正式结果集中于 `runs/formal_training_*/`，包含原始矩阵配置、每次实际 YAML、`pipeline.json`、`experiments/` 内六个训练目录及各自完整评估、汇总 CSV/JSON、图和 `report.md`。每个训练目录包含 config/data/source 记录与 `best.ckpt`。中断不会恢复 optimizer；tmux 仅保证 SSH 断开后进程继续。

严格确定性模式固定 cuBLAS 工作区、关闭 TF32，并使用确定性卷积。Torch 2.4 的 CUDA `cumsum` 不支持该模式，因此窗口积分项使用等价的三角矩阵前缀求和；loss 定义和梯度保持一致，在 CPU 与 GPU 分别核对数值/梯度。启动器默认使用 Agg 绘图后端，适配无显示器服务器。

配对指标按器件分开报告 rad/s、m/s²；按完整 80 点、stride 40 的窗口预测并合并重叠时间点，短片段和尾部排除量显式记录。经典基线只在训练集拟合。SmartPNT 仅评估 1.975 秒端点间的固定局部 ENU 位移，不提供完整轨迹 ATE/RPE。所有评估先核对训练时保存的配置及数据清单哈希。
