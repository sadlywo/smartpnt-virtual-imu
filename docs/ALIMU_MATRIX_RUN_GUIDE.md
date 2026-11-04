# ALiMU 留传感器矩阵运行指南

本指南用于严格的 ALiMU 留低成本传感器实验：仅在 BMX160 与 LSM9DS1 上训练/验证，并在未见过的 MPU9250 上测试。参考目标为 VN100，采样率保持 ALiMU 原生 40 Hz。

## 运行前检查

在项目根目录执行。先确认矩阵配置、基础模型配置以及严格切分的三个 Parquet 文件和 manifest 都存在：

```powershell
$env:PYTHONIOENCODING='utf-8'
Get-Content configs/alimu_leave_sensor_matrix_3seed.yaml
Test-Path configs/alimu_leave_sensor_matrix_3seed.yaml
Test-Path configs/alimu_leave_sensor_bias_drift_base.yaml
Test-Path data/processed/alimu_strict/leave_sensor_mpu9250/train.parquet
Test-Path data/processed/alimu_strict/leave_sensor_mpu9250/val.parquet
Test-Path data/processed/alimu_strict/leave_sensor_mpu9250/test.parquet
Test-Path data/processed/alimu_strict/leave_sensor_mpu9250/manifest.json
```

CLI 会再次预检这些输入，并在开始时打印 `READS`、`WRITES`、`SKIPS`。矩阵配置固定五个神经方法和三个种子；每次可复现实验都由配置快照、输入指纹、指标和报告关联到同一 `runs/<experiment_id>/` 目录。

## 可选 Smoke

Smoke 只用于检查端到端流程，采用更小的 CPU 设置，但保留完整的方法和种子轴。不要把它作为正式结果。

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_alimu_leave_sensor_matrix.py --smoke
```

## 正式训练

正式训练不会由实现或测试自动启动；请由实验执行者在完成预检后明确运行下面的命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_alimu_leave_sensor_matrix.py --config configs/alimu_leave_sensor_matrix_3seed.yaml
```

预期完成 15 条神经网络试验记录（5 个方法 x 3 个种子）和 4 条训练集拟合的解析基线记录，共 19 条结果。源目录 `runs/<experiment_id>/` 是唯一可信的实验工件目录。

严格协议中 `drift_only`、`bias_drift` 与 `fir_bias_drift_residual` 使用 `reference_scope: train_global`。2048 样本静态上下文只从训练分区的 BMX160/LSM9DS1 选择；绝不使用 MPU9250 的静止片段或任何其他样本进行静态校准、替换静态窗口或拟合变换。它是训练域先验，不是目标设备校准。

## 中断恢复

运行开始时会写入 ALiMU 专用活动指针。读取该指针取得完整 run 目录，再将其传给 `--resume-run`：

```powershell
$env:PYTHONIOENCODING='utf-8'
$runDir = (Get-Content outputs/active_alimu_matrix_run.txt -Raw).Trim()
Test-Path $runDir
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_alimu_leave_sensor_matrix.py --resume-run $runDir
```

恢复会先比对矩阵配置、基础配置、切分 manifest 与输入内容指纹。已完成且审计通过的基线和试验会记录为 `SKIPS`；任一不匹配或损坏都会在重新训练前失败。

## 查看结果

完整结果请从活动或完成后的 ALiMU 专用指针定位：

```powershell
$activeRun = (Get-Content outputs/active_alimu_matrix_run.txt -Raw).Trim()
Get-ChildItem $activeRun
Get-Content "$activeRun\report.md" -Encoding utf8

$latestRun = (Get-Content outputs/latest_alimu_matrix_run.txt -Raw).Trim()
Get-ChildItem outputs/latest_alimu_matrix
Get-Content "$latestRun\overall_summary.md" -Encoding utf8
```

检查 `trial_ledger.csv` 中是否有 15 条完成的神经试验，`trials.csv` 是否共 19 行，并结合 `overall_summary.*`、`report.md` 与 `provenance.json` 追溯配置、数据 manifest、git 提交和指标。
