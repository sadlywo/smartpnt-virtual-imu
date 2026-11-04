# Penn 四折三种子矩阵运行指南

正式训练在 Conda 环境 `pinn_imu` 中执行。所有命令都在项目根目录运行。

## 1. 先做配置检查

```powershell
$env:PYTHONPATH='src'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -c "from virtual_imu.config import load_config; from virtual_imu.training.penn_matrix_config import build_trial_specs; from pathlib import Path; c=load_config('configs/penn_loso_matrix_3seed.yaml'); t=build_trial_specs(c, Path.cwd()); print(len(t), t[0].trial_id, t[-1].trial_id)"
```

Expected output contains:

```text
60 test_af__tcn__seed_20260712 default_bs__fir_bias_drift_residual__seed_20260714
```

## 2. 可选 smoke

Smoke 会运行 60 个一 epoch 小模型 trial，主要用于验证路径、checkpoint、Hybrid FIR 和聚合逻辑：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_penn_loso_matrix.py --config configs/penn_loso_matrix_3seed.yaml --smoke
```

## 3. 正式 60-trial GPU 训练

正式运行不加 `--smoke`：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_penn_loso_matrix.py --config configs/penn_loso_matrix_3seed.yaml
```

程序会立即写入：

```text
outputs/active_penn_matrix_run.txt
```

该文件指向当前 source-of-truth run。训练完成且聚合报告成功后，才会更新：

```text
outputs/latest_run.txt
```

## 4. 中断后恢复

不要创建新的 run，使用 active pointer：

```powershell
$run=(Get-Content outputs/active_penn_matrix_run.txt -Encoding utf8).Trim()
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_penn_loso_matrix.py --config configs/penn_loso_matrix_3seed.yaml --resume-run $run
```

如果矩阵 YAML、base model config、strict manifest 或任意 fold manifest 被修改，程序会在 baseline/trial 工作开始前拒绝恢复。

## 5. 运行后检查

```powershell
$run=(Get-Content outputs/latest_run.txt -Encoding utf8).Trim()
Get-Content (Join-Path $run 'overall_summary.csv')
Get-Content (Join-Path $run 'fold_summary.csv')
Get-Content (Join-Path $run 'report.md')
```

关键数量应为：

- `trial_ledger.csv`：60 个神经 trial；
- `trials.csv`：76 行，即 60 个神经结果和 16 个解析基线结果；
- 四个 fold：`test_af`、`test_as`、`test_bf`、`default_bs`；
- 每个神经方法每个 fold：三个 seed；
- 每个预测文件：只包含对应 held-out sequence，六个 virtual 列均为有限值。

大型 checkpoint、prediction 和训练目录只保留在 source run；`outputs/latest/` 只保存轻量汇总文件。
