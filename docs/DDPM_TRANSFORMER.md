# ALiMU 条件 DDPM + Transformer

GitHub阅读：[最新结果汇总](results/2026-11-13/README.md) · [完整报告、CSV与图表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。本页工件链接指向Release，下载包内按原run名称查找；本地源路径仍用代码形式标识。


本实验将已有的 ALiMU 六轴低成本 IMU 映射到实测 VN100 参考信号，分别训练
`BMX160_to_VN100`、`MPU9250_to_VN100`、`LSM9DS1_to_VN100`。
SmartPNT 在当前统一数据中只有导航标签，因此不用于此六轴配对实验。

## 运行

在仓库根目录使用本机 `pinn_imu` 的解释器，无需重新安装环境：

```powershell
$env:PYTHONUTF8 = '1'
& 'D:/Anaconda3/envs/pinn_imu/python.exe' scripts/run_ddpm_experiment.py --config configs/ddpm/alimu.yaml
```

也可激活环境后执行 `python scripts/run_ddpm_experiment.py --config configs/ddpm/alimu.yaml`。
`--run-id NAME` 固定输出目录。使用原配置和原始源代码运行
`--run-id NAME --resume` 可复用完成的候选实验，未完成的候选保留归档后按种子重新训练。
恢复会核对原配置、源代码和数据清单，避免把不同实现或数据混为一组实验。

入口记录 READS / WRITES / SKIPS；明确配置 `device: cuda` 时 CUDA 不可用会报错。
本机 `pinn_imu` 为 Python 3.9 / PyTorch 2.3.1+cu121，本入口兼容这个组合，
不调用仓库其他路径中要求 Python 3.11 的日期工具。

## 模型与训练

- 条件输入：低成本 IMU 六轴和温度；ALiMU 没有温度时填常数 25℃。
- 扩散变量：`high - low` 六轴残差，逐轴使用训练集均值和标准差归一化。
- Transformer：拼接带噪残差与条件，加入序列位置编码和扩散时间嵌入，再通过编码器和六轴输出头。
- 完整 DDPM：余弦 beta 调度、前向加噪、后验均值与方差、从高斯噪声开始的全部 100 步祖先采样；最后一步不再加噪。
  可选线性调度仅在终点信号系数 `alpha_bar <= 0.001` 时允许采样，避免短链仍保留明显信号。
- 默认 `prediction_type: x0`，以 MSE 学习干净残差。这是 DDPM 的干净样本参数化。
  `prediction_type: epsilon` 也可配置，使用噪声预测目标；两种参数化均有数值与采样测试。
  x0 推理也经过完整反向链，不使用单次 Transformer 回归替代采样。
- Adam、CUDA 自动混合精度、梯度裁剪、固定配置种子；独立保存各候选的最佳检查点。
- 每种传感器独立搜索 3 组 Transformer 配置；上限 40 轮、早停耐心 7 轮。
  检查点保存实际最小验证分数，`min_delta` 仅用于早停耐心判断。

## 数据隔离与指标

使用 `data/processed/unified_imu_v2/manifest.json` 的现有冻结划分及 SHA256 校验。
每种传感器有 163,265 / 14,992 / 29,983 条训练／验证／测试样本，分别对应 6 / 1 / 2 段采集记录。
各集合按 `physical_group_id` 隔离，窗口不跨序列，所有归一化和 affine 参数仅从训练集估计。

调参和早停使用固定种子，在预先保存的 32 个验证窗口上做与最终测试相同的 4 次完整采样平均，
并采用相同的重叠窗口拼接，支持区域的每个时间点只计算一次误差。
评分为陀螺仪和加速度计各自相对原始输入的 RMSE 比值的等权平均。
只有保存 `selection.json` 后才读取测试信号。选定模型另行评估完整验证集和完整测试集。
最终每窗口做 4 次 Monte Carlo 采样平均，再将重叠窗口平均回原时间轴；
尾部追加窗口以覆盖完整序列，长度不足一个窗口的记录会明确排除并计数。

对比方法为原始输入、训练集拟合的 affine 和 DDPM。输出逐轴及 gyro/accel 分组的
RMSE、MAE、bias、残差标准差、SNR、SNR 提升，以及逐记录指标。
分组 RMSE 是对应三轴所有分量误差平方的均值开方，不混用角速度与加速度单位。
逐轴 bias 保留正负号；分组 bias 为各轴均值的 RMS，避免正负抵消。
分组残差标准差先逐轴去均值，再对轴方差取均值开方，避免把静态轴偏差计为噪声。

`RMSE 提升百分比 = 100 * (1 - 模型 RMSE / 原始输入 RMSE)`，负数原样保留。
VN100 是实测参考，不是无误差真值；沿用上游对应传感器坐标轴，未独立核验外参。
双向注意力和重叠窗口属于离线映射。首次实验只有一个数据划分和一个种子，
不能据此声称跨记录、跨环境的普遍改善。

## 结果与追溯

`runs/<experiment_id>/` 是唯一实验来源；本入口不修改 `outputs/latest` 或 `outputs/latest_run.txt`。

```text
config.yaml, provenance.json, working_tree.patch, source_snapshot/
comparison.csv, comparison.md, report.md, complete.json
figures/three_sensor_improvement.{png,svg,pdf}
<pair>/
  config.yaml, data_manifest.json, normalization.json, affine_state.json
  search_results.{csv,json}, selection.json, report.md, complete.json
  trials/trial_*/config.yaml, history.csv, best.pt, summary.json, validation_windows.json
  validation/, test/: predictions.parquet, metrics.{csv,json,md}, per_record_metrics.csv
  figures/: training_search, heldout_rmse, axis_improvement, time_series, error, PSD
```

图表均通过 `src/virtual_imu/viz` 生成，导出 300 dpi PNG、SVG、PDF 和图表索引。
配置、数据 SHA256、Git 基线、未提交差异、实际源文件快照、设备信息、模型检查点和结果互相关联。
所有数据、检查点和图表保存在 Git 忽略的目录中。

参考：[DDPM 原论文](https://arxiv.org/abs/2006.11239)。
