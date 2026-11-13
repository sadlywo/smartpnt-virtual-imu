# Matched ALiMU FIR / ARX Supplement

GitHub阅读：[最新结果汇总](results/2026-11-13/README.md) · [完整报告、CSV与图表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。本页工件链接指向Release，下载包内按原run名称查找；本地源路径仍用代码形式标识。


Configuration: `configs/analysis/alimu_context_classical.yaml`.
Source: `runs/alimu_context_mechanisms_20261108/`.
New authoritative result directory: `runs/alimu_context_classical_20261112/`.

The supplement reconstructs the original full LOW contexts and checks every
center40 support identity and TRAIN normalization against the sealed source.
It does not infer missing context by treating the saved center-target Parquet
files as continuous raw recordings.

Shared models cover40/80/160/320-point full-window conditions, plus320-point
zero-future and six-point-future conditions. Device-held-out models cover both
six-point and full-window information to match the source CNN-GRU and
BiGRU/Informer dependencies separately. State resets at each input anchor.

FIR uses LOW lags. Historical ARX-style uses LOW lags plus first differences;
it has no output autoregression. Recursive residual ARX uses lagged residual
outputs for TRAIN identification and only its own predicted outputs at
VALID/TEST. Teacher residual lags are restricted to the original supervised
TRAIN target rows. Every selected candidate is locked before new TEST access.

The main comparison has three declared packages per family. A separate
twelve-candidate history/regularization search evaluates search sensitivity.
These are deterministic closed-form fits, with no copied neural seed SD or
claim of equal wall-clock cost. Historical TEST use remains documented.

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_context_baselines.py --config configs/analysis/alimu_context_classical.yaml
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_context_baselines.py --config configs/analysis/alimu_context_classical.yaml --verify
```

Use a new `experiment_id` for another fit. The CLI refuses to overwrite an
existing result directory. Source run predictions, figures and checkpoints
remain read-only; no neural retraining is performed.

Outputs include config/data/source provenance, all candidate states and VALID
trials, a pre-TEST selection lock, exact support receipts, selected prediction
arrays, per-device/recording/axis metrics, strictly matched neural comparison
tables, independent metric/state checks, SVG/PDF/PNG300dpi figures and a sealed
artifact manifest. Figures use the repository IMU visualization skill/style.

## 2026-11-12 实测结果

12个比较条件已完成432个确定性候选拟合。主三候选预算、扩展十二候选预算
分别保留36个选型槽位，共72个槽位。两套预算复用相同候选拟合，不计为额外
独立重复。未新增神经训练。

以下为320点输入、相同中心40点TEST目标上的S，越低越好，Raw LOW为1。
FIR和递归残差ARX均按三个预声明候选的VALID成绩选阶数和正则化。

| 信息条件 | FIR | 递归残差ARX | 对应神经模型 |
|---|---:|---:|---|
| 零前视 | 0.785716 | 0.871371 | 固定前端GRU：0.522061 |
| 6点/150ms前视 | 0.711962 | 0.713568 | 受控Transformer后端：0.372671 |
| 整窗离线 | 0.710244 | 0.710594 | 40轮等预算CNN-BiGRU：0.361670 |

三行保留各自原始神经实验协议。另一个按17配置VALID选择的零前视模型为
线性CNN前端加GRU，S=0.477732；它属于配置选择实验，不当作与FIR三个候选
完全相同的家族搜索预算。历史LOW差分ARX-style在上述三条件分别为
0.785716、0.711962、0.710248，与FIR接近。

扩大到十二候选后，共享FIR三条件为0.785508、0.711953、0.710244；共享递归
残差ARX为0.853503、0.713559、0.710594。扩展搜索后，共享任务中神经模型
仍保持更低的综合测试误差。

整窗FIR在40/80/160/320点窗口上的S分别为0.770479、0.710250、0.710244、
0.710244。160和320点选择同一49点局部滤波配置，中心目标已经远离边界，
因此两者结果几乎相同。

器件留出有不同结果。整窗主预算下，BMX160上递归ARX为0.604949，低于原神经
组中最低的Informer0.664198；MPU9250上FIR为0.657879，略低于源TRAIN仿射
0.668838。LSM9DS1上FIR/递归ARX分别为1.888404/1.865569，Raw LOW仍最低。
三器件折等权平均，整窗FIR为1.050527，递归ARX为1.042997，Raw为1。

论文可据此表述：在训练覆盖的器件上，匹配时间信息范围后，所评估神经时序
模型比FIR及递归ARX获得更低的信号校正误差；器件留出时的相对表现随器件
改变，传统模型在部分目标器件上更优。

## 工件与核验

- [完整报告](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [逐条件TEST表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [严格匹配神经对照](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [所有候选及VALID分数](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [源运行、配置与代码追溯](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [完成封存](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [保存预测全量回放核验](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)

20项相关测试通过。两张图均完成PNG目视检查，SVG/PDF/PNG300dpi导出齐全。
741个工件哈希通过；72个选型槽位的144份VALID/TEST预测均从保存模型状态全量
回放，最大差异0。1296行物理/逐轴指标独立复算最大差异2.84e-14。原始和本轮
目标表的全部键、LOW/HIGH值一致；选型只使用VALID，源运行保持只读。

独立核验不写入已封存数值运行。复核时将输出指向新的核验文件：

```powershell
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/verify_context_baselines.py --out runs/alimu_context_classical_20261112_verification_repeat/verification.json
```
