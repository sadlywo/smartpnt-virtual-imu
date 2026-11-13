# ALiMU 研究结果汇总：2026-11-13

本页汇总已完成的共享信号校正、时间上下文、器件留出和匹配FIR/ARX实验。
报告、完整CSV和SVG/PDF/PNG图表收录于
[GitHub Release](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。

- [完整结果包，约27 MB](https://github.com/sadlywo/smartpnt-virtual-imu/releases/download/research-results-20261113/alimu_research_results_20261113.zip)
- [下载文件SHA256](https://github.com/sadlywo/smartpnt-virtual-imu/releases/download/research-results-20261113/SHA256SUMS.txt)
- [实验来源和发布包指纹](publication_evidence.json)
- [本次代码与结果包核验记录](VALIDATION.md)
- [上下文实验配置](../../../configs/ablations/alimu_context_mechanisms.yaml)
- [匹配传统基线配置](../../../configs/analysis/alimu_context_classical.yaml)

## 主要结论

1. 在训练覆盖的三种器件上，一个共享残差模型能降低相对于VN100的六轴信号误差，并在未见采集上保持收益。
2. 可用时间上下文与前视预算影响性能和结构排名。零前视、150 ms前视和整窗离线应分别报告。
3. 匹配目标样本和可用信息范围后，所评估神经时序模型在共享任务上优于FIR和递归ARX。
4. 器件留出时，传统模型在BMX160、MPU9250上具有优势；三器件折等权平均后，Raw LOW最低。
5. 较早的严格扩散消融中，18个设备×seed×信号组配对均由直接回归获得更低误差。

## 数据与评价口径

输入为BMX160、LSM9DS1、MPU9250的LOW六轴，目标为实测VN100六轴。
采样率40 Hz，最大输入320点，同一锚点中心40点监督和评价，stride40。
归一化由TRAIN拟合，VALID用于选择，模型状态锁定后计算本轮TEST。

ALiMU划分：TRAIN为dyn01/02/04/05/06和stat11，VALID为dyn07，TEST为dyn08/09。
共享TEST包含88,080个器件时刻，来自两条独立物理采集。器件留出折共享这两条采集。
三优化seed的SD描述训练随机性。历史工作已使用相同TEST，这些实验属于回顾性确认。

S先对每种器件计算0.5×(角速度RMSE/Raw角速度RMSE＋加速度RMSE/Raw加速度RMSE)，
再等权器件平均，越低越好，Raw LOW的S=1。物理RMSE为三轴分量池化RMSE，
角速度单位rad/s，加速度单位m/s²。评价目标是相对于VN100的信号校正。
模型零前视之外，现有LOW预处理另有25 ms已知可用性延迟。

## 共享模型：40轮等搜索预算

七家族各有三个学习率/dropout候选包；用pilot seed的VALID选择，再补齐三seed。
各家族相同40轮上限和早停，参数量、FLOPs和训练耗时分别保留。

| 家族 | TEST S，均值±优化seed SD | Gyro RMSE | Accel RMSE |
|---|---:|---:|---:|
| CNN-BiGRU | **0.361670 ± 0.004765** | 0.001101 | 0.183075 |
| CNN-Transformer | 0.376357 ± 0.003828 | 0.001165 | 0.187336 |
| CNN-GRU | 0.377581 ± 0.006788 | 0.001168 | 0.189674 |
| CNN-LSTM64 | 0.379226 ± 0.009039 | 0.001159 | 0.191052 |
| CNN-TCN | 0.388324 ± 0.001354 | 0.001172 | 0.196049 |
| Informer | 0.661412 ± 0.016070 | 0.001979 | 0.317325 |
| Transformer | 0.688200 ± 0.069368 | 0.002079 | 0.321833 |

CNN-BiGRU在本组的VALID和TEST均值最低，综合相对误差较Raw降低63.83%。

## 窗口长度：80轮上限组

窗口组保持相同中心目标，使用原80轮上限配方。CNN-BiGRU的结果为：

| 输入点数 | 名义长度 | TEST S | Gyro RMSE | Accel RMSE |
|---|---:|---:|---:|---:|
| 40 | 1 s | 0.403717 | 0.001232 | 0.204037 |
| 80 | 2 s | 0.370382 | 0.001135 | 0.187721 |
| 160 | 4 s | 0.360912 | 0.001109 | 0.184026 |
| 320 | 8 s | **0.355320** | 0.001085 | 0.179579 |

320点BiGRU的S较Raw降低64.47%；池化角速度、加速度RMSE分别降低67.91%和69.61%。
本80轮组中，GRU的VALID均值最低，BiGRU的TEST均值最低，分别保留这两个观察。

![窗口结果](https://github.com/sadlywo/smartpnt-virtual-imu/releases/download/research-results-20261113/context_window_scan.png)

## 匹配信息范围的FIR/ARX对照

传统方法在相同TRAIN中心目标上拟合，逐锚点重置，VALID选阶数和正则化。
主预算为每方法三个候选，扩展预算为十二个候选，两个预算分别报告。
递归ARX在VALID/TEST仅使用自身预测回馈；历史LOW滞后加差分实现另列为ARX-style。

| 320点输入下的信息范围 | FIR，主预算 | 递归ARX，主预算 | 原神经实验结果 |
|---|---:|---:|---|
| 零前视 | 0.785716 | 0.871371 | 固定前端GRU：0.522061 |
| 6点/150 ms前视 | 0.711962 | 0.713568 | 受控Transformer后端：0.372671 |
| 整窗离线 | 0.710244 | 0.710594 | 40轮CNN-BiGRU：0.361670 |

按17个已探索配置的VALID选择，零前视线性CNN前端加GRU达到S=0.477732。
该项是前端配置选择，单独保留其搜索范围。十二候选下，共享FIR在三种信息条件的
S为0.785508、0.711953、0.710244，递归ARX为0.853503、0.713559、0.710594。

![匹配传统基线](https://github.com/sadlywo/smartpnt-virtual-imu/releases/download/research-results-20261113/shared_classical.png)

## 同时留出器件与采集

用其余两种器件的TRAIN/VALID拟合，在目标器件dyn08/09测试。
以下传统项为整窗、三个候选主预算的TEST观察；Raw为1。

| 留出器件 | FIR S | 递归ARX S | 对应发现 |
|---|---:|---:|---|
| BMX160 | 0.605299 | 0.604949 | 传统项低于原神经组最低的Informer0.664198 |
| LSM9DS1 | 1.888404 | 1.865569 | Raw LOW最低 |
| MPU9250 | 0.657879 | 0.658472 | FIR略低于源TRAIN仿射0.668838 |
| 三折等权平均 | 1.050527 | 1.042997 | Raw LOW的1.000000最低 |

共享校正与未知器件迁移具有不同的结果。跨器件的相对表现需要逐器件评价。

![器件留出传统基线](https://github.com/sadlywo/smartpnt-virtual-imu/releases/download/research-results-20261113/held_device_classical.png)

## 归档来源

| 源运行 | 内容 |
|---|---|
| alimu_context_mechanisms_20261108 | 上下文、前视、前端机制、等预算比较、器件留出；227槽位、197新训、30复用 |
| alimu_context_mechanisms_20261109_figures_final | 上一行数值的最终9张图与结论；新增训练0 |
| alimu_context_classical_20261112 | 匹配FIR/ARX；432候选、两个预算共72选型槽位 |
| alimu_context_classical_20261112_verification | 144份保存预测全量回放与1296行指标独立检查 |
| cnn_bilstm_shared_ablation_20261106 | 共享ALiMU/联合ALiMU-Penn窗口与模块消融 |
| cnn_lstm_module_replacement_20261107 | 66条件前后端替换与容量对照 |
| alimu_matched_ddpm_20261106_final | 80点逐设备比较及严格去扩散配对 |
| alimu_nature_figures_20261108 | 已完成共享模型/逐设备实验的5图导读 |
| comprehensive_model_audit_20261106 | 较早ALiMU/Penn/SmartPNT分协议结果总审计 |

这些组保留各自的数据支持、训练预算与信息范围。完整逐seed/器件/采集指标见包内CSV。

## 核验与复现

原上下文数值封存核验2691个工件，独立复算426份预测指标最大差异4.44e-16。
传统基线封存741个工件；144份VALID/TEST预测从保存状态全量回放最大差异0，
1296行物理/逐轴指标最大复算差异2.84e-14。

Release包中每个成员均有SHA256，已声明封存哈希的文件在打包前逐一比对，
压缩后再次检查成员哈希和CRC。`publication_evidence.json`标识外层包与源运行回执。

Git保存代码、YAML、结果说明和来源指纹；报告、汇总CSV、图表与冻结源码在Release。
数据测量、处理后张量、逐样本预测和模型检查点保留在本地源运行。
包内原始报告沿用本地文件名；其中涉及预测/检查点的链接对应本地完整运行。

数据准备和环境见[复现指南](../../REPRODUCIBILITY.md)。原上下文/传统基线实验环境为
Python3.9.25、PyTorch2.3.1+cu121、RTX3060Ti；冻结来源记录和源码随包保留。
仓库发布代码同时整合既有框架，统一安装环境由`pyproject.toml`和`uv.lock`定义。
重新训练时复制对应YAML并指定新experiment_id，保留原封存运行。

```powershell
python scripts/package_research_results.py --config configs/analysis/github_results_20261113.yaml --source-root <local-project-with-sealed-runs> --out outputs/new-results-package
```

后续计划为器件异质性/误差机制分解、时延与前视对照，以及器件参数迁移仿真。
这些计划与本页已完成结果分别记录。
