# CNN-LSTM 前后模块替换：实测结论与结构解释

GitHub阅读：[最新结果汇总](results/2026-11-13/README.md) · [完整报告、CSV与图表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。本页工件链接指向Release，下载包内按原run名称查找；本地源路径仍用代码形式标识。


**CNN 前端在 ALiMU 域的价值强；LSTM 并不是单域唯一最佳后端。** 在本轮固定配方下，换掉 CNN 的四个前端在两个 AL 评价条件、两类物理 RMSE 上均有损失；保留 CNN 后，GRU/BiGRU 可以在 AL 单域取得更低的部分读数。按预先冻结的 VALID 规则，AL 单域选择 **CNN→GRU**，ALiMU＋Penn 联合共享模型选择 **CNN→单向 LSTM64**。

**训练、评估和数值核验已完成：66/66条件（48项新拟合＋18项复用控制）、132份 VALID/TEST 预测的全部样本独立 CUDA 回放 PASS，7组真实图及21份导出文件 QA PASS。** 独立解读审阅与完成封存分别见本轮 [independent_review.json](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[complete.json](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。

实际 TEST 观测中，AL 单域的最低归一化均值来自 CNN→BiGRU；联合总体最低归一化均值来自历史 CNN→BiLSTM32。Penn 存在陀螺、加速度和容量取舍。这些 TEST 观察用于描述，不更改 VALID 选择，也不为联合模型按 AL/Penn TEST 各选一种架构。

权威证据：[本轮完整报告](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[实测解读](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[7组真实图索引](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[全部样本数值核验](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[真实图 QA](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。配置见[本轮冻结快照](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)与[仓库 YAML](../configs/ablations/cnn_lstm_module_replacement.yaml)，实验设计见[规范](superpowers/specs/2026-11-07-cnn-lstm-module-replacement-design.md)。

## VALID 选择与 TEST 描述

下面各模型数字均为三个优化 seed 的均值 ± 样本SD（ddof=1，n=3）；归一化分数无量纲、越低越好。它先在每设备对内对陀螺/Raw 与加速度/Raw 比值各取一半，再等权设备对、等权数据集。Raw=1；Raw 是确定性读数，n=1、SD未定义。完整 VALID 排名见 [validation_ranking.csv](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。

| 训练范围 | VALID名次 | variant | 归一化分数：均值 ± SD |
| --- | --- | --- | --- |
| alimu_only | 1 | `cnn_gru` | 0.331870 ± 0.003258 |
| alimu_only | 2 | `cnn_bigru` | 0.333727 ± 0.001826 |
| alimu_only | 3 | `cnn_lstm64` | 0.336470 ± 0.006132 |
| pooled | 1 | `cnn_lstm64` | 0.321351 ± 0.002097 |
| pooled | 2 | `cnn_bilstm32` | 0.326514 ± 0.005184 |
| pooled | 3 | `cnn_gru` | 0.337095 ± 0.009363 |

AL 的 GRU、BiGRU 与 LSTM64 的 VALID 数字接近；这里按既定均值规则报告排序，不把排序当作显著性证明。联合模型的 VALID 按 AL/Penn 等权评分，所选 `cnn_lstm64` 同时用于两个 TEST 域。

三个域的 TEST 归一化分数摘自 [summary.csv](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)：

| variant | AL单域 TEST | 联合模型 TEST·AL | 联合模型 TEST·Penn |
| --- | --- | --- | --- |
| `cnn_lstm64` | 0.371365 ± 0.011599 | 0.397682 ± 0.005660 | 0.200760 ± 0.001079 |
| `cnn_bilstm32` | 0.373622 ± 0.002399 | 0.389734 ± 0.002453 | 0.202218 ± 0.003576 |
| `cnn_identity` | 0.451010 ± 0.004393 | 0.596264 ± 0.001927 | 0.187859 ± 0.004024 |
| `cnn_gru` | 0.365479 ± 0.003098 | 0.399060 ± 0.010842 | 0.213876 ± 0.005776 |
| `cnn_bigru` | 0.355320 ± 0.000817 | 0.402033 ± 0.001410 | 0.215624 ± 0.006112 |
| `cnn_transformer` | 0.369866 ± 0.010716 | 0.409577 ± 0.019165 | 0.216915 ± 0.008401 |
| `cnn_tcn` | 0.393262 ± 0.003227 | 0.419791 ± 0.005786 | 0.214876 ± 0.002398 |
| `linear_lstm64` | 0.632748 ± 0.020695 | 0.660071 ± 0.005663 | 0.184792 ± 0.012216 |
| `mlp_lstm64` | 0.582871 ± 0.015805 | 0.612352 ± 0.006458 | 0.187091 ± 0.001898 |
| `gru_lstm64` | 0.547425 ± 0.058041 | 0.560639 ± 0.015183 | 0.240202 ± 0.013722 |
| `transformer_lstm64` | 0.696662 ± 0.010286 | 0.673483 ± 0.039181 | 0.195612 ± 0.007641 |

联合**总体** TEST 归一化分数为：历史 `cnn_bilstm32` 0.295976 ± 0.002733，参照 `cnn_lstm64` 0.299221 ± 0.003024。总体分数对两个数据集等权；不能用总体物理 RMSE 或数据点数量权重替代它。BiLSTM32 在联合 AL 的加速度更低、陀螺略高，联合 Penn 的两项物理均值略高，没有逐指标的一致优势。

以下是关键物理读数；陀螺列以 ×10⁻³ rad/s 显示，加速度列以 m/s² 显示，均为分量 RMSE 的跨 seed 均值 ± SD。每次 RMSE 先池化相同支持上的样本/轴平方误差再开方，不以录制或设备 RMSE 的平均重建总体。

| 评价范围 | variant | 陀螺分量RMSE（×10⁻³ rad/s） | 加速度分量RMSE（m/s²） |
| --- | --- | --- | --- |
| AL单域 | `cnn_lstm64` | 1.1411 ± 0.0449 | 0.187031 ± 0.006508 |
| AL单域（VALID所选） | `cnn_gru` | 1.1245 ± 0.0246 | 0.184835 ± 0.002909 |
| AL单域（TEST观测低值） | `cnn_bigru` | 1.0847 ± 0.0201 | 0.179579 ± 0.001263 |
| 联合·AL（VALID所选） | `cnn_lstm64` | 1.2245 ± 0.0047 | 0.197937 ± 0.008677 |
| 联合·AL | `cnn_bilstm32` | 1.2346 ± 0.0019 | 0.189388 ± 0.003955 |
| 联合·Penn（VALID所选） | `cnn_lstm64` | 4.3229 ± 0.0574 | 0.077734 ± 0.002194 |
| 联合·Penn | `linear_lstm64` | 3.3756 ± 0.2235 | 0.092602 ± 0.006313 |
| 联合·Penn | `cnn_identity` | 3.9983 ± 0.0665 | 0.074373 ± 0.002541 |

AL 单域 `cnn_bigru` 的归一化均值低于 VALID 所选 `cnn_gru`，但它的双向结构可访问整窗未来输入。联合 Penn 的 `linear_lstm64` 在本轮11种结构中归一化均值最低，陀螺也最低，却提高加速度误差；`cnn_identity` 的加速度均值最低且两项均低于参照，但在联合 AL 明显变差。这里比较的是同一批联合训练模型，没有训练或选择 Penn 专用模型。

## 配对变化：CNN 前端与后端各贡献了什么

以下全部来自 [paired_summary.csv](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)，参照为精确相同 protocol、split、support、window、seed 的 `cnn_lstm64`。每个 seed 先算 `100×(替换−参照)/参照`，再取均值和样本SD；**不是两个跨 seed 均值之比**。表中每格为“陀螺百分比均值 ± SD / 加速度百分比均值 ± SD”，单位为百分比，正值表示误差增大，SD对应百分点；不是录制级置信区间。参照为零时百分比保留NaN，绝对差和有效计数仍保留于CSV。

### 保留 LSTM64，改变前端

| variant | AL单域：陀螺 / 加速度 | 联合·AL：陀螺 / 加速度 | 联合·Penn：陀螺 / 加速度 |
| --- | --- | --- | --- |
| `linear_lstm64` | +71.08 ± 11.34 / +59.73 ± 12.56 | +66.42 ± 2.32 / +50.78 ± 6.87 | -21.90 ± 5.40 / +19.20 ± 9.18 |
| `mlp_lstm64` | +57.39 ± 9.76 / +51.15 ± 5.55 | +54.73 ± 1.25 / +46.25 ± 4.75 | -18.06 ± 0.91 / +15.14 ± 5.61 |
| `gru_lstm64` | +50.37 ± 10.03 / +40.75 ± 9.49 | +42.04 ± 2.54 / +42.07 ± 2.62 | +2.51 ± 1.13 / +53.20 ± 20.69 |
| `transformer_lstm64` | +85.89 ± 8.00 / +80.07 ± 6.55 | +70.07 ± 10.20 / +62.91 ± 17.82 | -13.08 ± 2.00 / +17.97 ± 8.85 |

在 AL 单域与联合 AL，四种前端替换的陀螺及加速度在三个 seed 中均比参照更差。近似预算的逐点 MLP 仍在 AL 单域增加57.39%/51.15%，说明本轮增加通道非线性容量没有替代当前局部时间 CNN 的表现；GRU 与 Transformer 前端也没有在这两个 AL 条件达到参照。该证据支持当前 CNN 前端在 AL 上有价值。

Penn 上，Linear/MLP/Transformer 前端的陀螺误差分别下降约21.90%/18.06%/13.08%，但加速度增加约19.20%/15.14%/17.97%。结论有域和指标范围，不能外推为 CNN 在每个域都更好或 Transformer 家族在充分调优后仍必然更差。

### 保留 CNN，改变后端

| variant | AL单域：陀螺 / 加速度 | 联合·AL：陀螺 / 加速度 | 联合·Penn：陀螺 / 加速度 |
| --- | --- | --- | --- |
| `cnn_bilstm32` | +1.74 ± 5.57 / +0.47 ± 5.32 | +0.82 ± 0.45 / -4.21 ± 4.39 | +0.41 ± 3.19 / +1.46 ± 3.85 |
| `cnn_identity` | +21.17 ± 6.19 / +21.57 ± 5.25 | +52.61 ± 2.46 / +44.27 ± 6.26 | -7.49 ± 2.61 / -4.25 ± 4.97 |
| `cnn_gru` | -1.41 ± 1.81 / -1.10 ± 3.41 | +0.52 ± 4.75 / -0.95 ± 3.26 | +4.62 ± 3.26 / +10.30 ± 3.43 |
| `cnn_bigru` | -4.88 ± 2.30 / -3.92 ± 2.63 | +4.24 ± 0.98 / -0.96 ± 4.14 | +5.34 ± 3.34 / +11.46 ± 3.67 |
| `cnn_transformer` | +3.09 ± 6.91 / -0.12 ± 6.75 | +9.15 ± 7.26 / -0.29 ± 6.69 | +7.32 ± 2.93 / +9.41 ± 6.90 |
| `cnn_tcn` | +4.78 ± 4.72 / +7.16 ± 4.61 | +6.96 ± 2.36 / +5.90 ± 3.73 | +9.96 ± 1.91 / +1.46 ± 4.69 |

AL 单域 BiGRU 的两项配对变化均为负，三个 seed 都改善；GRU 的归一化配对变化为−1.54% ± 2.39%，只有2/3 seed改善，适合表述为可替代的候选而非确定优势。删除后端在两个 AL 条件均明显变差，支持窗口内时序聚合有用；LSTM 以外的门控单元也能实现这种作用。

联合 Penn 的 CNN-only 两项配对变化都更好，陀螺在3/3 seed改善、加速度在2/3 seed改善；联合 AL 却损失很大。联合整体 TEST 上，历史 BiLSTM32 的归一化配对变化为−1.07% ± 1.90%，2/3 seed改善。这些取舍不改变联合 VALID 选择 LSTM64，也不支持每个域分别按 TEST 换后端。

## 数据支持与不确定性

两个训练范围分别为 ALiMU 的 BMX160/MPU9250/LSM9DS1→VN100 共享模型，以及 ALiMU＋PennCOSYVIO Tango Top IMU→ADIS16488 的联合共享模型。AL/Penn 的后两列评价来自同一组联合模型。按本轮 `supports/<protocol>/records.json` 的 physical_group_id 去重：

| 数据集 | TRAIN物理录制 | VALID物理录制 | TEST物理录制 | TEST配对记录文件 | 每模型/seed的TEST目标点数 |
| --- | --- | --- | --- | --- | --- |
| ALiMU | 6 | 1 | 2 | 6（每物理录制3个LOW设备） | 88,080 |
| PennCOSYVIO | 2 | 1 | 1 | 1 | 6,280 |

同一次物理录制对应的多个设备对不是独立录制；联合模型也不会使同一 AL TEST 录制再增加一份独立样本。三个 seed 的 SD 描述训练优化变化，三个同号差值不是录制级显著性检验。设备对行数、目标点数或anchor数都不能充当独立录制数量，以上点数也不因三个 seed 而增加。

旧 TEST 已经被查看，本轮是在既有数据及划分上的后续研究。TEST 前冻结本轮结构、配方、分析与检查点可以约束后续选择自由度，但新录制/新场景验证仍是独立外推的必要后续证据。

## 参数与统一新测推理成本

下表摘自 [compute_summary.csv](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。参数量各 seed 相同（SD=0）；batch1列是每作业重复测时均值 `batch1_ms_mean` 再跨三个 seed 求均值 ± SD。全部66条件包括18个复用控制均在本轮 RTX3060Ti/CUDA 环境统一新测，warmup5、重复20次并同步GPU。

| variant | 参数量 | 相对参考参数量 | AL单域 batch1（ms） | 联合 batch1（ms） |
| --- | --- | --- | --- | --- |
| `cnn_lstm64` | 81,414 | +0.00% | 1.109 ± 0.054 | 1.089 ± 0.055 |
| `cnn_bilstm32` | 73,222 | -10.06% | 1.432 ± 0.023 | 1.481 ± 0.084 |
| `cnn_identity` | 48,134 | -40.88% | 0.598 ± 0.037 | 0.545 ± 0.012 |
| `cnn_gru` | 82,614 | +1.47% | 1.058 ± 0.107 | 1.067 ± 0.112 |
| `cnn_bigru` | 78,758 | -3.26% | 1.406 ± 0.105 | 1.493 ± 0.229 |
| `cnn_transformer` | 81,606 | +0.24% | 1.096 ± 0.029 | 1.214 ± 0.167 |
| `cnn_tcn` | 77,830 | -4.40% | 2.248 ± 0.369 | 2.040 ± 0.031 |
| `linear_lstm64` | 38,406 | -52.83% | 0.780 ± 0.020 | 0.743 ± 0.017 |
| `mlp_lstm64` | 82,502 | +1.34% | 1.051 ± 0.029 | 1.261 ± 0.227 |
| `gru_lstm64` | 79,622 | -2.20% | 1.155 ± 0.082 | 1.115 ± 0.052 |
| `transformer_lstm64` | 80,134 | -1.57% | 1.431 ± 0.167 | 1.360 ± 0.046 |

七个新训练非删除替换的实际总参数在参考81,414的±5%内；历史 `cnn_bilstm32=73,222` 比参考少10.06%，明确不在此预算组。`linear_lstm64=38,406` 和 `cnn_identity=48,134` 是有意的低容量删除控制。具体结构、信息范围及近似预算的限制见下文。

CNN-GRU 与 LSTM64 的均值均约1.1ms，当前重复数和波动不足以据小差距宣称稳定速度优势。BiGRU 约1.4–1.5ms，CNN-only约0.55–0.60ms；这些均是模型前向成本，不含采样、窗口缓冲和前视等待。18个控制的旧训练成本保留 `prior_run/source_training` 来源，48项新增拟合另记 `current_run/new_training`；统一推理成本标记 `current_run_uniform_measurement`。本轮并发训练下的单作业 elapsed_seconds 仅作描述，不能把旧训练耗时计作本轮重新训练或据此宣称受控训练速度优势。

## 为什么考察局部 CNN 与门控历史的组合

参考 `cnn_lstm64` 的路径是：六轴 LOW 经 TRAIN-only 输入标准化 → 三层 CNN64/k5/BN/GELU → 单向 LSTM64 → 公共读出头 → 六轴归一化残差。每个模型输出 `B×6×320` 的归一化 `HIGH−LOW` 残差，训练损失使用其中的公共中心40点；物理指标在同一中心支持上还原计算。读出头为 LayerNorm64 → Linear64 → GELU → Linear6。

```text
残差目标 r = HIGH − LOW
模型输出 r_hat_normalized
r_hat = denormalize_residual_with_TRAIN_statistics(r_hat_normalized)
HIGH_hat = LOW + r_hat
```

LOW 加法跳接和残差还原约定在所有结构中相同；公共残差读出头也保留。模块替换改变残差分支的表示，不将任务改为直接预测物理 HIGH 六轴或预测 LOW 本身。HIGH 只提供监督目标，不作为输入特征。

YAML 中 `model.lstm_hidden=32`、`bidirectional=true` 是旧工厂的构造模板；`cnn_lstm64` 明确映射到旧 `cnn_lstm_wide`，实际构造单向 hidden64。参考结构不能仅凭这两个模板字段误读为 BiLSTM32。

CNN 沿时间共享局部卷积权重，可学习相邻采样点的联合表示以及轴间混合。后续 LSTM 在局部表示上递归更新隐藏状态和记忆状态，允许信息有选择地跨时间保留。LSTM 原始研究提供了门控记忆与长时依赖学习的结构依据；它没有证明当前 IMU 任务中哪一层承担哪种物理误差校正。[Hochreiter & Schmidhuber, 1997](https://doi.org/10.1162/neco.1997.9.8.1735)，[原论文书目与作者摘要](https://pubmed.ncbi.nlm.nih.gov/9377276/)。

结构假设是局部共享表示与选择性历史聚合形成互补，帮助 LOW→HIGH 映射；本轮 AL 结果给出固定配方下的描述性支持。这里没有把 CNN 指定为低通滤波器，也没有把 LSTM 的状态指定为陀螺偏置。对称 padding 只限定依赖范围，未约束卷积权重镜像对称或零相位。

### 13点局部范围与6点前视

对步长1、膨胀1、三层 k5 卷积，结构允许的局部推理感受野为：

```text
R_CNN = 1 + 3 × (5 − 1) = 13 点
z_t 依赖 x_[t−6 : t+6]，越过输入窗口的部分由 padding 处理。
```

在40Hz下，13个采样点的首末时间差为 `12/40 = 0.300 s`，前视为 `6/40 = 0.150 s`。BN 在 `eval()` 中使用固定运行统计，GELU 和读出头不额外扩大时间依赖。

单向 LSTM 在位置 `t` 只接收特征前缀 `z_0,…,z_t`，但其中的 `z_t` 已含未来 LOW 输入。因此完整参考模型允许的原始输入依赖可延伸到 `x_min(319,t+6)`，历史端可延伸到窗口起点；**单向后端并不使整个 CNN-LSTM 严格因果**。递归状态在每个320点 anchor 内重新开始，不跨 anchor 累积无限历史。

这里计算的是结构允许的最大推理依赖范围，不是训练后测得的有效记忆长度。门控和学习到的权重可能只利用其中一部分。TRAIN 模式 BN 的均值/方差沿 batch 与 time 维估计，会通过统计量耦合多个位置；`eval()` 使用固定 running statistics，不再以当前窗口重算这些统计。上述感受野只描述 `eval()` 推理，不能用于声称训练计算严格局部或因果。BN 的运行统计与输入/残差的 TRAIN-only 标准化统计是不同的统计层次。

## 固定320点输入、中心40点支持

所有结构使用相同40Hz、320点输入窗口，名义长度8 s；监督和评估使用同一 anchor 的中心40点，名义长度1 s。以零起点索引表示，中心切片为 `140:180`，即位置140至179。配置固定 `max_window=320`、`block_size=40`、`stride=40`，不重做窗口搜索，也不改成整窗320点监督。

同样的输入长度、公共 anchor 和评价目标保证各结构在相同样本上比较，却不保证各结构真正能访问同样多的历史或未来。例如，参考模型在中心位置 `t` 最远访问 `t+6`，而无因果掩码的 Transformer 可以访问输入窗口末端319。更多未来 LOW 输入不等于读取未来 HIGH 目标，但会改变模型适用的信息条件。

batch1 前向延迟也不等于端到端系统延迟：它未包含传感器采集、anchor 缓冲或未来输入等待。6点前视只是参考模型相对单个目标位置的结构依赖，不是已测得的部署延迟。

## 两侧替换与容量控制

固定 CNN 替换后端，或固定单向 LSTM64 替换前端；保持相同64维模块接口和读出头。保留模块及其 BN buffer 在相同种子下复制相同初始化，新模块使用单独的确定性随机流。本轮核验已确认保留模块的初始化一致性。

“固定/保留”指结构及同 seed 初始化对齐，不表示训练中冻结这些参数；各模型仍端到端拟合其残差分支。

| variant | 改变的模块 | 保留模块 | 实际宽度/结构 | 来源与预算角色 |
| --- | --- | --- | --- | --- |
| `cnn_lstm64` | 主参照 | CNN＋单向 LSTM64 | CNN64×3/k5；LSTM64 | 复用 `cnn_lstm_wide/320`；参考参数量81,414 |
| `cnn_bilstm32` | 后端方向与宽度 | 相同 CNN | BiLSTM32/方向 | 复用 `cnn_bilstm/320`；73,222参数，比参考少10.06%，不属于±5%预算组 |
| `cnn_identity` | 删除循环后端 | 相同 CNN | Identity，直接接读出头 | 复用 `cnn_only/320`；48,134参数，低容量删除控制 |
| `cnn_gru` | 后端 | 相同 CNN | GRU72 → Linear64 | 本轮新拟合；七个预算匹配替换之一 |
| `cnn_bigru` | 后端 | 相同 CNN | BiGRU40/方向 → Linear64 | 本轮新拟合；七个预算匹配替换之一 |
| `cnn_transformer` | 后端 | 相同 CNN | 单层64维 Transformer，4头，FF128 | 本轮新拟合；七个预算匹配替换之一 |
| `cnn_tcn` | 后端 | 相同 CNN | 4个残差瓶颈块，宽32，k3，dilation 1/2/4/8 | 本轮新拟合；七个预算匹配替换之一 |
| `linear_lstm64` | 删除局部时间前端 | 相同单向 LSTM64 | 逐点 Linear6→64 | 新拟合；38,406参数，低容量删除控制 |
| `mlp_lstm64` | 前端 | 相同单向 LSTM64 | 逐点 MLP6→176→176→64，逐层 BN/GELU | 本轮新拟合；七个预算匹配替换之一 |
| `gru_lstm64` | 前端 | 相同单向 LSTM64 | 单向 GRU104 → Linear64 | 本轮新拟合；七个预算匹配替换之一 |
| `transformer_lstm64` | 前端 | 相同单向 LSTM64 | Linear6→64＋单层 Transformer，4头，FF192 | 本轮新拟合；七个预算匹配替换之一 |

约±5%的参数预算只适用于7个新训练、非删除替换：`cnn_gru`、`cnn_bigru`、`cnn_transformer`、`cnn_tcn`、`mlp_lstm64`、`gru_lstm64`、`transformer_lstm64`。这些宽度在训练前选定，本轮 `compute_summary.csv` 的实际参数量均落在81,414参数参考的±5%内，并与模型计数核验一致。这里的参数量是结构属性，不是训练性能结果。

历史控制 `cnn_bilstm32` 为73,222参数，相对参考差异为 `(73,222−81,414)/81,414×100≈−10.06%`，明确排除在±5%预算组之外。它同时改变容量和方向性，不能称为与参考严格匹配容量。`linear_lstm64` 的38,406参数和 `cnn_identity` 的48,134参数是有意设置的低容量删除控制，也不属于该预算组。不能据删除控制的损失单独区分“模块类型”与“参数减少”的影响。GRU72 与 BiGRU40/方向也不是同隐藏宽度比较。这些实际计数来自本轮计算成本表，并已纳入独立数值核验。

近似相同参数量仍可对应不同乘加量、激活内存、并行度、规范化方式及优化难度。固定配方比较回答的是这些具体实现和预算的表现，不是每个架构各自充分 HPO 后的性能上限。

### 替代家族检验的内容

GRU 使用 reset/update 门控制隐藏状态更新；它适合检验当前配方下是否必须使用 LSTM 这一具体门控/状态结构。原始 GRU 论文来自机器翻译，其实验结果不能转用为本任务的优劣结论。[Cho et al., 2014，第2.3节](https://aclanthology.org/D14-1179.pdf)。双向循环块还改变未来信息范围，不能只将差异归结于门的种类。

逐点 Linear 和逐点 MLP 在推理时只处理当前采样点的六轴特征；MLP 可以增加通道非线性容量，却没有卷积的相邻时间混合。GRU 前端则引入递归历史表示。将这些前端与相同 LSTM64 配合，检验的是前端表示方式的替代能力。

TCN 家族用膨胀卷积扩大有限时间范围。Bai 等的通用 TCN 使用因果卷积和残差块；本项目是**对称 padding 的自定义瓶颈版本**，每块只有一个 k3 时间卷积，并非原文的双层因果卷积块，不能把原论文的因果性或性能结论直接套用。[Bai, Kolter & Koltun, 2018，第3节](https://arxiv.org/abs/1803.01271)。按本配置计算：

```text
R_TCN_backend = 1 + 2 × (1 + 2 + 4 + 8) = 31 点
R_CNN+TCN    = 13 + 31 − 1 = 43 点
整体半径 = 21 点；未来输入范围为21点，即21/40 = 0.525 s。
```

Transformer 自注意力可让编码器位置与窗口中其他位置交互；是否遮蔽未来取决于 attention mask。本项目使用无因果掩码的单层编码模块、固定正弦位置编码和 dropout0，与原论文整套 encoder–decoder 的深度及预算不同。[Vaswani et al., 2017，第3.2.3与3.5节](https://arxiv.org/abs/1706.03762)。因此 Transformer 放在前端后，即使后接单向 LSTM，整体仍允许整窗依赖。

| 结构组 | 结构允许的最大推理依赖，限于当前320点 anchor | 未来范围 |
| --- | --- | --- |
| `cnn_lstm64`、`cnn_gru` | 从窗口起点到 `min(319,t+6)` 的前缀 | 最多6点 |
| `cnn_identity` | `t−6` 至 `t+6`，共13点 | 最多6点 |
| `cnn_tcn` | `t−21` 至 `t+21`，共43点 | 最多21点 |
| `linear_lstm64`、`mlp_lstm64`、`gru_lstm64` | 从窗口起点到 `t` 的前缀 | 0点，按推理计算图 |
| `cnn_bilstm32`、`cnn_bigru`、`cnn_transformer`、`transformer_lstm64` | 整个窗口0至319 | 随目标位置变化，最远到319 |

本轮同时改变算子、隐藏宽度、规范化和结构依赖范围。结果描述这些具体实现与固定配方的表现，不能识别脱离这些因素的纯粹“算子因果效应”。

## 固定训练、复用与实际核验

11结构×2训练范围×3优化 seed（20261010/20261011/20261012）已完成66条件，其中8种新结构对应48项新拟合，3种旧控制对应18项复用。固定历史 AdamW、Huber、最多80轮、patience15；学习率0.0022830268403520168、weight_decay0.007082535831697058、warmup5，完整数值以冻结 YAML 为准。没有每模型或每架构家族 HPO，结果不代表各家族充分调优后的性能上限。

输入与残差标准化仅使用 TRAIN 统计。相同 seed 的保留模块初始化、每epoch平衡采样顺序及 VALID 最小值检查点已通过核验；所有模块仍端到端更新。复用来源为[旧控制 SOURCE 运行](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)，本轮保留原生 job/checkpoint/初始化和来源哈希，登记 `reused=true`，保留旧 sealed 运行。

实际流程先完成 TRAIN/VALID 拟合与复用核对，冻结分析源码并锁定所有66项检查点/初始化、配置与计划身份，再写持久化 TEST 访问记录并评估。可追溯到 [analysis_plan.json](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[selections.lock.json](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) 与 [test_access.json](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。锁中的 test_loaded=false 是锁定时快照，后续访问由独立 test_access.json 记录。

[verification.json](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) 已记录：66条件、132文件全部样本 CUDA/AMP/batch64 独立检查点回放，重建原始支持与 TRAIN 统计、核验身份/初始化/采样顺序/VALID最小值，并独立核对1,496行物理指标及各汇总表。状态为数值PASS；[7组真实图 QA](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) 确认21份PNG300dpi/SVG/PDF导出、逐图PNG可见性与哈希。独立解读审阅与完成封存的状态分别以本轮 `independent_review.json`、`complete.json` 为准。

## 实测支持与机制边界

| 问题 | 本轮实际支持 | 仍未证明的机制或外推 |
| --- | --- | --- |
| CNN局部表示是否有用 | 两个AL条件的前端替换在两类RMSE、三个seed均劣于参照 | 卷积必为低通、专门去除白噪声，或在所有域都更好 |
| 时序后端是否有用 | AL上删除后端恶化；GRU/BiGRU可替代LSTM的部分表现 | LSTM状态等于物理零偏，或后端必须是LSTM |
| 哪种模型应被选择 | AL单域VALID选CNN-GRU；联合VALID选CNN-LSTM64 | 用TEST低值回选BiGRU/BiLSTM，或给联合模型按域各选一次 |
| 局部表示与历史聚合是否互补 | 两侧替换及删除控制提供固定配方下的描述性支持 | 纯算子因果效应、严格统计交互，或充分family HPO后的普遍冠军 |

正文实测表格直接由本轮 summary/paired_summary/validation_ranking/compute_summary 四张CSV提取，并核对其 SHA 与数值核验记录一致；图证据来自本轮真实QA记录。完整逐指标、逐设备对、逐录制与逐seed数据以[报告](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[逐seed配对](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)和[报告追溯记录](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)为准。

## 复现时使用新目录

本轮 `runs/cnn_lstm_module_replacement_20261107/` 用于保存本次实验的权威产物。重跑前先复制 YAML、修改 `experiment_id`，使用新的运行目录；训练、评估、成本测量、报告与核验均指向该新目录。不要把本轮或旧 sealed 目录作为重跑的输出位置，也不在其中重算或覆盖成本与报告。具体新目录示例见[实施计划](superpowers/plans/2026-11-07-cnn-lstm-module-replacement.md)。

## 原始论文

1. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation 9(8), 1735–1780. [DOI/原始出版物](https://doi.org/10.1162/neco.1997.9.8.1735)；[官方书目与作者摘要](https://pubmed.ncbi.nlm.nih.gov/9377276/)。用于门控记忆的结构动机，不用于证明本项目的物理机制。
2. Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP, 1724–1734. [ACL 原始论文页](https://aclanthology.org/D14-1179/)；[PDF](https://aclanthology.org/D14-1179.pdf)。第2.3节给出 reset/update 门。
3. Bai, S., Kolter, J. Z. & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. [作者原始预印本](https://arxiv.org/abs/1803.01271)。用于膨胀/残差卷积的比较背景；本文另外说明项目版本的非因果 padding 和单时间卷积瓶颈结构。
4. Vaswani, A. et al. (2017). *Attention Is All You Need*. [作者原始论文](https://arxiv.org/abs/1706.03762)。用于自注意力、未来 mask 与位置编码的概念来源，不移植原文翻译任务的性能结论。
