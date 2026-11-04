# unified_imu_v2 正式训练协议

日期：2026-10-12（Asia/Shanghai）。用途：在启动长训练前固定任务、数据拆分、基线、指标与解释边界。

本文根据当前 YAML、训练器、推理器和真实数据检查结果编写。文中“当前配置”是已实现内容；“正式训练前待补齐”是拟议修改，尚未执行。本文未启动训练、未修改数据或现有训练配置、未更新此前不可变服务器包。

2026-10-23 执行状态：用户已同意本协议并提供服务器连接。后文所列配对验证、完整窗口合并、物理指标、训练集基线与多随机种子入口已实现；使用 `configs/train_formal_paired.yaml`、`configs/train_formal_navigation.yaml` 和 `configs/train_formal_matrix.yaml` 执行。旧配置和旧服务器包保留。服务器独立环境的 RTX 4090 CUDA 运算及反向传播已验证。实际运行状态以新 `runs/formal_training_*/pipeline.json`、各训练目录和最终报告为准，本文不替代运行记录。

## 1. 研究问题与优先级

主问题：在共同 40 Hz 处理带宽、固定参考器件和未见过的物理录制上，学习到的六轴补偿能否降低低成本 IMU 相对实测参考 IMU 的误差，并优于经典标定/时序基线？

主任务使用 ALiMU 与 PennCOSYVIO。SmartPNT-MSF 是单独的导航监督探索任务，不提供主任务的高等级六轴标签，也不是主任务模型的下游实测验证平台。直接将主任务模型用于 SmartPNT 尚缺参考类型对应关系和域外验证。

本版本不训练实测温漂补偿，不启用静态上下文或独立零偏辅助分支，不纳入 FusionPortable/NavINST。目标是数据支持范围内的信号误差改善；达到某个硬件精度等级需要额外验证，不能由训练 loss 或参考器件名称推定。

## 2. 数据冻结与独立实验单位

正式目录：`data/processed/unified_imu_v2/`。944,840 行、145 个连续片段。manifest SHA256：`5e7f198a57c9370734c4ecce092aa11f797530dded935957324c725c4c752260`。

| 数据 | Train | Validation | Test | Label |
| --- | --- | --- | --- | --- |
| ALiMU | dyn_01/02/04/05/06、stat_11 | dyn_07 | dyn_08/09 | VN100 六轴 |
| PennCOSYVIO | af、as | bf | bs | ADIS16488 六轴 |
| SmartPNT-MSF | Traj3 | Traj4 | Traj1 | Fixed 筛选后的导航真值 |

ALiMU 每次录制对应的三种低成本器件共享参考信号，必须始终同属一个 split。已排除六个较短输入复用变体；SmartPNT Traj2 与 Traj1 完全重复，亦已排除，原文件保留。旧 v1 禁止用于正式训练/导出。

物理录制/轨迹是独立实验单位。窗口、数据片段和同一次录制的多个器件视图不是独立重复实验。三个随机种子衡量优化随机性，不能代替更多独立录制。

| 任务/器件 | Train 窗口 | Validation 窗口 | Test 窗口 |
| --- | ---: | ---: | ---: |
| BMX160 → VN100 | 4,071 | 373 | 746 |
| MPU9250 → VN100 | 4,071 | 373 | 746 |
| LSM9DS1 → VN100 | 4,071 | 373 | 746 |
| Tango Top IMU → ADIS16488 | 245 | 98 | 163 |
| 配对合计 | 12,458 | 1,217 | 2,401 |
| SmartPNT 导航 | 2,016 | 3,637 | 1,649 |

计数使用 80 点窗口、40 点 stride；不跨片段、缺口或 split。训练窗口重叠，但 split 按源录制隔离。全部窗口的精确跨 split 重用检查已经通过。最终连续信号指标须将重叠窗口合并为每个有效时间点一份预测，不能将重复窗口点当作额外独立样本。

## 3. 配对任务的模型和监督

输入：80 点低成本六轴时序，加参考器件类别编码，共 `9 × 80`。角速度为 rad/s，加速度为 m/s²。`reference_vn100`、`reference_adis16488` 是已知参考类型；`reference_stim300` 当前为零，不代表已经训练 STIM300 映射。低成本器件型号用于数据身份与评估分组，当前没有单独的型号编码输入。

监督残差：`r(t) = high(t) - low(t)`，输出 `6 × 80`。网络输出反归一化后得到 `virtual(t) = low(t) + predicted_r(t)`。参考测量仅用于监督/评估，推理只需要低成本信号及预先确定的参考类型。

模型为因果 TCN：64 隐藏通道、4 个残差扩张卷积块、kernel=5、dropout=0.05。采用基线结构便于解释结果；本轮不同时增加温度、静态分支、器件规格或更复杂架构。

两层卷积/块、dilation=1/2/4/8 的理论感受野为 121 点；当前输入仅 80 点，窗口外由因果零填充处理。窗口开始处的历史上下文较少，应检查端点误差。因果结构不自动证明整条预处理/分窗/推理链实时；总等待、缓存策略和时延需要单独测量。

归一化只在 train 拟合。实际残差目标的归一化由训练器计算并保存，不能直接拿数据目录里 HIGH 的均值/方差替代残差统计。

当前 sampler 按数据集等预期概率抽样：ALiMU/Penn 各约 50%，每 epoch 总抽样量 12,458 窗口，允许重复。Penn 只有 245 个独有训练窗口，因此每 epoch 预期被多次重采样；它增加曝光次数，不增加数据量或独立性。需单独监控 Penn 验证过拟合。

## 4. 配对训练 loss

实际在标准化的残差空间计算：

`L = L_signal + 0.02 L_difference + 0.02 L_spectrum + 0.005 L_integral`

| 项 | 当前定义 | 目的 |
| --- | --- | --- |
| Signal | `(2 × Huber(gyro) + Huber(accel))/3` | 学习参考与低成本的逐点差值，陀螺仪相对加权更高 |
| Difference | 一阶时间差分的 Huber，陀螺仪/加速度等权平均 | 约束误差补偿的时间变化 |
| Spectrum | 每个窗口内 log(1+FFT 幅值) 差异 | 约束补偿信号的频谱形状 |
| Integral | `cumsum(residual) × dt` 的 Huber，dt=0.025 s | 约束窗口内累积补偿误差 |

这里只是窗口差分/FFT/累积和约束。Integral 不是完整 SO(3) 预积分或导航滤波器；归一化空间的 loss 也不能直接解释为 rad、m/s 或米。Allan 权重为 0，独立 bias 辅助监督未开启。

## 5. 优化设置

| 设置 | 配对 | SmartPNT 导航 |
| --- | --- | --- |
| Optimizer | AdamW | AdamW |
| 初始学习率 | 0.001 | 0.001 |
| Weight decay | 0.00001 | 0.00001 |
| Batch size | 128 | 128 |
| 最多 epoch | 60 | 60 |
| Gradient clipping | norm=1.0 | norm=1.0 |
| Early stopping | 10 个 epoch 验证指标无改善 | 10 个 epoch 验证指标无改善 |
| Scheduler | ReduceLROnPlateau，patience=5，factor=0.5，min_lr=0.00005 | 当前未配置 scheduler |
| 初始 seed | 20261010 | 20261010 |

数据加载器不丢尾批，配对每 epoch 约 98 个 optimizer step，导航约 16 个。不能据此给出尚未测量的服务器训练时长。

## 6. 模型选择：当前行为与拟议正式标准

当前配对 checkpoint 监控 `val_score`：gyro 三维 RMSE + accel 三维 RMSE + 0.3 × gyro spectral + 0.2 × (1-coherence) + 0.2 × cumulative integration proxy。

当前实现把不同器件、不同记录和重叠窗口展开后拼接，再计算 FFT/coherence/cumsum；这些拼接边界不是真实连续信号。它还混合 rad/s、m/s² 和其他量的尺度。因此旧综合分数不得作为跨数据集的科学评价或正式比较标准。

拟议正式主选择分数为相对未校正信号的无量纲误差比。在每个器件配对 k 上分别计算：

`q_g(k) = RMSE_gyro_virtual(k) / RMSE_gyro_identity(k)`

`q_a(k) = RMSE_accel_virtual(k) / RMSE_accel_identity(k)`

`s(k) = (q_g(k) + q_a(k))/2`

先平均 ALiMU 的三个器件分数，再与 Penn 分数等权平均：

`S = (mean(s_AL_BMX160, s_AL_MPU9250, s_AL_LSM9DS1) + s_Penn)/2`

S 越低越好，identity 的 S=1。这样 ALiMU 的更多样本与三个器件视图不会压过 Penn，物理单位也不会直接相加。若某个 identity 误差近零导致比值不稳定，使用预声明的数值下限并标记该组，不把微小分母造成的极端比值解释为有效差异。

以上 S 尚未接入 trainer。正式训练前应补齐分组指标，令 checkpoint、配对 scheduler 与早停统一监控 S；窗口诊断按真实片段独立计算并重置累计量。最终报告仍须保留各组指标，单个 S 改善不能掩盖某个器件退化。

测试集用于固定方案的最终评估，不用于调整校正时间差、损失权重、基线参数、epoch 或候选模型。基线参数/候选超参数只由 train/validation 决定。

## 7. 主任务测试指标与基线

主指标为矢量误差 RMSE：`sqrt(mean(||prediction - reference||²))`，陀螺仪和加速度分别计算。不要与 `norm(prediction)-norm(reference)` 的标量误差混淆：后者不能可靠反映方向错误。

| 指标 | 单位/解释 | 优先级 |
| --- | --- | --- |
| Gyro 三维矢量 RMSE | rad/s，另提供 deg/s | 主指标 |
| Accel 三维矢量 RMSE | m/s² | 主指标 |
| 六轴逐轴 RMSE、MAE、平均误差 | 保留各自物理单位，发现单轴退化/零偏残余 | 必须报告 |
| 相对 identity / 经典基线的改善率 | `100 × (1 - RMSE_model/RMSE_baseline)` | 必须报告，负值代表退化 |
| 误差范数 P95、峰值段误差 | 物理单位，检查少量大误差 | 补充，需新增汇总 |
| 连续片段内 PSD、coherence、phase/lag | 检查过度平滑、动态响应和额外时差 | 辅助；静止低激励时标记不可用 |
| 2 s 窗口累积角增量/速度增量残余 | 分开报告 rad / m/s，窗口起点重置 | 辅助；不是完整导航精度 |
| 吞吐、推理时间、显存及已知处理等待 | 指明测量硬件和处理方式 | 工程指标 |

对每个独立连续测试片段生成 LOW/HIGH/VIRTUAL 同步输出，保留器件与数据身份。先逐序列计算，再逐器件和逐数据集汇总；频谱、Allan 和积分不能跨片段拼接，重叠窗口输出不能重复计数。滞后估计仅用于诊断，不按 test 拟合后移信号来改善主指标。

比较基线至少包括：

1. Identity：同一份已规整 40 Hz LOW，零校正。
2. Mean bias / affine / ridge affine：只用 train 拟合，验证选定参数。
3. 因果 learned FIR（初始 taps=9，ridge alpha=0.01），必要时 ARX：只用 train 拟合，逐记录重置历史。

基线按各器件配对拟合，避免不同器件坐标和目标参考混合。所有方法共用相同输入、测试时间点和标签。非因果平滑/频率变换方法可以另列离线参考，但不得混作实时基线。

主要评价表每行一种映射：BMX160→VN100、MPU9250→VN100、LSM9DS1→VN100、Tango→ADIS16488。分别列 gyro/accel 和逐轴结果，不能将 rad/s 与 m/s² 合成一个“六轴精度”作为结论。若仅优于 identity 但不能优于 affine/FIR，学习模型的额外价值尚未证明。

ALiMU stat_11 仅在 train；本版没有独立静态 test。其 Allan/零偏改善只能作为训练域诊断，不作为独立泛化结论。40 Hz 处理数据不能证明原始高带宽噪声性能或硬件温漂等级。

现有通用 `navigation_metrics` 是简化的无姿态变换积分代理；不能用它宣称真实 ATE、RPE 或完整惯导漂移。完整下游导航须另建立姿态/重力/初始状态一致的评估流程。

## 8. SmartPNT 导航探索任务

输入为原生解析后直接抗混叠处理的六轴 IMU，`6 × 80`。TCN 取窗口最后时刻的三个输出，监督标签为 `p_ENU(end)-p_ENU(start)`。端点间隔是 1.975 s（80 点的名义采样跨度约 2 s）。

当前 loss 是标准化三维位移的 Huber；速度/航向积分约束为 0。checkpoint 根据反归一化 `val_rmse` 选择，单位米，为三个分量的总体 RMSE。`val_vector_rmse` 和 `val_position_rmse_m` 是三维矢量 RMSE，等于前者的 sqrt(3) 倍；对同一三维标签，两种定义的排名一致。

报告 E/N/U 逐轴 RMSE、三维位移 RMSE、水平/垂直误差、误差中位数/P95（后两类须补齐输出）。比较零位移、仅训练标签的常数均值预测和同输入窗口的简单回归基线。恒速预测需要初始速度，若引入必须单列“额外状态输入”实验；测试 GT 初速/航向不能悄悄作为模型输入。

任务边界：局部 IMU 坐标到固定 ENU 的映射依赖初始方向，位移也包含窗口初速度。当前仅六轴的短窗口输入没有显式提供这些状态。初速度不同但加速度相同、水平整体旋转但机体系运动相同，都可能产生相似 IMU 窗口和不同 ENU 位移。因此本配置是学习该数据运动先验的探索基线，不能宣称任意运动下唯一恢复全局位移。

RoNIN 的坐标规范化方案明确将 IMU 与运动标签放入一致、重力对齐的坐标表示，可用于后续设计参考：[原论文，第 4.1 节](https://arxiv.org/pdf/1905.12853)。这是后续方案参考，当前代码未复现 RoNIN，也没有将 GT 姿态引入推理。

后续如需要可靠完整轨迹，需明确可用初始状态/姿态估计、输入输出坐标转换和安装关系，或另设计与初始状态相容的监督。仅将位移 label 改为航向相对坐标不能自动消除初速度问题。

当前 80 点窗口 stride=40，有重叠且端点间隔 79 个采样周期；不能直接累加全部窗口位移成轨迹。完整 ATE/RPE 需额外定义不重复覆盖的预测区间、时间对齐、初始姿态、连接方法和固定对齐规则，并在 gap 处重置。本轮以窗口位移误差为准。

## 9. 执行顺序和复现实验

1. 补齐正式指标/评估联动，使用小数据回归测试验证分组、归一化、片段边界和 metadata。
2. 新建训练协议对应配置与服务器包，记录与原包不同的源码指纹；formal v2 数据保持不变。
3. 服务器校验归档/hash/frozen 依赖；GPU 实际计算与真实 batch 反向传播；运行短 epoch 验证 loss、checkpoint、预测评估链。
4. 拟合经典基线，记录 train/validation 指标；运行 seed=20261010 的配对主实验，最多 60 epoch、早停选最佳 checkpoint。SmartPNT 独立运行，不联训。
5. 固定候选方案后使用 seed=20261011、20261012 重复，分别保留 run；这些是优化随机重复，不是额外轨迹。固定配置完成 test 评价，每个 run 不再用 test 调参。报告三次结果与 mean±std。
6. 后续可比较相同 TCN 的“只信号 loss”与“当前组合 loss”、AL/Penn 单数据集与联合模型。实验矩阵和验证选择规则应先写 YAML；一旦观察正式 test，不基于 test 反馈继续反复挑方案。

主判定：各器件 gyro/accel 误差相对 identity 的变化，联合模型对经典基线的价值，是否存在稳定退化、动态失真或额外时差。首次训练不预先承诺改善百分比，不以某一个 seed 的最佳值替代重复结果。

## 10. 正式训练前待补齐清单

- 接入第 6 节的分组、无量纲 checkpoint 分数，替换当前跨记录拼接的验证诊断；保持旧结果可追溯。
- 配对 trainer 当前训练结束后不自动执行 test；补齐最佳模型逐片段推理和固定 test 报告。
- 推理输出目前未保留 `dataset_id` / `high_sensor` 等全部分组 metadata，现有 `paired_device_metrics` 会因此缺组；保留身份并验证报告表实际生成。
- 统一连续片段内频谱/lag/积分评价，避免旧通用评估器对多序列整体计算；Allan 仅用于满足条件的连续静态段。
- 导航逐轴/水平垂直/尾部误差以及 seed 汇总，需要新增输出；不把已有 summary 的缺项默认为已实现。
- 如果代码/配置改变，重建服务器源码包；2026-10-10 的旧包完整性已验证，但不包含本文拟议的指标补齐。

## 11. 已验证证据与产物

此前本地 readiness 使用真实数据进行了每任务两步 CUDA 更新，有限 loss/梯度、参数更新、标签形状和数据哈希检查通过：`runs/unified_training_readiness_20261010_141709_308068/`。本地 Python 3.9 / Torch 2.3.1 CUDA 环境不是将来 Linux frozen Python 3.12 环境的验证替代。

现有配置：`configs/train_unified_paired.yaml`、`configs/train_unified_navigation.yaml`。数据说明：`docs/UNIFIED_IMU_DATASET.md`。服务器说明：`docs/SERVER_TRAINING.md`。

每个新训练 run 应保存：实际 YAML、seed、数据和 split manifests、数据 build_run/hash、源码提交与 dirty 指纹/来源包、环境/GPU、每 epoch 指标、best.ckpt、训练曲线、逐片段预测、器件和序列测试表、基线参数及报告。使用 `runs/<experiment_id>/` 为唯一来源定位，不能从 `outputs/latest/` 猜来源。

训练图通过 `.agents/skills/imu-research-visualization` 或 `src/virtual_imu/viz` 生成。当前 checkpoint 只保存最佳模型，没有优化器/scheduler/RNG 完整状态；tmux 能保留断开 SSH 后的进程，进程终止后仍需新 run 重跑，不能宣称无损续训。
