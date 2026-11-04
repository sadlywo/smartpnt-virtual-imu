# 扩展 IMU 模型对照协议（2026-10-24）

## 研究问题和比较边界

现有 TCN 在四种器件八个 gyro/accel 测试 RMSE 中改善七个，但全部八项均不及固定 FIR/ARX。扩展研究检验不同序列骨干在同一训练和标签协议下的表现。由于原测试结果已经查看，本轮属于回顾性的扩展架构比较，不能写成一个全新未见测试集上的预注册 SOTA 竞赛。新增候选及预算在启动前由 `configs/train_benchmark_matrix.yaml` 固定；不按测试分数增删候选或调参。

“论文骨干适配”与“原方法完整复现”分别陈述。这里不会重现论文的数据、采样率、损失、姿态/速度标签、滤波器、协方差或全部训练细节，因此最终表格不能冒称所有方法的原论文 SOTA 性能。比较结果只适用于当前数据和本协议。

## 冻结的任务

- **配对信号任务**：80 点、40 Hz（首末点间隔 1.975 秒）；六轴 LOW 加参考器件类别的三维 one-hot，合计 9 通道，预测六轴标准化残差。反归一化后加到 LOW 得到 HIGH 预测。ALiMU 的 BMX160/MPU9250/LSM9DS1 分别映射 VN100，PennCOSVIO 的 Tango Top 映射 ADIS16488。无真实温度特征，无独立静态分支。
- **SmartPNT 导航任务**：80 点六轴输入，预测窗口首末真实 ENU 位置差 3 通道。继续使用固定的训练归一化及 last 输出接口。该数据没有 HIGH 六轴标签；不累计重叠位移，不计算或宣称完整轨迹 ATE/RPE。

录制/轨迹划分、归档去重和信号对齐全部沿用已验证的 unified_imu_v2 数据，manifest SHA256 为 `5e7f198a57c9370734c4ecce092aa11f797530dded935957324c725c4c752260`。Traj2 仍是 Traj1 的归档副本，FusionPortable 仍未纳入。

## 固定候选和可复现程度

| 候选 | 配对 / 导航 | 可复现程度与适配 | 配对逐点是否因果 |
| --- | --- | --- | --- |
| 原 TCN | 是 / 是 | 复用原固定训练的六次运行；核对配置、数据和模型，不重新抽取最好种子 | 是 |
| Temporal CNN | 是 / 是 | 通用四层卷积，64 通道、k7；不以 Calib-Net 的完整方法命名 | 否 |
| GRU | 是 / 是 | 通用单向两层 GRU，hidden 64 | 是 |
| Transformer encoder | 是 / 是 | 通用两层注意力，d_model 64、4 heads、FFN 2048；无因果 mask | 否 |
| CNN–BiLSTM | 是 / 是 | 通用 CNN 加双向 LSTM 对照；不是 ALiMU 原方法的完整复现 | 否 |
| DeepGyro backbone adapter | 是 / 是 | 核对公开四层空洞卷积骨干；调整输入/输出头和共同损失，未实现原姿态积分训练与可学习 gyro 矩阵 | 是 |
| TLIO ResNet18 backbone adapter | 是 / 是 | 核对完整四阶段 ResNet18 骨干；改为稠密时间输出头，无原方法的不确定度头和融合滤波器 | 否 |
| TartanIMU single-window adapter | 否 / 是 | 核对公开 ResNet-LSTM 骨干；单窗口、单头、无协方差；LSTM 跨窗口序列长度为 1 | 不适用 |

DeepGyro 原骨干感受野为 511，而这里窗口只有 80 点，其左侧较多历史由边界样本重复填充；这是明确的任务适配限制。TLIO 的时间下采样特征以最近邻映射到 80 点，稠密输出能力不等同于原生逐点高频骨干。TartanIMU 原本利用多个窗口和多平台头；本轮单窗口适配不验证其原生长期/跨平台能力。共同 last 接口下，CNN–BiLSTM 的反向分支在最后时间点只具有该点的卷积上下文，不能当作原论文导航头。

配对组：7 个神经方法（包括原 TCN）加 6 个经典基线，共 13 个方法。导航组：8 个神经方法加 zero/train-mean/ridge，共 11 个方法。新增 39 次训练，原 TCN 复用 6 次，总计 45 次神经运行。没有强行将不输出六轴序列的 TartanIMU 单窗口位移头放进配对信号任务。

CTIN 作者仓库没有公开可运行模型；Calib-Net 没找到可核查的官方网络代码，现有四层 CNN 与论文拓扑不一致；ALiMU 官方仓库提供数据但未核实模型代码；AirIMU/AirIO 的原方法涉及当前共同任务未复现的动力学/不确定度和融合过程。因此本轮不使用这些名称虚报原方法成绩。现有 FSDA 命名模块没有实现域适配目标，而且 reconstruction 分支在共同损失下无梯度，也不作为 FSDA 论文复现纳入。未修改的 U-Net 线性插值 CUDA 反向与严格确定性不兼容，因此不纳入本轮。

## 相同的训练预算与选择规则

三种子固定为 20261010、20261011、20261012。每次最多 60 轮，验证连续 10 轮不改善则早停；AdamW、lr 0.001、weight decay 1e-5、batch size 128，与原任务 YAML 相同。配对使用原标准化 Huber、2:1 gyro/accel 权重、差分 0.02、频域 0.02、积分代理 0.005，以及原 ReduceLROnPlateau；导航使用原标准化 Huber，不额外添加不同的方法损失。配对依据验证集分组相对 RMSE 保存模型，导航依据验证分量 RMSE 保存模型。两任务分别训练，不混合六轴和位移标签。

PyTorch 2.4.0+cu121 / Python 3.12 / NumPy 1.26.4，严格确定性；关闭 TF32，CUBLAS 工作区固定，Transformer 使用数学 SDPA 后端。DeepGyro 避免 CUDA 反向不确定的 ReplicationPad 算子，采用等价的边界展开与拼接；GPU 上实际验证梯度后才启动。CPU 和 GPU 的依赖及源码快照进入独立部署记录。

种子控制每次随机过程，但模型初始化消耗随机数的数量不同，导航 shuffle 不保证跨架构逐批完全一致；不得声称仅靠相同种子就实现了相同 minibatch 顺序。报告均值及样本 SD（ddof=1），它们只表示优化随机性，不是独立录制的置信区间。

## 评价和交付

同一物理测试点上的 gyro rad/s、accel m/s² 分开报告 vector RMSE、MAE、均值残差/偏差及相对 identity 和强经典基线的变化，另提供分轴、器件和连续片段结果。不要将不同单位直接相加。导航提供三维、水平、垂直位移 RMSE/MAE 与基线；这些是 1.975 秒窗口指标。

完整表以候选、任务、种子为身份，附参数量、上下文/因果性、最优 epoch、早停、配置、数据/source/checkpoint/预测指纹。经典基线只归纳一份，重复输出必须数值一致，不能把“每个网络重复拟合的同一个经典方法”算作许多独立重复。保留失败、退化和未完成的运行；完整表只在全部预期项有有效 artifact 后标为完成。

科研图沿用 `nature-figure` 与 `src/virtual_imu/viz` 的 Python 样式，输出可编辑 SVG、嵌入字体的 PDF、PNG 300 dpi、TIFF 600 dpi、源数据及图注。区分因果方法和窗口方法；展示每种子而非只展示测试集最好的种子。原固定候选的 Nature 图和结果保存在独立 `runs/nature_baseline_analysis_20261024/`。

## 论文和实现来源

- [DeepGyro / Brossard et al., RA-L 2020](https://doi.org/10.1109/LRA.2020.3003256)；[作者实现，固定提交 9948b664](https://github.com/mbrossar/denoise-imu-gyro/tree/9948b6646882c547d5ba6eda19a95ea4cba5e89e)，MIT（文件名 LICENCE）。
- [TLIO, RA-L 2020](https://doi.org/10.1109/LRA.2020.3007421)；[作者实现，固定提交 d7051c4f](https://github.com/CathIAS/TLIO/tree/d7051c4ff14834b93931984b75b40f894e39654d)，BSD-3-Clause。
- [TartanIMU, CVPR 2025 作者仓库，固定提交 8a802500](https://github.com/superxslam/TartanIMU/tree/8a8025000f85d2c6b0692b88e15c2c91e761f7d7)，Apache-2.0。
- [ALiMU（作者仓库标为 2025；Crossref 卷期为 2026-01，未独立确认 online 日期）](https://doi.org/10.1016/j.measurement.2025.118736)；[作者数据仓库](https://github.com/cmsahu97/Low_Grade_High_Grade_IMU_dataset)。
- [IMU Calib-Net（2022-01-03）](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.772583/full)。
- [CTIN 作者仓库和未公开代码的说明](https://github.com/bingrao/ctin)。

## 运行命令

```bash
printf 'READS: frozen bundle, v2 data and configs\nWRITES: new benchmark run only\nSKIPS: old formal results and test-based tuning\n'
bash scripts/server/run.sh benchmark --config configs/train_benchmark_matrix.yaml
```

原有运行、下载包和最终报告保持各自身份。新结果写入 `runs/benchmark_training_<UTC timestamp>/`；每次启动关联 `server_launch`、新源码完整 bundle 清单与原数据 manifest。原六次 TCN 只作为经核验的外部来源引用，其旧源码身份不能改写成新源码。
