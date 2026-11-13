# 共享 CNN-BiLSTM：窗口与模块消融

GitHub阅读：[最新结果汇总](results/2026-11-13/README.md) · [完整报告、CSV与图表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。本页工件链接指向Release，下载包内按原run名称查找；本地源路径仍用代码形式标识。


状态：138/138 正式拟合及 TEST 评估已完成，独立数值重算与结果解释审阅均通过（未解决P1/P2为0）；105 项回归测试通过，六组18张图表完成检查。结果来源为 `runs/cnn_bilstm_shared_ablation_20261106/`，不是 `outputs/latest`。最终封存以 `complete.json`、`verification.json` 与 `independent_review.json` 为准。

结果入口：[中文结论与选型](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[完整数值导读](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[逐项技术报告](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)、[所有指标CSV](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。

主要发现：ALiMU 上时间 CNN 与循环模块均有稳定贡献，但双向没有普遍优势。五个主结构中，ALiMU 专域的 VALID 最低为单向CNN-LSTM32/320，联合模型最低为完整CNN-BiLSTM/320；纳入三个补充控制后，单向CNN-LSTM64/320在两种训练范围中的VALID均值最低。整窗监督CNN-BiLSTM的TEST域平均最低，但Penn误差上升，且监督量更多。不能把完整双向模型称为所有场景下的最好模型。

完整模型按 VALID 选择的窗口为 ALiMU 专域160点、联合320点。固定320点去CNN后，ALiMU专域陀螺／加速度RMSE变差28.56%／28.01%，联合ALiMU变差40.47%／36.93%；Penn陀螺却改善27.35%。这些是固定训练配方、三种子下的域间取舍，不支持静态或未见器件的外推。

此前未找到严格的共享 CNN-BiLSTM 窗口／模块消融。历史 HPO 同时改变结构、窗口、batch、优化器超参和损失配方；July context/bias/drift 消融属于 TCN。既有证据索引保存在本运行 `preparation/prior_ablation_audit.md/json`。

## 实验范围

两种训练范围：ALiMU 三个 LOW 设备共享一个模型；ALiMU+PennCOSYVIO 按数据集平衡采样、共享一个模型。分别报告 ALiMU 专域、联合模型的 ALiMU 域和 Penn 域，不把它们算作三个独立数据集。

冻结六轴 LOW 输入与 TRAIN-only 标准化 HIGH−LOW 残差目标，无温度、器件ID或导航GT输入。ALiMU 输出参考为 VN100，Penn 为 ADIS16488，均是实测 IMU 参照。录制级互斥分组，ALiMU 静态 stat11 仅训练。TEST 只有 ALiMU dyn08/09 和 Penn bs，不能据此建立静态场景或未见器件结论。

主矩阵：5个变体×4个窗口×3个优化种子×2个训练范围=120次拟合；3个320点补充控制×3种子×2范围=18次，总计138次。

| 变体 | 变化 | 参数量 |
| --- | --- | ---: |
| cnn_bilstm | CNN64×3层/k5 + BiLSTM32/方向 + 64维读出头 | 73,222 |
| bilstm_only | 去时间CNN，保留逐点投影与BiLSTM | 30,214 |
| cnn_only | 去循环块，保留CNN与相同64维读出头 | 48,134 |
| mlp_only | 去两个时间模块，仅逐点投影与读出头 | 5,126 |
| cnn_lstm | 去反向LSTM，保留单向hidden32 | 58,566 |
| cnn_lstm_wide | 单向hidden64，同输出宽度，容量更大 | 81,414 |
| pointwise_bilstm | CNN的k5改k1，保留三层深度/BN/非线性 | 38,918 |
| cnn_bilstm_dense | 同完整模型，改为整窗监督 | 73,222 |

保留且同形状的参数、BN缓冲与读出头复制相同 seed 的完整模型初始化；新形状组件使用独立可复现初始化。模型工厂与归档原始类的初始权重及前向输出已逐项核对一致。删除模块的子网络不单独重新调参，回答的是固定共享配方下的模块作用。

## 公平支持与训练

窗口40/80/160/320点，40 Hz下标称1/2/4/8秒。所有主变体使用相同320点anchor、stride40，较短输入从中间对称裁剪；监督与主评估均为相同中心40点。训练标签、抽样权重、逐seed minibatch序列和评估时间点一致；主评估目标块不重叠。首尾采样跨度比标称时长少0.025秒。

AdamW、Huber、batch32、最多80轮、patience15、5轮warmup及cosine调度；学习率和weight decay沿用历史共享CNN-BiLSTM的VAL选定配置。使用pinn_imu和本机CUDA。各条件按完整共同VALID支持选择最优checkpoint，全部拟合及其身份核验完成后才加载TEST。

主支持设计在相同中心标签上比较上下文长度，因此绝对数值不应直接与历史整窗训练、重叠平均的结果混排。dense控制仅考察训练监督范围；额外固定检查点的多锚窗融合诊断不新增拟合、不重新选epoch，保留完整窗口输出用于独立重算。融合还扩大单点的输入并集（8个320点锚窗可达600点）并使用非中心输出，不能将差值全部归因为纯平均收益。单向LSTM仍保留对称CNN，不构成完全因果部署方案。纯CNN的推理感受野为13点，训练窗口变化还会影响BN统计与优化轨迹。

## 产物与复建

主结果：`summary.csv`、同窗口模块配对差值、对320点的窗口配对差值、CNN×循环块交互、仅VALID选窗、计算成本、逐设备与逐录制指标。三seed样本标准差只衡量优化随机性。所有误差同时保留component/vector RMSE；不将rad/s与m/s²直接相加。

```powershell
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_cnn_bilstm_ablation.py --mode all
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/report_cnn_bilstm_ablation.py --run-dir runs/cnn_bilstm_shared_ablation_20261106
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/evaluate_cnn_bilstm_overlap.py --run-dir runs/cnn_bilstm_shared_ablation_20261106
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/verify_cnn_bilstm_ablation.py --run-dir runs/cnn_bilstm_shared_ablation_20261106
```

配置见 `configs/ablations/cnn_bilstm_shared.yaml`。训练源码、配置、历史来源、分析规则、manifest、支持键、初始化、采样顺序、checkpoint与预测的哈希保留在运行目录。已完成拟合只在配置和checkpoint哈希一致时跳过；未完成项从头重新拟合。预运行默认写独立 `_preflight` 目录。

`*_guard_diagnostic` 为首次入口回归检查后保留的中断诊断，未访问TEST、不计入正式结果。`*_preparation` 内的合成报表检查和计时探查也不属于正式消融结果。本研究复用原TEST划分，是由已有结果提出的后续机制研究，未增加独立确认录制。
