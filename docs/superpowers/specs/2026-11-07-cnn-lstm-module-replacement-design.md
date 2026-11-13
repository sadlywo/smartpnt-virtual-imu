# CNN-LSTM 前后模块替换实验

用户已明确要求分析 CNN-LSTM 优势，并执行前端与后端替换消融。本方案据此直接实现、在本机CUDA训练和汇报，不增加审批或提交步骤。既有已封存运行及其源码保持原样。

## 问题和设计

在相同 LOW 六轴→HIGH 六轴映射中，固定 CNN 替换后端，或固定单向 LSTM64 替换前端。两个训练范围仍为 ALiMU三LOW共享、ALiMU+Penn共享；TEST域分别报告AL专域/联合AL/联合Penn。320点/40Hz为唯一输入窗口，监督与评估均为相同中心40点。三优化种子20261010/11/12，固定历史AdamW、Huber、80轮上限、patience15等配方；不对子网络单独HPO，不按TEST选结构。

主参考为上一轮容量控制CNN-LSTM64（81,414参数），其在两训练范围的三种子平均VALID均值均最低。选择它是上一轮结果提出的后续研究，旧TEST已经被查看，不能称为全新独立确认。原数据与录制划分、TRAIN-only标准化、公共anchor和采样序列完全沿用。

## 处理矩阵

| 名称 | 前端 | 后端 | 来源 |
| --- | --- | --- | --- |
| cnn_lstm64 | CNN64×3/k5/BN/GELU | 单向LSTM64 | 复用旧cnn_lstm_wide/320 |
| cnn_bilstm32 | 相同CNN | BiLSTM32/方向 | 复用旧cnn_bilstm/320 |
| cnn_identity | 相同CNN | 无时序后端 | 复用旧cnn_only/320 |
| cnn_gru | 相同CNN | GRU72→Linear64 | 新训练 |
| cnn_bigru | 相同CNN | BiGRU40/方向→Linear64 | 新训练 |
| cnn_transformer | 相同CNN | 单层Transformer64/head4/FF128 | 新训练 |
| cnn_tcn | 相同CNN | 四个残差瓶颈TCN块/宽32/k3/dilation1,2,4,8 | 新训练 |
| linear_lstm64 | 逐点Linear6→64 | 保留相同LSTM64 | 新训练，删除前端局部时序基线 |
| mlp_lstm64 | 逐点MLP6→176→176→64，逐层BN/GELU | 保留相同LSTM64 | 新训练 |
| gru_lstm64 | GRU104→Linear64 | 保留相同LSTM64 | 新训练 |
| transformer_lstm64 | Linear6→64＋单层Transformer/head4/FF192 | 保留相同LSTM64 | 新训练 |

每个可替换模块输出64维，每个模型输出B×6×320；相同LayerNorm64/Linear64/GELU/Linear6读出头。无额外温度、器件ID、导航标签或未来目标输入。Transformer使用固定正弦位置编码，dropout0。TCN块使用64→32逐点投影、单层k3膨胀卷积、32→64投影和残差LayerNorm；其四层膨胀加前CNN的推理感受野为43点。逐点MLP用Linear实现，等价逐点运算，不把它称为时间CNN。

隐藏宽度在训练前按参数量选定，目标让七个新训练的非删除替代模型接近参考总参数量（约±5%，实际由模型清单核实）；历史BiLSTM32控制为73,222参数、比参考少10.06%，不属于这七个，减少把容量差异误作结构作用。线性前端及无后端是明确的低容量删除控制。结构、宽度、规范化、有效感受野和未来信息范围仍共同变化，因此这是固定预算配方下的模块替换，不是抽象算子的单因素因果实验。后端GRU/BiGRU的宽度明确报告为72/40，不能简称相同宽度对照。

固定初始化：以同seed旧CNN-LSTM64实例为参考，保留的模块和读出头逐张量完全一致；新模块使用独立确定性CPU RNG，不污染调用者CPU/CUDA/NumPy/Python随机流。所有训练使用相同逐seed逐epoch平衡抽样序列。

## 复用与新拟合

总矩阵11结构×2范围×3种子=66项，其中48项新拟合、18项旧控制复用。复用必须核验旧complete/verification/manifest/source/config/checkpoint/fit/init/history哈希，且新工厂与旧工厂的控制模型初始state/前向一致。复用项显式标记reused，保留旧检查点原生job和来源，禁止伪造新训练或覆盖旧目录。新运行可复制旧控制的best.pt、history和初始化供追踪；新的import.json/fit.json绑定来源哈希。旧VALID/TEST预测在新矩阵锁定之后才复制/评估，不能驱动新训练。

新模型训练结束前仅访问TRAIN/VALID。所有66项完成后固定检查点哈希锁，再统一导出VALID/TEST预测与物理分量/vector RMSE、MAE、bias、逐设备/逐录制指标。归一化分数与上一轮一致：每设备对(G/RawG+A/RawA)/2，再等设备对、等数据集。归一化VALID用于记录候选排序，TEST只作报告；两个训练范围分别排序，联合模型不能按两个TEST域分别选结构。

同seed配对相对CNN-LSTM64计算RMSE及归一化分数差值（正为变差）。报告三seed均值、样本SD和方向次数，包含参数、训练轮数/时长、同步batch1延迟。所有表保留reused字段，复用控制的旧训练成本明确属于旧运行；新的统一推理计时可重测。无需重做四窗搜索或增加输出融合，这两项上一轮已测。

## 科学解释和核验

CNN局部共享滤波/非线性与LSTM门控历史聚合是结构上合理的解释；只有消融损失支持模块有用，不能直接证明某层专门消除了偏置或随机噪声。本轮新增误差频段、残差均值/波动等诊断可作描述，不把它们当作隐藏状态的机制识别。优先汇报两侧替换、域间取舍和精度成本。

独立数值核验从同一原始数据/公共支持重建并回放全部132份预测，检查18复用/48新拟合、TRAIN-only统计、逐epoch采样、最小VALID检查点、初始化、配对摘要与源哈希；预先冻结分析规则和源码再读取本轮TEST。核心测试包括控制模型等价、保留参数一致、参数预算、局部/逐点感受野、实际梯度传播、无TEST训练、复用身份拒绝、未完成项禁止解锁和摘要重算。

图通过src/virtual_imu/viz生成，采用仓库风格，每组PNG300dpi/SVG/PDF，轴标物理单位并保留图注。实验权威目录runs/cnn_lstm_module_replacement_20261107；最终config/source/data/CP/predictions/metrics/report通过SHA和git commit联结。输出不进入Git，不改outputs/latest。
