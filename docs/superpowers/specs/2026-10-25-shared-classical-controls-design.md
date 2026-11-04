# 共享经典对照的补充实验设计

日期：2026-10-25。状态：设计已复核，按当前对话已授权的完整公平比较目标进入实施；尚未部署或运行共享对照。

原当前 HPO、训练源码/配置、core、aux1、aux2及冻结 bundle 均保持原身份。实现只增加独立CPU共享对照和其交付工具；不新增 GPU 候选。新共享状态全部冻结前不解码新的 TEST 数据/预测/指标载荷。当前神经搜索按原合同继续。设计和实施记录留在工作区，运行产物不提交Git。

## 1. 问题与能够回答的问题

当前复核神经模型每个协议只有一个统一映射，数值输入为六个 LOW 通道，无器件/参考类型 one-hot、温度或静态分支。原 in-domain 经典控制按 `(dataset_id, low_sensor, high_sensor)` 分别拟合、分别在 VAL 选参数，并在预测时用已知 identity 选择状态。这是一组 **known-device bank**：显式设备身份路由和多套系数是额外条件。它仍然有工程意义，但不宜作为统一六通道神经映射的唯一主对照。

增加 **protocol-shared classical**：每个 paired 协议只保留一套跨该协议 TRAIN 器件的状态，每个经典方法只有一组系数/超参数，预测只使用六通道 LOW 及其同流历史。协议本身决定模型/状态，和神经协议相同；不能在协议内部按器件、参考类型或录制选择系数。

主表比较同 TRAIN 物理拆分、六通道数值输入、无器件 ID 选状态的神经与共享经典；原 bank 另组完整保留。这个对照检查共享状态/身份路由条件的影响，不宣称不同算法的训练目标、窗口曝光次数、参数容量或未来上下文完全相同，也不保证任何方向的性能差异。

历史部分测试已看过。本补充设计在本轮新 TEST 全量读取前冻结，定位仍是 retrospective common-task controls，不能改称全球预注册、全球 SOTA 或完整原论文系统复现。

## 2. 备选方案与选择

| 方案 | 价值与代价 | 选择 |
| --- | --- | --- |
| A. 只把原 bank 改名，继续作为唯一经典主对照 | 不需新拟合，但已知身份路由、多状态与统一映射的条件差异仍存在 | 不采纳；bank 保留为另组 |
| B. 补充每协议共享经典，保留神经训练与旧 bank | 复用现有数学方法/网格，隔离共享状态条件；只新增 CPU 后处理和来源审计 | 采纳 |
| C. 给神经加器件 ID，或逐器件重训神经以匹配 bank | 可回答另一个条件化问题，但改变固定候选、信息输入和 GPU 预算，要求新的训练/选择协议 | 不纳入本轮 |

“一组全协议参数”明确为：**每个 protocol 内，为每种 tuned 方法仅用本 protocol 的 VAL 选一组参数，供全部器件对共同使用**。六个协议独立选择；不能把它们的 VAL 混合选一组跨协议参数。Penn 外层角色轮换时，一个协议的 VAL 可能是另一个协议的 TEST，混合会产生跨 fold 泄漏。

## 3. 范围、身份与计数

只新增六个 paired 协议：`pooled`、`alimu_only`、`penn_only`、`penn_loso_af`、`penn_loso_as`、`penn_loso_bf`。三种 SmartPNT navigation rotation 的模型、基线、标签和指标全部沿用旧结果，不因本补充更改。

每协议的逻辑方法固定为：

| 方法 | 参数状态/固定规则 | 选择范围 |
| --- | --- | --- |
| identity | 原 LOW；零拟合、零学习系数，复用原 canonical identity 一次 | 不选择 |
| bias_mean | TRAIN 六轴 `mean(HIGH-LOW)` | 不选择 |
| affine | 六入六出全矩阵加截距的 OLS，不是逐轴 affine | 不选择 |
| ridge_affine | 六入六出 ridge，alpha=0.01，截距不惩罚 | 固定 |
| learned_fir_causal | LOW 因果 taps=9，alpha=0.01 | 固定 |
| arx_linear | LOW taps=12 加当前一阶差分，alpha=0.01 | 固定 |
| ridge_tuned | 共享六入六出 ridge | 本协议 VAL 选 alpha |
| fir_tuned | 共享因果 LOW lag ridge | 本协议 VAL 选 taps、alpha |
| arx_tuned | 共享 LOW lag + 一阶差分 ridge | 本协议 VAL 选 taps、alpha |

`arx_linear/arx_tuned` 是本代码的历史命名：特征为 LOW 历史与 LOW 差分，不使用先前 HIGH/预测 HIGH 的反馈，不存在推理时 HIGH teacher forcing。图表显示名使用 **ARX-style LOW lag/delta**，canonical method id 不改；这不是声称具有真实输出自回归项的完整ARX系统。

审计计数是 **6×9=54 logical controls**，其中 **6×8=48 learned-state slots**、6 个 identity reference slots。identity 不产生新训练或新的独立重复。48 不是数值求解次数：每协议有5种固定 learned 方法，加7个 ridge、49个 FIR、35个 ARX tuned 候选，未缓存时为96个候选/固定计算项；所有真实求解、缓存命中、失败和选中状态另列，不伪装为48次训练运行。

预先固定的 tuned 网格与冻结 `configs/reviewer_hpo.yaml:145-150` 一致：

- alpha：`[1e-6, 1e-4, 0.01, 0.1, 1, 10, 100]`。
- FIR taps：`[1, 3, 5, 9, 17, 33, 65]`。
- ARX taps：`[3, 6, 12, 24, 48]`。
- 遍历顺序按上列；严格更小分数才替换最优。相等时保留原顺序第一项，不能看 TEST 破 tie，也不事后改 epsilon/tolerance 产生新的选择。

所有经典行的训练重复数为1，SD留空。重复映射到60个神经 family/variant/seed 槽位不能使经典 n 变成3、60或更多。48 learned slots 可通过经验证的别名引用既有等价数值状态；逻辑数、实际计算数、唯一系数状态数和缓存别名数分列。

## 4. 源码核对与可复用边界

本次只读核对了以下实现，后续 aux3 必须记录其 SHA 并与冻结来源核对，不修改这些来源：

| 来源 | 已知事实/复用方式 |
| --- | --- |
| `scripts/evaluate_reviewer_transfer.py:168-268` | `_macro_score`、`_select_shared`、`fit_source_classical`、`predict_source_classical` 已有共享拟合/预测数学；作为优先复用的纯函数 |
| 同文件 `_load_direction`、`_classical_record` | 按跨域方向组织，且会读取目标 TEST；不作为 aux3 的装载/队列入口 |
| `src/virtual_imu/eval/formal_evaluation.py:169-200` | fixed bias、OLS、ridge 与 fixed system-ID；仅 helper copy 的 IDENTITY 常量化以满足单 identity 断言 |
| `src/virtual_imu/eval/baselines.py:398-478` | `_lag_features/_sequence_lag_features` 只按 sequence_id 分组、按时间排序；不验证物理身份或 gap，缺 sequence_id 时会退化为整表历史，因此 wrapper 必须先验证 |
| `src/virtual_imu/training/paired_validation.py:20-223` | NN `val_grouped_score` 的覆盖、pair/dataset 宏平均与 near-zero 分母规则 |
| `src/virtual_imu/eval/reviewer_evaluation.py:289-340` | 原 in-domain bank 是逐 pair fit/选参数，不能改名冒充 shared |
| 同文件 `select_ridge_train_validation` | 直接用于全 pooled 数组时先 pooled RMSE，缺 dataset/pair 宏平均，不可作为 aux3 共享选择器 |
| `src/virtual_imu/eval/downstream_attitude.py` | 复用原 SO(3)/strapdown 传播与初始化，不重写积分器 |
| `scripts/benchmark_reviewer_classical_latency.py` | 复用/薄适配经验证的 float64 数学 kernel、同段预载 history 和计数规则；新合同、新目录，原计时不改 |

数学复用不等于复用原流程的 TEST 读取或来源判断。aux3 负责新增严格装载、缓存、选择冻结、覆盖/别名审计。拟合 API 只接收 TRAIN/VAL，不接收 TEST 数据、TEST 路径或 TEST 指标。

## 5. 物理流边界：合并器件 ID 之前必须成立的合同

这是实施的首要正确性门槛。

1. 从冻结 split/strict manifest 和 canonical frame 关联源文件 SHA、dataset、原 LOW/HIGH、pair、原 recording、segment/原 sequence_id。保留 `source_*` 审计信息和原始列，不从数值近似、文件 basename 或字符串猜测录制身份。
2. 每个原 `sequence_id` 必须唯一映射至一个 `(dataset, original low, original high, original recording, original segment)` 物理流。AL 三种 LOW 共用的 `physical_group_id` 是录制统计分组，不能单独用作 history stream key。
3. 原 sequence_id 非空、原身份一致、时间戳有限且唯一，段内严格40Hz；采用既有容差 `atol=1e-6, rtol=1e-5` 检查 `diff(t)=0.025`。缺 identity/sequence/时间、同 seq 对应多个 sensor/record/segment、重复时间或 gap 均失败，不能自动重命名、静默拼段或自动补值掩盖坏 schema。
4. 保留原 sequence_id 值。只在 `fit_pair_baselines` 所需的独立 TRAIN helper copy 中把 `IDENTITY=[dataset_id,low_sensor,high_sensor]` 设为同一 shared 标记；**绝不把 sequence_id 或 recording/segment 常量化**。原 frame 与 VAL 的 identity 保持不变。
5. TRAIN/VAL/TEST 各自在自己的原流内建 lag/delta；不能从上一器件、上一录制、上一 segment 或另一 split 借历史。流首沿用原 edge padding，首差分为0。连续推理中不能每个窗口重置；计时预载窗口前的同段 history，不允许用其他窗口或更早已断段数据。
6. 保存完整 stream registry、逐流时间范围/行数、源文件/role、边界检查结果、物理行 SHA。DATASET/LOW/HIGH/record/segment 只用于装载验证、断流和指标分组，不用于预测时选择系数。

不引入自动修复是本设计选择。当前 canonical 已经经过原严格验证；若新增验证发现冲突，应阻止 aux3 并保留证据，不改既有 canonical、split 或神经训练来配合经典。

## 6. TRAIN、VAL、TEST 支持与选择目标

### 6.1 明确沿用的支持政策

对每个有限、连续且 `N>=320` 的原段：

- TRAIN 拟合沿用既有 bank/transfer 的全部 N 行，包含末尾最多39行；不跨流重复或把同 LOW 复制成三个 seed。
- VAL/TEST 使用 `C=floor(N/40)*40` 个前缀行。该段80/160/320点、stride40的完整窗口首尾共同覆盖恰好是这些行；短于320的段不进入比较，记录跳过。
- 原 NN 归一化统计可以包含 TRAIN 全部 N 行，神经梯度来自完整窗口覆盖；经典最小二乘直接使用 N 行。因此只称同 TRAIN 物理拆分，不能称逐行监督曝光、重复权重或训练损失完全相同。
- 共享数学保持原 raw SI 最小二乘/正则化，未新增输入/目标 z-score。`normalization.mode=none_raw_si`，学习的归一化系数数为0。不能为了看起来“与NN一致”暗改 ridge 的尺度或 residual 参数化。

TRAIN/VAL 支持可在门禁后的 fit 阶段装载。TEST/新神经 TEST 预测载荷只在93 selections/558 final evaluations 与 aux2 成功之后装载。不得用会默认读取全部 splits 的 loader 绕过此顺序；显式指定 `split_names`。本次设计没有执行任何装载。

### 6.2 VAL 选择分数

在原 identity 分组仍存在、同 C 行覆盖且预测按原行位置对齐时，对每个 dataset d、device pair p 和物理量 u∈{gyro,accel}：

`R[d,p,u] = sqrt(sum_t ||pred_u(t)-HIGH_u(t)||² / n[d,p])`。

`B[d,p,u] = sqrt(sum_t ||LOW_u(t)-HIGH_u(t)||² / n[d,p])`。

`score = mean_d mean_p_in_d mean_u (R[d,p,u] / max(B[d,p,u], 1e-12))`。

同 pair 的多个段先合并平方误差和行数，再开方；不是段 RMSE 简单平均。gyro/accel 各自除同单位 identity 后才求均值。记录 near-zero 标记；不偷偷排除、改分母或跨单位相加。

这与 NN 的 `val_grouped_score` 在相同覆盖上的物理聚合定义一致；共享 OLS/ridge 的 TRAIN objective 仍是全部 TRAIN 行的未加权平方误差，不是 dataset-balanced training loss。不同协议不能共用别的协议 VAL 分数。每种 tuned 方法保存全部候选 alpha/taps、score、pair/dataset 分解、有限性、失败/耗时与选择顺序；选中后写入不可变 selection 文件及 SHA，再允许 TEST。原helper遇非有限或求解错误会失败，aux3保持此fail-closed规则；不能静默跳过失败候选后把不完整网格标为complete。

### 6.3 信号输出与原结果逐项对齐

每协议在同一冻结 canonical VAL/TEST 前缀上输出一次 shared 六轴预测。用源 manifest 直接确定支持，再与该协议全部60个神经最终输出的原物理列/覆盖指纹核对；不能依测试分数选一个代表确定支持。六个协议共360个 paired final mappings；原198个 navigation 最终评估仍参与558完整性门槛，但不重新计算。

保存原 identity、sequence/record、timestamp/source timestamp/availability、calibration 与 LOW/HIGH，加 shared 方法各6轴。完整原行顺序/物理列精确相等、输出有限、每物理时间点一份，才计算 dataset/pair/sequence 指标。gyro、accel 的向量/逐轴 RMSE、MAE、bias 及相对 identity 规则沿用旧定义，单位分开。

54个 logical controls 在 VAL/TEST 各有一份，形成108个 `(protocol,method,split)` logical prediction/metric slots；这不是108次拟合。identity 的数据列与原 LOW 精确相等，直接 alias 原输入/原 identity 产物，不能作为额外实验重复。

## 7. Penn 单 pair 的等价核验与别名

四个 Penn-only 协议（`penn_only` 和三个 `penn_loso_*`）都预期只有一个原 pair；实际装载后必须验证，不能凭名称假设。

若一个协议的 TRAIN/VAL 只有同一 pair，则共享选择分数退化为原 bank 单 pair 选择分数。等价还要求同 TRAIN 行/顺序、同 VAL 覆盖、同 history 特征、相同 alpha/taps 网格与 tie 顺序、相同 solver/precision。fixed helper 的 identity 元数据不同，不影响系数；整个 JSON state SHA 因标签/来源不同可以不同。

`alias_audit` 对每个协议/方法记录：

- 原 bank cache/state SHA、shared state SHA、TRAIN/VAL 原物理行 SHA、history 特征定义/输入顺序；确认未跨 protocol 借状态。
- 同 alpha/taps、同候选 history 与选中 VAL score；系数、截距和同支持预测的精确/数值比较。
- 精确 coefficient/hyperparameter/normalization 内容 SHA 相等且预测精确相等时，可记 `exact_alias` 并只储存/计时一个数值状态，两个 scope 只是逻辑引用。
- 精确字节不等但 `rtol=1e-10, atol=1e-12` 下系数、预测、VAL score 一致时，记 `numerically_equivalent`，保留两份原状态与差异证据，不伪造相同 SHA；不能把微小差异当作独立重复。所有容差在 TEST 前冻结。
- 不等价时保留实值和原因，不能据优劣选取或用共享结果覆盖 bank。来源/物理支持不符合本协议，或选中参数不属于本方法冻结网格/不能复现本方法自己的选择规则，属于合同错误，阻止交付。两个分别合法的选择若因实现/浮点排序差异选到不同参数，则记 `not_alias` 并完整披露，不能为了得到别名强行替换状态，也不能仅因合法结果不相等就伪造失败或成功。

不能因为四个协议都是 Penn 就跨协议 alias：TRAIN/VAL 录制不同。pooled 的 shared 状态由 AL+Penn TRAIN 共同拟合，通常不应与 pooled 内原 Penn 专用 bank 状态等同；是否相等也只凭审计，绝不预写结论。

源迁移本来已经 shared。本轮复用其纯数学函数，不用跨域目录缓存替代本次六协议的选择合同，也不以包含目标TEST载荷的整目录校验绕过fit阶段读取限制。既有source state可以在六协议全部冻结后作为额外等价证据，须证明源角色、N/C支持、候选顺序、状态与依赖 SHA 完全匹配；原迁移120个神经记录不变。aux3自身的完整resume缓存可复用，54/48逻辑覆盖不减少。

## 8. 共享经典的独立 GT 下游

新增 shared arm 的5个视图，沿用原 `(protocol,recording)`：af/as/bf 三个 LOSO、`penn_only/bs`、`pooled/bs`。前四个为 Penn 外层录制，pooled/bs 单列为同 bs 的训练范围对照，不能把两份 bs 作为额外独立场景。

每视图新增9个 shared logical controls，其中 identity 复用。SO(3)与strapdown两个任务、4个跨度、9个方法：每视图72个 summary slots，全部5视图360个。原 aux1 的5视图、30方法/70流及全部负面结果保持原文件与 SHA；新 arm 不是新录制或新种子。

主比较视图为20个神经 family/variant（每个3训练种子）+9个 shared经典+HIGH，共30 logical method variants/70 logical streams。known-device bank 另组或标明 exact alias，不能把两组相同 identity/Penn bank 别名画成独立重复，不混成新的 seed SD。

以下全部逐项复用原冻结协议，并核对 aux1 的支持/窗口 key SHA：

- 独立 AprilTag rig-body GT、原刚体标定与原固定源时钟 origin；不拟合新的 lag、旋转、bias 或 target optical calibration。
- 80/200/400/800点非重叠窗口，跨度1.975/4.975/9.975/19.975 s；同起始端点、同 GT 有效支持、同 gap 分段。
- midpoint SO(3) gyro；同冻结 strapdown 算法与 `[0,0,-9.80665]` 重力。
- 每窗/连续段初态只用一次 GT q/p；v只用开始前0.1s已存在的GT位置差，无过去支持则同样跳过；传播中不更新GT。
- 保留 endpoint/all-samples 姿态误差、heading/tilt可用性、3D/水平位置、full supported segments、窗口级结果和全部不支持/退化情况。

Shared 每方法 n=1、SD空；神经继续原三训练随机种子 sample SD。原旧 summary 中单确定性方法写0 SD的历史序列化不回写修改；新综合表明确 n=1/SD不适用，并保留原文件链接。无需重新推理神经或再次评估旧 bank，复用其已验证原结果；若新 shared 输出改变共同支持，应拒绝而不是删掉不利窗口。

## 9. 参数、归一化与 CPU float64 计时

新增的是实际状态审计，不把 JSON 文件大小当参数量，不按神经 family/seed/context 的重复引用乘经典系数。

| 方法 | 六轴输出的 stored learned coefficient scalars（含截距） |
| --- | --- |
| identity | 0 |
| bias_mean | 6 |
| affine / ridge_affine / ridge_tuned | 各36+6=42 |
| fixed FIR(9) | 36×9+6=330 |
| fixed ARX(12,加delta) | 36×(12+1)+6=474 |
| fir_tuned(T) | 36T+6，使用真实选中T |
| arx_tuned(T,加delta) | 36(T+1)+6，使用真实选中T |

从真实 ndarray 逐项验证上表，报告 coefficient_count/bytes（float64）、normalizer_count/bytes、非拟合超参数、序列化 state bytes、实际存储去重映射。这是存储的 learned coefficient scalar 数量，不是统计自由度；lag与delta可能有线性冗余，不能把存储数量等同独立可识别参数/有效DoF。raw-SI共享经典无 fitted normalizer，数为0；若读到与合同不符的 normalization 状态，失败而不是漏计。神经的 FP32权重/缓冲区与 TRAIN输入/残差统计分列，不把它们和 float64系数混成单一“训练参数”数。

原 bank 的 per-method/per-protocol 总量为该方法全部原 pair **唯一有效状态之和**，即每pair只记一次，去掉family/seed/context的重复映射，同时列出各pair和部署需知道的身份路由。数值内容相同的exact alias实际存储去重另列，不能以其替代pair-sum总量。不能把60个神经槽位、80/160/320计时 context 或三 seed 重复计入。不同协议的模型不是单次部署同时必需，跨协议 artifact 总存储另列。

计时在 aux2成功且原训练/神经/经典计时均结束后串行进行。采用原 `reviewer_classical_latency.yaml` 合同：CPU NumPy float64、threadpoolctl=3.6.0核验实际单线程、seed=20261024、warmup30、100 calls×3 repeats、B=1/128、W=80/160/320。只从本协议 TRAIN取预声明连续输入；输入池/row SHA/真实同段 history 长度均冻结。

每个状态报告真实 P50/P95、计时调用 sample SD、独立 loop吞吐和原始时间。计入 lag/delta 构造及矩阵/偏置运算，排除 I/O、fit、候选选择、状态载入、history检索与初始化；同段所需 past values 已预载，不能每窗冷启动。与原 float64 predictor 对齐后方可使用薄 NumPy kernel。

另给协议级整 bank 工作负载：固定每个原pair各B个同段窗口，按冻结pair次序逐pair调用；shared用同一系数重复调用，bank用各pair状态，二者处理完全相同 P×B输入。对整个循环实际计时并报告总 ms/处理窗口数；不能把各pair P95相加，不能把“整bank同时处理P个pair”冒充单器件一次查询成本。单状态计时与整bank工作负载分开；P=1的等价状态按 alias 审计去重。

新计时不与训练、aux1神经计时或aux2经典计时并发。资源或依赖无法验证时保留partial并等待，不启动争抢计时。神经原结果为FP32 network-only、已归一化驻留输入；shared/known-bank 为float64算法前向，precision与scope差异必须继续披露，不宣称普遍硬件优劣、40Hz保证或跨硬件bitwise。

## 10. 独立 aux3、缓存与完成顺序

建议新工程 `/root/imu-work/reviewer_shared_controls_20261025_v1`，源码/依赖/新配置单独SHA清单；运行cwd仍冻结 core，PYTHONPATH使用冻结src与已固定的独立CPU依赖。不覆盖 core、aux1、aux2任何文件、依赖环境或已生成结果。

aux3状态 `runs/reviewer_shared_controls_20261025_core_122113_893148/shared_controls_state.json`，阶段固定为：

1. `waiting_for_aux2`：只看明确 core/aux状态、marker、来源代码/配置/manifest元数据；93 selected、558唯一final evaluated、`REVIEWER_CORE_PASSED`/`WORKFLOW_EXIT=0`、aux1与aux2均声明全部成功后才进入后续阶段。此时验证来源/状态/manifest本身的SHA与身份，不解码或读取TEST数值/预测/指标文件；完整原产物的载荷审计延至新selection-set冻结之后。失败时阻止，不能因“部分模型已有test”放宽。
2. `fit_select`：仅TRAIN/VAL；验证原流，先完成全部六协议并冻结54逻辑registry、48 learned状态、所有候选history与selection hashes，生成global selection-set SHA。不能在逐协议循环内fit完一个协议就读它的TEST。
3. `signal_evaluation`：全部六协议选中状态均不可变且global selection-set SHA已验证后，先执行完整原core/aux1/aux2产物SHA、身份及有限性审计；缺任何原93/558/120/5门槛即阻止。之后才装载同支持VAL/TEST并保存108 logical slots与完整指标、360 paired final mappings（每个神经job同时绑定VAL/TEST，合计720个支持引用）；不重推理神经。不把早期元数据门禁冒称完整载荷验证。
4. `downstream`：新增shared5视图；复用已验证原GT与窗口支持，保存360 summary slots及逐窗/段结果。
5. `cpu_latency`：独占任务范围内串行测单状态与整bank工作负载，完成参数/归一化/实际时间清单。
6. `complete`：所有原与新增门槛、有限值、来源和alias审计完成后输出 `REVIEWER_SHARED_CONTROLS_PASSED`。

除只读状态等待外，aux3的全部实际阶段均在aux2成功后执行，不做CPU prefit并发。各阶段新目录与 `artifact_manifest.json`；合同绑定 core pipeline、aux1/aux2状态/manifest、数据split/strict/source SHA、源函数SHA、本文/实施草案SHA、新配置/网格/行支持/alias容差/计时规则。完整cache只有同合同且全部文件hash吻合才 `--resume` 跳过；变更不覆盖，失败尝试保留 `.partial` 和错误，不留下可冒充complete的文件。

## 11. Export v3与本地工具扩展

新 export v3在独立目录等待aux3成功，再完整导出原core+aux1+aux2+aux3，不替换其历史内容。原 export等待器证据保留；root仅在确认尚未导出时，未来可把旧等待器标为superseded并替换等待流程。若v2已完成则保留完整v2，不改写旧包、清单或分片；另建v3目录/新manifest。

包内增加：

- `roots.aux3_project=server/aux3`、`roots.aux3=server/aux3/runs/<实际run id>`；`source_roots`同key记录原绝对路径。
- 增加aux3 project的最长prefix path mapping；所有新文件进入 `files` SHA与 `file_metadata`，generated审计的source_path仍为null。
- completion audit保留原93/558、120 transfer、5原downstream视图、所有checkpoint/prediction有限与SHA审计；新增 `shared_protocols=6`、`shared_logical_controls=54`、`shared_learned_controls=48`、`shared_identity_controls=6`、`shared_signal_slots=108`、`shared_downstream_views=5`、`shared_downstream_summary_slots=360`、`shared_paired_final_mappings=360`，以及来源/selection/stream/alias/CPU审计。
- 新 `analysis_profile=reviewer_shared_classical_v1` 或等价明确版本字段，使正式消费者知道aux3是必要部分。不得只增加若干JSON数字而不回绑真实文件/行、真实状态、全候选与marker。
- 外部仍是经校验的tar.gz流、2GiB分片及逐分片/全流SHA；v3分片不能因文件名相同就复用v2，只能按新manifest校验。原132MB bulk canonical数据仍用既有已验证bundle引用，来源/GT/manifest不缺失。

| 工具 | 后续必要扩展 |
| --- | --- |
| exporter / export queue | 显式aux3工程/状态/marker、等待及完整四工程源/结果审计；新v3源码/配置SHA、新目录，不改已部署v2源码 |
| `reviewer_artifact_package.py` | 当前roots/path_mappings本来是通用字典，可接第四prefix；保持其generic读/校验职责。增加四root跨平台映射测试，不在helper伪装业务完整性 |
| receiver | 仍按manifest安全解包/校验全部成员与SHA，支持新root而不放宽路径安全；v3新mirror与新receipt，resume只接受同新manifest。不能把只含旧三工程的mirror当新完整交付 |
| figures | 必须同时验证原门槛和aux3新增门槛；主比较画shared经典，known-device bank单组，Penn精确别名标注；原93训练曲线/120迁移/5视图完整保留。全部源表/派生CSV/QA/四格式SHA继续可查 |
| report | 新profile必须要求aux3成功，严格保留原93/558/120/5，不以54个controls替代558。补共同信息条件、N/C差异、bank额外身份、Penn别名、参数/normalizer/整bank工作负载与所有退化；完整CSV/Markdown/PDF/HTML链接 |

旧report当前对audit.counts使用整字典相等，不能简单给旧manifest追加字段然后声称兼容。后续版本显式校验原required字段+新required字段和profile；缺任何字段拒绝正式完成，不删除/弱化原门槛。新图和新PDF仍需实际渲染、逐图/逐页视觉检查，不能继承已有synthetic QA passed或用旧成绩补缺。

### 11.1 消费接口与分组键

aux3根目录的 `shared_controls_artifacts.json` 只在完整成功后生成，包含 `schema_version=1`、`analysis_profile=reviewer_shared_classical_v1`、`status=complete`、上述counts、global selection-set SHA、各阶段manifest引用，以及 `tables`（每项的工程/运行相对路径、SHA、行数、列名）。表角色至少为 `control_registry`、`candidate_history`、`metrics_long`、`signal_registry`、`neural_support_registry`、`alias_audit`、`method_scope`、`downstream_summary`、`downstream_windows`、`downstream_segments`、`state_inventory`、`bank_parameter_totals`、`latency_summary`、`latency_raw`、`workload_latency`。具体内部文件位置可由生产者在此索引中声明，不从basename推断来源。

所有新汇总、排名、别名和绘图必须以 `(protocol, information_scope, canonical_method, variant)` 等完整键区分方法，不能仅按canonical method把bank与shared合并平均。`information_scope`明确区分统一六轴映射、已知器件bank与观测HIGH参考；identity是共同零拟合控制，只是一份确定性输出。原神经三种子是三个分别训练的模型，部署/前向计时用其中一份固定代表，不是三模型ensemble。各自n、系数总量、误差SD与计时分布不得互相替代。

54/108/360等是逻辑slot计数，需由真实registry唯一键核对；长表可因每slot多个metric而有更多行。不能用CSV总行数冒充逻辑覆盖。原bank、原全部神经/导航结果完整保留并带原信息范围；共同信息主表之外给bank控制组，Penn合法非别名结果如实保留。

## 12. 验收与自我审查

| 必须成立 | 自我审查结果 |
| --- | --- |
| 原神经训练不变，未读新TEST | 本次仅读源码/已有协议，未执行拟合、预测、测试数据读取、服务器操作或工具实现 |
| 同信息条件的含义准确 | 六LOW、同物理拆分、单共享状态；明确不等同逐行曝光/训练loss/时间感受野 |
| 先保存原物理流再collapse ID | 第5节规定原seq一对一映射和40Hz验证，不重命名/合并坏schema；fit和predict都受保护 |
| 每协议只用本VAL，所有候选可查 | 公式、原网格、tie与history合同均固定；不混六协议VAL |
| 54/48计数不伪装成独立重复 | identity复用、候选求解与最终slots分开；经典n=1，Pennalias需实证 |
| 原bank没有消失或被偷偷改善 | 文件/SHA不改，完整另组；不重新拟合目标GT偏置或挑有利结果 |
| 新信号/下游与原物理支持相同 | 360 paired mapping与共同C行、原window/GT key hash必须一致；不为shared删窗口 |
| actual参数/归一化/时延可复查 | 实际array计数、raw-SI normalizer=0、per-state和整bank workload分列，float64/FP32 scope明确 |
| 正式交付不绕开原门槛 | aux3等aux2，新export v3等aux3；93/558/120/5均保留，新增controls不替代神经记录 |

残留限制是研究结论的边界，不是待挑分的自由度：共享最小二乘受TRAIN行分布影响；神经训练窗口/未来上下文不同；有限录制、HIGH自身偏置、SmartPNT可观测性及独立GT精度限制继续存在。任何实现若不能证明原流、支持或alias，应失败并说明证据，不以良好结果替代合同证明。
