# Shared Classical Controls Implementation Plan

> **For agentic workers:** 按当前对话已授权的完整公平比较目标执行，使用 `subagent-driven-development` 分组件实施并分别做规格与代码审查。任务以checkbox追踪。只改独立辅助/交付代码；原固定神经/core/aux1/aux2及其结果不改，不提交运行产物或原数据。

**Goal:** 在不改变冻结神经/core/aux1/aux2的前提下，补齐六paired协议的54个共同信息条件经典控制、独立GT下游和CPU状态/计时审计。

**Architecture:** 独立aux3等待原aux2成功后，先仅从严格TRAIN/VAL拟合并冻结全部共享状态，再验证完整原产物与读取新TEST/下游。复用冻结共享fit/predict与传播/计时纯数学函数，通过新wrapper保存物理流、原共同覆盖、原bank与Penn别名证据；独立export v3和本地新profile消费完整四工程产物。

**Tech Stack:** 冻结Python/NumPy/Pandas/Scipy与既有CPU依赖；经典float64，threadpoolctl=3.6.0；无新GPU模型或训练环境安装。报告仍用独立文档runtime。完整科学定义见[设计规格](../specs/2026-10-25-shared-classical-controls-design.md)。

---

## 0. 文件边界与不可变对象

以下是未来实施的文件分工，当前没有创建这些代码/配置：

| 文件 | 未来职责 |
| --- | --- |
| `scripts/reviewer_shared_classical.py` | 流验证、N/C支持、共享纯fit/predict适配、统一候选/状态和alias registry；不读TEST的fit API |
| `scripts/evaluate_reviewer_shared_classical.py` | 完整core门禁、每协议fit/选择冻结、共同支持信号输出与5个shared下游视图 |
| `scripts/benchmark_reviewer_shared_classical_latency.py` | 实际float64系数/归一化、逐状态与整bank工作负载计时；不拟合、不读TEST |
| `scripts/server/run_reviewer_shared_controls.py` | 等aux2成功、验证全部SHA、串行阶段、partial/resume与终止marker |
| `configs/reviewer_shared_classical.yaml` | 未来独立合同：6协议、9逻辑方法、原网格、N/C、score、alias与支持定义 |
| `configs/reviewer_shared_classical_latency.yaml` | 未来独立CPU计时合同，复用原预算/seed/thread规则 |
| `tests/test_reviewer_shared_classical.py` | 源流、宏平均、TRAIN/VAL-only、原预测等价与Pennalias单元/合成集成 |
| `tests/test_reviewer_shared_classical_latency.py` | history预载数值等价、实际计数、bank总量与计时scope |
| `tests/test_reviewer_shared_controls_queue.py` | 93/558与aux2等待、失败阻断、source SHA/resume/partial |
| 现有exporter/receiver/helper/figures/report及其测试 | 仅未来本地工具新版扩展aux3 profile；已部署旧版本与旧产物不覆盖 |

原 `src/virtual_imu/**`、`configs/reviewer_hpo.yaml`、已冻结core/aux1/aux2、旧export等待器源码/证据、原checkpoint/预测/状态/指标不改。代码实现只在独立aux3工程打包部署，不向冻结PYTHONPATH塞一个同名virtual_imu包覆盖训练源码。

## 1. 合同与synthetic schema先行

- [ ] 先建立不涉及真实数据的full-scope fixture：93个选中单元、558个final、原120迁移与5下游视图、6个共享协议、54逻辑控制/48learned/6identity、108信号slots、5新增shared视图/360summary slots、360 paired-final mappings；synthetic双标记始终可见。
- [ ] 新 `contract.json` 显式绑定 core pipeline、aux1/aux2状态与源码manifest、split/strict/source文件、本文/草案SHA、依赖源码SHA、原网格、支持/alias容差/CPU预算。未绑定的来源拒绝。
- [ ] logical registry 每行至少包含 `control_id, protocol_id, method, information_scope, fit_split, selection_split, identity_alias, selected_parameters, numeric_state_sha256, provenance_state_sha256, train_row_sha256, validation_row_sha256, selection_sha256, aliases`；保留逻辑与实际fit/cache/唯一状态的不同计数。
- [ ] 验收反例：只有53个control、丢一个learned、identity被记fit、缺原558/120/5、来源hash变化、synthetic走正式入口，均必须失败；不能用缩小scope参数放宽production CLI。

未来预期验证命令：`python -m pytest tests/test_reviewer_shared_classical.py tests/test_reviewer_shared_controls_queue.py -q`。先有合成反例失败，再实现必要门禁；当前不执行这些未存在测试。

## 2. 原物理流与支持装载

- [ ] 使用冻结 split/strict manifest 按明确role读取，fit阶段只请求 `train,validation`。不调用默认同时读取3个split的便捷入口；记录READS/WRITES/SKIPS。
- [ ] 验证每个原sequence_id一对一映射dataset/LOW/HIGH/record/segment；原seq保持原值。保留source记录，跨split物理录制不重用；同HIGH录制的不同LOW不能因physical_group相同被合并。
- [ ] 段内排序并核验唯一有限时间、40Hz、六LOW/HIGH有限、N>=320；缺字段/冲突/gap拒绝，不改名或补值。TRAIN留N，VAL按C前缀。保存输入/stream registry、原序与排序后的行SHA、N/C及跳过原因。
- [ ] 合成边界反例：同名seq的两个sensor同clock、不同record、gap、缺seq、重复timestamp必须在fit调用前失败；两个合法独立seq的首lag只edge-pad本段，delta首值为0。
- [ ] 验收NN同覆盖：N=320、359、360、399等边界段的80/160/320点、stride40覆盖均精确为floor(N/40)*40；不足320段跳过并列数，不把尾部补identity。

## 3. 共享fit/VAL选择与纯函数复用

- [ ] 以只接受TRAIN/VAL数组及原metadata的API复用冻结 `fit_source_classical/_select_shared/predict_source_classical`；不复用会装载目标TEST的 `_load_direction/_classical_record`。依赖文件SHA列入aux3 manifest。
- [ ] 仅fixed helper copy的IDENTITY常量化，原seq与VAL原IDENTITY不动。保存原frame不被helper原地改写的证明；预测不得按device/reference选state。
- [ ] 五种fixed learned与三种tuned严格沿原网格；raw-SI训练、无新normalizer。每协议一组方法参数，不把六协议VAL混合。完整候选history和strict tie次序落盘，所有状态有限。
- [ ] 分数回归使用不均衡toy：两个dataset各一个pair、样本100:1、identity的两物理量RMSE均为1；候选A在两域误差0.1/1，B为0.2/0.2。宏平均应选B（0.2<0.55），全数组pooled可能选A；测试必须防止回退成pooled选择。另核对多pair、多个seq、near-zero分母与frozen NN聚合一致。
- [ ] 同一六LOW物理流只改非预测metadata时，共享预测不得改变；改变TEST标签不能影响任何fit/selection文件SHA。未来真实TEST只在门禁后打开，设计/单元阶段使用synthetic标签扰动。
- [ ] selection文件先写一次并哈希，再进入评估；alpha/taps候选失败留完整错误、不得删除不利候选或改网格。48是最终learned逻辑slots，候选真实求解数另报。
- [ ] 六协议全部完成并冻结54逻辑/48learned的global selection-set SHA之前，不读取任何协议TEST；禁止逐协议fit→test交替。ARX显示名明确为ARX-style LOW lag/delta，canonical id与原数学方法不变，无过去HIGH输入。

## 4. Penn/bank别名与共同信号输出

- [ ] 原bank每协议/方法状态从已验证cache读取，原文件不写；每protocol只比自己的TRAIN/VAL支持与state，不能跨Pennfold引用。
- [ ] 用单pair synthetic frame同时走shared纯函数与原bank函数；比同网格顺序/选中值、系数/截距、VAL score、预测和源行SHA。整JSON可因标签不同而异。
- [ ] exact_alias要求真实数值状态/超参数/normalization内容与预测精确匹配；数值容差 `rtol1e-10/atol1e-12` 仅标numerically_equivalent，保留原SHA与最大差异。缺来源/物理支持不符失败，不按分数挑alias。两个合法选择不等时记not_alias，保留实值；只在自身网格/选择/来源合同失败时阻止，不能强迫Penn数值相等。
- [ ] 待完整93/558与aux2成功，装载每协议canonical TEST C行；与全部60个对应神经最终预测的原物理列精确核对。形成360 paired-final mappings，原navigation只验证完成而不改变。
- [ ] 每协议输出9方法一次，identity引用原LOW；VAL/TEST共108logical slots，指标沿原dataset/pair/sequence定义。classic n=1/SD空，method scope与原known-bank分开；candidate/seed/context不能产生重复经典指标。
- [ ] 保存完整预测Parquet、SI指标CSV/JSON、支持/alias审计与artifact hashes；有限值、原六轴/时钟/校准身份不符时拒绝完成。

## 5. 新shared下游arm

- [ ] 读取aux1原5视图的protocol/recording/physical support、window/segment keys与GT SHA；shared `pooled`取其跨全部TRAIN pair共享状态，再在Penn bs同原支持应用，不能取原Penn专用bank代替。
- [ ] 复用冻结 `evaluate_attitude_stream/evaluate_strapdown_stream`，9个shared方法/视图，两任务四跨度共360summary slots；identity不重新训练/算成独立重复。
- [ ] 强制原GT时间origin、刚体frame、最大gap、midpoint/trapezoid算法、重力、一次GT q/p及只用过去0.1s v初始化；实际window/segment keys与原aux1匹配，不因shared误差删窗口。
- [ ] 全部窗口/段和失败/不支持原因落盘；主图shared+NN+HIGH，known-bank另组。四外层录制与同bs pooled参考分离；无新GT bias拟合或神经推理。
- [ ] synthetic验收：恒速率旋转与已知加速度复用现有积分器科学测试，不重写算法；重点新增wrapper层的source clock/frame/支持、别名n和冻结来源拒绝测试。

## 6. 实际系数/normalizer与隔离CPU计时

- [ ] 从实际float64状态array统计stored learned coefficient scalars、截距、normalizer、超参数与序列化bytes；核对bias6、affine/ridge42、FIR36T+6、ARX36(T+1)+6；raw-SI normalizer0不得漏写。明确lag/delta冗余使存储数量不等同统计DoF。
- [ ] 对bank按method/protocol对各pair唯一状态求和；三seed、family和context映射只作来源，不乘系数总量。单状态、整bank部署状态和所有artifact存储分列。
- [ ] 固定TRAIN输入池、seed20261024、W80/160/320、B1/128、warmup30/100×3、实际BLAS单线程与threadpoolctl3.6.0；不读TEST或拟合状态。
- [ ] lag/delta kernel对同段preloaded history与frozen predictor数值等价；段首/中段窗口都验证，ARX差分不能丢首个真实past值。长度不足的TRAIN流明确跳过而不借其他流。
- [ ] 每状态实际p50/p95/调用分布SD/独立loop吞吐；整bank用同P×B输入与同逐pair调用scheduler，对完整循环实际计时，不能把pair P95相加。P=1 exact-alias只测一次并映射。
- [ ] 拟合/模型载入/I/O/history检索/初始化排除，lag/delta与矩阵运算计入；旧神经FP32 network-only scope另列。测量前验证core/aux2已终止成功与无本实验竞争计时；未满足时等待/partial，不强行计时。

## 7. aux3队列、immutable resume与服务器交付草案

- [ ] 新aux3工程独立manifest，绑定冻结core源码与aux1纯函数版本；cwd为core、PYTHONPATH冻结src，额外CPU依赖只用已固定独立target，不修改训练venv。
- [ ] 队列检查core全部93/558、三原terminal markers、aux1/aux2状态/来源代码与manifest元数据SHA；aux2失败明确阻断。fit前不解码TEST/预测/指标载荷，完整原阶段载荷SHA/有限性审计在新global selection-set冻结后、signal_evaluation开始前完成。部分final已完成不允许读新TEST。
- [ ] aux2成功之前只读等待状态，不CPU prefit；成功后串行运行 `fit_select → signal_evaluation → downstream → cpu_latency → complete`。signal_evaluation先核验全部六协议global selection-set SHA；`shared_controls_state.json`写阶段合同、开始/完成时间、command、artifact inventory，完成marker=`REVIEWER_SHARED_CONTROLS_PASSED`。
- [ ] 新run路径，已完成阶段同contract/全SHA才resume跳过；同名旧输出改变、同长度CSV篡改、source/grid/stream支持改变均拒绝。失败`.partial`保留，不能覆盖成成功。

后续拟议服务器命令如下，**本次不执行**；API由上述队列实现后再核对：

```text
python -B /root/imu-work/reviewer_shared_controls_20261025_v1/scripts/server/run_reviewer_shared_controls.py
  --project-root /root/imu-work/server_bundle_reviewer_hpo_20261024_v1
  --pipeline-root /root/imu-work/server_bundle_reviewer_hpo_20261024_v1/runs/reviewer_hpo_20261024_122113_893148
  --aux-root /root/imu-work/reviewer_shared_controls_20261025_v1
  --preceding-state /root/imu-work/reviewer_final_analysis_20261024_v1/runs/reviewer_final_analysis_20261024_core_122113_893148/final_analysis_state.json
  --out /root/imu-work/reviewer_shared_controls_20261025_v1/runs/reviewer_shared_controls_20261025_core_122113_893148
```

这不是重启/停止core的命令；不新增神经候选、不改固定超参数、不创建新的训练并发。源码/配置部署须完整审查与冻结SHA后才能进入执行阶段。

## 8. export v3、reader与完整报告

- [ ] exporter新版显式增加aux3 root/state/manifest/marker、54/48/108/360及全部来源审计；原93/558/120/5门槛全部保持。v3新目录、新源码/配置SHA，保留v2等待器和任何已完成包。
- [ ] root未来仅在确认尚未导出时替换旧export等待器；记录superseded身份/时间/原receipt，不删除证据，不对core/aux1/aux2进程做停止操作。已完成v2不覆盖，另导v3。
- [ ] 内manifest保留schema1通用files/file_metadata，增加aux3_project/aux3与第四path mapping；新analysis profile声明aux3必需。completion audit逐项回绑真实state/prediction/metric/stream/selection/alias/CPU文件和SHA。aux3生产者按规格11.1提供shared_controls_artifacts.json与有名table角色，消费者不猜文件路径。
- [ ] helper通用四prefix映射/最长prefix/source_path:null测试；receiver验证新manifest的全部成员和安全路径、分片+全流SHA，新mirror/resume合同。旧三工程包缺aux3不能作为新研究完成。
- [ ] Figures/report未来新版显式接受新profile的原必需counts和新增counts；当前整counts字典相等逻辑须按版本扩展，不能删原检查。主表shared共同信息，bank另组；全108信号slots、5shared views/360summary、系数/normalizer/整bank测量完整可查。
- [ ] 所有方法聚合/排名/误差条按information_scope与canonical method/variant完整键分组，禁止把同名bank/shared平均；identity只一份，NN种子均值不是ensemble。逻辑slot按唯一键计数，不用多metric长表行数替代覆盖。
- [ ] 完整source CSV/Markdown、图source/QA/四格式hash、93训练appendix、所有退化、Chinese MD/HTML/PDF和源/model/config/pred/metric链接保留。synthetic一条链验证后，真实包重新生成并实际逐图/逐页看，旧fixture QA不得迁移为真实通过。

## 9. 完成定义与草案自检

以下全部完成才满足扩展后的用户目标：原core+aux1+aux2成功；aux3 54logical/48learned覆盖及全部候选/来源/stream/别名/有限性审计成功；108signal slots、360paired mappings与5shared下游视图支持一致；实际参数/normalizer/per-state/整bank CPU计时完整；export v3、回传/安全解包SHA成功；全部新真实图/完整报告与实际视觉QA通过。

自检结果：

- [x] 每个设计要求能对应第1–8任务；没有把54控制替代558最终训练或改变120迁移。
- [x] 原sequence_id不重命名；先审计再collapse，fit与predict均防混流。
- [x] TRAIN N、VAL/TEST C和各自 SHA 清楚；不宣称同训练loss/曝光。
- [x] 逐protocol VAL宏分数、原网格、ties、候选history固定；不跨fold汇合VAL。
- [x] Penn等价由实际数值/支持证明，逻辑/数值/缓存重复与n=1分开。
- [x] 原bank、SmartPNT、原图表/服务器冻结产物不回写；仅后续新aux3与工具版本扩展。
- [x] 设计编写阶段没有实施代码、配置、科学测试、服务器命令或Git提交；现按本计划进入独立实施任务，后续代码/测试/部署证据按各任务更新。
