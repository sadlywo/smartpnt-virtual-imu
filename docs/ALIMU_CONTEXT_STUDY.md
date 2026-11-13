# ALiMU 上下文、机制与泛化实验

GitHub阅读：[最新结果汇总](results/2026-11-13/README.md) · [完整报告、CSV与图表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。本页工件链接指向Release，下载包内按原run名称查找；本地源路径仍用代码形式标识。


实验配置：`configs/ablations/alimu_context_mechanisms.yaml`。
源证据目录：`runs/alimu_context_mechanisms_20261108/`，不使用outputs/latest推断来源。
设计：`docs/superpowers/specs/2026-11-08-alimu-context-mechanisms-design.md`。

运行（pinn_imu；Windows Unicode路径启用UTF-8）：

```powershell
$env:PYTHONUTF8 = '1'
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_context_study.py
```

初始215个训练槽位：窗口72、受控前视36、前端机制45、等预算HPO35、严格泛化27。30个精确复用槽位，预计185次新训练；HPO验证选择产生14个最终重复槽位。另有预TEST声明的双单向GRU容量对照6个槽位，见`configs/ablations/alimu_causal_dual_gru.yaml`；另有同参数/同层数/同初始权重的TCN RF43→181对照6槽位；合计227槽位、197次新训练、30次精确复用。各项状态以run中的fit.json/phase文件/complete.json为准，不能把配置槽位当作已完成实验。

- 40/80/160/320点窗口保留同一中心40点目标和320点锚点。
- 0/6点前视与完整双向上下文，验证网络未来扰动、Jacobian与逐样本到达前缀回放。
- 前端局部RF、近参数匹配深度、BN/LN/无归一化、GELU/线性完整交叉。
- 七类模型每类三个LR/dropout候选；统一40轮上限、patience10；选后补全三seed。窗口组沿用旧80轮/patience15配方。
- 三折留出设备，同时留出物理采集；VALID排除目标设备，采用预声明固定配方。
- TRAIN-only affine校准；六轴和时间段的精确分组Shapley；损失/验证/梯度/隐藏表示中间态。

正式结果由run的`report.md`、`answers.json`、`verification.json`与`complete.json`共同确认。进行中曲线见`progress_figures`，不能用来替代完整TEST排名。

数据边界：LOW预处理清单记载25ms已知可用性延迟，网络0点前视不等于传感器零总延迟；上游外参未独立验证。三seed是优化重复，TEST只有两条独立物理采集。严格泛化指ALiMU内新设备+新采集，不宣称跨数据集泛化。

只读检查进度：`python -X utf8 scripts/check_context_study.py`；完成后用同脚本的`--verify`校验封存工件，不重新执行训练CLI修改已完成实验。结果图表和核心表格另打包为run中的`ALiMU_context_mechanisms_results.zip`。

执行并行记录：主协调器保留配置中的两个 worker；`scripts/assist_context_study.py` 可在 window/future 阶段使用一个额外独立进程，提前完成已锁定的 18 个 depth=1 机制配置。它不改变模型、种子、训练预算或数据支持，不读取 TEST。进程、源代码哈希和完成列表写入 `execution_assistant.json`，主协调器随后逐项校验并跳过已完成拟合。

在内存和显存核验后，另一个有界辅助进程 `scripts/assist_context_generalization.py` 可提前拟合27个固定配方的设备留出配置，并在设备切换时释放数据缓存；主协调器到达 search 阶段前停止领取新工作，以避免与 generalization 队列竞争。全程跳过 TEST 载荷，执行记录见 `execution_generalization_assistant.json`。主流程仍负责锁定、测试、审计和封存。

2026-11-09 01:05执行更新：有界辅助进程已统一停止，由 `scripts/run_context_study_execution.py` 使用单个四worker进程池续跑。冻结YAML中的worker=2保留作为原始配置，实际并发数及纯执行覆盖详见 `execution_pool_override.json`；`execution.py` 仅在设备切换时释放缓存，不改变数据、随机种子或训练计算。旧辅助进程的未完拟合由主队列恢复完整优化器/RNG状态。训练wall time受到并发影响，不能用于跨模型的独立速度排名。状态检查器将超过120秒未更新的未完历史列为pending_resume。

部署配置分析补充（2026-11-09，已查看VALID、尚未读取本轮TEST）：保持227个拟合槽位，把B组7个零前视后端与机制组10个不同的非BN零前视配置汇总为17个候选，按三seed平均VALID锁定配置。机制组默认GRU0精确复用只计一次。B组仍用于固定前端的后端公平比较；新增表是已探索配置的选择，GRU前端组合较多，不能据此宣称各家族拥有相同搜索预算。对新增配置也运行训练/推理未来扰动、Jacobian与逐样本前缀复核。规则和源码哈希见causal_configuration_reporting.lock.json。

2026-11-09 数值恢复记录：最后一个严格泛化Informer（留出MPU9250，seed20261012）在epoch7第194批次出现FP16稀疏注意力溢出。已从epoch6完整状态复现；输入/目标/参数均有限，首个非有限模块为第二层ProbSparseAttention。原始证据保留在`preparation/nonfinite_informer_probe/diagnostic.json`。预TEST修复配置`configs/ablations/alimu_numeric_precision_recovery.yaml`仅在有限输入的AMP注意力输出非有限时，将这一注意力运算以FP32重算，正常路径计算图不变，所有触发写入逐实验JSONL。没有改变模型、数据、seed、优化器或训练预算；这是一项需要明确披露的数值精度例外。使用`python -X utf8 -u scripts/run_context_study_safe_execution.py`继续未封存实验。数值修复和报告/审计的源代码分别冻结在`numeric_precision_recovery.lock.json`和带历史的`causal_configuration_reporting.lock.json`。源码、诊断、触发计数与测试结果纳入结果包。

最终完成（2026-11-09）：数值源run `alimu_context_mechanisms_20261108` 已封存，227槽位/197新训/30严格复用，213槽位评估，426份预测独立重算最大差异4.44e-16，48项因果复核通过，111个锁定源码哈希一致，2691个数值工件哈希通过校验。实际训练中间态63份，精确分组SHAP原始结果及完整性验证已保留。

最终图表与选型报告：`runs/alimu_context_mechanisms_20261109_figures_final/conclusions.md`；完整报告同目录`report.md`，9张SVG/PDF/PNG300dpi及数据表收录在`ALiMU_context_mechanisms_results_revised.zip`。此目录新增训练0次，数值CSV与数值源run逐字节一致；它修正机制热图密度、补全Raw LOW泛化基线、缩短训练SHAP纵轴并消除面板标题碰撞。先前figure_revision目录是保留的排版检查版本，最终使用figures_final。

本轮选型结论：共享离线CNN-BiGRU最优（80轮上限窗口组320点S0.355320；40轮等预算组S0.361670）；固定前端零前视后端GRU最优（S0.522061）；17个可因果配置按VALID选择三层线性CNN前端加GRU（无前端归一化/GELU，TEST S0.477732）；6点前视时因果CNN加Transformer后端最优（S0.372671）。40点窗口BiLSTM与BiGRU接近，80/160/320点BiGRU最低。未知器件测试：BMX160为Informer，MPU9250为源TRAIN拟合affine，LSM9DS1为Raw；三fold等权平均Raw最低，所有校正方法在留出LSM9DS1时均退化。三fold共享两条TEST物理采集，3seed SD仅代表优化随机性；不能把共享训练优势外推成可靠未知器件升级。

只读复核最终图表目录：`python -X utf8 scripts/check_context_study.py --run runs/alimu_context_mechanisms_20261109_figures_final --verify`。数值源仍用默认`--verify`。两个已封存目录均不应重新运行训练/绘图生成命令；新的呈现修改另建图形修订run。

2026-11-12 匹配FIR/ARX补充已完成：[协议与实测结果](ALIMU_CLASSICAL_BASELINES.md)。
权威运行`runs/alimu_context_classical_20261112/`覆盖共享窗口、0/6点前视及器件
留出条件，保持原中心40点样本和完整输入锚点；432个候选按VALID选型后评估。
共享320点整窗FIR/递归ARX的S为0.710244/0.710594，原40轮CNN-BiGRU为0.361670。
零前视FIR为0.785716，原固定前端GRU为0.522061；6点前视FIR为0.711962，原
受控Transformer为0.372671。器件留出时传统方法在BMX160、MPU9250上有竞争力，
三折等权平均仍由Raw最低。144份保存预测全量回放最大差异0，1296行指标复算
最大差异2.84e-14；独立核验记录位于`runs/alimu_context_classical_20261112_verification/`。
