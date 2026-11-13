# 2026-11-06 全部模型与消融结果索引

GitHub阅读：[最新结果汇总](results/2026-11-13/README.md) · [完整报告、CSV与图表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。本页工件链接指向Release，下载包内按原run名称查找；本地源路径仍用代码形式标识。


权威汇总目录为 `runs/comprehensive_model_audit_20261106/`；本页是索引，不把 `outputs/latest` 当作结果来源。

- [完整报告](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)：任务范围、各域全部模型、旧消融、迁移、导航及未验证方案。
- [各场景最低信号误差](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [近期 ALiMU 逐设备对照](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [全部汇总图](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)
- [数值核验](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)与[来源/配置/代码哈希](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)

当前可支持的选型：历史 ALiMU 共享模型与较长上下文选择 CNN-BiLSTM；近期固定80点逐设备训练，角速度选 Informer，加速度选独立调参的无扩散 Transformer。Penn 优先保留 FIR/ARX，af/as 角速度的 GRU 也有优势。DDPM 严格消融没有收益。SmartPNT 无实测 HIGH 六轴标签，位移任务应单独评价；33个 tuned 配置均未胜过同轮 zero，default TartanIMU 仅在第三轮小幅领先。

2026-11-07 后续证据：[共享CNN-BiLSTM窗口与模块消融](CNN_BILSTM_ABLATION.md)完成138次拟合，结果见该运行的[结论](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。在相同中心监督的五个主结构中，完整CNN-BiLSTM/320的联合表现最均衡；纳入容量控制后，单向CNN-LSTM64/320的VALID均值更低，整窗监督的TEST域平均最低但Penn误差上升，不能将双向模型称为普遍最优。这是不同监督与评估支持的后续实验，保留上述历史汇总，不能将两轮绝对指标混排。

所有最低值是指定 TEST 和已测候选内的描述性结果。历史 vector RMSE 在主表中除以√3转换为 component RMSE；附录保留原指标单位，不跨协议、预算、因果范围或器件身份范围做总排名。minimal 分量选择存在索引与模型状态追溯缺口，只列为待核实候选。

PowerShell 复建本汇总（不新增训练）：

```powershell
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/summarize_all_experiments.py --config configs/analysis/comprehensive_model_audit.yaml
```

需保留源运行目录与三份 `support/*_audit.json/.md`；脚本先检查原封存表与引用证据，再写新的表、图、报告和核验记录。大型运行产物均留在被忽略的 `runs/` 下。
