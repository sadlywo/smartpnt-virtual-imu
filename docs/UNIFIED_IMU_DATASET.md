# 统一 IMU 数据集与训练标签

正式数据版本位于 `data/processed/unified_imu_v2/`，构建来源由该目录的 `manifest.json` 中 `build_run` 指向 `runs/`。原始下载文件及之前的处理文件保留。`archive/duplicates.json` 记录重复源文件的内容哈希、保留轨迹及排除理由，Traj2 不会被重新分配到另一个 split。v1 已登记为 superseded_diagnostic，导出和训练会拒绝它。

## 数据范围与器件映射

| 数据 | 输入 | 标签 | 正式用途 |
| --- | --- | --- | --- |
| ALiMU | BMX160、MPU9250、LSM9DS1 | 同时采集的 VN100 六轴 | 配对信号训练 |
| PennCOSYVIO | Tango top 低成本 IMU | ADIS16488 六轴，经刚体外参/杆臂处理 | 配对信号训练 |
| SmartPNT-MSF | IMU，器件等级未确认 | 位置、速度、姿态导航真值 | 导航监督训练 |
| FusionPortable | DAVIS/Ouster IMU | STIM300 | 当前版本排除；严格连续测试窗口不足 |
| NavINST | 下载的 IMU 流 | 参考 IMU | 当前版本排除；缺官方空间标定 |

低成本 IMU → 更高等级 IMU 的直接监督使用 ALiMU 和 PennCOSYVIO。VN100 和 ADIS16488 是不同参考器件；配对配置提供参考器件条件输入，防止混淆目标分布。SmartPNT 没有第二套高等级 IMU 六轴测量，不能把导航真值、位置差分或速度差分冒充该任务的参考 IMU。

ALiMU 的三种低成本器件共享 11 段 VN100 原始参考序列，因此旧 leave-MPU9250 划分存在参考信号跨 split 重用，仅保留为诊断历史。原始 CSV 进一步确认 dyn_03 是 dyn_04 的 14,991 行输入前缀，stat_10 是 stat_11 的 14,999 行输入前缀，三个器件均如此；参考加速度也完全重用、少数参考陀螺仪值略有差异，不能称为完全相同的配对文件。v2 归档这些六个较短的输入重复变体，保留更长的 dyn_04/stat_11。

ALiMU 正式训练录制为 dyn_01、dyn_02、dyn_04、dyn_05、dyn_06、stat_11，验证为 dyn_07，测试为 dyn_08、dyn_09；同一参考录制的三种低成本器件始终在同一 split，并记录参考内容哈希与物理组。每个 split 均含三个低成本器件。这评估未见过的录制，不宣称未见过的器件泛化。Penn 保留 af/as 训练、bf 验证、bs 测试；SmartPNT 使用 Traj3 训练、Traj4 验证、Traj1 测试。所有切出的片段继承原轨迹 split，窗口不会跨片段。不同平台采用不同坐标约定：ALiMU 采用上游配对轴约定，尚未额外独立验证外参；Penn 为标定的刚体参考系。应同时报告各数据集/器件的测试指标。

## 信号与 Label

共同采样率为 40 Hz，角速度为 rad/s，加速度为 m/s²。统一列为 `gx_low` 至 `az_low`，实测信号标签为对应的 `*_high`。`has_signal_label` 和 `has_navigation_label` 明确区分两种任务；缺少的物理标签不会填零或伪造。参考器件条件列为 `reference_vn100`、`reference_adis16488`、`reference_stim300`。

较高采样率输入先进行 63 tap FIR 抗混叠低通，再规整到共同时间网格；配对输入和实测目标使用相同新阶段滤波。切掉完整滤波启动段，单个缺采样、非有限值、真值质量切换都划分独立片段。最大时间间隔为 1.5 个原生周期；不跨缺口插值。`timestamp_s` 为相对有效信号时间，`source_timestamp_s` 保留源时钟，`available_timestamp_s` 记录已知处理等待与群延迟。Penn 的继承重采样延迟按通道记录，另剔除 1.2 s 上游启动段；这不是设备完整频率相关相位响应的估计。

SmartPNT 从已解析的原生 200 Hz IMU 直接处理，避开旧 100 Hz 对齐文件在降采样前缺少抗混叠的问题。IMU 单位沿用原解析器的角增量/速度增量除以包周期的解释，该解释通过周期与重力幅值检查，但未据此推定器件等级。真值时间插值只使用两个端点均为 Fixed 且间隔有效的区间，角度插值先展开再包装。

SmartPNT 每条轨迹以第一条 ECEF 真值位置及经纬度为固定 ENU 原点，同一旋转用于位置和 ECEF 速度。上游 `X-LL/Y-LL/Z-LL` 等局部字段作为 `upstream_*` 留存，避免未经确认直接当作 ENU。转换约定见 [ESA Navipedia](https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates)。姿态列提供度和弧度；姿态仍遵循上游 heading/pitch/roll 约定，不宣称已知 IMU 安装旋转。

导航配置默认使用 `position_delta`：80 个采样点窗口的末端 ENU 位置减起点 ENU 位置，三个标签单位均为米。80 点代表 2 s 采样跨度，首末时间差为 1.975 s。`label_schema.json` 明确字段与单位；序列 Parquet 中实际存储标签，窗口标签由真实端点生成。导航真值未被反向用于补齐 IMU 标签。

## 可复现与服务器

处理命令：`python scripts/prepare_unified_imu.py --config configs/unified_imu_dataset.yaml`。已有版本拒绝覆盖，改处理方式时应使用新的 dataset_id/output_root。目录中保存 manifest、标签 schema、训练集统计量、配置快照和两套 split manifest。源文件哈希、AL split 依赖、Penn 上游诊断、SmartPNT 双源和实际工作区源码指纹均可追溯。

配对训练：`python scripts/train.py --config configs/train_unified_paired.yaml`；导航训练：`python scripts/train_gt_supervised.py --config configs/train_unified_navigation.yaml`。配对训练按数据集平衡抽样，随机数由 YAML seed 控制。归一化参数仅由训练 split 拟合；验证及测试复用。训练 run 保存数据和 split manifest 快照并链接构建 run；服务器 launcher 另记录来源训练包。

服务器打包与 SSH 操作见 [SERVER_TRAINING.md](SERVER_TRAINING.md)。上传包只包含选定处理数据、所需代码配置及对应构建记录，不携带原始大袋、旧实验、旧 checkpoint 或本地虚拟环境。训练入口支持最优权重保存；当前没有优化器/RNG/调度器状态恢复，不能宣称完整断点续训。
