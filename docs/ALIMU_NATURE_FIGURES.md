# ALiMU Nature 风格结果图集（2026-11-08）

GitHub阅读：[最新结果汇总](results/2026-11-13/README.md) · [完整报告、CSV与图表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。本页工件链接指向Release，下载包内按原run名称查找；本地源路径仍用代码形式标识。


本图集整理已完成并核验的实验，不新增模型训练。运行目录为 `runs/alimu_nature_figures_20261108/`，源实验目录保持只读。

| 比较范围 | 选型/观察结果 | 解释 |
| --- | --- | --- |
| ALiMU-only，三设备共享模型，320点中心监督 | **VALID 选择 CNN-GRU** | VALID 综合分数0.3319±0.0033 |
| 同一共享实验的已观察 TEST | **CNN-BiGRU 的均值最低** | 角速度0.001085 rad/s，加速度0.1796 m/s²，综合分数0.3553±0.0008 |
| ALiMU+Penn 联合共享训练 | **联合 VALID 选择 CNN-LSTM64** | 两数据集等权选择；图中只展示 ALiMU TEST |
| ALiMU 三设备分别训练，固定80点 | **Informer 角速度最低；独立调参无扩散 Transformer 加速度最低** | 物理误差分别为0.001018 rad/s、0.1642 m/s² |
| 严格匹配骨干、超参数与初始化的 DDPM 消融 | **去掉扩散的18个配对结果均更好** | 当前条件没有扩散收益 |

这里的“最低”限定于同一实验范围的已观察均值。320点共享实验与80点逐设备实验具有不同训练范围、监督支持和优化配方，不能混成统一排名。三个优化 seed 反映训练随机性；TEST只有两条物理录制，没有做显著性检验。

## 图片

| 图片 | 用途 | 可编辑版本 |
| --- | --- | --- |
| [图1：验证选择与测试排名](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) | 回答共享模型中哪个应选、哪个测试均值最低 | [SVG](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) / [PDF](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) |
| [图2：物理误差](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) | 全部11种共享候选与Raw对照 | [SVG](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) / [PDF](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) |
| [图3：消融证据](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) | 前后模块、窗口长度与设备表现 | [SVG](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) / [PDF](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) |
| [图4：DDPM对照](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) | 80点逐设备比较与严格去扩散消融 | [SVG](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) / [PDF](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) |
| [图5：训练范围与成本](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) | ALiMU-only/联合训练敏感性与统一CUDA延迟 | [SVG](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) / [PDF](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) |

每图同时提供300dpi PNG、600dpi LZW TIFF与可编辑SVG/PDF，宽7.2英寸。图形按 `nature-figure` 技能与项目 IMU 样式制作，采用统一方法配色、逐seed散点、均值±样本SD、明确单位与分面；未添加推断性星号或p值。

[中文完整报告](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) · [英文图注](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) · [逐模型数值表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) · [核验记录](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) · [图集压缩包](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)

## 复现

```powershell
& 'D:/Anaconda3/envs/pinn_imu/python.exe' scripts/build_alimu_nature_figures.py --config configs/analysis/alimu_nature_figures.yaml
& 'F:/python/python.exe' scripts/finalize_alimu_nature_figures.py --run-dir runs/alimu_nature_figures_20261108
```

绘图使用现有 Python/Matplotlib，PDF检查使用Python/PyMuPDF。完成后的图集目录会封存；修订图形时通过新的YAML experiment_id/output_dir创建新目录。绘图CLI会记录读取、写入与跳过项，保存源运行输入、配置、逐seed源数据、代码快照、数据清单与SHA256溯源。
