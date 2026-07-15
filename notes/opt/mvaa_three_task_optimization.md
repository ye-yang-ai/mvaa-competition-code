# MVAA 三个 Task 总体优化策略

本文档是总策略和路线索引。三个任务的详细实验路线分别放在：

- [Task1 CT 路线](task1_route.md)
- [Task2 TEE 路线](task2_route.md)
- [Task3 视频帧路线](task3_route.md)

最终目标不是把三个任务套进同一个模型，而是分别吃满每个任务最适合的路线：Task1 走 3D CT 强医学分割，Task2 走充分训练的 3D 多类别超声分割，Task3 走强 2D 视频帧分割加时序后处理。

## 1. 总体判断

| Task | 数据规模与类型 | 当前最关键问题 | 主路线 |
| --- | --- | --- | --- |
| Task1 | 27 个有标注 3D CT，1040 个无标注 CT，30 个验证 CT | 有标注极少，半监督伪标签质量不稳定 | 先建立 supervised/半监督 3D UNet 强基线，再做 nnU-Net v2、SegResNet/DynUNet、连通域后处理和 ensemble |
| Task2 | 105 个有标注 3D TEE，20 个验证 TEE | baseline 训练还没吃满，多类别边界和小组件需要后处理 | large 3D UNet 充分训练，再做 SegResNet/DynUNet、class-wise 后处理和 checkpoint/seed ensemble |
| Task3 | 6 个有标注视频、180 帧，1379 个无标注帧，48 个验证帧 | 半监督容易塌向背景，验证视频少，时序一致性未利用 | supervised-only 强 2D baseline，阈值搜索，视频级时序后处理，再谨慎加半监督 |

推荐总优先级：

1. **Task2 先充分训练**：当前代码已支持 large 3D UNet，最直接，最可能快速涨分。
2. **Task3 先禁用半监督建立上限**：先确认 `unetplusplus + efficientnet-b4 + imagenet` 的 supervised-only 能力，再决定半监督是否值得开。
3. **Task3 做时序后处理**：视频任务天然有帧间连续性，这是当前代码没有利用的明显收益点。
4. **Task1 跑稳强基线和 nnU-Net v2**：Task1 有标注少，单 split 波动大，不能只看一次内部验证。
5. **Task1/Task2 增加 MONAI 架构**：SegResNet/DynUNet 需要先改 `model_factory.py`，不是当前脚本可直接跑的实验。
6. **最后做 ensemble**：先收集可靠单模型，再做概率平均、checkpoint/seed/model ensemble。

## 2. 对现有想法的评审

### 2.1 合理的部分

- 不把 MedSAM2 当主路线是合理的。当前 MedSAM2 若依赖 baseline coarse mask 自动生成 prompt，容易把 baseline 错误精修得更像真错误。
- Task1 优先考虑 nnU-Net v2 合理。3D CT 二分类分割非常适合 nnU-Net 的自动 spacing、patch、normalization 和后处理经验。
- Task2 优先“训练吃满”合理。105 个 3D 样本对 baseline 仍有训练空间，过早换架构容易误判。
- Task3 先 supervised-only 合理。当前代码已有半监督塌背景保护，但视频少、阈值敏感，半监督仍可能放大错误。
- 后处理和 ensemble 必须进入主路线。三个任务都不是只靠单模型 argmax 就能拿满的类型。

### 2.2 需要修正或补强的部分

- **SegResNet/DynUNet/V-Net 不是当前 Task1/Task2 脚本直接支持的模型**。`task1/model_factory.py` 和 `task2/model_factory.py` 目前只支持 MONAI `UNet`。文档和实验计划必须标为“需先开发”。
- **Task1 半监督不应默认越强越好**。27 个有标注 case 下，伪标签错了会快速污染训练；需要先跑 supervised-only 对照，再逐步开启 `unsup_weight`、`pseudo_threshold` 和 `unsup_ratio`。
- **Task2 固定最后 20 例做验证可能有偏差**。当前 `task2` 按排序取最后 `val-count`，后续应增加随机 split/k-fold，否则内部验证可能不能代表 leaderboard。
- **Task3 的 `val_only_fg=True` 会高估模型**。只看有前景帧能更稳定地选阈值，但会掩盖空帧误报；需要同时记录 foreground-only 和 all-frame 指标。
- **Task3 阈值搜索容易过拟合 1-2 个验证视频**。阈值、时序平滑和面积规则都必须做 leave-one-video-out 或多 seed 验证。
- **后处理必须基于训练集统计调参**。例如连通域数量、最小体积、面积突变阈值都不能拍脑袋固定，否则 private set 容易掉分。

## 3. 当前代码能力边界

当前已经可直接跑：

- Task1：3D UNet `small/base/large`，半监督 EMA teacher，spacing resample 开关，内部 split，sliding-window 推理。
- Task2：3D UNet `small/base/large`，全监督训练，spacing resample 开关，sliding-window 推理。
- Task3：`unet/unetplusplus/fpn/deeplabv3plus`，SMP encoder，ImageNet 权重，Dice/Focal，EMA 半监督，阈值搜索，TTA。

当前需要新增开发后才能跑：

- Task1/Task2：SegResNet、DynUNet、V-Net。
- Task1/Task2：概率保存、概率 ensemble、后处理脚本。
- Task2：随机 split/k-fold。
- Task3：视频级概率保存、时序平滑、面积曲线修正、video-wise threshold。
- 三个 task：统一的实验表、结果汇总脚本、提交文件自动校验脚本。

## 4. 最小高性价比实验矩阵

如果训练资源有限，建议先跑下面这些，不要一开始铺太多模型。

| 优先级 | Task | 实验 | 目的 |
| --- | --- | --- | --- |
| P0 | Task2 | large 3D UNet, 150-200 epoch, `train-crops=2` | 吃满当前 baseline |
| P0 | Task3 | supervised-only `unetplusplus + efficientnet-b4 + imagenet` | 建立不被半监督污染的上限 |
| P0 | Task3 | 扩展 threshold candidates + TTA | 找稳定阈值 |
| P1 | Task3 | temporal smoothing + component filtering | 利用视频连续性 |
| P1 | Task1 | supervised-only 与保守半监督对照 | 判断无标注是否有效 |
| P1 | Task1 | nnU-Net v2 3d_fullres fold 0 | 获取强外部 baseline |
| P2 | Task1/2 | SegResNet/DynUNet | 架构增益 |
| P2 | 三个任务 | checkpoint/seed/model ensemble | 最终冲分 |

## 5. 推荐执行纪律

- 每个实验目录必须保存 `config.json`/命令、`history.csv`、`best_metrics.json`、split 信息和生成提交的 checkpoint 路径。
- 不同路线不要共用验证假设。Task3 要按视频维度验证；Task1/Task2 要记录 case 级别表现。
- 每次后处理调参都要在固定验证集上比较 before/after，并保留失败结果。
- 只把“内部验证稳定提升”的模型放进 ensemble；不要把弱模型平均进去。
- 最终提交前检查 zip 根目录必须直接包含 `t1_ct/`、`t2_tee/`、`t3_vid/`。

## 6. MedSAM2 的定位

MedSAM2 不建议作为 Task1/2/3 的主输出模型。更合理的定位是：

- Task3 的 pseudo-label/refiner 候选，但必须经过验证集证明。
- 对少数高置信 frame 或 mask 边界做 refinement，而不是全量替代主模型。
- 若 prompt 来自 baseline mask，则必须先筛掉明显错误的 baseline 预测。

目前最值得投入的是强基线、后处理、ensemble 和验证体系，而不是把 MedSAM2 作为核心提分路线。
