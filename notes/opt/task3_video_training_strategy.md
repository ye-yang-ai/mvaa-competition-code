# Task3 Video-Level Training Strategy

本文档记录 Task3 下一阶段优化策略。结论基于 v12、v15、v20、v21 的线上结果和本地帧级诊断。

## Current Evidence

- v12: Task3 DSC 最高，但 HD/ASD 明显变差，说明召回提高后远端误差和假阳性风险增加。
- v15/v19: Task3 DSC 略低于 v12，但 HD/ASD 最均衡，说明保守阈值和 TTA 能减少极端错误。
- v20: `threshold=0.28, min_area=20` 线上三项均略差，轻量小连通域过滤没有泛化收益。
- v21: `min_total_area=1500` 线上与 v15/v19 完全一致；本地逐 PNG 比较确认门控没有触发，48 张预测最小前景面积为 15982。

因此，继续做轻量 threshold/postprocess 的收益很低，下一步应转向训练策略。

## Main Goal

训练一个兼具 v12 召回能力和 v15 稳定性的 Task3 模型：

- 提高 foreground frame 的 Dice。
- 降低空帧 false positive。
- 降低视频间泛化波动。
- 控制 HD/ASD 的极端 outlier。

## Priority Strategy

### P0: Video-Level Validation

Task3 只有 6 个标注训练视频，单 split 很容易过拟合。后续模型选择必须按视频验证：

- 至少使用多个 `seed` 做 video-level split。
- 更理想是 leave-one-video-out：每次留 1 个视频验证，训练其余 5 个视频。
- 选择模型时同时看平均指标和最差视频指标。
- 不只看 foreground Dice，还要记录 all-frame Dice、empty-frame false positive rate、foreground miss rate。

### P0: Strong Supervised-Only Baseline

第一步先禁用半监督，建立稳定 supervised-only 上限：

- architecture: `unetplusplus`
- encoder: `efficientnet-b4`
- encoder weights: ImageNet
- image size: `512 x 896`
- loss: `dice_focal`
- sampling: foreground-balanced sampling
- validation: include empty frames, use all-frame Dice for threshold/checkpoint selection

半监督先不开，避免伪标签污染。

### P1: Empty-Frame Robustness

当前后处理无法解决大面积空帧误报，训练时应显式建模空帧：

- batch 中保留一定比例 empty frames。
- 记录 empty-frame false positive rate。
- 后续可加入 image-level presence loss。
- 后续可对空帧预测面积加入 penalty。

### P1: Temporal Consistency

视频帧不是独立样本。后续优先考虑低风险时序约束：

- 单帧模型先保持不变。
- 在训练或验证后处理中加入相邻帧 probability consistency。
- 避免某帧预测突然爆炸、突然消失。

直接改成 3-frame 输入或 3D 视频模型作为后续方案，不作为第一步。

### P2: Conservative Semi-Supervised Training

只有 supervised-only 稳定后才启用半监督：

- warmup 足够长。
- unsup weight 小。
- pseudo positive threshold 高。
- pseudo labels 需过滤过小区域和低置信区域。
- 如果 best epoch 总在半监督开启前，则半监督视为伤害模型。

## First Experiment

输出目录：

```text
outputs/opt/task3/t3_sup_unetpp_effb4_img512x896_allframe_s42
```

核心配置：

```text
--arch unetplusplus
--encoder-name efficientnet-b4
--encoder-weights imagenet
--image-size 512 896
--epochs 120
--batch-size 2
--unsup-weight 0
--semi-warmup-epochs 999
--no-val-only-fg
--threshold-selection-metric all_dice
--score-use-all-frame-dice
--val-tta
```

判断标准：

- 如果 all-frame Dice 明显提高且 empty FP rate 下降，再生成线上提交。
- 如果 foreground Dice 提高但 empty FP rate 上升，下一步加入 presence/empty penalty。
- 如果整体不如 v15，则回退到 ResNet34 或调整 loss/sampling。
