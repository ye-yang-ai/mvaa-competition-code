# Task3 视频帧优化路线

Task3 是 2D surgical video frame 二值分割。它和 Task1/Task2 本质不同：输入是 RGB 视频帧，标注视频只有 6 个、180 帧，验证视频 2 个、48 帧。主线应是强 2D encoder、严格按视频验证、阈值搜索、时序后处理，再谨慎加入半监督。

## 1. 当前代码状态

已支持：

- `unet`、`unetplusplus`、`fpn`、`deeplabv3plus`。
- SMP encoder，例如 `efficientnet-b4`、`efficientnet-b3`、`resnet34`。
- `encoder-weights=imagenet`。
- Dice/Focal 或 Dice/BCE。
- foreground-balanced sampling。
- EMA teacher 半监督。
- threshold candidates 搜索。
- TTA 推理。

未支持但建议开发：

- 保存每帧 probability map。
- video-wise threshold。
- temporal probability smoothing。
- 面积曲线平滑、突然消失/暴涨修正。
- 同时报告 foreground-only 与 all-frame 指标。
- leave-one-video-out 结果汇总。

## 2. 路线优先级

### P0：先禁用半监督，建立 supervised 上限

当前最重要的是确认强 2D baseline 的上限，避免半监督伪标签污染。

```bash
conda run -n mvaa python task3/train.py \
  --labeled-root data/reference_data/t3_vid/train \
  --unlabeled-root data/images \
  --output-dir outputs/opt/task3/t3_sup_unetpp_effb4_img_e120_s42 \
  --epochs 120 \
  --batch-size 1 \
  --unlabeled-batch-size 1 \
  --image-size 448 800 \
  --val-video-count 1 \
  --arch unetplusplus \
  --encoder-name efficientnet-b4 \
  --encoder-weights imagenet \
  --unsup-weight 0 \
  --semi-warmup-epochs 999 \
  --threshold-candidates 0.05 0.075 0.10 0.125 0.15 0.175 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 \
  --val-tta \
  --num-workers 0
```

如果显存足够，第二组尝试：

```text
image-size: 512 896
encoder-name: efficientnet-b4 或 efficientnet-b5
```

### P0：多视频验证

只有 6 个标注视频，必须按视频做稳定性检查。建议至少跑 3 个 seed：

| 实验 | `seed` | 目的 |
| --- | --- | --- |
| T3-S42 | 42 | 基准 |
| T3-S3407 | 3407 | 换验证视频 |
| T3-S2026 | 2026 | 换验证视频 |

更理想的是 leave-one-video-out：每次留 1 个视频验证，6 个视频轮流做验证。所有 threshold、后处理参数都看平均结果和最差视频。

### P1：模型对照

优先级：

1. `unetplusplus + efficientnet-b4 + imagenet`
2. `deeplabv3plus + efficientnet-b4 + imagenet`
3. `fpn + efficientnet-b4 + imagenet`
4. `unetplusplus + efficientnet-b3 + imagenet`
5. `unetplusplus + resnet34 + imagenet`

不建议把 Task3 改成 3D nnU-Net/V-Net。视频帧数少，直接 3D 建模工程复杂且容易过拟合；更现实的收益来自时序后处理。

### P1：阈值策略

当前脚本支持 `--threshold-candidates`，应扩大低阈值范围。原因是目标可能偏小或模型校准偏低，默认从 0.2 开始可能错过最佳点。

推荐候选：

```text
0.05, 0.075, 0.10, 0.125, 0.15, 0.175,
0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
```

后续开发 video-wise threshold：

- 每个视频独立选择候选阈值。
- 用训练视频交叉验证选择规则，而不是直接对最终验证视频过拟合。
- 最终提交若没有标签，只能使用从训练验证得到的规则，不应人工看验证输出后手调。

### P1：视频时序后处理

推荐处理 probability map，而不是只处理二值 mask：

```text
for each video:
  load frame probabilities
  temporal smooth with window 3 or 5
  threshold
  remove tiny connected components
  keep top-k components
  compute area curve
  repair sudden disappearance if neighbors are stable
  suppress sudden explosion if area is far above local median
```

第一版保守规则：

- temporal window：3。
- min component area：从训练 mask 面积分布估计。
- keep top-k：1 或 2。
- 如果某帧面积接近 0，但前后帧均非空且面积稳定，用平滑 probability 重新阈值。
- 如果某帧面积超过邻域 median 的 3-5 倍，优先保留和邻帧位置相近的组件。

### P2：谨慎半监督

只有 supervised-only 稳定后才开半监督。建议从很保守的参数开始：

```bash
conda run -n mvaa python task3/train.py \
  --labeled-root data/reference_data/t3_vid/train \
  --unlabeled-root data/images \
  --output-dir outputs/opt/task3/t3_semi_unetpp_effb4_img_conservative_s42 \
  --epochs 160 \
  --batch-size 1 \
  --unlabeled-batch-size 1 \
  --image-size 448 800 \
  --val-video-count 1 \
  --arch unetplusplus \
  --encoder-name efficientnet-b4 \
  --encoder-weights imagenet \
  --semi-warmup-epochs 50 \
  --unsup-ramp-epochs 40 \
  --unsup-weight 0.1 \
  --pseudo-pos-thr 0.9 \
  --pseudo-neg-thr 0.05 \
  --pseudo-min-area 120 \
  --pseudo-min-pos-ratio 0.001 \
  --val-tta \
  --num-workers 0
```

判断半监督是否有效：

- `pseudo_pos_ratio` 不能长期接近 0。
- `val_pred_pos_ratio` 不应明显低于 `val_gt_pos_ratio`。
- 如果 best epoch 总在半监督开启前，说明半监督在伤害模型。
- 半监督只要不稳定，就保留 supervised-only 作为主提交模型。

## 3. Ensemble 路线

推荐 probability ensemble：

```text
model A probability
model B probability
model C probability
average probability
temporal smoothing
threshold
component filtering
save *_label_bin.png
```

候选来源：

- 不同 encoder：efficientnet-b4、efficientnet-b3、resnet34。
- 不同 arch：unet++、deeplabv3+、fpn。
- 不同 image size：448x800、512x896。
- 不同 seed/验证视频。
- supervised-only 与保守半监督模型。

不要做 mask hard-vote 作为第一选择。概率平均更适合后续时序平滑和阈值搜索。

## 4. 风险点

- `val_only_fg=True` 会隐藏空帧误报风险，后续必须同时看 all-frame 指标。
- 1 个验证视频选出的 threshold 可能严重过拟合。
- ImageNet 权重需要网络下载或本地缓存；如果环境离线，要提前确认。
- 半监督伪标签若塌向背景，会让模型越来越不预测前景。
- 时序后处理过强可能让真实快速变化被抹平，要用 leave-one-video-out 检查。

## 5. 最近下一步

1. 跑 supervised-only `unetplusplus + efficientnet-b4 + imagenet`。
2. 扩大 threshold candidates。
3. 增加 all-frame validation 指标。
4. 写 probability 保存推理脚本。
5. 写 temporal smoothing 后处理脚本。
6. 再尝试保守半监督。
