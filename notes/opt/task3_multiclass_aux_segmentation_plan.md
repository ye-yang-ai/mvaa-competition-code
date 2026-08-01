# Task3 多类辅助分割优化方案

日期：2026-07-27

## 背景

Task3 当前最稳定的路线仍然是逐帧二值分割模型加融合后处理。

已经验证过的结论：

| 版本 | Task3 方案 | DSC | HD | ASD | 结论 |
|---|---|---:|---:|---:|---|
| v15 | UNet++ ResNet34 单模型 | 0.770197 | 95.627329 | 14.537786 | DSC 不高，但距离指标相对稳定 |
| v29 | v15/v22/B5 ensemble | 0.802186 | 101.142118 | 13.298139 | 当前 DSC 最强 |
| v30 | v15/v22/B5 局部搜索 ensemble | 0.800279 | 97.443801 | 13.185380 | 当前综合较稳 |
| v32 | v30 + Cutie gate | 0.800005 | 88.494260 | 14.071110 | HD 明显改善，但 ASD 没有同步改善 |
| v33 | 两阶段 ROI refine | 0.714739 | 107.004474 | 18.459779 | DSC 明显下降，暂时放弃 |

目前的主要问题不是单纯 threshold 不合适，而是模型在某些帧上会把细线、缝线、器械边缘或组织边界误连到二尖瓣上。一旦这些细长结构和目标区域相连，二值后处理很难完全切开，HD/ASD 会被远端错误显著拉高。

因此，下一步值得尝试一个更有针对性的训练方向：多类辅助分割。

## 核心想法

当前 binary 训练方式是：

```text
二尖瓣 label 10 = 1
其他所有类别 = 0
```

这个设定的问题是：线、器械、心房内面虽然都是背景，但它们不是普通背景。它们视觉上很像、距离二尖瓣很近，而且经常和二尖瓣发生接触或遮挡。

在 binary 训练里，模型只知道“这些都不是二尖瓣”，但不知道“线是一个特别危险、特别容易误分的负类”。所以模型容易把线或细长器械边缘并进二尖瓣 mask。

多类辅助分割的目标是让模型显式学习：

- 二尖瓣是什么。
- 线/缝线是什么。
- 器械是什么。
- 心房内面/组织是什么。
- 哪些区域是普通背景或无关结构。

提交时仍然只输出二尖瓣类别。其他类别只作为训练辅助和推理抑制信号。

## 推荐类别设计

第一版不建议直接做 16 类。样本量有限，类别越细，每类有效样本越少，训练会更不稳定。

推荐第一版使用 5 类：

| 训练类别 | 含义 | 原始 label |
|---:|---|---|
| 0 | background / ignore | 其他未使用标签 |
| 1 | mitral valve | 10 |
| 2 | wire / suture | 7 |
| 3 | instruments | 1, 2, 3, 4, 5, 6, 8, 15, 16 |
| 4 | atrial / tissue / hard negative | 9, 11, 12, 13, 14 |

说明：

- label 10 是最终提交目标。
- label 7 单独作为 line 类，因为它最可能导致细线 false positive 和 HD 飙升。
- 器械类合并，不追求具体器械种类，只要求模型知道它们不是瓣膜。
- 组织/心房内面合并成 hard negative，帮助模型学习瓣膜边界附近的相似结构。
- label 13 第一版可以并入 class 4，不单独设类，避免类别太碎。

如果第一版 5 类训练不稳定，可以退一步做 4 类：

| 训练类别 | 含义 |
|---:|---|
| 0 | background |
| 1 | mitral valve |
| 2 | line |
| 3 | hard negative，包括器械、组织、心房内面 |

但优先建议先试 5 类，因为 line 单独建模是这个方案的关键。

## 模型配置

第一版建议从稳定骨干开始，而不是直接上更大的 EfficientNet。

推荐配置：

```text
model: UNet++
encoder: ResNet34
encoder weights: ImageNet
classes: 5
image size: 448 x 800
loss: 0.5 CrossEntropy + 0.5 Multiclass Dice
batch size: 2
```

原因：

- ResNet34 单模型虽然 DSC 不最高，但线上 HD 一直比较稳。
- 先用 ResNet34 验证“多类监督是否能减少线误分”，风险比直接换大 backbone 更低。
- 如果有效，再迁移到 EfficientNet-B4/B5 或做多 seed ensemble。

## Loss 设计

推荐第一版：

```text
loss = 0.5 * CrossEntropyLoss(class_weight)
     + 0.5 * MulticlassDiceLoss
```

class weight 建议：

| 类别 | 建议权重 | 原因 |
|---:|---:|---|
| background | 0.2 ~ 0.5 | 背景像素太多，避免主导 loss |
| mitral valve | 1.5 ~ 2.0 | 最终目标类别，需要重点学习 |
| line | 1.5 ~ 2.5 | 危险负类，需要显式惩罚 |
| instruments | 1.0 ~ 1.5 | 重要负类 |
| tissue | 1.0 ~ 1.5 | 目标边界附近 hard negative |

注意：line 类像素可能很少，权重不能无限加大。权重太高可能导致模型过度抑制细长瓣膜结构，反而掉 DSC。

## 推理方式

模型输出 5 通道 logits，推理时使用 softmax：

```text
prob = softmax(logits)
mitral_prob = prob[class_mitral]
line_prob = prob[class_line]
```

基础提交方式：

```text
mask = mitral_prob > threshold
```

更推荐第一版同时搜索 line suppression：

```text
score = mitral_prob - alpha * line_prob
mask = score > threshold
```

候选搜索范围：

```text
threshold: 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
alpha:     0.00, 0.25, 0.50, 0.75
```

也可以测试硬抑制：

```text
mask = mitral_prob > threshold
mask[line_prob > line_threshold] = 0
```

候选：

```text
line_threshold: 0.30, 0.40, 0.50, 0.60, 0.70
```

第一版更建议使用 `score = mitral_prob - alpha * line_prob`，因为它比硬删除更温和，不容易误删真实瓣膜边界。

## 后处理

多类模型之后仍然保留轻量后处理：

1. threshold / alpha sweep。
2. 删除小连通域。
3. 填洞。
4. 保留主连通域或保留与稳定模型主区域重叠的连通域。

但这次后处理不应该是核心。核心是让模型在训练阶段学会区分 line 和 mitral valve。

## 验证重点

不要只看总 DSC。这个实验最重要的验证目标是：

1. 原来把线分进去的帧，是否明显改善。
2. worst-frame HD 是否下降。
3. ASD 是否下降。
4. DSC 是否持平或小幅提升。
5. 是否出现过度抑制，导致瓣膜细边界被删掉。

推荐输出 per-video / worst-frame 分析：

```text
video_id
frame_id
DSC
HD
ASD
pred_area
gt_area
line_prob_mean_inside_pred
num_components
```

如果发现 HD 降了但 DSC 大幅下降，说明 line suppression 太强或多类模型欠拟合。

如果 DSC 提升但 HD/ASD 没变，说明多类监督学到了主体区域，但没有解决远端细线错误，需要加强 line 类采样或后处理 gate。

## 与现有方案的关系

这个方案不是替代当前最佳 ensemble，而是先作为一个新候选模型加入对比。

推荐实验顺序：

1. 训练 `UNet++ ResNet34 5-class` 单模型。
2. 在本地验证集搜索 `threshold + alpha`。
3. 对比 ResNet34 binary 单模型和 v29/v30 ensemble。
4. 如果单模型有效，再与 v15/v22/B5/B5-seed ensemble 融合。
5. 如果 line 类确实能提供有效抑制，再考虑 EfficientNet-B4/B5 多类版本。

第一版线上提交不建议直接替换当前最优方案。更稳的做法是：

```text
p_final = w1 * current_ensemble_prob + w2 * multiclass_mitral_score
```

其中：

```text
multiclass_mitral_score = mitral_prob - alpha * line_prob
```

这样可以利用当前 ensemble 的稳定性，同时引入 line-aware 抑制。

## 风险

主要风险有四个：

1. 原始多类 label 质量不够稳定。
   - 如果 line 或器械标注不完整，模型会学到噪声。

2. line 类样本太少。
   - 需要考虑 hard-frame sampling，让含有 label 7 的帧更频繁进入训练。

3. 多类任务比二值任务更难。
   - 可能出现二尖瓣主类学习变慢，前期 DSC 不如 binary。

4. 过度抑制细结构。
   - 如果 line_prob 抑制太强，可能误删真实瓣膜边界，导致 DSC 下降。

## 第一版建议实验

实验名建议：

```text
t3_multiclass_res34_5class_s50
```

配置：

```text
model: UNet++
encoder: resnet34
classes: 5
image size: 448 800
epochs: 100
batch size: 2
seed: 50
loss: 0.5 CE + 0.5 multiclass dice
sampling: foreground balanced + line/hard-negative frame boost
validation: all-frame validation
```

推理搜索：

```text
threshold: 0.25 ~ 0.50
alpha: 0.00 ~ 0.75
min_area: 0, 100, 300
keep_main_component: true / false
```

推荐判断标准：

```text
如果 DSC >= 0.79 且 HD/ASD 比 v30/v32 更低，则值得生成线上 submission。
如果 DSC 明显低于 0.78，则先不提交，检查类别映射、采样和 line suppression。
```

## 结论

多类辅助分割是一个值得尝试的实质性训练方向。

它比继续微调 threshold 更有针对性，因为它直接处理当前最可能影响 HD/ASD 的错误来源：细线、器械、组织边界与二尖瓣误连。

第一版建议用 ResNet34 做 5 类多类训练，验证 line 类是否能有效减少 false positive。若有效，再将其作为新模型加入当前 v29/v30 ensemble，而不是一开始就完全替换现有最优方案。
