# Task3 Cutie VOS 时序后处理方案

日期：2026-07-26

## 目标

本文档记录 Task3 的一个新后处理方向：使用 Cutie video object segmentation 作为现有逐帧模型预测的时序 prior/gate。

这个方案的目标不是替换当前的逐帧分割模型，而是减少视频级 outlier 错误，尤其是由突然假阳性、断裂连通域、面积跳变、质心跳变导致的 HD/ASD 飙升。

## 当前证据

Task3 线上结果呈现出比较稳定的模式：

| Version | Task3 来源 | DSC | HD | ASD | 解释 |
|---|---|---:|---:|---:|---|
| v15/v19/v21 | ResNet34 风格稳定逐帧模型 | 0.770197 | 95.627329 | 14.537786 | DSC 较低，但距离指标相对稳定 |
| v22 | UNet++ EfficientNet-B4 supervised-only | 0.790266 | 216.022344 | 22.657026 | DSC 提高，但出现严重距离 outlier |
| v23/v26 | v15/v22 probability ensemble `0.5/0.5, thr=0.4` | 0.798404 | 122.469457 | 14.554252 | 当前 DSC 最好；ensemble 压住了不少 v22 outlier |
| v24 | v15/v22 ensemble `0.55/0.45, thr=0.42` | 0.797286 | 118.965305 | 14.092329 | DSC 小降，HD/ASD 小幅改善 |
| v27 | EfficientNet-B4 EMA semi single model | 0.791728 | 219.513061 | 23.135393 | 半监督略微提高单模型 DSC，但没有解决 HD/ASD |

结论：更强的逐帧模型容易提高 DSC，但也容易引入不稳定的远端错误。当前的 threshold 和简单连通域后处理不能完全解决 worst-frame HD 问题。因此，引入视频级传播 prior 是一个合理的下一步方向。

## Cutie 的角色

Cutie 应该作为视频目标分割 prior 使用，而不是作为唯一预测来源。

推荐角色：

- 使用可靠的逐帧预测作为 anchor mask。
- 让 Cutie 在短连续 clip 内传播 object mask。
- 用 Cutie 输出识别时序上合理的目标区域。
- 用逐帧 probability 保留细边界。
- 主要用 Cutie 抑制远端假阳性连通域和不稳定跳变。

第一版不要把 Cutie 当成逐帧 mask 的硬替代。如果 anchor 错了，直接传播可能会把错误扩散到多帧。

## 输入假设

对每个 video folder：

- RGB 帧已经按帧编号排序。
- 已有逐帧预测，可以是 probability map，也可以是 binary mask。
- 输出必须保持现有 MVAA submission 格式：

```text
t3_vid/<video_folder>/<frame_name>_label_bin.png
t3_vid/task3_predictions.json
```

第一版推荐的逐帧来源：

- 如果有概率图，优先使用 v23/v26 probability ensemble，因为 v23 是当前 Task3 DSC 最好且比 v22/v27 单模型更稳定。
- 如果暂时没有 probability map，就先用 v23/v26 binary mask 做第一版实现，后续再扩展到 probability。

## Clip 切分

Cutie 默认假设视频帧具有时间连续性。但 Task3 的帧编号可能存在跳跃。

对每个视频：

1. 从文件名解析 frame number。
2. 按 frame number 排序。
3. 如果相邻帧编号间隔过大，就切成独立 clip。

初始规则：

```text
if frame_number[i] - frame_number[i - 1] > gap_threshold:
    start a new clip
```

推荐第一版设置：

```text
gap_threshold = 8
```

原因：

- 小间隔，例如 `000190 -> 000192 -> 000194`，仍可认为连续。
- 大跳跃应该重置传播，因为目标运动、视角、器械遮挡都可能变化太大。

## 多 Anchor 选择

不要每个视频只用一个 anchor。

对每个 clip，选择多个可靠 anchor：

1. 计算每帧预测特征：
   - 前景面积
   - 面积比例
   - 最大连通域面积
   - 连通域数量
   - 前景区域平均概率
   - 高置信前景比例
   - 质心
   - 帧间面积变化
   - 帧间质心移动

2. 满足以下条件时，将该帧标记为可靠：
   - mask 面积在该视频的合理范围内
   - 前景置信度高
   - 连通域数量不过多
   - 与相邻帧相比，面积没有突变
   - 与相邻帧相比，质心没有突变

3. anchor 选择规则：
   - 每个 clip 选择第一个可靠帧
   - 之后每隔 `N` 帧选择一个可靠 anchor
   - 遇到低置信或跳变片段时，额外启动一个新的局部 anchor

推荐第一版设置：

```text
anchor_stride = 8 to 12 frames
min_anchor_area_ratio = video median area ratio * 0.35
max_anchor_area_ratio = video median area ratio * 2.5
max_area_jump_ratio = 0.75
max_centroid_jump = 0.25 * image_diagonal
max_components = 3
```

如果第一帧不可靠：

- 使用最早的可靠帧作为第一个 anchor。
- 如果 Cutie 能稳定支持 backward inference，则从该 anchor 向前传播。
- 否则，anchor 之前的帧更依赖逐帧预测，Cutie 只作为弱 gate 使用。

## Propagation 策略

对每个 clip：

1. 用选中的 anchor mask 初始化 Cutie。
2. 向后传播，直到：
   - 到达下一个 anchor
   - 到达 frame gap 边界
   - propagated mask 变得不可靠
3. 在下一个 anchor 处重新初始化。
4. 如果可行，也从每个 anchor 向前传播到之前帧。

如果某一帧存在多个 propagated mask：

- 优先使用 frame index 上距离最近的 anchor 所在传播片段。
- 可以尝试平均最近 forward 和最近 backward 的 Cutie probability。
- 避免让距离很远的 anchor 主导当前帧，尤其是在已有更近可靠 anchor 的情况下。

## 融合策略

### 基线加权融合

第一版可以从这个公式开始：

```text
p_final = 0.65 * p_per_frame + 0.35 * p_cutie
```

对检测为不稳定的帧，提高 Cutie 权重：

```text
p_final = 0.45 * p_per_frame + 0.55 * p_cutie
```

不稳定帧判断信号：

- 逐帧前景置信度低
- 连通域数量过多
- 面积突然跳变
- 质心突然跳变
- 与附近 Cutie propagation 的 overlap 很低
- 存在远离 Cutie 区域的孤立远端组件

### 更推荐的保守 Gate

更有希望降低 HD 的策略，是把 Cutie 当作区域 gate：

1. 对逐帧 probability 做 threshold，得到候选连通域。
2. 对 Cutie mask 做 dilation。
3. 如果有稳定的 ResNet34/v15 mask，也做 dilation。
4. 只保留与以下任一区域有重叠的候选连通域：
   - `dilate(mask_cutie)`
   - `dilate(mask_resnet34_or_v15)`

这样可以保留逐帧模型的细边界，同时抑制远端假阳性连通域。

推荐第一版规则：

```text
keep component C if:
    overlap(C, dilate(mask_cutie)) > 0
    or overlap(C, dilate(mask_v15)) > 0
```

对高风险帧可以尝试更严格规则：

```text
keep component C if:
    overlap_ratio(C, dilate(mask_cutie or mask_v15)) >= 0.05
```

## 二值后处理

融合或 gate 后做小规模网格搜索：

```text
thresholds = [0.25, 0.30, 0.35, 0.40]
min_component_area = [0, 50, 100, 300]
fill_holes = [False, True]
gate_dilate_iters = [3, 5, 8]
```

后处理顺序：

1. threshold `p_final`
2. connected-component labeling
3. 删除小连通域
4. 保留与 Cutie/v15 gate 重叠的连通域
5. fill holes
6. 只有在不伤 DSC 且改善 ASD 时，才尝试轻量 closing

默认不要激进使用 keep-top-1，因为真实 mask 可能存在合理的分离区域。只有当逐帧诊断证明 extra components 是 HD 主因时，再考虑 keep-top。

## 验证协议

不要只看全局平均分。

每个候选配置都要报告：

- global DSC, HD, ASD
- per-video DSC, HD, ASD
- worst-frame HD list
- worst-frame ASD list
- empty-frame false positives
- foreground misses
- 面积跳变统计
- 质心跳变统计

主要成功标准：

```text
HD 相比 v23/v26 明显下降
ASD 不变差
DSC 下降不超过约 0.002 到 0.004
worst-frame HD cases 真的被修复或降低
```

强成功目标：

```text
Task3 DSC >= 0.795
Task3 HD < 115
Task3 ASD <= 14.0
```

最低有用目标：

```text
Task3 DSC 接近 v24
Task3 HD 低于 v24
Task3 ASD 不差于 v24
```

## 实现计划

### Step 1: 准备 Cutie 接入

使用官方 Cutie 代码和预训练权重，不从零实现 Cutie。

预期本地结构：

```text
external/Cutie
checkpoints/pretrained/cutie/<cutie_checkpoint>
```

创建 wrapper 脚本：

```text
task3/run_cutie_propagation.py
```

职责：

- 读取一个 video folder
- 读取 anchor masks
- 按 clip 运行 Cutie propagation
- 保存 Cutie probability 或 mask 输出

### Step 2: 构建 Anchor 和 Clip 元数据

创建：

```text
task3/build_task3_cutie_anchors.py
```

输出：

```text
outputs/analysis/task3_cutie/<source_name>/anchors.json
outputs/analysis/task3_cutie/<source_name>/clips.json
```

元数据需要记录：

- frame index
- frame gap
- anchor score
- anchor reason
- area
- confidence
- component count
- centroid

### Step 3: 评估 Cutie 融合

创建：

```text
task3/evaluate_task3_cutie_postprocess.py
```

输入：

- RGB frame root
- validation ground-truth root
- per-frame probability root 或 mask root
- Cutie output root
- 可选 v15 mask/probability root，用于 gate

输出：

```text
outputs/analysis/task3_cutie/<candidate_name>/metrics.csv
outputs/analysis/task3_cutie/<candidate_name>/per_video_metrics.csv
outputs/analysis/task3_cutie/<candidate_name>/worst_frames.csv
```

### Step 4: 有本地证据后再生成 Submission

创建：

```text
task3/generate_task3_cutie_submission.py
```

只有当本地验证显示 worst frames 的 HD/ASD 有改善时，才生成线上候选。

预期第一个线上候选：

```text
submit_v28_task1_v25_task2_v19_task3_v23_cutie_gate
```

## 第一版候选配置

推荐第一版候选：

```text
source per-frame: v23/v26 probability ensemble
stable gate source: v15 mask/probability
Cutie source: official pretrained Cutie
clip gap threshold: 8
anchor stride: 8
fusion:
  normal frame: 0.65 per-frame + 0.35 Cutie
  unstable frame: 0.45 per-frame + 0.55 Cutie
gate:
  keep components overlapping dilated Cutie or dilated v15
threshold sweep: 0.25, 0.30, 0.35, 0.40
min component area sweep: 0, 50, 100, 300
fill holes sweep: false, true
```

## 预期结果

最好情况：

- Cutie 修复少数 worst-HD foreground frames。
- HD 明显下降。
- ASD 小幅改善或基本持平。
- DSC 接近 v23/v26。

失败模式：

- anchor 错误在 clip 内传播。
- Cutie 过度平滑目标。
- gate 删除了真实的细边界或细小区域。
- DSC 下降，但 HD/ASD 改善不够。

决策：

- 如果 worst-frame HD 改善且 DSC 只小幅下降，继续细化 Cutie gating。
- 如果 global DSC/ASD 变差且 worst-HD frames 仍未修复，停止 Cutie 后处理路线，回到训练侧 temporal consistency 或 fixed-teacher distillation。
