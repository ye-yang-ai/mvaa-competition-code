# Task3 引入 MedSAM2 的优化方案

## 1. 目标与结论

Task3 是手术视频帧二值分割任务，输入为 RGB 帧图像，输出为 `*_label_bin.png` 二值 mask。当前项目已经有基于 `segmentation_models_pytorch` 的 UNet/UNet++ 半监督训练流程，训练入口为 `task3/train.py`。

MedSAM2 更适合在 Task3 中作为离线伪标签生成器，而不是直接替换现有提交模型。推荐方案是：

1. 使用已标注帧的 mask 或由当前模型生成的粗 mask 作为 MedSAM2 prompt。
2. 利用 MedSAM2/SAM2 的视频传播能力，在同一视频序列中向前后帧传播 mask。
3. 对传播得到的 mask 做质量过滤，形成高置信伪标签。
4. 将伪标签按 Task3 现有 `*_label_bin.png` 格式落盘。
5. 复用现有 `task3/train.py` 训练 UNet++，把 MedSAM2 伪标签当作扩充后的 labeled data。

这样做的好处是改动小、风险低、容易和当前 baseline 对比；MedSAM2 负责扩大训练监督信号，最终提交仍由现有 Task3 模型生成，输出格式和评测流程不变。

## 2. 当前 Task3 现状

### 2.1 数据与代码接口

当前 Task3 主要文件：

- `task3/dataset.py`：发现有标注帧、读取 tar 或 `*_label_bin.png` 标签、读取无标注帧。
- `task3/train.py`：半监督训练入口，支持 labeled root 和 unlabeled root。
- `task3/model_factory.py`：创建 2D UNet、UNet++、FPN、DeepLabV3+。
- `task3/generate_task3_predictions.py`：生成提交用二值 mask 和 JSON。

现有训练命令示例：

```bash
CUDA_VISIBLE_DEVICES=3 python task3/train.py \
  --labeled-root data/reference_data/t3_vid/train \
  --unlabeled-root data/images \
  --output-dir outputs/exp/task3_unetpp_effb4_none_e100_bs2 \
  --epochs 100 \
  --batch-size 2 \
  --unlabeled-batch-size 1 \
  --image-size 448 800 \
  --val-video-count 1 \
  --num-workers 0 \
  --arch unetplusplus \
  --encoder-name efficientnet-b4 \
  --encoder-weights none
```

### 2.2 为什么 Task3 最适合 MedSAM2

MedSAM2 的核心优势包括 promptable segmentation 和视频/序列 mask propagation。Task3 的数据天然是视频帧序列，已有标注帧可以直接作为初始 mask prompt，未标注帧可以通过传播得到伪标签。这比 Task1/Task2 的 3D 多类 NIfTI 自动分割更容易落地。

## 3. 总体架构

建议新增一个独立的伪标签生成模块，不直接修改训练主循环：

```text
tools/
  medsam2_task3/
    prepare_medsam2_inputs.py
    run_medsam2_propagation.py
    filter_pseudo_masks.py
    build_pseudo_labeled_root.py
    README.md

outputs/
  pseudo/task3_medsam2/
    raw_masks/
    filtered_masks/
    pseudo_labeled_train/
    reports/
```

推荐数据流：

```text
已有标注帧 + 标签
        |
        v
构造 seed frame / seed mask
        |
        v
MedSAM2 视频传播
        |
        v
raw pseudo masks
        |
        v
质量过滤 + 连通域清理 + 面积约束
        |
        v
pseudo_labeled_train
        |
        v
task3/train.py
        |
        v
最终 Task3 提交模型
```

## 4. MedSAM2 伪标签生成方案

### 4.1 输入组织

MedSAM2 视频推理通常需要按视频组织连续帧。Task3 文件名形如：

```text
video_id_000001.png
video_id_000002.png
...
video_id_000001_label_bin.png
```

建议将每个视频整理成：

```text
outputs/pseudo/task3_medsam2/medsam2_inputs/
  video_001/
    frames/
      000001.png
      000002.png
      ...
    seeds/
      000010_label_bin.png
      000080_label_bin.png
    metadata.json
```

`metadata.json` 记录原始路径映射：

```json
{
  "video_id": "xxx",
  "frames": [
    {
      "frame_idx": 1,
      "src": "data/reference_data/t3_vid/train/xxx_000001.png",
      "dst": "frames/000001.png"
    }
  ],
  "seeds": [
    {
      "frame_idx": 10,
      "mask": "seeds/000010_label_bin.png"
    }
  ]
}
```

### 4.2 Prompt 来源

优先级建议如下：

1. **真实标签 mask prompt**：对已有标注帧，直接把 `*_label_bin.png` 作为初始 mask prompt。这是质量最高的方式。
2. **当前最佳 UNet++ 预测 prompt**：对没有真实标签但模型很自信的帧，用当前模型预测 mask 作为 MedSAM2 seed。
3. **box prompt**：从真实或预测 mask 的最大连通域生成 bounding box。适合作为 MedSAM2 的补充 prompt。
4. **point prompt**：从 mask 内部采样正点，从 mask 外部边界附近采样负点。可作为失败 case 的 fallback。

第一阶段只建议使用真实标签 mask prompt，避免引入当前模型错误造成的级联污染。

### 4.3 传播窗口

不要一次性从一个 seed 传播到整个长视频。手术视频中器械、组织、视角和遮挡变化较快，长距离传播容易漂移。

建议配置：

- 初始窗口：向前 20 帧、向后 20 帧。
- 如果相邻 seed 较密，可以传播到两个 seed 的中点。
- 如果视频帧间隔较大或画面变化剧烈，窗口缩短到 10 帧。
- 如果 raw pseudo mask 的面积变化稳定，再尝试扩大到 40 帧。

推荐输出文件：

```text
outputs/pseudo/task3_medsam2/raw_masks/
  video_001/
    000001_label_bin.png
    000002_label_bin.png
    ...
```

## 5. 伪标签质量过滤

MedSAM2 的传播结果不能无条件加入训练。建议至少做以下过滤。

### 5.1 面积过滤

对每个 mask 计算前景比例：

```text
fg_ratio = mask_foreground_pixels / image_pixels
```

推荐规则：

- `fg_ratio < 0.0002`：认为几乎空 mask，丢弃。
- `fg_ratio > 0.25`：认为过度扩张，丢弃。
- 面积相对 seed mask 变化超过 4 倍：丢弃或降权。

具体阈值需要根据 Task3 标注对象实际大小调整。

### 5.2 连通域过滤

对二值 mask 做连通域分析：

- 只保留最大连通域，或保留面积大于最大连通域 10% 的连通域。
- 删除面积小于 80 像素的小噪声区域。
- 做一次 closing/opening 平滑边界，但不要过度腐蚀细长结构。

### 5.3 时序一致性过滤

同一视频内相邻帧的 mask 面积和中心位置应连续变化。建议计算：

- `area_t / area_{t-1}`
- mask centroid distance
- 与上一帧 mask 的 IoU

推荐规则：

- 相邻帧面积变化超过 3 倍：标记为异常。
- centroid 跳变超过图像对角线的 20%：标记为异常。
- 与上一帧 IoU 低于 0.05 且两帧都非空：标记为异常。

异常帧先不加入训练集，保留在报告中人工抽查。

### 5.4 模型一致性过滤

如果已有 Task3 baseline checkpoint，可以把当前模型预测与 MedSAM2 mask 做一致性检查：

```text
iou = IoU(medsam2_mask, baseline_model_mask)
```

推荐策略：

- `iou >= 0.3`：高置信伪标签。
- `0.1 <= iou < 0.3`：中置信，先不用于第一轮训练。
- `iou < 0.1`：丢弃或人工抽查。

第一轮可以只使用高置信伪标签，第二轮再考虑中置信样本。

## 6. 训练接入方式

### 6.1 最小改动方案

最小改动是不改 `task3/train.py`，而是构造一个新的 labeled root：

```text
outputs/pseudo/task3_medsam2/pseudo_labeled_train/
  original/
    原始有标注训练帧和标签
  pseudo/
    MedSAM2 伪标签帧和 *_label_bin.png
```

由于 `task3/dataset.py` 使用 `rglob("*.png")` 搜索图像，并识别同名 `*_label_bin.png`，只要保持命名规则即可被现有训练代码读取。

训练命令：

```bash
CUDA_VISIBLE_DEVICES=3 python task3/train.py \
  --labeled-root outputs/pseudo/task3_medsam2/pseudo_labeled_train \
  --unlabeled-root data/images \
  --output-dir outputs/exp/task3_medsam2_pseudo_unetpp_effb4_e100 \
  --epochs 100 \
  --batch-size 2 \
  --unlabeled-batch-size 1 \
  --image-size 448 800 \
  --val-video-count 1 \
  --num-workers 0 \
  --arch unetplusplus \
  --encoder-name efficientnet-b4 \
  --encoder-weights none \
  --warmup-epochs 5 \
  --semi-warmup-epochs 15 \
  --unsup-ramp-epochs 30 \
  --save-every 5 \
  --early-stop-patience 30
```

### 6.2 更稳的改动方案

后续可以给 `task3/dataset.py` 增加 pseudo sample 权重或 source 字段：

- 原始真实标签 loss 权重：`1.0`
- 高置信 MedSAM2 伪标签 loss 权重：`0.5`
- 中置信 MedSAM2 伪标签 loss 权重：`0.25`

这样可以降低伪标签噪声对模型的影响。但这需要改训练 loss 逻辑，建议在最小改动方案验证有效后再做。

## 7. 实验设计

### 7.1 Baseline

先固定一个当前最强 Task3 baseline，作为对照：

```text
Run ID: task3-baseline-current
模型: UNet++ + efficientnet-b4
输入尺寸: 448 x 800
训练数据: 原始 labeled + 原始 unlabeled
```

### 7.2 MedSAM2 伪标签实验

建议按增量做三组：

| Run ID | 伪标签来源 | 过滤强度 | 加入比例 | 目的 |
| --- | --- | --- | --- | --- |
| task3-ms2-001 | 真实标签 seed，前后 10 帧 | 严格 | 只用高置信 | 验证是否稳定提升 |
| task3-ms2-002 | 真实标签 seed，前后 20 帧 | 严格 | 只用高置信 | 扩大伪标签规模 |
| task3-ms2-003 | 真实标签 seed + baseline 高置信 seed | 中等 | 高置信 + 部分中置信 | 测试更大覆盖率 |

每组都记录：

- 生成 raw mask 数量。
- 过滤后 mask 数量。
- 被过滤原因统计。
- 加入训练的 foreground / empty mask 比例。
- 内部验证 Dice、HD、ASD、综合 score。
- 预测结果可视化抽查图。

### 7.3 推荐第一轮实验命令

假设已经生成：

```text
outputs/pseudo/task3_medsam2/pseudo_labeled_train
```

第一轮训练：

```bash
CUDA_VISIBLE_DEVICES=3 python task3/train.py \
  --labeled-root outputs/pseudo/task3_medsam2/pseudo_labeled_train \
  --unlabeled-root data/images \
  --output-dir outputs/exp/task3_ms2_001_unetpp_effb4_e100 \
  --epochs 100 \
  --batch-size 2 \
  --unlabeled-batch-size 1 \
  --image-size 448 800 \
  --val-video-count 1 \
  --num-workers 0 \
  --arch unetplusplus \
  --encoder-name efficientnet-b4 \
  --encoder-weights none \
  --warmup-epochs 5 \
  --semi-warmup-epochs 15 \
  --unsup-ramp-epochs 30 \
  --save-every 5 \
  --early-stop-patience 30
```

## 8. 需要新增的脚本职责

### 8.1 `prepare_medsam2_inputs.py`

职责：

- 扫描 `data/reference_data/t3_vid/train` 中的视频帧。
- 按 `video_id` 和 `frame_idx` 分组。
- 复制或软链接帧到 MedSAM2 输入目录。
- 查找已有 `*_label_bin.png` 或 tar 标签。
- 生成 `metadata.json`。

输出：

```text
outputs/pseudo/task3_medsam2/medsam2_inputs/
```

### 8.2 `run_medsam2_propagation.py`

职责：

- 调用 MedSAM2 官方视频推理接口。
- 对每个 video 的 seed frame 做前后窗口传播。
- 保存 raw mask。
- 记录每个 mask 的来源 seed、传播距离、MedSAM2 score。

输出：

```text
outputs/pseudo/task3_medsam2/raw_masks/
outputs/pseudo/task3_medsam2/reports/raw_manifest.json
```

### 8.3 `filter_pseudo_masks.py`

职责：

- 读取 raw mask。
- 做面积、连通域、时序一致性、模型一致性过滤。
- 保存 filtered mask。
- 输出过滤统计 CSV/JSON。

输出：

```text
outputs/pseudo/task3_medsam2/filtered_masks/
outputs/pseudo/task3_medsam2/reports/filter_report.csv
outputs/pseudo/task3_medsam2/reports/filter_summary.json
```

### 8.4 `build_pseudo_labeled_root.py`

职责：

- 构造可被 `task3/dataset.py` 直接读取的 labeled root。
- 包含原始真实标签样本和过滤后的伪标签样本。
- 保持 `image.png` 与 `image_label_bin.png` 同目录同前缀。

输出：

```text
outputs/pseudo/task3_medsam2/pseudo_labeled_train/
```

## 9. 风险与规避

### 9.1 伪标签漂移

风险：MedSAM2 长距离传播后目标漂移，产生错误 mask。

规避：

- 限制传播窗口。
- 做时序一致性过滤。
- 第一轮只使用真实标签 seed。
- 对每个视频抽查可视化结果。

### 9.2 错误伪标签压垮训练

风险：伪标签数量远大于真实标签，模型学到错误边界或错误前景。

规避：

- 第一轮伪标签数量控制在真实标签数量的 1 到 3 倍。
- 优先加入高置信样本。
- 后续增加 pseudo loss weight，而不是把伪标签等权处理。

### 9.3 空 mask 比例失衡

风险：伪标签中空 mask 或近空 mask 太多，模型倾向预测背景。

规避：

- 过滤近空 mask。
- 保持 foreground 样本比例。
- 继续使用现有 foreground-balanced sampling。

### 9.4 评测分布不一致

风险：MedSAM2 伪标签来自训练视频邻近帧，可能提升内部验证但不泛化。

规避：

- 按 video split 验证，不按 frame 随机 split。
- 记录 CodaBench 或外部验证结果。
- 不要用验证视频生成伪标签再参与训练。

## 10. 实施顺序

推荐按以下顺序推进：

1. 安装并跑通 MedSAM2 官方 video inference demo。
2. 写 `prepare_medsam2_inputs.py`，只处理 1 个视频。
3. 用真实标签 seed 传播前后 10 帧。
4. 写可视化检查脚本，人工看 20 到 50 张 overlay。
5. 写 `filter_pseudo_masks.py`，生成严格过滤后的伪标签。
6. 构造 `pseudo_labeled_train`，确认 `task3/dataset.py` 能发现样本。
7. 跑 `task3-ms2-001`。
8. 如果内部验证提升，再扩大传播窗口和伪标签规模。
9. 如果内部验证下降，优先检查伪标签质量，而不是继续调训练超参。

## 11. 成功标准

第一阶段成功标准：

- MedSAM2 能在至少 1 个视频上从真实标签 seed 成功传播 mask。
- 过滤后伪标签可被现有 Task3 dataloader 正确读取。
- 加入伪标签后内部验证综合 score 不低于 baseline。

第二阶段成功标准：

- `task3-ms2-001` 相比 baseline 的 Dice 或综合 score 有稳定提升。
- 可视化抽查中伪标签明显增加了有效前景样本，而不是增加噪声。
- 最终提交结果不低于当前最佳 Task3 checkpoint。

## 12. 推荐结论

Task3 引入 MedSAM2 的最佳切入点是离线伪标签生成，尤其是利用真实标注帧作为 seed，在同视频邻近帧做 mask propagation。不要一开始把 MedSAM2 塞进训练循环，也不要直接用 MedSAM2 输出作为最终提交。先用严格过滤的小规模伪标签验证收益，再逐步扩大覆盖范围。
