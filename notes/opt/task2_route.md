# Task2 TEE 优化路线

Task2 是 3D TEE 多类别分割。数据有 105 个有标注训练 case 和 20 个验证 case。当前最重要的不是立刻换复杂模型，而是把现有 large 3D UNet 充分训练，并建立稳定的多类别后处理和 ensemble。

## 1. 当前代码状态

已支持：

- MONAI 3D UNet：`small/base/large`。
- 多类别 Dice + CE loss，默认 `num_classes=3`。
- `train-crops` 多 crop 训练。
- 可选 spacing resample。
- sliding-window 验证和推理。
- best 模型按 DSC/HD/ASD 综合分数保存。

未支持但建议开发：

- SegResNet、DynUNet、V-Net。
- 随机 split/k-fold；当前固定取排序最后 `val-count` 个 case。
- class-wise 连通域后处理。
- 概率保存和概率 ensemble。
- per-class 指标记录；当前主要是整体平均。

## 2. 路线优先级

### P0：吃满 large 3D UNet

第一组强 baseline：

```bash
conda run -n mvaa python task2/train.py \
  --data-dir data/reference_data/t2_tee/train \
  --output-dir outputs/opt/task2/t2_unet_large_e200_c2_s42 \
  --epochs 200 \
  --batch-size 1 \
  --roi-size 128 128 128 \
  --train-crops 2 \
  --sw-batch-size 1 \
  --val-count 20 \
  --model-size large \
  --lr 5e-4 \
  --seed 42 \
  --num-workers 0
```

建议优先消融：

| 变量 | 候选值 | 说明 |
| --- | --- | --- |
| `epochs` | 150, 200, 250 | 看验证是否仍在涨 |
| `train-crops` | 1, 2, 4 | crop 越多越慢，但对 3D 目标覆盖有帮助 |
| `lr` | 5e-4, 3e-4, 1e-4 | large 模型稳定性对学习率敏感 |
| `roi-size` | 128^3, 160x160x128 | 视显存决定 |
| `val-interval` | 1, 2 | 训练后期建议更频繁保存最佳点 |

### P1：修正验证体系

当前 Task2 的 `split_train_val` 使用最后 `val-count` 个 case 做验证。如果编号或采集顺序存在偏差，内部验证会误导。

建议新增：

- `--split-seed` 随机划分。
- `--fold-index`/`--num-folds` 做 k-fold。
- 保存每个 val case 的 DSC/HD/ASD，找出拖后腿 case。

在未开发前，至少固定 `val-count=20`，不要反复换验证集挑结果。

### P1：class-wise 后处理

Task2 是多类别分割，后处理要按类别做：

1. 对 class 1 和 class 2 分别提取连通域。
2. 每类保留最大连通域或 top-k 连通域。
3. 删除小于训练集该类别体积分布下限的小组件。
4. 如果后处理过程中出现重叠，用 softmax probability 的 argmax 决定归属。
5. 对预测为空、体积异常小或异常大的 case 打标，必要时回退原始预测。

推荐第一版保守规则：

```text
class 1: keep top-2 components, remove very small components
class 2: keep top-2 components, remove very small components
overlap: choose higher softmax probability
```

### P2：MONAI 架构扩展

需要先改 `task2/model_factory.py`：

- `segresnet`：第一优先级。
- `dynunet`：第二优先级。
- `vnet`：对照实验。

实验设计：

| 实验 | 目的 |
| --- | --- |
| T2-UNet-large | 当前强基线 |
| T2-SegResNet | 看 residual encoder-decoder 是否更稳 |
| T2-DynUNet | 看 nnU-Net 风格结构是否提升 |
| T2-ensemble | 概率平均后 class-wise 后处理 |

### P2：nnU-Net v2

Task2 也可以跑 nnU-Net v2，但优先级低于 Task1。原因是：

- 现有 Task2 baseline 还没充分训练。
- TEE 超声噪声更重，后处理和训练设置收益可能先于完整 nnU-Net。

如果 P0/P1 后仍有资源，再转 nnU-Net v2：

- 数据集设为 3 类。
- 保持原 spacing/header。
- 输出转回 `t2_tee/*.nii.gz + task2_predictions.json`。

## 3. Ensemble 路线

推荐做 softmax probability average：

```text
model A softmax
model B softmax
model C softmax
average softmax
argmax
class-wise postprocess
save NIfTI
```

候选来源：

- 同一模型不同 seed。
- 不同 epoch checkpoint，例如 best score、best DSC、late checkpoint。
- UNet + SegResNet + DynUNet。
- 不同 `train-crops` 或 ROI。

弱模型不要硬加进 ensemble。每个候选必须先在固定内部验证集上证明不是明显拖后腿。

## 4. 风险点

- `batch-size` 默认很大，但 3D 大 ROI 通常实际只能用 1；命令里要显式写 `--batch-size 1`。
- 多类别平均指标可能掩盖某一类完全失败，需要补 per-class 指标。
- 固定最后 20 例验证可能与 leaderboard 分布不同。
- 后处理过强会误删真实小结构，必须用训练集体积统计调参。

## 5. 最近下一步

1. 跑 `large UNet e200 train-crops=2`。
2. 对比 `train-crops=1/2/4`。
3. 写训练集 class 体积统计脚本。
4. 写 class-wise 后处理脚本。
5. 增加随机 split/k-fold。
6. 开发 SegResNet/DynUNet。
