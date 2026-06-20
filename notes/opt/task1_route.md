# Task1 CT 优化路线

Task1 是 3D Cardiac CT 二分类分割。数据特点是有标注 case 很少、无标注很多、验证集 30 个 CT。主线应围绕 3D 医学分割强基线、谨慎半监督、nnU-Net v2、连通域后处理和 ensemble。

## 1. 当前代码状态

已支持：

- MONAI 3D UNet：`small/base/large`。
- 半监督 EMA teacher：`unsup_warmup_epochs`、`unsup_rampup_epochs`、`unsup_weight`、`pseudo_threshold`、`unsup_ratio`。
- CT 强度裁剪到 `[-1000, 1000]` 后归一化。
- 可选 spacing resample。
- sliding-window 验证和推理。

未支持但建议开发：

- SegResNet、DynUNet、V-Net。
- 保存概率图或 logits，用于 ensemble 和阈值/后处理调参。
- 训练集体积统计和连通域后处理脚本。
- nnU-Net v2 数据转换和提交格式转换脚本。

## 2. 路线优先级

### P0：先建立可靠 baseline

先跑 supervised-only 和保守半监督对照，判断无标注是否真的有效。

supervised-only：

```bash
conda run -n mvaa python task1/train.py \
  --data-root data/reference_data/t1_ct \
  --output-dir outputs/opt/task1/t1_unet_large_sup_e120_s42 \
  --epochs 120 \
  --batch-size 1 \
  --roi-size 128 128 128 \
  --train-crops 2 \
  --sw-batch-size 1 \
  --model-size large \
  --unsup-warmup-epochs 999 \
  --unsup-weight 0 \
  --val-count 3 \
  --split-seed 42 \
  --num-workers 0
```

保守半监督：

```bash
conda run -n mvaa python task1/train.py \
  --data-root data/reference_data/t1_ct \
  --output-dir outputs/opt/task1/t1_unet_large_semi_e160_s42 \
  --epochs 160 \
  --batch-size 1 \
  --roi-size 128 128 128 \
  --train-crops 2 \
  --sw-batch-size 1 \
  --model-size large \
  --unsup-warmup-epochs 40 \
  --unsup-rampup-epochs 40 \
  --unsup-weight 0.3 \
  --unsup-ratio 1 \
  --pseudo-threshold 0.8 \
  --val-count 3 \
  --split-seed 42 \
  --num-workers 0
```

判断标准：

- 如果半监督的 `val_dsc` 提升，但 HD/ASD 明显变差，要优先检查小孤岛和过分割。
- 如果 `pseudo_valid_ratio` 长期过低，说明阈值太高或 teacher 不可靠。
- 如果半监督开启后 DSC 下降，先降 `unsup_weight` 和 `unsup_ratio`，不要直接加 epoch。

### P1：多 split 稳定性

Task1 只有 27 个有标注 case，单 split 不可信。至少跑：

| 实验 | `split-seed` | 目的 |
| --- | --- | --- |
| T1-S42 | 42 | 基准 |
| T1-S3407 | 3407 | 看验证波动 |
| T1-S2026 | 2026 | 看验证波动 |

优先比较同一配置的多 seed 平均和最差表现，不要只看某个 seed 的 best。

### P1：nnU-Net v2

Task1 非常值得单独跑 nnU-Net v2。它的自动 spacing、patch、normalization、fold ensemble 和后处理经验适合 3D CT。

数据目标结构：

```text
nnUNet_raw/
  Dataset101_MVAA_Task1_CT/
    imagesTr/
      case_0001_0000.nii.gz
    labelsTr/
      case_0001.nii.gz
    imagesTs/
      val_0001_0000.nii.gz
    dataset.json
```

计划：

1. 写转换脚本，把 `data/reference_data/t1_ct/train/labeled` 转为 nnU-Net 格式。
2. 跑 `3d_fullres` fold 0。
3. 如果 fold 0 明显强，再跑 5-fold 或至少多 fold ensemble。
4. 写转换脚本，把 nnU-Net 输出转成 `t1_ct/*.nii.gz + task1_predictions.json`。

### P2：MONAI 架构扩展

需要先改 `task1/model_factory.py`：

- `segresnet`：优先级最高，通常比普通 UNet 更稳。
- `dynunet`：优先级第二，可借鉴 nnU-Net 的 kernel/stride 设计。
- `vnet`：只作为对照，不建议主押。

开发完成后，实验应保持同一 split、同一 ROI、同一训练轮数，避免架构对比被训练设置污染。

## 3. 后处理路线

Task1 是二分类器官/结构分割，后处理应从保守开始：

1. 保留最大连通域。
2. 删除小于训练集前景体积 1%-3% 的小组件。
3. 填洞。
4. 统计训练集前景体积分布，标记异常小/异常大的预测。
5. 对异常预测回退到另一个 checkpoint 或 ensemble 结果。

不要直接用过强体积阈值。Task1 有标注少，体积分布估计不稳定。

## 4. Ensemble 路线

推荐顺序：

1. 同架构不同 split/seed 的 probability average。
2. supervised-only 与保守半监督模型 average。
3. nnU-Net 与 MONAI 模型 average。
4. average 后再做连通域后处理。

因此需要新增推理能力：保存每个 case 的 foreground probability，而不是只保存 argmax mask。

## 5. 风险点

- 内部验证只有 3 个 case 时，best epoch 可能是噪声。
- 半监督可能把错误伪标签放大，尤其是 `unsup_weight=1.0` 时。
- spacing resample 可能显著增加显存和内存，要先小规模验证。
- nnU-Net 输出需要保持原 affine/header 和提交命名，否则分数会异常。

## 6. 最近下一步

1. 跑 `large supervised-only` 与 `large conservative semi` 对照。
2. 写 Task1 前景体积统计脚本。
3. 写 Task1 连通域后处理脚本。
4. 写 nnU-Net v2 转换脚本。
5. 再开发 SegResNet/DynUNet。
