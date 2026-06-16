# 使用预训练 Backbone / Foundation Model 的优化思路

## 1. 核心理解

师兄建议的重点不是单纯把模型做大，而是不要让每个模态都从随机初始化开始学习底层特征。更合理的做法是：

1. 针对每种数据模态选择合适的预训练 backbone 或 foundation model。
2. 把它作为特征提取器，接当前任务需要的分割头或分类头。
3. 在本项目数据上微调，必要时只微调后几层或 decoder，降低小数据过拟合风险。

这个方向对比赛小数据尤其重要，因为预训练模型已经学到了一部分通用图像或医学图像表征，训练时更容易稳定收敛，也更可能提升泛化。

## 2. 当前项目现状

### 2.1 Task1: CT 3D 体数据

当前入口：

- `task1/train.py`：训练参数里有 `--model` 和 `--model-size`。
- `task1/model_factory.py`：`get_model()` 目前只支持 MONAI `UNet`。

当前问题：

- `UNet` 是随机初始化，等于让 CT 模态从零学特征。
- 对标注数据较少的 3D 医学分割任务，这通常不是最优。

适合方向：

- 先把 `UNet` 替换或扩展为 MONAI 的 `SwinUNETR` / `UNETR`。
- 再考虑加载 BTCV、MedicalNet、Med3D、SwinUNETR 相关医学预训练权重。
- 如果预训练权重类别数或输入尺寸不匹配，优先加载 encoder/backbone 权重，decoder/head 随机初始化。

### 2.2 Task2: 3D TEE 超声

当前入口：

- `task2/train.py`：训练参数里有 `--model` 和 `--model-size`。
- `task2/model_factory.py`：`get_model()` 目前也只支持 MONAI `UNet`。

当前问题：

- TEE 是 3D 超声，和 CT 的成像机制不同，直接用 CT 预训练权重不一定最优。
- 但完全随机初始化也不是最优，可以先用 3D 医学影像预训练 backbone 做低成本尝试。

适合方向：

- 第一阶段与 Task1 共用 `SwinUNETR` / `UNETR` 分支，先验证 transformer-style 3D backbone 是否优于当前 `UNet`。
- 第二阶段寻找超声、心脏超声、心脏分割相关预训练权重。
- 如果没有可用权重，可以用未标注 TEE 做自监督预训练，再迁移到有标注训练。

### 2.3 Task3: 手术视频帧 / RGB 图像

当前入口：

- `task3/train.py`：已经支持 `--arch`、`--encoder-name`、`--encoder-weights`。
- `task3/model_factory.py`：使用 `segmentation_models_pytorch` 创建 `Unet`、`UnetPlusPlus`、`FPN`、`DeepLabV3Plus`。

当前最直接的问题：

- 默认 `--encoder-weights none` 时，SMP encoder 也是随机初始化。
- 对 RGB 图像任务，ImageNet 预训练是最容易落地的 baseline。

低成本优化：

```bash
conda run -n mvaa python task3/train.py \
  --labeled-root data/t3_vid/train \
  --unlabeled-root data/images \
  --output-dir outputs/exp/task3_unetpp_effb4_imagenet \
  --arch unetplusplus \
  --encoder-name efficientnet-b4 \
  --encoder-weights imagenet
```

这一步就是把 EfficientNet-B4 从随机初始化切换为 ImageNet 预训练特征提取器，分割头仍在本项目上训练。

## 3. 推荐实施优先级

### Priority 1: 先跑 Task3 ImageNet 预训练

原因：

- 代码已经支持，不需要改模型结构。
- RGB 图像使用 ImageNet encoder 是成熟做法。
- 可以快速和当前 `--encoder-weights none` 的结果对比。

建议实验：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n mvaa python task3/train.py \
  --labeled-root data/t3_vid/train \
  --unlabeled-root data/images \
  --output-dir outputs/exp/task3_unetpp_effb4_imagenet_e100_bs2 \
  --epochs 100 \
  --batch-size 2 \
  --unlabeled-batch-size 1 \
  --image-size 448 800 \
  --val-video-count 1 \
  --num-workers 0 \
  --arch unetplusplus \
  --encoder-name efficientnet-b4 \
  --encoder-weights imagenet
```

对比对象：

- 当前 `efficientnet-b4 + encoder_weights none`。
- 如果显存不足，可试 `efficientnet-b3`、`resnet34` 或减小 batch size。

### Priority 2: 给 Task1/Task2 增加 3D 预训练模型接口

建议先在 `task1/model_factory.py` 和 `task2/model_factory.py` 增加新分支，例如：

```python
from monai.networks.nets import SwinUNETR, UNETR

if name == "swinunetr":
    return SwinUNETR(
        img_size=roi_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=48,
        use_checkpoint=True,
    )

if name == "unetr":
    return UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=roi_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name="instance",
        res_block=True,
    )
```

注意点：

- 当前 `get_model()` 没有 `roi_size` 参数，`SwinUNETR` / `UNETR` 需要知道输入 patch size，因此需要从 `train.py` 把 `args.roi_size` 传给 `get_model()`。
- 预测脚本 `generate_task1_predictions.py`、`generate_task2_predictions.py` 也要能从 checkpoint 的训练参数恢复模型类型和 ROI size。
- 第一步可以先随机初始化 `SwinUNETR` / `UNETR`，验证训练流程和显存。
- 第二步再增加 `--pretrained-path` 加载医学预训练权重。

建议训练命令形态：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n mvaa python task1/train.py \
  --model swinunetr \
  --model-size base \
  --roi-size 96 96 96 \
  --output-dir outputs/exp/task1_swinunetr_base
```

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n mvaa python task2/train.py \
  --model swinunetr \
  --model-size base \
  --roi-size 96 96 96 \
  --output-dir outputs/exp/task2_swinunetr_base
```

具体 `roi-size` 需要以现有训练脚本支持的参数和显存为准。

### Priority 3: 加载医学预训练权重

新增参数建议：

```text
--pretrained-path /path/to/pretrained_weights.pt
--pretrained-strict false
--freeze-backbone-epochs 0
```

加载策略：

1. 优先只加载 shape 匹配的权重。
2. 跳过输出 head、分类层、类别数不一致的层。
3. 打印成功加载和跳过的 key 数量，方便确认是否真的加载到了 backbone。
4. 若训练不稳定，可先冻结 encoder/backbone 5 到 10 个 epoch，再全量微调。

伪代码：

```python
def load_pretrained_partial(model, path):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model_state = model.state_dict()

    matched = {
        k: v
        for k, v in state.items()
        if k in model_state and model_state[k].shape == v.shape
    }
    model_state.update(matched)
    model.load_state_dict(model_state, strict=False)
    print(f"Loaded {len(matched)} pretrained tensors from {path}")
```

## 4. 各任务可选 Backbone

| 任务 | 数据模态 | 当前模型 | 第一阶段可试 | 更进一步 |
| --- | --- | --- | --- | --- |
| Task1 | CT 3D NIfTI | MONAI UNet | SwinUNETR / UNETR | BTCV/SwinUNETR/MedicalNet/Med3D 预训练 |
| Task2 | TEE 3D 超声 | MONAI UNet | SwinUNETR / UNETR | 超声或心脏相关预训练，自监督预训练 |
| Task3 | RGB 视频帧 | SMP UNet++ | `encoder_weights=imagenet` | MedSAM2 伪标签、SAM/MedSAM 特征或更强 encoder |

## 5. 实验记录建议

每个实验至少记录：

- 模型名、backbone、是否预训练。
- 预训练权重来源和加载方式。
- ROI size / image size / batch size。
- 学习率、epoch、是否冻结 backbone。
- 验证集 Dice、HD、ASD 或当前项目使用的综合 score。
- 和对应随机初始化 baseline 的差值。

推荐命名：

```text
outputs/exp/task1_swinunetr_btcv_e100_roi96
outputs/exp/task2_swinunetr_med3d_e100_roi96
outputs/exp/task3_unetpp_effb4_imagenet_e100_bs2
```

## 6. 风险与注意事项

1. 预训练权重不一定和当前数据域一致，尤其 Task2 超声和 CT 差异明显，需要用验证集说话。
2. 3D transformer backbone 显存开销大，可能需要减小 ROI size、batch size，或开启 gradient checkpointing。
3. 只改模型不等于一定涨分，数据划分、伪标签质量、后处理和阈值搜索仍然重要。
4. 加载权重时必须确认日志里真的加载到了大量 backbone 参数，否则可能只是随机初始化。
5. Task1/Task2 修改模型后，训练脚本和预测脚本必须保持一致，否则提交推理时无法正确恢复模型。

## 7. 当前最建议的下一步

最稳妥顺序：

1. 先跑 Task3 `efficientnet-b4 + imagenet`，与当前 `none` baseline 对比。
2. 给 Task1/Task2 的 `model_factory.py` 增加 `SwinUNETR` 分支，并让训练/预测脚本传入 `roi_size`。
3. 跑 Task1/Task2 的随机初始化 `SwinUNETR` smoke test，确认显存和流程。
4. 再接入 `--pretrained-path`，尝试加载医学 3D 预训练权重。
5. 若 Task2 没有合适现成权重，再考虑用未标注 TEE 做自监督预训练。

一句话总结：Task3 先打开 ImageNet 预训练，Task1/Task2 先扩展 3D backbone 接口，再逐步加载医学预训练权重；不要一次性重写全流程，先做能快速验证收益的实验。
