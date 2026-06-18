# Task3 MedSAM2 Stage A Implementation Plan

本文档记录 Task3 第一阶段 MedSAM2 encoder backbone 的具体实现细节。

主策略文档：

```text
notes/Optimization/medsam2_peft_strategy.md
```

当前阶段目标：

```text
Frozen MedSAM2 image_encoder + LightFPNDecoder
```

第一阶段只验证 MedSAM2 encoder 特征是否能支撑 Task3 自动二值分割。暂时不做 semi-supervised teacher、不做 LoRA、不做 adapter、不扩展 Task1/Task2。

## 1. Probe 结果

已新增并运行：

```text
scripts/probe_medsam2_encoder_features.py
```

运行命令：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  scripts/probe_medsam2_encoder_features.py \
  --device cuda \
  --gpu-id 0
```

AMP 也已通过：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  scripts/probe_medsam2_encoder_features.py \
  --device cuda \
  --gpu-id 0 \
  --amp
```

关键输出：

```text
Input:  B x 3 x 512 x 512
Output: image_encoder(x)

out["backbone_fpn"]:
  level 0: B x 256 x 128 x 128, stride 4
  level 1: B x 256 x 64 x 64,   stride 8
  level 2: B x 256 x 32 x 32,   stride 16

out["vision_features"]:
  B x 256 x 32 x 32

image_encoder params:
  27,219,136

CUDA peak memory for single forward:
  about 0.23 GiB
```

因此后续 decoder 的 feature contract 固定为：

```python
features = image_encoder(x)["backbone_fpn"]
```

其中：

```text
features[0] = stride 4,  channels 256
features[1] = stride 8,  channels 256
features[2] = stride 16, channels 256
```

## 2. 新增代码结构

建议新增：

```text
medsam2_backbone/
  __init__.py
  build_encoder.py
  encoder.py
  decoder.py
  task3_model.py

task3/
  train_medsam2_encoder.py
  generate_task3_medsam2_encoder_predictions.py
```

第一版只服务 Task3。Task1/Task2 的 2.5D 模型后置。

## 3. Encoder 构建

`medsam2_backbone/build_encoder.py` 只负责加载 MedSAM2 model 并返回 `image_encoder`。

默认参数：

```text
cfg:  external/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml
ckpt: external/MedSAM2/checkpoints/MedSAM2_latest.pt
```

Hydra 配置名要使用：

```text
configs/sam2.1_hiera_t512.yaml
```

而不是裸文件名 `sam2.1_hiera_t512.yaml`。

实现要点：

```python
MEDSAM2_ROOT = REPO_ROOT / "external" / "MedSAM2"
sys.path.insert(0, str(MEDSAM2_ROOT))

from sam2.build_sam import build_sam2

model = build_sam2(
    config_file="configs/sam2.1_hiera_t512.yaml",
    ckpt_path=str(ckpt_path),
    device=device,
    mode="eval",
    apply_postprocessing=False,
)
return model.image_encoder
```

`apply_postprocessing=False` 即可，因为这里只使用 image encoder，不使用 SAM mask decoder 后处理。

## 4. Encoder Wrapper

`medsam2_backbone/encoder.py` 封装成 `MedSAM2ImageEncoder`。

职责：

```text
1. 保存 image_encoder。
2. 管理 freeze/eval 状态。
3. forward 返回 backbone_fpn。
```

第一阶段默认：

```text
freeze_encoder = true
image_encoder.eval()
requires_grad = false
```

forward：

```python
def forward(self, x):
    if self.freeze_encoder:
        with torch.no_grad():
            out = self.image_encoder(x)
    else:
        out = self.image_encoder(x)
    return out["backbone_fpn"]
```

注意：Task3 dataset 已经做 ImageNet normalization，因此 wrapper 里第一版不重复 normalize。

## 5. LightFPNDecoder

`medsam2_backbone/decoder.py` 实现第一版轻量 decoder。

输入：

```text
features[0]: B x 256 x H/4  x W/4
features[1]: B x 256 x H/8  x W/8
features[2]: B x 256 x H/16 x W/16
```

输出：

```text
B x 1 x H x W
```

推荐第一版结构：

```text
per-level projection:
  1x1 conv: 256 -> decoder_channels

fusion:
  upsample all levels to stride-4 resolution
  sum fusion

refine:
  3x3 conv + GroupNorm + SiLU
  3x3 conv + GroupNorm + SiLU
  1x1 conv -> out_channels

final:
  bilinear upsample to input H x W
```

默认参数：

```text
decoder_channels = 128
out_channels = 1
norm = GroupNorm(num_groups=8)
activation = SiLU
upsample = bilinear
```

选择 sum fusion 的原因：

```text
1. 参数少。
2. 训练样本只有 180 帧，先避免 decoder 过大。
3. 三层 feature 都已是 256 channel，sum 前只需投影到同一 decoder channel。
```

后续如果 Stage A 欠拟合，再尝试 concat fusion：

```text
concat -> 3 * decoder_channels -> decoder_channels
```

## 6. Task3MedSAM2EncoderSeg

`medsam2_backbone/task3_model.py` 组合 encoder 和 decoder。

接口：

```python
class Task3MedSAM2EncoderSeg(nn.Module):
    def __init__(
        self,
        encoder_cfg,
        encoder_ckpt,
        decoder_channels=128,
        freeze_encoder=True,
    ):
        ...

    def forward(self, x):
        features = self.encoder(x)
        logits = self.decoder(features, output_size=x.shape[-2:])
        return logits
```

输出必须和现有 Task3 loss 兼容：

```text
logits: B x 1 x H x W
label:  B x 1 x H x W
```

第一版 H/W 固定为 512，但实现上仍保留 `output_size=x.shape[-2:]`，方便后续实验其他尺寸。

## 7. 训练脚本改造

复制：

```text
task3/train.py
```

为：

```text
task3/train_medsam2_encoder.py
```

尽量复用现有逻辑：

```text
discover_samples
split_train_val_by_video
foreground-balanced sampler
Dice/Focal loss
threshold search
HD/ASD
history.csv
checkpoint
TTA validation
```

主要改动：

```text
1. 不再使用 task3/model_factory.py 的 get_model。
2. 改用 Task3MedSAM2EncoderSeg。
3. 默认 image_size 改为 512 512。
4. 默认 batch_size 先设 4 或 8。
5. 第一版关闭 semi-supervised。
6. optimizer 只优化 decoder 参数。
7. checkpoint 增加 model_type = medsam2_encoder。
```

建议新增参数：

```text
--encoder-cfg
--encoder-ckpt
--decoder-channels
--freeze-encoder / --no-freeze-encoder
--no-semi
```

Stage A 默认：

```text
--freeze-encoder
--no-semi
```

关闭 semi-supervised 的原因：

```text
Stage A 目的是隔离验证 frozen MedSAM2 feature 是否有用。
如果一开始加入无标注 teacher，会同时引入 teacher 质量、pseudo threshold、strong augmentation 等变量，不利于判断 encoder backbone 价值。
```

## 8. Optimizer

Stage A：

```python
optimizer = torch.optim.AdamW(
    model.decoder.parameters(),
    lr=1e-3,
    weight_decay=1e-5,
)
```

不要把 frozen encoder 参数传给 optimizer。

后续 Stage B 再启用参数组：

```python
optimizer = torch.optim.AdamW(
    [
        {"params": model.decoder.parameters(), "lr": 1e-3},
        {"params": model.encoder.image_encoder.neck.parameters(), "lr": 1e-5},
    ],
    weight_decay=1e-5,
)
```

Stage B 不在第一版实现范围内，但类结构要避免写死。

## 9. Checkpoint

checkpoint 至少保存：

```python
{
    "model_type": "medsam2_encoder",
    "model_state": model.state_dict(),
    "args": vars(args),
    "val_metrics": metrics,
    "epoch": epoch,
}
```

`args` 中要包含：

```text
image_size
encoder_cfg
encoder_ckpt
freeze_encoder
decoder_channels
loss_type
use_imagenet_norm
target_label
```

预测脚本从 `val_metrics["val_threshold"]` 恢复最佳 threshold。

## 10. 推理脚本

复制：

```text
task3/generate_task3_predictions.py
```

为：

```text
task3/generate_task3_medsam2_encoder_predictions.py
```

主要改动：

```text
1. 根据 checkpoint args 构建 Task3MedSAM2EncoderSeg。
2. 加载 model_state。
3. 使用 checkpoint 中的 image_size 和 val_threshold。
4. 输出格式完全保持现有 Task3 submission 格式。
```

输出仍然是：

```text
t3_vid/<video_folder>/<stem>_label_bin.png
task3_predictions.json
```

不要改变打包脚本。

## 11. 验证顺序

实施后按以下顺序验证：

```text
1. py_compile 新增模块和脚本。
2. model forward smoke:
   input  = 1 x 3 x 512 x 512
   output = 1 x 1 x 512 x 512
3. dataloader smoke:
   取少量 labeled samples，跑 forward + loss.backward。
4. 1 epoch debug train。
5. 正式 Stage A 训练。
6. 生成 Task3 predictions，检查 JSON 和 mask 路径。
```

debug train 命令：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  task3/train_medsam2_encoder.py \
  --labeled-root data/t3_vid/train \
  --output-dir outputs/medsam2_stageA/task3_debug \
  --epochs 1 \
  --batch-size 2 \
  --image-size 512 512 \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --no-semi \
  --num-workers 0
```

正式 Stage A 命令：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  task3/train_medsam2_encoder.py \
  --labeled-root data/t3_vid/train \
  --output-dir outputs/medsam2_stageA/task3_frozen \
  --epochs 50 \
  --batch-size 8 \
  --image-size 512 512 \
  --lr 1e-3 \
  --loss-type dice_focal \
  --no-semi \
  --num-workers 4
```

如果 GPU 2 被占用，改用当前空闲 GPU：

```bash
CUDA_VISIBLE_DEVICES=6 ...
```

## 12. 成功标准

Stage A 要回答：

```text
1. train loss 是否稳定下降？
2. val Dice 是否接近当前 Task3 baseline？
3. HD/ASD 是否没有明显恶化？
4. 预测 mask 是否不是全空或全前景？
```

进入 Stage B 的条件：

```text
Stage A val Dice 接近 baseline，或 HD/ASD 明显更好。
```

如果 Stage A 明显失败，排查顺序：

```text
1. label resize 是否和 image resize 对齐。
2. ImageNet normalization 是否重复或遗漏。
3. decoder 是否太弱。
4. threshold search 是否合理。
5. frozen encoder 是否不足，需要 neck fine-tune。
6. Task3 手术视频域和 MedSAM2 预训练域差异是否过大。
```

## 13. 当前实现边界

第一版明确不做：

```text
Task1/Task2
semi-supervised pseudo label
MedSAM2 prompt refine
LoRA
adapter
full fine-tune
多尺度输入/TTA 扩展
```

第一版必须做：

```text
Task3 supervised train
frozen MedSAM2 image_encoder
LightFPNDecoder
checkpoint
prediction script
submission-compatible output
baseline metrics comparison
```

## 14. 实现记录

已新增代码：

```text
medsam2_backbone/
  __init__.py
  build_encoder.py
  encoder.py
  decoder.py
  task3_model.py

task3/
  train_medsam2_encoder.py
  generate_task3_medsam2_encoder_predictions.py
```

当前模型：

```text
Task3MedSAM2EncoderSeg
  MedSAM2 image_encoder frozen
  LightFPNDecoder trainable
```

forward smoke 已通过：

```text
input:  1 x 3 x 512 x 512
output: 1 x 1 x 512 x 512
trainable params: 394,241
```

1 epoch debug training 已通过：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  task3/train_medsam2_encoder.py \
  --labeled-root data/t3_vid/train \
  --output-dir outputs/medsam2_stageA/task3_debug \
  --epochs 1 \
  --batch-size 2 \
  --image-size 512 512 \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --no-semi \
  --num-workers 0 \
  --no-val-tta \
  --print-freq 1 \
  --save-every 1
```

debug 结果：

```text
train_loss: 0.6020
train_dice: 0.2095
val_loss:   0.5514
val_dice:   0.5424
val_hd:     252.6261
val_asd:    47.9643
threshold:  0.20
```

注意：以上结果只用于验证训练闭环，不代表正式 Stage A 质量。

预测 smoke 已通过：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  task3/generate_task3_medsam2_encoder_predictions.py \
  --ckpt-path outputs/medsam2_stageA/task3_debug/checkpoints/best.pt \
  --data-dir data/t3_vid/val/images \
  --submission-task-dir outputs/medsam2_stageA/task3_debug_submission/t3_vid \
  --video-folders REC_20250428_101313_571A \
  --no-tta
```

输出已确认：

```text
outputs/medsam2_stageA/task3_debug/checkpoints/best.pt
outputs/medsam2_stageA/task3_debug_submission/t3_vid/task3_predictions.json
outputs/medsam2_stageA/task3_debug_submission/t3_vid/<video_folder>/*_label_bin.png
```
