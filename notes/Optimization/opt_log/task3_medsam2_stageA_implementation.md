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

## 15. 碎片化问题后续实验

观察：

```text
task3_frozen 的预测 mask 不够连续，存在一小块一小块的碎片。
```

已新增后处理评估脚本：

```text
task3/evaluate_medsam2_postprocess.py
```

评估命令：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  task3/evaluate_medsam2_postprocess.py \
  --run-dir outputs/medsam2_stageA/task3_frozen \
  --device cuda \
  --batch-size 4 \
  --num-workers 0 \
  --thresholds 0.2 0.3 0.4 \
  --min-areas 0 100 400 \
  --keep-components 0 1 2 \
  --close-iters 0 1 \
  --no-fill-holes
```

当前结果：

```text
Raw best:
  threshold = 0.40
  dice      = 0.6510
  hd        = 130.10
  asd       = 18.34
  pred_pos  = 0.1213

Post best:
  threshold       = 0.20
  min_area        = 400
  keep_components = 2
  close_iters     = 1
  fill_holes      = false
  dice            = 0.6599
  hd              = 87.51
  asd             = 17.96
  pred_pos        = 0.1283
```

结论：

```text
后处理能改善 HD，并轻微提升 Dice，但提升幅度有限。
碎片化问题不只是 mask 清理问题，decoder 表达能力也需要加强。
```

已新增 decoder v2：

```text
LightFPNDecoderV2
```

结构变化：

```text
v1: sum fusion + 2 conv blocks
v2: concat fusion + residual conv blocks + stronger refine head
```

v2 forward smoke：

```text
input:  1 x 3 x 512 x 512
output: 1 x 1 x 512 x 512
trainable params: 1,575,809
```

v2 debug training 已通过：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  task3/train_medsam2_encoder.py \
  --labeled-root data/t3_vid/train \
  --output-dir outputs/medsam2_stageA/task3_decoder_v2_debug \
  --epochs 1 \
  --batch-size 2 \
  --image-size 512 512 \
  --decoder-version v2 \
  --max-train-samples 8 \
  --max-val-samples 4 \
  --no-semi \
  --num-workers 0 \
  --no-val-tta \
  --print-freq 1 \
  --save-every 1
```

v2 预测脚本恢复 checkpoint 已通过。

正式 v2 训练完成：

```text
run_dir: outputs/medsam2_stageA/task3_decoder_v2

best_epoch:    6
best_score:    0.4635135266855435
best_val_dice: 0.6645892858505249
best_val_hd:   90.83099365234375
best_val_asd:  17.928552627563477
threshold:     0.20
```

和 v1 对比：

```text
v1 task3_frozen:
  best_epoch:    6
  best_val_dice: 0.6654625535011292
  best_val_hd:   104.75625610351562
  best_val_asd:  17.102874755859375
  threshold:     0.40

v2 task3_decoder_v2:
  best_epoch:    6
  best_val_dice: 0.6645892858505249
  best_val_hd:   90.83099365234375
  best_val_asd:  17.928552627563477
  threshold:     0.20
```

v2 后处理评估：

```text
Raw best:
  threshold = 0.20
  dice      = 0.6420
  hd        = 97.85
  asd       = 18.98
  pred_pos  = 0.0980

Post best:
  threshold       = 0.20
  min_area        = 0
  keep_components = 0
  close_iters     = 1
  fill_holes      = false
  dice            = 0.6421
  hd              = 97.78
  asd             = 18.95
  pred_pos        = 0.0979
```

注意：后处理评估脚本当前未使用 TTA，因此 Dice 和训练日志中的 val Dice 不完全一致，适合用于同脚本下的横向比较。

阶段性判断：

```text
1. v2 增强 decoder 后，Dice 没有超过 v1。
2. v2 的 HD 比 v1 好，但 ASD 略差。
3. v2 后处理收益几乎没有，说明碎片化/不连续问题不能靠简单形态学修复。
4. 继续只加 decoder 容量的收益有限。
```

下一步建议进入 Stage B：

```text
frozen:   MedSAM2 trunk
trainable: MedSAM2 neck + decoder
```

原因：

```text
当前 backbone_fpn 由 neck 产生。
如果 frozen feature 对 Task3 域不够适配，只训练 decoder 很难恢复连续、稳定的目标结构。
neck-only fine-tune 是比 full fine-tune 更稳的下一步。
```

## 17. Stage B neck-only fine-tune 实现

已将训练模式从布尔开关扩展为：

```text
--encoder-train-mode frozen | neck | full
```

模式含义：

```text
frozen:
  trainable: decoder
  frozen:    MedSAM2 trunk + neck

neck:
  trainable: MedSAM2 neck + decoder
  frozen:    MedSAM2 trunk

full:
  trainable: MedSAM2 trunk + neck + decoder
```

新增参数：

```text
--encoder-lr
```

Stage B 第一版选择：

```text
--encoder-train-mode neck
--decoder-version v1
--lr 1e-3
--encoder-lr 1e-5
```

参数冻结检查已通过：

```text
trainable trunk:   0
trainable neck:    369,664
trainable decoder: 394,241
trainable total:   763,905
```

Stage B debug training 已通过：

```bash
CUDA_VISIBLE_DEVICES=2 /home/wuyongji/miniconda3/envs/mvaa/bin/python \
  task3/train_medsam2_encoder.py \
  --labeled-root data/t3_vid/train \
  --output-dir outputs/medsam2_stageB/task3_neck_v1_debug \
  --epochs 1 \
  --batch-size 2 \
  --image-size 512 512 \
  --decoder-version v1 \
  --encoder-train-mode neck \
  --lr 1e-3 \
  --encoder-lr 1e-5 \
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
train_loss: 0.6009
train_dice: 0.2153
val_loss:   0.5096
val_dice:   0.6103
threshold:  0.20
```

Stage B checkpoint prediction smoke 已通过。

## 16. 当前测试网站最优基准

后续实验需要和测试网站上的当前最优提交对比，而不是只看内部 val split。

当前最优提交指标：

```text
task1_ct:
  DSC: 0.8218615271354927
  HD:  5.606846043478218
  ASD: 0.36295904757726943
  num_cases: 30
  missing_cases: 0

task2_tee:
  DSC: 0.7968176708654592
  HD:  18.419442984555477
  ASD: 1.0316021748510564
  num_cases: 20
  missing_cases: 0

task3_vid:
  DSC: 0.7617500534965712
  HD:  91.84735430968526
  ASD: 15.172139087804789
  num_cases: 48
  missing_cases: 0
```

Task3 当前优化目标：

```text
DSC > 0.7617500534965712
HD  < 91.84735430968526
ASD < 15.172139087804789
```

说明：

```text
内部 val split 只能用于开发选择方向。
最终是否有效必须以测试网站 task3_vid 指标为准。
```
