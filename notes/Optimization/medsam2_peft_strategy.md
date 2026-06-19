# MedSAM2 Encoder Backbone Implementation Guide

本文档是项目落地指导文档。当前主线不再使用 MedSAM2 prompt refine / pseudo-label teacher，而是使用：

```text
MedSAM2 image encoder + custom segmentation decoder
```

目标：把 MedSAM2 改造成一个 **无 prompt 自动分割模型的预训练 backbone**。

## 1. 总体框架

新架构分成四层：

```text
Input Adapter
  -> MedSAM2 Image Encoder
  -> Segmentation Decoder
  -> Task Head
```

模块职责：

```text
Input Adapter:
  把不同任务输入整理成 encoder 可接受的 3-channel 2D image。
  Task3 是 RGB frame，可直接使用。
  Task1/Task2 后续用 2.5D slices: [z-1, z, z+1]。

MedSAM2 Image Encoder:
  只负责提取预训练视觉特征，输出 backbone_fpn 多尺度特征。
  第一阶段冻结；后续可训练 neck 或加入 adapter/LoRA。

Segmentation Decoder:
  读取 backbone_fpn，做上采样和多尺度融合。
  第一版使用 LightFPNDecoder，不做复杂 UNet。

Task Head:
  根据任务输出类别数。
  Task3 输出 1-channel binary logits。
  Task1 后续输出 2-class logits。
  Task2 后续输出 3-class logits。
```

整体数据流：

```text
Task3 frame / Task1-2 2.5D slice
  -> input adapter / normalization
  -> MedSAM2 image_encoder
  -> backbone_fpn features
  -> LightFPNDecoder
  -> task logits
  -> loss / metrics / submission mask
```

保留：

```text
MedSAM2 image_encoder
```

舍弃：

```text
sam_prompt_encoder
sam_mask_decoder
memory attention propagation
prompt-based inference
```

第一阶段只做 Task3，目标是先验证 MedSAM2 encoder 特征是否能提升或接近当前 Task3 baseline。Task1/Task2 不直接输入 3D volume，后续统一走 2.5D prototype。

## 2. 权重和路径

### 2.1 已有本地权重

当前项目已有：

```text
external/MedSAM2/checkpoints/MedSAM2_latest.pt
```

这是第一阶段 Task3 默认使用的权重。

默认配置文件：

```text
external/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml
```

### 2.2 官方下载地址

官方仓库：

```text
https://github.com/bowang-lab/MedSAM2
```

官方模型页：

```text
https://huggingface.co/wanglab/MedSAM2/tree/main
```

推荐权重：

| 用途 | 文件 | 下载 URL | 存放路径 |
| --- | --- | --- | --- |
| 默认主线 | `MedSAM2_latest.pt` | `https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt` | `external/MedSAM2/checkpoints/MedSAM2_latest.pt` |
| Task2 超声可选 | `MedSAM2_US_Heart.pt` | `https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_US_Heart.pt` | `external/MedSAM2/checkpoints/MedSAM2_US_Heart.pt` |
| Task1 CT 可选 | `MedSAM2_CTLesion.pt` | `https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_CTLesion.pt` | `external/MedSAM2/checkpoints/MedSAM2_CTLesion.pt` |
| 原始 2024 base | `MedSAM2_2411.pt` | `https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_2411.pt` | `external/MedSAM2/checkpoints/MedSAM2_2411.pt` |

官方脚本也可下载全部权重：

```bash
cd external/MedSAM2
bash download.sh
```

第一阶段不需要下载全部权重；已有 `MedSAM2_latest.pt` 即可。

## 3. 推荐新增代码结构

```text
medsam2_backbone/
  __init__.py
  build_encoder.py          # 加载 MedSAM2 model，只返回 image_encoder
  encoder.py                # MedSAM2Encoder wrapper: forward -> backbone_fpn
  decoder.py                # LightFPNDecoder: backbone_fpn -> fused logits
  heads.py                  # BinaryHead / MultiClassHead，可选；简单时可并入 decoder
  task3_model.py            # Task3MedSAM2EncoderSeg
  slice_model.py            # 后续 Task1/Task2 2.5D slice segmentation model
  adapters.py               # 后续 input/feature adapter, LoRA hooks

scripts/
  probe_medsam2_encoder_features.py

task3/
  train_medsam2_encoder.py
  generate_task3_medsam2_encoder_predictions.py
```

Task1/Task2 后置：

```text
task1/train_medsam2_25d.py
task1/generate_task1_medsam2_25d.py

task2/train_medsam2_25d.py
task2/generate_task2_medsam2_25d.py
```

## 4. 第一阶段：Probe Encoder

先不要训练。先验证 encoder 能正常加载和输出多尺度特征。

新增：

```text
scripts/probe_medsam2_encoder_features.py
```

功能：

```text
1. 加载 cfg: external/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml
2. 加载 ckpt: external/MedSAM2/checkpoints/MedSAM2_latest.pt
3. 构造或读取一张 Task3 RGB frame
4. resize 到 512 x 512
5. 使用 ImageNet normalization
6. forward image_encoder
7. 打印 backbone_fpn 每层 shape
8. 打印参数量和显存占用
```

通过标准：

```text
checkpoint 正常加载
backbone_fpn 非空
feature map shape 可用于 decoder
单卡显存可接受
```

## 5. Task3 模型落地

### 5.1 模型结构

```text
Task3 RGB frame
  -> MedSAM2 image_encoder
  -> LightFPNDecoder
  -> 1-channel logits
```

输出必须兼容现有 Task3 loss：

```text
logits: B x 1 x H x W
label:  B x 1 x H x W
```

### 5.2 LightFPNDecoder

第一版 decoder 保持简单：

```text
for each feature in backbone_fpn:
  1x1 conv -> decoder_channels
  upsample to highest feature resolution
sum or concat
3x3 conv + norm + activation
1x1 conv -> output classes
upsample to input image size
```

Task3：

```text
output classes = 1
```

Task1 2.5D：

```text
output classes = 2
```

Task2 2.5D：

```text
output classes = 3
```

## 6. Task3 训练计划

### Stage A：冻结 encoder，只训练 decoder

```text
frozen: MedSAM2 image_encoder
trainable: LightFPNDecoder
```

目的：判断 MedSAM2 frozen features 是否有价值。

建议参数：

```text
image_size: 512 512
loss: dice_focal
lr: 1e-3
epochs: 30-50
batch_size: 按显存设置
encoder_ckpt: external/MedSAM2/checkpoints/MedSAM2_latest.pt
encoder_cfg: external/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml
```

验证对象：

```text
当前 Task3 baseline: unetplusplus + efficientnet-b4
```

继续条件：

```text
Stage A val DSC 接近 baseline，且 HD/ASD 不明显恶化。
```

### Stage B：训练 neck + decoder

```text
frozen: image_encoder.trunk
trainable: image_encoder.neck + LightFPNDecoder
```

建议学习率：

```text
lr_decoder = 1e-3
lr_neck = 1e-5 或 3e-5
```

继续条件：

```text
超过 baseline，或至少 HD/ASD 明显改善。
```

### Stage C：Adapter / LoRA

后置，不作为第一阶段。

优先级：

```text
1. feature adapter + decoder
2. input adapter + decoder
3. encoder LoRA + decoder
4. full fine-tune
```

不建议一开始 full fine-tune，因为 Task3 labeled frame 少，容易过拟合。

## 7. Task1 落地方案

Task1 是 3D CT，MedSAM2 image encoder 是 2D。后续优先做 2.5D：

```text
input = [slice z-1, slice z, slice z+1]
target = label of center slice z
model = MedSAM2 image_encoder + LightFPNDecoder
output = 2-class logits for center slice
```

推理：

```text
逐 slice 预测 center slice
stack predictions -> 3D mask
resize / restore affine -> save NIfTI
```

第一阶段不做 Task1，等 Task3 验证有效后再做。

## 8. Task2 落地方案

Task2 是 3D TEE，后续也优先做 2.5D：

```text
input = [slice z-1, slice z, slice z+1]
target = label of center slice z
output = 3-class logits
```

可选权重：

```text
external/MedSAM2/checkpoints/MedSAM2_US_Heart.pt
```

Task2 需要先确认 class 1 / class 2 的官方解剖含义。当前代码只能确认：

```text
0 = background
1 = foreground class 1
2 = foreground class 2
```

## 9. Normalization 和尺寸

第一阶段 Task3 使用：

```text
image_size = 512 x 512
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

推理保存时必须 resize 回原图尺寸，保持现有提交格式：

```text
t3_vid/<video_folder>/*_label_bin.png
```

Task1/Task2 2.5D 需要先把医学强度映射到 `[0, 1]`，再扩展/归一化到 encoder 输入格式。

## 10. 验证指标

保持现有项目指标：

```text
DSC
HD
ASD
```

Task3 继续使用 threshold search：

```text
0.2, 0.25, 0.3, ..., 0.65
```

必须输出和 baseline 的对比：

```text
baseline val metrics
MedSAM2 encoder Stage A val metrics
MedSAM2 encoder Stage B val metrics
```

当前测试网站最优提交指标需要作为最终对比基准：

```text
Task1 DSC 0.8218615271 | HD 5.6068460435  | ASD 0.3629590476
Task2 DSC 0.7968176709 | HD 18.4194429846 | ASD 1.0316021749
Task3 DSC 0.7617500535 | HD 91.8473543097 | ASD 15.1721390878
```

Task3 后续优化目标：

```text
DSC > 0.7617500535
HD  < 91.8473543097
ASD < 15.1721390878
```

## 11. 最小落地顺序

```text
1. 写 probe_medsam2_encoder_features.py
2. 确认 MedSAM2 image_encoder 输出 shape 和 feature 顺序
3. 实现 medsam2_backbone/build_encoder.py，只负责加载权重和返回 image_encoder
4. 实现 medsam2_backbone/encoder.py，统一 normalization 和 forward 输出
5. 实现 medsam2_backbone/decoder.py，先做 LightFPNDecoder
6. 实现 medsam2_backbone/task3_model.py，把 encoder + decoder 组合成 Bx1xHxW logits
7. 复制 task3/train.py 为 task3/train_medsam2_encoder.py，并替换 model_factory
8. 跑 smoke training，确认 loss 能下降、输出尺寸正确
9. 跑 Task3 internal val，保存 best checkpoint 和 metrics
10. 和当前 Task3 baseline 对比，决定是否进入 Stage B
```

当前 probe 已完成，MedSAM2 image encoder 在 Task3 RGB frame 上输出：

```text
backbone_fpn[0]: B x 256 x H/4  x W/4
backbone_fpn[1]: B x 256 x H/8  x W/8
backbone_fpn[2]: B x 256 x H/16 x W/16
```

Task3 Stage A 的详细实现计划记录在：

```text
notes/Optimization/opt_log/task3_medsam2_stageA_implementation.md
```

Task3 Stage C semi-supervised 训练实现和实验记录在：

```text
notes/Optimization/opt_log/task3_medsam2_stageC_semi.md
```

第一版实现边界：

```text
只支持 Task3。
只支持 MedSAM2_latest.pt。
只支持 frozen encoder + LightFPNDecoder。
不做 Task1/Task2，不做 LoRA，不做 pseudo labels。
```

## 12. 当前结论

当前项目正确落地路线：

```text
先做 Task3。
先 frozen MedSAM2 image_encoder。
只训练 LightFPN decoder。
先证明 encoder 特征有效，再考虑 neck fine-tune / adapter / LoRA。
Task1/Task2 后续走 2.5D，不直接把 3D volume 输入 MedSAM2 encoder。
```

Stage C 已实现 MedSAM2 EMA teacher 半监督训练，但 hard pseudo-label BCE 的第一版不理想：

```text
v1 best 仍在 supervised warmup epoch 6: Dice 0.6661 | HD 98.83  | ASD 16.97
v1b best 仍在 supervised warmup epoch 6: Dice 0.6658 | HD 101.78 | ASD 17.13
```

主要失败模式：

```text
teacher pseudo positive area only about 0.03-0.04
validation GT area about 0.1136
unsupervised BCE is dominated by confident background pixels
student prediction area becomes too small after semi starts
```

Stage C v2 已验证 positive-only pseudo supervision 是更好的方向：

```text
best epoch 59
Dice 0.6911 | HD 109.54 | ASD 16.38 | threshold 0.65
```

Stage C v2 后处理评估：

```text
thr=0.70 | min_area=400 | keep_components=2 | close_iters=0 | fill_holes=False
Dice 0.6887 | HD 83.65 | ASD 16.36
```

结论：

```text
positive-only semi-supervision 明显优于 hard pseudo-label BCE。
当前 MedSAM2 路线的 Dice 仍低于网站最佳 Task3 DSC 0.76175。
HD 可以通过后处理降到网站目标 91.85 以下。
下一步应把后处理参数加入 MedSAM2 prediction script，并生成一次网站提交验证泛化。
```
