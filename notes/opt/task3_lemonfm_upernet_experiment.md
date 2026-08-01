# Task3 LemonFM + UPerNet 实验记录

日期：2026-07-29

## 目的

尝试把 LemonFM ConvNeXt-Large encoder 从之前的 FPN decoder 换成 UPerNet-style decoder，验证更强多尺度 decoder 是否能提升 Task3 二尖瓣分割。

## 实现

新增模型分支：

```text
arch = lemonfm_upernet
```

代码位置：

```text
task3/model_factory.py
```

结构：

```text
LemonFM ConvNeXt-Large encoder
+ PPM on stride-32 feature
+ FPN top-down fusion
+ 1-channel segmentation head
```

训练脚本已支持：

```text
task3/train.py
task3/train_task3_multiclass.py
```

## 验证

已完成：

- Python syntax check
- GPU forward smoke test
- 1 epoch tiny smoke training

smoke training 输出：

```text
outputs/opt/task3/debug_lemonfm_upernet_smoke_gpu3
```

## Stage A 正式探针

输出目录：

```text
outputs/opt/task3/t3_lemonfm_upernet_stageA_dec_s53
```

配置摘要：

- encoder：LemonFM ConvNeXt-Large
- decoder：UPerNet-style，decoder channels = 128
- image size：512 x 896
- batch size：2
- encoder：freeze
- lr：1e-4
- loss：Dice + Focal + 0.02 boundary
- seed：53
- early stop patience：20

训练正常结束，early stop 于 epoch 48。

best epoch：28

```text
score: 0.288768
val_dice_fg: 0.431718
val_dice_all: 0.387812
val_hd: 206.959396
val_asd: 46.534698
threshold: 0.25
empty_fp_rate: 0.700000
fg_miss_rate: 0.000000
```

last epoch：48

```text
score: 0.266599
val_dice_fg: 0.398827
val_dice_all: 0.382552
val_hd: 226.803238
val_asd: 51.076332
threshold: 0.55
empty_fp_rate: 0.650000
fg_miss_rate: 0.075000
```

## 结论

这轮结论是：UPerNet 代码接入成功，但当前 frozen LemonFM encoder + random UPerNet decoder 的表现明显不够强，不建议继续直接做 tail unfreeze Stage B，也不建议生成线上 submission。

主要问题：

- foreground Dice 只有约 0.43，说明前景主体学习不稳定。
- HD/ASD 非常高，说明形状和边界质量远弱于现有 v35/v39 ensemble。
- empty FP rate 仍高，presence gate 只能修空帧误报，不能解决 foreground 帧内部漏分/边界问题。

对当前路线的判断：

- LemonFM 仍可以作为小权重互补分支使用，v39 已经证明这一点。
- LemonFM + UPerNet 这一版 decoder-only 方案暂时不进入主候选。
- 后续如果继续 LemonFM，更值得尝试的是改训练目标和数据策略，而不是单纯把 FPN 换成更重 decoder。
