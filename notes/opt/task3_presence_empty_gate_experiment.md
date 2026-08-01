# Task3 Presence/Empty Gate 实验记录

日期：2026-07-29

## 目的

这次不再把 empty/background loss 直接加到分割头上，而是单独训练一个 frame-level presence classifier：

- 输入：Task3 RGB 帧。
- 输出：这一帧是否存在 label 10 二尖瓣。
- 推理 gate：如果 `presence_prob < gate_threshold`，则把这一帧分割结果整体置空。
- 关键约束：阈值选择优先保证验证集 foreground 帧 `fg_miss=0`，避免牺牲二尖瓣召回。

## 实现

新增脚本：

```text
task3/train_presence_gate.py
```

模型：

```text
LemonFM ConvNeXt-Large encoder
+ binary presence classification head
```

本次正式训练：

```text
outputs/opt/task3/t3_lemonfm_presence_gate_s51
```

配置摘要：

- encoder：LemonFM ConvNeXt-Large
- encoder 权重：`checkpoints/pretrained/lemonfm/lemonfm.pth`
- encoder：冻结
- trainable params：4609
- image size：448 x 800
- train split：120 帧，79 foreground / 41 empty
- val split：60 帧，39 foreground / 21 empty
- lr：3e-4
- epochs：25
- presence positive loss weight：1.5
- empty loss weight：1.0
- gate threshold selection：`max_fg_miss=0`

同时给单模型生成脚本新增可选 gate 参数：

```text
task3/generate_task3_predictions.py
  --presence-gate-ckpt
  --presence-gate-threshold
  --presence-gate-tta / --no-presence-gate-tta
```

默认不开启 gate，不影响原有 submission 流程。

## 本地结果

best epoch：25

presence classifier 验证结果：

```text
threshold: 0.54
AUC: 0.9988
accuracy: 0.9833
foreground miss: 0 / 39
empty cleared: 20 / 21
```

唯一没清掉的 empty frame：

```text
REC_20250205_102353_979A_000495
presence_prob: 0.5488
threshold: 0.54
```

## 对 LemonFM probe segmentation 的 gate 效果

分割 checkpoint：

```text
outputs/opt/task3/t3_lemonfm_fpn_probe_s51/checkpoints/best.pt
```

## v39 线上验证

v39 使用方式不是把 presence gate 直接用于最终 mask，而是只 gate LemonFM ensemble 分支：

```text
Task3 = v35 five-model ensemble + LemonFM formal s52 gated branch
weights = 0.27 / 0.19 / 0.17 / 0.1275 / 0.1425 / 0.10
threshold = 0.36
presence gate threshold = 0.54
```

线上 Task3 结果：

| Version | DSC | HD | ASD | 结论 |
|---|---:|---:|---:|---|
| v35 | 0.804892 | 73.787839 | 12.248903 | 原五模型均衡最好 |
| v38 | 0.808549 | 99.392005 | 13.441782 | 最高 DSC，但距离退化 |
| v39 | 0.806144 | 72.862321 | 12.207524 | 新的均衡最好 |

相对 v35：

```text
DSC +0.001252
HD  -0.925518
ASD -0.041379
```

判断：

- LemonFM 单模型/本地 split 并不强，但小权重加入线上有正互补。
- presence gate 只作用于 LemonFM 分支是关键；它限制了 LemonFM 的空帧/低置信风险，没有破坏主 ensemble。
- v39 不如 v38 的 DSC 高，但 HD/ASD 明显优于 v38，因此更适合作为当前主提交。

对比：

| 设置 | Dice all | Dice fg | HD | ASD | empty FP rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| LemonFM probe baseline | 0.4552 | 0.7003 | 151.93 | 18.86 | 1.000 |
| + presence gate | 0.7885 | 0.7003 | 151.93 | 18.86 | 0.0476 |

结论：

- gate 成功解决 LemonFM probe 的空帧误报问题。
- foreground Dice 完全不变，说明 gate 没有误杀 GT 非空帧。
- HD/ASD 不变，因为当前 HD/ASD 主要由 GT 非空帧内部的漏分/边界问题决定，empty gate 不会修复 foreground 帧的形状。

## 决策

presence/empty gate 是一个有效的安全模块，适合用于：

- LemonFM 这类容易在空帧产生 FP 的模型；
- LemonFM 与 v35/v39 主体 ensemble 结合时，可以先用 gate 过滤明显空帧；
- 生成 submission 时建议保留 `presence-gate-tta`，因为阈值也是按 TTA 验证选择的。

它不是单独主突破点，但作为 LemonFM 小权重分支的安全模块已经被 v39 线上验证有效。当前线上均衡最强 v39 的核心瓶颈仍然更像是 foreground 内部漏分、边界定义和细长区域召回，而不是 empty FP。
