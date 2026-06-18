# MedSAM2 Methods Comparison for MVAA

本文档用于保留两种 MedSAM2 使用方法的对比，并说明为什么当前项目的主线切换到 **MedSAM2 encoder backbone**。

主线定位：MedSAM2 不再作为“根据 prompt 输出 mask 的模型”，而是作为“提供医学视觉多尺度特征的 encoder”。项目落地时，decoder 和 task head 负责学习 MVAA 的固定输出类别。

相关落地文档：`notes/Optimization/medsam2_peft_strategy.md`。

## 1. 两种方法

### 方法 A：Prompt Refine / Pseudo-label Teacher

```text
baseline model -> coarse mask / box prompt -> MedSAM2 refine / propagate -> final mask or pseudo label
```

这个方法把 MedSAM2 当作 promptable refiner。baseline 负责自动找到目标，MedSAM2 负责根据 prompt 精修边界或做视频/切片传播。

### 方法 B：Encoder Backbone + Custom Decoder

```text
image / slice / frame -> MedSAM2 image encoder -> custom segmentation decoder -> task logits
```

这个方法把 MedSAM2 当作医学视觉预训练 encoder。训练后模型不需要 prompt，直接输出比赛需要的分割结果。

## 2. 核心对比

| 维度 | 方法 A：Prompt Refine | 方法 B：Encoder Backbone |
| --- | --- | --- |
| MedSAM2 角色 | 交互式分割器 / refiner / teacher | 预训练特征提取器 |
| 是否需要 prompt | 需要，由 baseline 自动生成 | 不需要 |
| 是否依赖 baseline | 依赖 baseline 生成 prompt | 不依赖 baseline，可独立训练 |
| 输出形式 | 通常是 binary target mask | 直接输出任务类别 logits |
| Task3 适配 | 很好，适合 video propagation | 很好，适合 2D RGB frame segmentation |
| Task1 适配 | 可做 slice propagation / pseudo label | 需 2D/2.5D/3D aggregation 适配 |
| Task2 适配 | class-wise refine 后合并，复杂 | decoder 可直接输出 3-class logits |
| 工程复杂度 | prompt 生成、质量筛选、fallback | encoder 特征提取、decoder 对接、训练策略 |
| 推理成本 | baseline + MedSAM2，较慢 | 单模型推理，结构更清晰 |
| 主要风险 | prompt 错则 refine 错 | 2D encoder 适配 3D volume、过拟合 |

## 3. 为什么选择 Encoder Backbone 作为主线

当前项目目标是比赛自动提交，不是交互式分割。最终需要：

```text
输入测试图像 / volume / frame -> 自动输出固定格式 mask
```

方法 A 的主要瓶颈是 prompt 质量和在线 refine 稳定性。即使 MedSAM2 很强，它仍然需要 baseline 给出正确目标位置。如果 baseline prompt 错，MedSAM2 往往会把错误目标分得更完整。

方法 B 更符合自动分割比赛模型形态：

```text
MedSAM2 encoder 提供医学视觉预训练特征
custom decoder 学习 MVAA 的固定类别定义
最终单模型推理，不需要 prompt
```

尤其对 Task2 这类多类别语义分割，方法 B 可以直接输出：

```text
0 = background
1 = class 1
2 = class 2
```

避免 class-wise prompted binary mask 的合并冲突。

## 4. 方法 B 的提升点

相对原始 baseline 的潜在提升来自：

```text
1. 使用 MedSAM2 的医学图像预训练 encoder，而不是只依赖 ImageNet encoder。
2. 利用 MedSAM2 image encoder 的 Hiera + FPN 多尺度特征。
3. decoder 直接学习比赛类别，推理时不需要 prompt。
4. 后续可通过 neck fine-tune、feature adapter、LoRA 注入任务/模态知识。
```

## 5. 方法 B 的主要限制

MedSAM2 image encoder 是 2D encoder。Task3 可以直接使用，Task1/Task2 不能直接输入完整 3D volume。

因此 Task1/Task2 需要：

```text
slice-wise 2D
2.5D: [z-1, z, z+1] -> center slice label
2D features + 3D aggregation
```

短期推荐先在 Task3 验证 encoder backbone 的价值，再扩展到 Task1/Task2 的 2.5D。

## 6. 三条路线的最终关系

虽然当前主线选择 Encoder Backbone，但方法 A 不是完全无价值。后续可以作为辅助 teacher：

```text
Track A: Direct Prompt Refine
  baseline -> prompt -> MedSAM2 -> final submission candidate

Track B: Pseudo-label Teacher
  baseline -> prompt -> MedSAM2 -> high-quality pseudo labels

Track C: Encoder Backbone Student
  MedSAM2 image encoder -> custom decoder -> supervised / semi-supervised training
```

当前落地主线是 Track C。

在 Track C 跑通并证明有效后，可以再考虑把 Track B 生成的高质量伪标签加入训练。

## 7. 推荐结论

```text
第一阶段只做 Task3 Encoder Backbone。
先冻结 MedSAM2 image encoder，只训练 LightFPN decoder。
如果接近或超过当前 Task3 baseline，再解冻 neck 或加入 adapter/LoRA。
Task1/Task2 后续走 2.5D prototype，不作为第一阶段重点。
```
