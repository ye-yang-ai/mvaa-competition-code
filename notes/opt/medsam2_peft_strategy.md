# MedSAM2 Strategy for MVAA

本文档给出当前项目中使用 MedSAM2 优化 MVAA 三个任务的推荐策略。根据项目代码、数据规模和本地 MedSAM2 仓库能力，当前结论是：

```text
主线: baseline 自动定位目标 -> MedSAM2 离线 refine / propagate -> 质量筛选 -> 伪标签蒸馏回现有 baseline/student
后置实验: 在 refine 已经验证有效后，再尝试 MedSAM2 LoRA / PEFT
不推荐: 无 prompt 直接用 MedSAM2 生成最终提交
```

核心原因：

- MedSAM2 是 promptable segmentation model，不是固定类别语义分割模型。
- 比赛测试阶段没有人工 prompt，必须由现有 baseline 自动产生 box/mask/point prompt。
- MedSAM2 适合“给定目标后精修边界或跨帧/跨切片传播”，但不天然知道比赛类别。
- 伪标签不能盲用，必须经过 internal validation 和质量规则筛选。
- 最终提交优先使用蒸馏后的 student，减少推理时间和工程风险。
- 每个优化阶段必须能回答一个问题：是否在稳定 internal metric 上超过当前 best baseline；否则不进入下一阶段。

## 1. 当前任务和数据

| Task | 输入 | 输出 | 监督形式 | 本地数据规模 | MedSAM2 适配度 |
| --- | --- | --- | --- | --- | --- |
| Task1 | 3D cardiac CT `.nii.gz` | 3D mask `.nii.gz` | 半监督 | 27 labeled + 1040 unlabeled + 30 val | 中等 |
| Task2 | 3D TEE ultrasound `.nii.gz` | 3D multi-class mask `.nii.gz` | 全监督 | 105 train + 20 val | 较低到中等 |
| Task3 | Surgical video frames `.png` | Binary mask `*_label_bin.png` | 半监督 | 180 labeled + 1379 unlabeled + 48 val | 最高 |

Task2 的 `num_classes=3` 在当前代码中表示：

```text
0 = background
1 = foreground class 1
2 = foreground class 2
```

当前本地代码和文档没有明确写出 class 1 / class 2 的官方解剖名称。后续若要精细优化 Task2，必须先从官网说明、数据说明文件或标签可视化确认类别语义。

## 2. 本地 MedSAM2 资源

当前仓库已经包含 MedSAM2：

```text
external/MedSAM2/
  checkpoints/MedSAM2_latest.pt
  sam2/configs/sam2.1_hiera_t512.yaml
  medsam2_infer_3D_CT.py
  medsam2_infer_video.py
```

推荐默认配置：

```text
checkpoint: external/MedSAM2/checkpoints/MedSAM2_latest.pt
config:     external/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml
```

注意：

- 原始视频脚本默认扫描 `.jpg/.jpeg`，Task3 是 `.png`，需要适配。
- 原始视频脚本支持 `add_new_mask`，Task3 优先使用 baseline 二值 mask prompt；box prompt 只作为 fallback / ablation。
- 原始 3D CT 脚本本质是把 3D volume 当作 slice video，在关键 slice 上加 prompt 后前后传播。
- 3D 脚本不是多类别 3D semantic segmentation 模型，需要由 baseline 提供类别和目标区域。
- 不建议直接修改 `external/MedSAM2` 内部脚本；应在项目内新增适配层，便于记录实验、回滚和复用。

## 2.1 当前代码风险和已知约束

在进入 MedSAM2 闭环前，先固定以下工程边界：

```text
1. 默认路径必须指向当前 repo 根目录，而不是父目录。
2. Task3 的默认数据路径应使用 data/reference_data/t3_vid/train、data/reference_data/t3_vid/val/images 和 data/images。
3. Task3 内部验证不能只看前景帧，否则会高估实际提交表现。
4. Task3 threshold / postprocess 必须同时统计 empty-frame false positive。
5. 所有 refined 输出必须保留 baseline fallback，不允许无条件覆盖。
```

Task3 评估建议同时报告两组结果：

```text
foreground-only metrics:
  用于观察目标区域边界质量

all-frame metrics:
  用于观察空帧误报、提交风险和真实泛化
```

如果两者结论冲突，以 all-frame / video-level 结果作为是否进入下一阶段的主要依据。

## 3. MedSAM2 工作方式和策略边界

MedSAM2 的典型流程：

```text
image/video/volume slices -> image encoder
prompt(point/box/mask) -> prompt encoder
image feature + prompt feature -> mask decoder
video/slice sequence -> memory attention propagation
```

它更适合回答：

```text
这里有一个目标，请把这个目标分割得更准。
```

它不适合直接回答：

```text
请自动输出比赛要求的所有语义类别。
```

因此本项目中的合理使用方式是：

```text
现有 baseline/student:
  负责自动定位比赛目标和类别

MedSAM2:
  负责对指定目标做二值 refine、边界修正、视频/切片传播

最终 student:
  学习 GT + 经过筛选的 MedSAM2 pseudo labels，并保持比赛提交格式
```

## 4. 什么是离线 Refine

离线 refine 指 MedSAM2 不一定参与最终提交推理，而是在训练前或训练中提前生成更好的候选 mask：

```text
1. 训练现有 baseline
2. baseline 对 labeled / unlabeled / val 生成 coarse mask 或 probability
3. coarse mask 转成 MedSAM2 prompt
4. MedSAM2 refine / propagate 得到候选 mask
5. 用验证集和质量规则判断候选 mask 是否可信
6. 保存 accepted pseudo labels
7. 用 GT + accepted pseudo labels 训练最终 student
8. 最终提交优先使用 student 输出
```

这样做的优势：

- 最终推理快，不依赖 MedSAM2 大模型在线运行。
- 伪标签可以人工抽查和批量筛选。
- 如果 MedSAM2 某些 case 失败，可以 fallback 到 baseline，不污染最终提交。
- student 可以把 MedSAM2 的局部边界改进吸收进标准分割模型。

## 5. 伪标签质量控制

MedSAM2 生成的 mask 只是候选伪标签，不默认可信。必须做质量控制。

### 5.1 Internal Validation

先在有 GT 的 internal validation 上比较：

```text
baseline prediction vs GT
MedSAM2 refined prediction vs GT
```

只有当 refined mask 在 DSC / HD / ASD 上稳定优于 baseline，才进入批量伪标签生成。

建议记录：

```text
case_id, video_id, frame_idx, class_id,
baseline_dsc, refined_dsc,
baseline_hd, refined_hd,
baseline_asd, refined_asd,
baseline_fg_ratio, refined_fg_ratio, gt_fg_ratio,
agreement_dice, area_ratio, accepted, reject_reason
```

Task3 至少额外记录：

```text
empty_gt_frames
baseline_false_positive_frames
refined_false_positive_frames
accepted_frames
fallback_frames
video_mean_dsc
video_mean_hd
video_mean_asd
```

### 5.2 Baseline Agreement

MedSAM2 输出不能和 baseline 差异过大，否则可能是 prompt 漂移。

建议初始规则：

```text
Dice(baseline_mask, medsam2_mask) >= 0.4
```

Task3 可适当提高到 `0.5`。如果前景很小，Dice 会不稳定，可以结合 IoU、面积比例和连通域判断。

初始决策规则：

```text
accept refined:
  agreement_dice 达标
  fg_ratio 在任务范围内
  refined 不明显增大 empty-frame false positive
  相邻帧/slice 面积变化可解释

fallback baseline:
  refined 与 baseline 差异过大
  refined 面积暴涨/暴跌
  refined 连通域碎片过多
  keyframe prompt 本身置信度不足

skip pseudo:
  baseline 和 refined 都低置信
  无法判定目标是否存在
  该 case/frame 在 internal val 中同类规则表现差
```

### 5.3 形态学规则

候选 mask 需要满足基本形态约束：

```text
foreground ratio 不小于任务下限
foreground ratio 不大于任务上限
最大连通域面积超过 min_area
连通域数量不过多
相邻 frame/slice 面积变化不过大
```

推荐初始阈值：

```text
Task1:
  fg_ratio in [0.0005, 0.35]
  keep largest connected component when appropriate

Task2:
  class-wise fg_ratio 根据训练标签统计 p1-p99 设置
  overlap 区域优先采用 baseline probability 决策

Task3:
  fg_ratio in [0.001, 0.7]
  min_area >= 80 pixels at training resolution
  neighboring frame area change ratio <= 3.0 unless baseline also changes
```

### 5.4 低权重蒸馏

伪标签不应和人工 GT 同权重。

推荐：

```text
GT loss weight = 1.0
pseudo loss weight = 0.2 - 0.5
```

如果 pseudo labels 主要来自 MedSAM2 且未经过充分验证，先用 `0.2` 或 `0.3`。

### 5.5 阶段推进标准

每个阶段都需要明确进入下一阶段的门槛：

```text
进入 Task3 批量 pseudo:
  one-video MVP 成功
  至少 2 个 internal val video 上 refined/fallback 组合优于 baseline
  all-frame false positive 没有明显变差

进入 Task1 批量 pseudo:
  5 - 10 个 internal val case 上多数 case 不退化
  失败 case 能被质量规则识别
  spacing/affine/shape 保存无误

进入 Task2 refine:
  class-wise internal val ablation 明确优于 baseline
  overlap 合并规则稳定

进入 PEFT / LoRA:
  无 LoRA refine 已经稳定收益
  prompt 自动生成和筛选规则已经固定
  student-only 蒸馏收益接近瓶颈
```

## 6. 推荐新增模块

```text
medsam2_integration/
  __init__.py
  config.py
  build_medsam2.py
  prompt_utils.py
  quality_filter.py
  pseudo_label.py
  nifti_utils.py
  video_utils.py
  refine_3d.py
  refine_video.py
  lora.py                 # 后置实验，不是第一阶段必需
```

职责：

| 文件 | 作用 |
| --- | --- |
| `config.py` | 统一保存 MedSAM2 checkpoint/config/device 路径 |
| `build_medsam2.py` | 加载 MedSAM2 predictor |
| `prompt_utils.py` | coarse mask -> box/mask/point prompt |
| `quality_filter.py` | 伪标签质量筛选 |
| `pseudo_label.py` | 保存 mask 和 metadata |
| `nifti_utils.py` | NIfTI 读写、spacing/shape/affine 保持 |
| `video_utils.py` | Task3 视频帧排序、关键帧选择 |
| `refine_3d.py` | Task1/Task2 slice propagation/refine |
| `refine_video.py` | Task3 video propagation/refine |
| `lora.py` | LoRA 注入、保存、加载，后置使用 |

## 7. Task3 推荐主线

Task3 是 MedSAM2 最值得优先投入的任务。

原因：

- Task3 是视频帧二值分割，和 SAM2/MedSAM2 video propagation 匹配。
- 输出是 binary mask，不需要多类别合并。
- 无标签视频帧较多，适合生成 pseudo labels 后蒸馏。
- 现有 Task3 baseline 可以提供自动 keyframe prompt。

### 7.1 Task3 流程

```text
1. 训练 Task3 baseline/student
2. baseline 对每个视频帧输出 probability map
3. 选择高置信 keyframes
4. keyframe coarse mask -> MedSAM2 mask prompt / box prompt
5. MedSAM2 向前/向后传播
6. 对低置信片段补 prompt
7. 质量筛选
8. 保存 accepted pseudo labels
9. 用 GT + pseudo labels 训练最终 Task3 student
```

### 7.2 Keyframe 选择

推荐初始规则：

```text
foreground_ratio between 0.001 and 0.7
mean foreground probability >= 0.7
max connected component area >= 80
keyframe interval K = 5 or 10
```

更细的排序规则：

```text
优先选择:
  baseline foreground confidence 高
  foreground area 不接近上下限
  最大连通域占 foreground 比例高
  与上一 keyframe 距离 >= K
  相邻帧 baseline 面积变化平滑

避免选择:
  mask 极小或极大
  多个碎片连通域
  边界贴图像边缘且疑似漂移
  前后帧预测突然跳变
```

如果一个视频没有合格 keyframe：

```text
fallback = baseline 原始输出
或者 skip 该视频的 pseudo label
```

### 7.3 最小可行实验

先不要做 LoRA。先跑一个 val video：

```text
1. baseline 输出 val video coarse masks
2. coarse masks 转 keyframe prompt
3. MedSAM2 propagate
4. 与 GT 比较 DSC / HD / ASD
5. 可视化 baseline vs refined vs GT
```

如果 refined 没有明显提升，不进入批量伪标签阶段。

MVP 必须输出以下产物：

```text
outputs/medsam2_mvp/task3/<run_name>/
  baseline_masks/
  medsam2_refined_masks/
  fallback_masks/
  overlays/
  metrics_frame.csv
  metrics_video.json
  config.json
```

`metrics_frame.csv` 至少包含：

```text
video_id, frame_idx,
baseline_dsc, refined_dsc, fallback_dsc,
baseline_hd, refined_hd, fallback_hd,
baseline_asd, refined_asd, fallback_asd,
baseline_fg_ratio, refined_fg_ratio, gt_fg_ratio,
agreement_dice, accepted, reject_reason
```

优先比较三条线：

```text
baseline only
MedSAM2 refined only
quality-filtered refined + baseline fallback
```

真正有意义的是第三条线是否稳定超过 baseline only。

### 7.4 推荐新增脚本

```text
task3/generate_task3_medsam2_refined.py
task3/train_task3_with_medsam2_pseudo.py
task3/evaluate_task3_refined.py
```

`generate_task3_medsam2_refined.py` 应支持两种模式：

```text
--submission-task-dir   生成比赛提交格式
--pseudo-output-dir     生成伪标签训练集
```

实现要求：

```text
1. 支持 png 视频帧，不依赖 MedSAM2 demo 的 jpg 扫描逻辑。
2. 支持 baseline probability / binary mask 作为 keyframe mask prompt。
3. 支持 --video-folders 限定单个或多个视频。
4. 支持保存 baseline、refined、fallback 三套 mask。
5. 支持质量规则 dry-run，只评估不写 pseudo。
```

## 8. Task1 推荐主线

Task1 可作为第二优先级。

原因：

- 有 1040 个 unlabeled CT，伪标签蒸馏潜在收益大。
- MedSAM2 的 3D medical image demo 主要面向 CT/MRI slice propagation，和 Task1 有一定匹配。
- 但 Task1 labeled case 很少，prompt 失败时容易产生系统性噪声。

### 8.1 Task1 流程

```text
1. 训练 Task1 baseline 3D UNet
2. baseline 对 CT volume 输出 coarse mask/probability
3. 每个 foreground component 生成 key slice / box / mask prompt
4. MedSAM2 把 volume slice 当 video 做前后传播
5. 合成 3D refined mask
6. 质量筛选
7. accepted pseudo labels 蒸馏回 3D UNet
```

### 8.2 Prompt 设计

推荐：

```text
primary: coarse mask prompt on key slice
secondary: expanded 2D box prompt on key slice
fallback: baseline original mask
```

key slice 选择：

```text
选择 foreground area 最大或 baseline confidence 最高的 slice
```

box 扩张：

```text
expand 5 - 15 pixels in resized 2D plane
```

### 8.3 Task1 风险

MedSAM2 的 3D refine 是 slice propagation，不是完整 3D semantic model。因此：

- 不要直接相信跨很长 z 轴传播。
- 对 slice 间面积变化要做平滑检查。
- 对 refined 与 baseline 差异过大的 case 使用 baseline fallback。
- 首先在 internal val 上证明 DSC/HD/ASD 提升。
- 保留原始 NIfTI affine/header/shape；任何 resize 后 mask 都必须回写到原图空间。
- 对空预测或极小预测要直接 fallback，不要强行构造 mean box。

### 8.4 推荐新增脚本

```text
task1/generate_task1_medsam2_refined.py
task1/train_task1_with_medsam2_pseudo.py
```

## 9. Task2 推荐策略

Task2 暂时不作为 MedSAM2 主攻方向。

原因：

- 3D TEE 是 ultrasound，和 MedSAM2 当前 CT/通用医学图像先验存在域差异。
- Task2 是 3 类语义分割，而 MedSAM2 输出更自然的是单目标 binary mask。
- 需要 class-wise 调用后再合并，容易产生 overlap、漏类和类别漂移。

### 9.1 可选小规模实验

如果要尝试 Task2，先只做 internal val ablation：

```text
1. baseline 3D UNet 输出 3-class probability
2. class 1 coarse mask -> MedSAM2 binary refine
3. class 2 coarse mask -> MedSAM2 binary refine
4. overlap 区域用 baseline probability 决策
5. 与 GT 比较 DSC / HD / ASD
```

只有当 class-wise refine 稳定提升，才考虑生成 pseudo labels 或蒸馏。

Task2 在 MedSAM2 前应优先做 baseline 侧优化：

```text
1. 固定 internal split，避免只看单一 val-count 结论
2. 统计 class-wise label ratio 和空间范围
3. 尝试 class-balanced crop / loss weight
4. 调整 ROI、spacing、TTA 和后处理
5. 检查每类 DSC/HD/ASD，而不只看 mean score
```

### 9.2 合并规则

```text
refined_1 only -> label 1
refined_2 only -> label 2
overlap -> argmax baseline probability between class 1 and class 2
neither -> background
```

如果某类 baseline coarse mask 为空：

```text
默认不输出该类
不要使用 mean box 强行生成，除非 validation 证明有效
```

### 9.3 推荐新增脚本

```text
task2/generate_task2_medsam2_refined.py
```

`train_task2_medsam2_peft.py` 和 `train_task2_with_medsam2_pseudo.py` 暂时后置。

## 10. PEFT / LoRA 后置策略

当前不建议第一阶段训练 MedSAM2 LoRA。

只有满足以下条件后，再考虑 PEFT：

```text
1. 无 LoRA 的 MedSAM2 refine 已经在 internal val 上提升指标
2. prompt 自动生成流程稳定
3. pseudo label 筛选规则已经建立
4. 当前 student 训练收益接近瓶颈
```

### 10.1 LoRA 目标模块

如果进入 PEFT，推荐顺序：

```text
Priority 1: sam_mask_decoder attention / MLP
Priority 2: memory_attention q_proj / v_proj / out_proj
Priority 3: image_encoder q_proj / v_proj
```

不建议一开始训练 image encoder LoRA，因为参数多、数据少、过拟合风险高。

### 10.2 默认 LoRA 参数

```text
r = 4 or 8
alpha = 8 or 16
dropout = 0.05
lr = 1e-4 or lower
trainable = LoRA parameters + optional decoder norm/bias
frozen = original MedSAM2 backbone
```

### 10.3 各任务 PEFT 优先级

```text
Task3 memory attention + mask decoder LoRA: 可尝试
Task2 class-wise mask decoder LoRA: 小规模验证后再尝试
Task1 LoRA: 最后考虑
```

## 11. 最终提交策略

推荐最终提交格式保持现有项目结构：

```text
outputs/submission/
  t1_ct/
    task1_predictions.json
    *.nii.gz
  t2_tee/
    task2_predictions.json
    *.nii.gz
  t3_vid/
    task3_predictions.json
    <video_folder>/
      *_label_bin.png
```

推荐提交模型：

```text
Task1: distilled 3D UNet
Task2: strengthened 3D UNet; optional class-wise MedSAM2 refine only if val proves better
Task3: distilled Task3 student; optional MedSAM2 refine only if inference budget allows and val proves better
```

默认优先级：

```text
1. student-only final submission
2. MedSAM2 offline pseudo labels for training
3. MedSAM2 online refine for final submission only as optional high-cost variant
```

提交前必须做：

```text
1. 重新生成三任务 submission 到 outputs/submission/<run_name>/
2. 检查 zip root 直接包含 t1_ct/、t2_tee/、t3_vid/
3. 抽查 NIfTI shape/affine 和 PNG 命名
4. 记录每个 task 的 checkpoint、threshold、postprocess、pseudo 数据版本
5. 将实验结论精简追加到 notes/opt/log.md
```

## 12. 推荐实验优先级

当前推荐顺序：

```text
Priority 1:
  Task3 MedSAM2 video refine on one validation video

Priority 2:
  Task3 pseudo labels on unlabeled videos + train final student

Priority 3:
  Task1 MedSAM2 refine on internal validation cases

Priority 4:
  Task1 accepted pseudo labels + train final 3D UNet

Priority 5:
  Task2 class-wise MedSAM2 refine ablation

Priority 6:
  Task3 / Task2 LoRA PEFT
```

不建议一开始做：

```text
Task1 MedSAM2 LoRA
Task2 full MedSAM2 training
无 prompt MedSAM2 final submission
无筛选伪标签全量训练
```

## 13. 第一阶段具体落地计划

第一阶段目标不是直接全量训练，而是验证 MedSAM2 是否真的能提高本比赛指标。

### 13.1 Task3 MVP

```text
1. 训练或选择一个 Task3 baseline checkpoint
2. 先用 baseline checkpoint 生成 internal val coarse probability/mask
3. 新增 medsam2_integration/prompt_utils.py
4. 新增 medsam2_integration/video_utils.py
5. 新增 task3/generate_task3_medsam2_refined.py
6. 新增 task3/evaluate_task3_refined.py
7. 跑 1 个 validation video
8. 输出 baseline/refined/fallback/GT 对比可视化
9. 计算 foreground-only 和 all-frame DSC / HD / ASD
```

通过标准：

```text
refined_dsc > baseline_dsc
refined_hd <= baseline_hd
refined_asd <= baseline_asd
没有明显大面积错误传播
quality-filtered fallback 组合优于 baseline only
```

### 13.2 Task1 MVP

```text
1. 训练或选择一个 Task1 baseline checkpoint
2. 对 internal val 的 5 - 10 个 case 生成 coarse mask
3. 用 key slice prompt 做 MedSAM2 refine
4. 比较 baseline/refined/GT 指标
5. 检查 slice propagation 是否稳定
```

通过标准：

```text
至少多数 case 的 DSC 或 HD/ASD 有提升
失败 case 可被质量规则识别并 fallback
```

## 14. 需要修正的原策略点

相对原始策略，本版做了以下原则调整：

```text
1. 将 PEFT 从主线改为后置实验
2. 将 Task3 明确为第一优先级
3. 将 Task2 降为小规模 ablation
4. 强化伪标签质量控制，而不是默认相信 MedSAM2 输出
5. 修正本地 MedSAM2 checkpoint/config 路径
6. 明确 MedSAM2 3D 能力是 slice/video propagation，不是真正多类别 3D 语义分割
7. 明确最终提交优先使用 distilled student
8. 增加 Task3 all-frame 评估，避免 foreground-only val 偏乐观
9. 增加 refined/fallback 双轨输出，避免 MedSAM2 失败污染提交或伪标签
10. 明确每阶段推进标准和实验日志要求
```

这更符合比赛目标：优先拿到稳定、可验证、能提升 leaderboard 指标的结果，而不是过早投入高风险 PEFT 工程。
