# main.pdf 更新到 v35 的 Task1/Task3 修改清单

整理日期：2026-08-01

本文基于当前 `main.pdf`、`scripts/final_infer_v35.py`、v35 submission 记录和 Docker 评分记录整理。这里的“task13”按 Task1 + Task3 理解。

## 1. 结论

当前 `main.pdf` 仍主要描述 v19/早期配置，不等于最终 v35：

- Task1 现在不是 `Dataset101_MVAA_Task1` 纯有标注 27 例训练，而是 v25 组件：`Dataset111_MVAA_Task1_PseudoTop100`，即 27 个真实标注 CT + 100 个高置信伪标签 CT，nnU-Net v2 `3d_fullres` 5-fold `checkpoint_best.pth` ensemble。
- Task3 现在不是单个 UNet++ ResNet34、阈值 `0.285`，而是 v35 组件：5 个 UNet++ 分支的 probability ensemble，权重 `0.30/0.20/0.20/0.15/0.15`，阈值 `0.40`，TTA 和 AMP 开启，无额外后处理。
- 论文里的结果表需要区分“验证阶段 online mask-zip v35 分数”和“最终 Docker v35 分数”。不要把二者都称作同一个 final result。

## 2. v35 最终实现确认

### Task1: v25 pseudo-top100 nnU-Net

最终 Docker v35 的 Task1 设置来自 `scripts/final_infer_v35.py`：

- Dataset：`Dataset111_MVAA_Task1_PseudoTop100`
- Framework：nnU-Net v2
- Configuration：`3d_fullres`
- Trainer/plans：`nnUNetTrainer__nnUNetPlans`
- Folds：`0,1,2,3,4`
- Checkpoint：`checkpoint_best.pth`
- 推理：nnU-Net 5-fold ensemble，输出 NIfTI 后转回 `t1_ct/*-pred.nii.gz`
- 后处理：无额外后处理

Dataset111 的训练数据：

- `numTraining = 127`
- 27 个官方 labeled CT
- 100 个从 1040 个 unlabeled CT 中筛选出的 pseudo-labeled cases
- 每个 fold 的验证集仍然是原始 labeled cases；100 个 pseudo cases 都只进入训练集
- fold 训练/验证规模：
  - fold0/fold1：121 train / 6 val，其中 train 含 100 pseudo
  - fold2/fold3/fold4：122 train / 5 val，其中 train 含 100 pseudo

需要写进论文的关键变化：Task1 最终确实使用了 unlabeled CT，但不是 EMA teacher-student 直接半监督训练，而是 self-training / pseudo-label augmentation 后重新训练 nnU-Net。

### Task3: v35 five-model probability ensemble

最终 Docker v35 的 Task3 设置：

| Weight | Branch | Encoder | Seed | Input size | Selected epoch | Checkpoint |
|---:|---|---|---:|---|---:|---|
| 0.30 | v15 ResNet branch | ResNet34 ImageNet | 42 | 448 x 800 | 31 | `outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt` |
| 0.20 | ResNet stability branch | ResNet34 ImageNet | 49 | 448 x 800 | 93 | `outputs/opt/task3/t3_v36_unetpp_res34_img448x800_s49/checkpoints/best.pt` |
| 0.20 | high-recall branch | EfficientNet-B4 ImageNet | 42 | 512 x 896 | 32 | `outputs/opt/task3/t3_sup_unetpp_effb4_img512x896_allframe_s42/checkpoints/best.pt` |
| 0.15 | B5 branch | EfficientNet-B5 ImageNet | 42 | 512 x 896 | 14 | `outputs/opt/task3/t3_v32_unetpp_effb5_sup_allframe_s42/checkpoints/best.pt` |
| 0.15 | B5 branch | EfficientNet-B5 ImageNet | 44 | 512 x 896 | 68 | `outputs/opt/task3/t3_v35_unetpp_effb5_sup_allframe_s44/checkpoints/best.pt` |

Inference logic:

1. Read each RGB frame and normalize with ImageNet statistics.
2. Resize the input tensor to each branch's own training size.
3. Run each model with TTA and AMP.
4. Interpolate each probability map back to the original frame size.
5. Compute weighted probability sum.
6. Binarize with threshold `0.40`.
7. Save `*_label_bin.png` and `task3_predictions.json`.

Important: v35 does not use LemonFM, presence gate, Cutie, recall rescue, or extra connected-component postprocessing. Those belong to later experiments such as v38/v39 and should not be described as v35.

## 3. 分数应该如何改

### v35 validation/online mask-zip result

如果论文的 Results 表仍然报告 validation/online mask-zip 结果，应把 Table 3 改成：

| Task | DSC | HD | ASD |
|---|---:|---:|---:|
| Task 1 | 0.8578127264 | 4.6716293728 | 0.2741234909 |
| Task 2 | 0.8463245905 | 11.1968886295 | 0.6406354095 |
| Task 3 | 0.8048920178 | 73.7878394899 | 12.2489033269 |

此结果来自 `outputs/submissions/submit_v35_task1_v25_task2_v19_task3_5model_ens/checkpoint_record.md`。

### v35 final Docker result

如果论文要报告最终 Docker 阶段结果，应使用：

| Task | mean DSC | mean HD | mean ASD |
|---|---:|---:|---:|
| Task 1 | 0.838233 | 6.243079 | 0.557513 |
| Task 2 | 0.858118 | 9.297447 | 0.494240 |
| Task 3 | 0.760200 | 494.150328 | 381.538430 |

此结果来自 `mvaa_submission_scores_docker.md`。建议在正文明确写成 “Docker final evaluation” 或 “hidden-test Docker evaluation”，不要和 validation server score 混用。

## 4. main.pdf 逐处需要修改

### Abstract

当前问题：

- 写成 Task3 单个 ResNet34 checkpoint。
- 分数还是 v19：Task3 `0.7702/95.6273/14.5378`。
- Task1 DSC 写 `0.8575`，未反映 v25/v35 的 `0.8578127` validation result。

建议替换为：

```text
For surgical video frames, we use an ensemble of five 2D UNet++ models with ResNet34, EfficientNet-B4, and EfficientNet-B5 ImageNet-pretrained encoders. At inference time, the branch probabilities are resized to the original frame resolution, combined with fixed weights, and binarized using a threshold of 0.40 with test-time augmentation.
```

如果报告 validation/online v35：

```text
On the organizer-side online validation server, the v35 submission achieves DSC/HD/ASD of 0.8578/4.6716/0.2741 for Task 1, 0.8463/11.1969/0.6406 for Task 2, and 0.8049/73.7878/12.2489 for Task 3.
```

如果报告 Docker v35：

```text
In the Docker final evaluation, the v35 submission achieves mean DSC/HD/ASD of 0.8382/6.2431/0.5575 for Task 1, 0.8581/9.2974/0.4942 for Task 2, and 0.7602/494.1503/381.5384 for Task 3.
```

### Contributions / Introduction

当前第三条贡献仍写“Task3 UNet++ ResNet34 + threshold tuning + TTA”。应改为“多 encoder、多 seed probability ensemble”。

建议句子：

```text
Third, for surgical video frames, we build a five-branch 2D UNet++ probability ensemble using ResNet34, EfficientNet-B4, and EfficientNet-B5 encoders, which improves the online Task3 validation score over the single ResNet34 model.
```

Task1 的贡献也要补一句：

```text
For Task 1, the final v35 system further incorporates high-confidence pseudo labels from unlabeled CT volumes through a self-training nnU-Net dataset.
```

### Section 2.1 Overall Framework / Fig. 1

当前写法：

- Task1 uses Dataset101。
- Task3 是单模型 UNet++ ResNet34。

应改为：

- Task1：`Dataset111_MVAA_Task1_PseudoTop100`
- Task3：five-model UNet++ probability ensemble
- Fig. 1 中 Task3 框图改成 `UNet++ ResNet34/B4/B5 branches -> probability ensemble -> threshold 0.40`
- Fig. 1 中 Task1 框图改成 `Dataset111 pseudo-top100 preprocessing`

建议段落：

```text
Task 1 is formulated as binary CT foreground segmentation and uses Dataset111_MVAA_Task1_PseudoTop100, which combines the 27 labeled CT cases with 100 high-confidence pseudo-labeled CT cases selected from the unlabeled set. Task 3 is handled by a five-model 2D UNet++ probability ensemble operating on individual RGB surgical frames. The ensemble contains two ResNet34 branches, one EfficientNet-B4 branch, and two EfficientNet-B5 branches.
```

### Section 2.3 Task 1

当前 Task1 小节仍描述 `Dataset101` 和“unlabeled CT not used”。这需要重写。

建议替换核心段落：

```text
We first train a supervised nnU-Net v2 model on the 27 labeled CT cases and use the resulting five-fold ensemble to generate pseudo labels for the unlabeled CT volumes. A metadata and mask-quality screening step selects the top 100 high-confidence pseudo-labeled cases. The final Task 1 dataset is Dataset111_MVAA_Task1_PseudoTop100, containing 27 labeled cases and 100 pseudo-labeled cases.

The original five validation splits are preserved for labeled cases. In each fold, the validation set contains only labeled cases, while all 100 pseudo-labeled cases are added to the training set. The fold sizes are 121/6, 121/6, 122/5, 122/5, and 122/5 for training/validation. We train nnU-Net v2 with the 3d_fullres configuration and use checkpoint_best.pth from folds 0-4 during inference.
```

保留原来的 spacing、patch、batch、CTNormalization 描述基本可以，但 Dataset 名和训练样本数必须改。

### Section 2.5 Task 3

当前 Task3 小节把最终方法写成单个 ResNet34，并详细描述 EMA pseudo-labeling。v35 中这最多只能描述其中一个历史/分支，不应作为最终 Task3 主方法。

建议重写为：

```text
The final Task 3 system is a probability ensemble of five UNet++ models. The first two branches use ResNet34 encoders pretrained on ImageNet and operate at 448 x 800 resolution with seeds 42 and 49. The remaining branches use EfficientNet-B4 and EfficientNet-B5 encoders pretrained on ImageNet and operate at 512 x 896 resolution. The ensemble weights are 0.30, 0.20, 0.20, 0.15, and 0.15 for ResNet34-s42, ResNet34-s49, EfficientNet-B4-s42, EfficientNet-B5-s42, and EfficientNet-B5-s44, respectively.

At inference time, each frame is normalized with ImageNet statistics and resized independently for each branch. The predicted probability maps are resized back to the original frame resolution and averaged using the fixed ensemble weights. The final binary mask is obtained with a probability threshold of 0.40. Test-time augmentation and automatic mixed precision are enabled, and no additional connected-component postprocessing is applied.
```

EMA pseudo-labeling 的旧段落建议降级为：

```text
The ResNet34-s42 branch follows the earlier semi-supervised training recipe with EMA pseudo-labeling, while later supervised all-frame EfficientNet branches are added to increase model diversity and recall. The final submitted output, however, is determined by the weighted probability ensemble rather than by any single branch.
```

不要继续写 “At inference time ... threshold 0.285”。

### Section 2.6 Data Usage / Table 1

当前写法错误点：写 Task1 unlabeled CT volumes are not used。

建议 Table 1 改为：

| Task | Modality | Supervision | Data usage |
|---|---|---|---|
| Task 1 | Cardiac CT | Self-training nnU-Net | 27 labeled cases + top100 pseudo-labeled CT cases selected from 1040 unlabeled volumes; 5-fold validation uses labeled cases only |
| Task 2 | 3D TEE | Supervised nnU-Net 5-fold | 105 labeled cases; each fold uses 84 training and 21 validation cases |
| Task 3 | Surgical frames | 2D ensemble | 150 labeled training frames, 30 labeled validation frames, 1379 unlabeled frames used in earlier semi-supervised branch; final v35 output is five-model probability ensemble |

### Section 2.7 Reproducible Inference

当前写法：Task3 selected UNet++ ResNet34 checkpoint with threshold 0.285。

应改为：

```text
Task 3 inference uses five fixed UNet++ checkpoints. For each frame, model probabilities are computed with TTA and AMP, resized to the original image size, combined with weights 0.30/0.20/0.20/0.15/0.15, and thresholded at 0.40.
```

Docker 描述可以补充：

```text
The submitted Docker image executes scripts/final_infer_v35.py. It discovers task-specific inputs under /input and writes the three required output folders under /output.
```

### Table 2

Task1 行需要从 Dataset101 改成 Dataset111 pseudo-top100。

Task3 行建议改成：

| Task | Model and training | Preprocessing and augmentation | Inference |
|---|---|---|---|
| Task 3 | Five UNet++ branches: ResNet34-s42, ResNet34-s49, EfficientNet-B4-s42, EfficientNet-B5-s42, EfficientNet-B5-s44 | RGB resize to branch-specific sizes 448 x 800 or 512 x 896; ImageNet normalization; standard frame augmentation during training | Probability ensemble weights 0.30/0.20/0.20/0.15/0.15; TTA; AMP; threshold 0.40; no postprocessing |

### Results / Table 3

当前 Table 3 是 v19 validation result。若论文标题或正文说 final v35，就必须替换。

建议加一句防止混淆：

```text
We report validation-server results and Docker final-evaluation results separately because the two phases use different evaluation sets and submission mechanisms.
```

然后按第 3 节选择一个或两个表。

### Discussion and Limitations

需要新增：

- Task1 self-training 使用伪标签，受伪标签筛选质量影响。
- Task3 五模型 ensemble 推理成本明显高于单模型。
- Task3 仍是 frame-wise，没有显式时序建模。
- Docker v35 的 Task3 ASD 相对 v19 Docker 变差，说明提高 overlap/HD 不一定同步改善平均表面距离；论文如果报告 Docker 分数，需要诚实讨论这个 trade-off。

建议句子：

```text
The Task1 improvement relies on selected pseudo labels, so pseudo-label noise and distribution shift remain potential risks. For Task3, the five-model ensemble improves the online validation result but increases inference cost and still treats frames independently. In the Docker final evaluation, Task3 improves DSC and HD over the previous Docker v19 result, while ASD becomes worse, indicating a metric-specific trade-off.
```

### Conclusion

当前结论仍写 semi-supervised UNet++ ResNet34 pipeline。建议改为：

```text
We presented a task-specific MVAA 2026 solution using nnU-Net v2 ensembles for the two volumetric tasks and a five-model UNet++ probability ensemble for surgical frames. The final v35 system uses pseudo-label self-training for Task1 and multi-encoder, multi-seed probability ensembling for Task3, with fixed Docker inference settings for reproducible evaluation.
```

## 5. 参考依据

- `scripts/final_infer_v35.py`: v35 Docker 推理入口，定义 Task1/Task2 datasets、Task3 checkpoints、weights、threshold。
- `task3/generate_task3_multi_ensemble_predictions.py`: Task3 多模型概率融合和 threshold 逻辑。
- `outputs/submissions/submit_v35_task1_v25_task2_v19_task3_5model_ens/checkpoint_record.md`: v35 validation submission 组件和 online 结果。
- `mvaa_submission_scores_docker.md`: Docker final-submission 阶段 v19/v35 结果。
- `outputs/nnunet_task1_pseudo_top100/pseudo_splits_record.json`: Dataset111 pseudo-top100 split 和 pseudo case 记录。
- `outputs/nnunet_task1_pseudo_top100/nnUNet_raw/Dataset111_MVAA_Task1_PseudoTop100/dataset.json`: Dataset111 `numTraining=127`。

## 6. 最小修改优先级

如果时间很紧，优先改这 6 处：

1. Abstract 的 Task3 方法和分数。
2. Overall Framework 中 Task1 Dataset101 -> Dataset111 pseudo-top100，Task3 single model -> five-model ensemble。
3. Task1 Method 中加入 100 pseudo CT，自训练而非“unlabeled not used”。
4. Task3 Method 中删除最终单模型阈值 `0.285` 叙述，改成五模型 ensemble + threshold `0.40`。
5. Table 1 / Table 2 / Table 3。
6. Conclusion 中把 “semi-supervised UNet++ ResNet34 pipeline” 改成 “five-model UNet++ probability ensemble”。
