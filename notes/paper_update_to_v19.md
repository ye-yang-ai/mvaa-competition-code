# main.pdf 改稿说明：更新到当前最优 v19 方案

本文档用于指导修改 `notes/main.pdf` 对应论文稿。当前 PDF 仍描述旧方案：

- Task1：SegResNet-base 半监督 EMA teacher-student。
- Task2：fully supervised 3D U-Net large。
- Task3：UNet++ ResNet34，阈值 `0.25`，best epoch `48`。
- 结果表：只写本地验证结果，official hidden-test/online results 仍为 `TBD`。

当前线上最优提交是 `v19`：

- Task1：v17，nnU-Net v2 5fold ensemble，`checkpoint_best.pth`，无后处理。
- Task2：v19，nnU-Net v2 5fold ensemble，`checkpoint_best.pth`，无后处理。
- Task3：v15，UNet++ ResNet34 ImageNet，TTA，threshold `0.285`。

主提交路径：

`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission.zip`

## 1. 总体改稿方向

论文需要从“baseline-derived task-specific semi-supervised/supervised methods”改成：

> A task-specific solution combining nnU-Net v2 self-configuring 3D segmentation for CT and 3D TEE, and a semi-supervised UNet++ ResNet34 frame-segmentation pipeline for surgical videos.

核心变化：

| Task | 旧论文方法 | 当前最优方法 | 是否需要重写 |
|---|---|---|---|
| Task1 CT | SegResNet-base + EMA semi-supervised | nnU-Net v2 3d_fullres 5fold ensemble | 需要大幅重写 |
| Task2 TEE | 3D U-Net large, ROI160/旧配置 | nnU-Net v2 3d_fullres 5fold ensemble | 需要大幅重写 |
| Task3 VID | UNet++ ResNet34 + semi-supervised | 仍是 UNet++ ResNet34，但 checkpoint/阈值更新 | 小幅修改 |

旧稿里关于 Task1 使用 unlabeled CT 的半监督训练，已经不再是最终最优方案。需要删除或降级为“early exploratory experiment”。最终方法中，Task1 与 Task2 都应描述为 fully supervised nnU-Net 5fold cross-validation ensemble。

## 2. 摘要需要修改

PDF 当前摘要中需要替换的旧表述：

- “For cardiac CT, we train a semi-supervised 3D segmentation model ... unlabeled volumes.”
- “For 3D TEE, we train a fully supervised 3D U-Net ...”
- “Our best local validation DSC/Dice values are 0.8303, 0.7796, and 0.7785 ...”
- “Official hidden-test results will be updated ...”

建议改成：

```text
For cardiac CT and 3D TEE, we use nnU-Net v2 as a self-configuring 3D segmentation framework and train five-fold ensembles with task-specific preprocessing, spacing, patch size, and normalization automatically determined from the training data. For surgical video frames, we retain a 2D UNet++ model with a ResNet34 ImageNet-pretrained encoder and improve inference with test-time augmentation and threshold tuning. On the organizer-side online evaluation, the final v19 submission achieves DSC/HD/ASD of 0.8575/4.6312/0.2793 for Task 1, 0.8463/11.1969/0.6406 for Task 2, and 0.7702/95.6273/14.5378 for Task 3.
```

如果论文严格区分 “hidden test” 和 “online validation”，不要把这些结果写成 hidden-test；可写成 “online validation/evaluation server results”。只有确认这是最终 hidden-test 后，才写 “hidden-test results”。

## 3. Introduction 和 Contributions 需要修改

当前 Introduction 的贡献点写了：

- Task1 和 Task3 使用 semi-supervised teacher-student。
- 使用 local validation metrics 选择 checkpoint。

需要改成：

1. 使用 nnU-Net v2 对 Task1 CT 和 Task2 3D TEE 进行自配置 3D 分割。
2. 对 Task1 和 Task2 采用 5fold ensemble，提高体数据分割泛化能力。
3. Task3 保留半监督 2D UNet++ ResNet34，并通过 TTA 与阈值搜索得到稳定提交。
4. 提供统一的 submission 组装流程，分别导出 NIfTI 与 PNG/JSON 格式。

建议替换贡献段：

```text
The main contributions of this technical report are as follows. First, we adopt nnU-Net v2 as a self-configuring 3D segmentation framework for both CT and 3D TEE tasks, replacing manually tuned 3D backbones with automatically planned preprocessing, network topology, patch size, and inference settings. Second, we use five-fold model ensembling for the two volumetric tasks, which substantially improves the online DSC for both Task 1 and Task 2. Third, for surgical video frames, we retain a task-specific 2D UNet++ ResNet34 pipeline and improve the final submission through threshold tuning and test-time augmentation. Finally, we document the complete validation-submission workflow with fixed checkpoints and output-format conversion.
```

## 4. Figure 2 需要重画

当前 Figure 2 仍写：

- Task 1: SegResNet / EMA teacher-student
- Task 2: Supervised 3D U-Net
- Task 3: UNet++ ResNet34 / EMA pseudo-labeling

建议更新为：

```text
Task 1 CT volumes
  -> nnU-Net v2 Dataset101 preprocessing
  -> 3d_fullres PlainConvUNet, folds 0-4
  -> checkpoint_best ensemble
  -> CT masks

Task 2 3D TEE volumes
  -> nnU-Net v2 Dataset102 preprocessing
  -> 3d_fullres PlainConvUNet, folds 0-4
  -> checkpoint_best ensemble
  -> multi-class TEE masks

Task 3 surgical frames
  -> resize 448 x 800 + ImageNet normalization
  -> UNet++ ResNet34 + semi-supervised training
  -> TTA + threshold 0.285
  -> frame masks
```

图注建议改成：

```text
Figure 2: Overall framework of the final v19 MVAA solution. Task 1 and Task 2 use nnU-Net v2 self-configuring 3D full-resolution models with five-fold checkpoint-best ensembling. Task 3 uses a 2D UNet++ ResNet34 model with semi-supervised training, test-time augmentation, and a tuned inference threshold.
```

## 5. Method 2.1 Overall Framework 需要修改

旧文强调 Task1/Task3 半监督、Task2 3D U-Net。应改成：

- Task1 和 Task2 都是 nnU-Net v2 3d_fullres。
- Task1 是 binary foreground segmentation。
- Task2 是 3-class segmentation，foreground labels 1 and 2。
- Task3 仍是 2D binary segmentation。
- 最终 v19 submission 是 Task1 v17 + Task2 v19 + Task3 v15。

建议替换核心段落：

```text
The final system consists of three task-specific pipelines. The two volumetric tasks are handled by nnU-Net v2. Task 1 is formulated as binary CT foreground segmentation and uses Dataset101_MVAA_Task1 with a 3d_fullres configuration. Task 2 is formulated as three-class 3D TEE segmentation and uses Dataset102_MVAA_Task2 with a 3d_fullres configuration. For both tasks, we train all five cross-validation folds and average the five checkpoint-best models during inference. Task 3 is handled by a 2D UNet++ ResNet34 model operating on individual RGB surgical frames, with test-time augmentation and a tuned binarization threshold.
```

## 6. Method 2.2 Baseline-Derived Implementation 需要修改

当前 2.2 说扩展了 server-side Task1 SegResNet。这个不再适合最终方法。

建议改成：

```text
Our implementation started from the official MVAA baseline repository for data conventions, prediction formats, and task-specific submission packaging. For Task 1 and Task 2, we added nnU-Net v2 dataset-conversion scripts that convert the official NIfTI files into nnU-Net raw datasets, run nnU-Net planning and preprocessing, train five 3d_fullres folds, and convert nnU-Net predictions back to the MVAA submission naming convention. For Task 3, we retained the segmentation-models-pytorch based UNet++ ResNet34 training pipeline and used the selected checkpoint with fixed TTA and threshold settings.
```

实现脚本可在论文附录或 reproducibility 段落中提：

- Task1 nnU-Net 数据准备：`scripts/prepare_task1_nnunet.py`
- Task1 预测转换：`scripts/convert_task1_nnunet_predictions.py`
- Task2 nnU-Net 数据准备：`scripts/prepare_task2_nnunet.py`
- Task2 预测转换：`scripts/convert_task2_nnunet_predictions.py`
- submission 打包：`scripts/package_submission.py`

## 7. Method 2.3 Task1 CT 需要重写

旧 Task1 内容应删除或移到 ablation/history。最终方法不是 SegResNet，也不是 EMA 半监督。

推荐新内容：

```text
Task 1 is treated as 3D binary foreground segmentation. We convert the official labeled CT training cases into nnU-Net v2 Dataset101_MVAA_Task1, where the CT image is stored as channel 0 and the segmentation contains background label 0 and target label 1. The validation images are copied into the nnU-Net imagesTs folder for prediction.

We use nnU-Net v2 with the 3d_fullres configuration. The planner selects a PlainConvUNet with feature widths [32, 64, 128, 256, 320, 320], target spacing [0.5, 0.357421875, 0.357421875], patch size [112, 128, 160], batch size 2, and CTNormalization. The five generated cross-validation splits contain 21/6, 21/6, 22/5, 22/5, and 22/5 training/validation cases, respectively. Each fold is trained for 1000 epochs using the standard nnU-Net training schedule. For the final v17/v19 Task1 prediction, we use checkpoint_best.pth from folds 0-4 and ensemble the five folds during inference.

Compared with the previous SegResNet baseline, nnU-Net 5fold ensembling substantially improves the online Task1 DSC from 0.8106 to 0.8575.
```

需要写入的 Task1 关键实现细节：

| Item | Value |
|---|---|
| Dataset | `Dataset101_MVAA_Task1` |
| Framework | nnU-Net v2 |
| Configuration | `3d_fullres` |
| Network | `PlainConvUNet` |
| Labels | background 0, target 1 |
| Normalization | `CTNormalization` |
| Spacing | `[0.5, 0.357421875, 0.357421875]` |
| Patch size | `[112, 128, 160]` |
| Batch size | `2` |
| Folds | `0,1,2,3,4` |
| Checkpoint | `checkpoint_best.pth` |
| Postprocess | none |
| Online result | DSC `0.8575385873`, HD `4.6311747737`, ASD `0.2792648483` |

## 8. Method 2.4 Task2 TEE 需要重写

旧 Task2 3D U-Net large、ROI160/ROI192 的描述不再是最终最优。可以作为历史候选或 ablation 简述，但主方法要换成 nnU-Net。

推荐新内容：

```text
Task 2 is treated as 3D multi-class segmentation with background label 0 and two foreground classes 1 and 2. We convert all 105 annotated TEE volumes into nnU-Net v2 Dataset102_MVAA_Task2. The ultrasound image is stored as channel 0, and the original labels are preserved as classes 0, 1, and 2.

We use nnU-Net v2 with the 3d_fullres configuration. The planner selects a PlainConvUNet with feature widths [32, 64, 128, 256, 320, 320], target spacing [0.5395808816, 0.2322079986, 0.3726583719], patch size [96, 128, 160], batch size 2, and ZScoreNormalization. The five cross-validation folds each use 84 training cases and 21 validation cases. All folds are trained for 1000 epochs. During inference, we ensemble folds 0-4 using checkpoint_best.pth.

This nnU-Net 5fold ensemble replaces our earlier manually tuned 3D U-Net large model. It improves Task2 online DSC from 0.8143 to 0.8463 and improves ASD from 0.6924 to 0.6406, while HD increases slightly from 10.9041 to 11.1969.
```

需要写入的 Task2 关键实现细节：

| Item | Value |
|---|---|
| Dataset | `Dataset102_MVAA_Task2` |
| Framework | nnU-Net v2 |
| Configuration | `3d_fullres` |
| Network | `PlainConvUNet` |
| Labels | background 0, class_1 1, class_2 2 |
| Normalization | `ZScoreNormalization` |
| Spacing | `[0.5395808816, 0.2322079986, 0.3726583719]` |
| Patch size | `[96, 128, 160]` |
| Batch size | `2` |
| Fold split | each fold 84 train / 21 validation |
| Folds | `0,1,2,3,4` |
| Checkpoint | `checkpoint_best.pth` |
| Postprocess | none |
| Online result | DSC `0.8463245905`, HD `11.1968886295`, ASD `0.6406354095` |

## 9. Method 2.5 Task3 VID 需要小幅更新

Task3 主体仍可保留，但以下值需要改：

- best epoch：旧文 `48`，当前记录为 `31`。
- threshold：旧文 `0.25`，当前最优 v15 为 `0.285`。
- 线上结果需要更新。
- 可补充 “threshold search found 0.285 best online-balanced setting”。

推荐替换句子：

```text
The selected Task 3 checkpoint is the UNet++ ResNet34 ImageNet model at epoch 31. At inference time, we enable test-time augmentation and use a tuned probability threshold of 0.285. This threshold was selected from a focused validation search and improved the online HD and ASD compared with the original 0.25 threshold.
```

Task3 关键值：

| Item | Value |
|---|---|
| Model | UNet++ |
| Encoder | ResNet34 |
| Encoder weights | ImageNet |
| Input size | `448 x 800` |
| Best checkpoint | `outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt` |
| Best epoch | `31` |
| TTA | enabled |
| Threshold | `0.285` |
| Online result | DSC `0.7701966980`, HD `95.6273285081`, ASD `14.5377862258` |

## 10. Section 2.6 Use of Labeled and Unlabeled Data 需要修改

旧文说 Task1 和 Task3 使用 unlabeled data；当前最终最优中 Task1 不再使用 unlabeled data。

建议改成：

```text
In the final v19 configuration, Task 1 and Task 2 use only the labeled volumetric training data within nnU-Net v2 five-fold cross-validation. The unlabeled Task 1 CT volumes were explored in earlier semi-supervised SegResNet experiments but were not used in the final best-performing submission. Task 3 retains the semi-supervised branch and uses unlabeled surgical frames through EMA pseudo-labeling.
```

Table 1 建议更新：

| Task | Modality | Supervision in final v19 | Data usage |
|---|---|---|---|
| Task 1 | Cardiac CT | Supervised nnU-Net 5fold | 27 labeled cases; 5fold splits of 21/6, 21/6, 22/5, 22/5, 22/5 |
| Task 2 | 3D TEE | Supervised nnU-Net 5fold | 105 labeled cases; each fold 84 train / 21 validation |
| Task 3 | Surgical frames | Semi-supervised 2D | 150 labeled train, 30 internal validation, 1379 unlabeled; threshold search on validation split |

如果论文要求严格描述 challenge-provided data，而不是最终实际使用，可以另加一句：

```text
Although unlabeled CT volumes are provided by the challenge, they are not used in the final nnU-Net Task 1 submission because the supervised 5fold nnU-Net ensemble achieved better online performance.
```

## 11. Section 2.7 Reproducible Inference 需要修改

旧文说 selected checkpoints are Table 4 local-validation checkpoints。现在应说明 final v19 的固定 checkpoints：

```text
For Task 1 and Task 2, inference is performed with nnUNetv2_predict using folds 0-4 and checkpoint_best.pth. The five-fold probabilities are ensembled by nnU-Net before exporting the final NIfTI segmentation. The resulting filenames are converted back to the MVAA submission convention with task-specific conversion scripts. For Task 3, the selected UNet++ ResNet34 checkpoint is evaluated with TTA and threshold 0.285. The final v19 validation zip contains 30 Task1 NIfTI masks, 20 Task2 NIfTI masks, 48 Task3 PNG masks, and one JSON manifest per task.
```

建议补充路径：

```text
Task1 predictions are generated from Dataset101_MVAA_Task1 imagesTs and converted to t1_ct/*-pred.nii.gz. Task2 predictions are generated from Dataset102_MVAA_Task2 imagesTs and converted to t2_tee/val_XXX-pred.nii.gz. Task3 predictions follow the baseline video-frame folder structure.
```

## 12. Table 2 需要替换

旧 Table 2：

- Task1 SegResNet base, 128^3 ROI, 150 epochs, epoch 58
- Task2 3D U-Net large, 160^3 ROI, 150 epochs, epoch 74
- Task3 threshold 0.25, epoch 48

建议新 Table 2：

| Task | Model | Input / patch size | Training | Selection |
|---|---|---|---|---|
| Task 1 | nnU-Net v2 3d_fullres PlainConvUNet | patch `[112,128,160]`, spacing `[0.5,0.3574,0.3574]` | 5 folds, 1000 epochs/fold, batch 2 | folds 0-4 `checkpoint_best.pth` ensemble |
| Task 2 | nnU-Net v2 3d_fullres PlainConvUNet | patch `[96,128,160]`, spacing `[0.5396,0.2322,0.3727]` | 5 folds, 1000 epochs/fold, batch 2 | folds 0-4 `checkpoint_best.pth` ensemble |
| Task 3 | UNet++ ResNet34 | `448 x 800` | 100 epochs configured, selected epoch 31 | TTA, threshold `0.285` |

## 13. Table 3 需要替换

旧 Table 3 的 Task1/Task2 都是手写 crop/sliding-window 配置。建议改为：

| Task | Preprocessing | Training augmentation | Inference |
|---|---|---|---|
| Task1 | nnU-Net CTNormalization; target spacing `[0.5,0.3574,0.3574]`; patch `[112,128,160]` | nnU-Net default 3D augmentation | nnU-Net 5fold ensemble, `checkpoint_best.pth`, no postprocess |
| Task2 | nnU-Net ZScoreNormalization; target spacing `[0.5396,0.2322,0.3727]`; patch `[96,128,160]` | nnU-Net default 3D augmentation | nnU-Net 5fold ensemble, `checkpoint_best.pth`, no postprocess |
| Task3 | resize to `448 x 800`, ImageNet normalization | flips, rotations, weak/strong photometric augmentation, foreground-balanced sampling, EMA pseudo-labeling | TTA, threshold `0.285` |

## 14. Experiments 3.1 Dataset and Splits 需要修改

旧文 Task1 split 是 24/3 + unlabeled；Task2 是固定 85/20。这不符合最终 nnU-Net 5fold。

建议改成：

```text
For Task 1, all 27 labeled CT cases are used in nnU-Net five-fold cross-validation. The generated folds contain 21 or 22 training cases and 5 or 6 validation cases. For Task 2, all 105 annotated 3D TEE cases are used in nnU-Net five-fold cross-validation, with 84 training cases and 21 validation cases in each fold. For Task 3, we keep the previous internal video/frame split for model and threshold selection.
```

不要再说 Task1 final uses 1040 unlabeled volumes，除非作为 early experiment。

## 15. Experiments 3.2 Implementation Details 需要修改

建议补充：

```text
For Task 1 and Task 2, nnU-Net v2 planning and preprocessing are run after converting the official files into Dataset101_MVAA_Task1 and Dataset102_MVAA_Task2. We use the default nnUNetTrainer and nnUNetPlans 3d_fullres configuration. Each fold is trained independently, and inference uses the default nnU-Net five-fold ensemble with checkpoint_best.pth.
```

Task2 fold validation pseudo Dice 可作为实现确认，不建议放主表：

| Fold | Latest pseudo Dice |
|---|---|
| fold0 | `[0.8410, 0.8576]` |
| fold1 | `[0.8294, 0.8480]` |
| fold2 | `[0.8322, 0.8620]` |
| fold3 | `[0.8517, 0.8617]` |
| fold4 | `[0.8260, 0.8344]` |

## 16. Table 4 Best Local Validation Results 需要改写或删除

旧 Table 4 已不适合当主结果，因为现在最重要的是线上 v19 结果，而 Task1/Task2 使用 nnU-Net 5fold 后本地 fold pseudo Dice 不等同于旧 local validation。

建议把 Table 4 改为 “Model-selection summary”：

| Task | Final selected model | Selection criterion | Notes |
|---|---|---|---|
| Task1 | nnU-Net 5fold `checkpoint_best.pth` | Online DSC comparison: v17 > v16 for DSC | v16 has better HD/ASD, v17 selected by DSC |
| Task2 | nnU-Net 5fold `checkpoint_best.pth` | Online DSC/ASD improvement in v19 | v14 has slightly better HD |
| Task3 | UNet++ ResNet34 epoch 31 | threshold search, TTA, online balance | threshold `0.285` selected |

如果必须保留 local validation 表，请把旧结果标为 “previous local baseline” 而不是 “best current”。

## 17. Table 5 Official Results 需要填写

旧 Table 5 是 TBD。建议替换为 v19 线上结果：

| Task | DSC/Dice | HD | ASD | Notes |
|---|---:|---:|---:|---|
| Task1 CT | 0.8575385873 | 4.6311747737 | 0.2792648483 | v17 Task1 nnU-Net 5fold `checkpoint_best.pth` |
| Task2 TEE | 0.8463245905 | 11.1968886295 | 0.6406354095 | v19 Task2 nnU-Net 5fold `checkpoint_best.pth` |
| Task3 VID | 0.7701966980 | 95.6273285081 | 14.5377862258 | v15 UNet++ ResNet34, TTA, threshold 0.285 |

注意：如果这不是最终 hidden-test，而是 validation server，要把表名写成：

```text
Table 5: Online validation results of the final v19 submission.
```

不要写成 hidden-test，除非已经确认。

## 18. Results 4.1 和 4.2 需要修改

旧 4.1 解释 local validation：

- Task1 local DSC 0.8303。
- Task2 local DSC 0.7796。
- Task3 local Dice 0.7785。

建议改成：

```text
The final v19 submission substantially improves the two volumetric tasks after replacing manually tuned models with nnU-Net five-fold ensembles. For Task 1, the online DSC increases from the previous SegResNet-based result of 0.8106 to 0.8575. For Task 2, replacing the 3D U-Net large model with a 5fold nnU-Net ensemble increases online DSC from 0.8143 to 0.8463 and decreases ASD from 0.6924 to 0.6406, although HD slightly increases from 10.9041 to 11.1969. Task 3 keeps the UNet++ ResNet34 pipeline and uses threshold 0.285, achieving DSC 0.7702, HD 95.6273, and ASD 14.5378.
```

旧 4.2 “Hidden-Test Results not available” 应改为在线评估结果。如果仍未最终 hidden test，写：

```text
At the time of this revision, the available organizer-side feedback corresponds to the online evaluation server. We therefore report these values as online evaluation results and keep final hidden-test results separate if the challenge later releases a distinct final-test phase.
```

## 19. Discussion and Limitations 需要修改

旧文强调 Task1/Task3 半监督 limitations。现在应改成：

- Task1 和 Task2 的主要限制是 nnU-Net 5fold 推理成本更高。
- Task2 的 HD 仍略差于 v14，需要后处理或 v14/v19 fusion 改善边界 outliers。
- Task3 仍未利用视频时序。
- 内部验证与线上结果不完全一致，尤其 nnU-Net fold pseudo Dice 不能直接等同线上 DSC。

建议新增段落：

```text
A practical limitation of the final solution is inference cost. The two volumetric tasks use five-fold nnU-Net ensembling, which improves robustness but requires running five 3D models per case. Another limitation is metric-specific trade-off. For Task 2, the nnU-Net ensemble gives the best DSC and ASD but slightly worsens HD compared with the previous v14 3D U-Net large model, suggesting that rare boundary outliers remain. Future work should investigate postprocessing or hybrid ensembling between the nnU-Net and ROI192 U-Net predictions. For Task 3, the frame-based model still ignores temporal continuity.
```

## 20. Conclusion 需要重写

旧 Conclusion：

- “semi-supervised 3D cardiac CT segmentation, fully supervised 3D TEE segmentation...”
- “local validation DSC/Dice values 0.8303, 0.7796, 0.7785...”

建议改成：

```text
We presented a task-specific MVAA 2026 solution in which the two volumetric tasks are solved with nnU-Net v2 five-fold ensembles and the surgical-frame task is solved with a semi-supervised UNet++ ResNet34 pipeline. Replacing the previous manually tuned Task1 SegResNet and Task2 3D U-Net models with nnU-Net substantially improved online performance. The final v19 submission achieved DSC/HD/ASD of 0.8575/4.6312/0.2793 for Task1, 0.8463/11.1969/0.6406 for Task2, and 0.7702/95.6273/14.5378 for Task3. Future work will focus on reducing ensemble inference cost, improving Task2 HD with postprocessing or hybrid ensembling, and exploiting temporal consistency for surgical videos.
```

## 21. 参考文献需要调整

nnU-Net 论文 `[6]` 已经存在，应在 Task1/Task2 方法中更核心地引用。

SegResNet `[19]` 不再是最终主方法。如果正文不再描述 SegResNet，可以：

- 删除 `[19]` 引用，或
- 保留在 “previous experiments / baseline” 里作为历史候选。

3D U-Net `[4]` 仍可保留，因为 nnU-Net 的网络家族仍源自 U-Net/3D U-Net，但不要把 Task2 主模型写成手工 3D U-Net large。

半监督相关 `[22,23]` 仍适用于 Task3；不再应用于最终 Task1。

## 22. 需要避免的错误表述

以下旧表述不要继续出现在最终方法中：

- “Task 1 uses SegResNet-base as the best current configuration.”
- “Task 1 uses unlabeled CT volumes in the final method.”
- “Task 2 uses fully supervised 3D U-Net large as the final best method.”
- “Task 2 input ROI is 160 x 160 x 160 in the final method.”
- “Task3 best threshold is 0.25.”
- “Official hidden-test results are TBD.” 如果现在报告 v19，则至少应报告 online evaluation results。
- “Selected checkpoints are the best local-validation checkpoints listed in Table 4.” 对 Task1/Task2，应改成 folds 0-4 `checkpoint_best.pth` ensemble。

## 23. 推荐最终结果汇总表

建议在 Results 或 Conclusion 前放一个最终表：

| Task | Final method | DSC | HD | ASD |
|---|---|---:|---:|---:|
| Task1 CT | nnU-Net v2 3d_fullres 5fold `checkpoint_best.pth` | 0.857539 | 4.631175 | 0.279265 |
| Task2 3D TEE | nnU-Net v2 3d_fullres 5fold `checkpoint_best.pth` | 0.846325 | 11.196889 | 0.640635 |
| Task3 Video | UNet++ ResNet34, TTA, threshold 0.285 | 0.770197 | 95.627329 | 14.537786 |

## 24. 推荐消融/历史对比表

为了说明为什么要“丢弃之前论文里的最优方法”，建议新增一张对比表：

| Task | Previous paper method | Previous online DSC | Final v19 method | Final online DSC | Change |
|---|---|---:|---|---:|---:|
| Task1 | SegResNet-base bestcfg | 0.810562 | nnU-Net 5fold best checkpoint | 0.857539 | +0.046977 |
| Task2 | 3D U-Net large ROI192 + postprocess | 0.814276 | nnU-Net 5fold best checkpoint | 0.846325 | +0.032049 |
| Task3 | UNet++ ResNet34 threshold 0.25 | 0.768802 | UNet++ ResNet34 threshold 0.285 | 0.770197 | +0.001395 |

补充说明：

- Task1 的 v16 final checkpoint ensemble 有更好的 HD/ASD，但 v17 checkpoint_best 的 DSC 最高，因此最终按 DSC 优先选 v17。
- Task2 的 v14 有更好的 HD，但 v19 的 DSC 和 ASD 更优，且 DSC 提升很大，因此最终按 DSC 优先选 v19。
- Task3 的 v12 DSC 更高，但 HD/ASD 明显变差；v15 是三项指标更均衡的选择。

## 25. 论文中可保留的旧内容

以下内容基本可以保留：

- Introduction 中关于多模态 MVAA 挑战难点的描述。
- Task2 中关于 3D TEE 噪声、dropout、边界模糊的医学背景。
- Task3 的数据读取、label 10、UNet++ ResNet34、Dice-Focal、EMA pseudo-labeling 大部分描述。
- Evaluation metrics 对 DSC/HD/ASD 的解释。
- Docker/submission 输出格式说明，但要更新 checkpoint 和推理配置。
- Discussion 中关于 Task3 不利用时序的 limitation。

## 26. 实现细节索引

当前最优结果对应的本地记录：

- 总结文档：`best_task_results_summary.md`
- 分数文档：`mvaa_submission_scores.md`
- v19 记录：`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/checkpoint_record.md`

Task1 nnU-Net：

- Dataset root：`outputs/nnunet_task1_fold0`
- Results：`outputs/nnunet_task1_fold0/nnUNet_results/Dataset101_MVAA_Task1/nnUNetTrainer__nnUNetPlans__3d_fullres`
- Final submission source：`outputs/submissions/submit_v17_task1_nnunet5fold_bestckpt_task2_v14_task3_v15/submission/t1_ct`

Task2 nnU-Net：

- Dataset root：`outputs/nnunet_task2_fold0`
- Results：`outputs/nnunet_task2_fold0/nnUNet_results/Dataset102_MVAA_Task2/nnUNetTrainer__nnUNetPlans__3d_fullres`
- Final submission source：`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission/t2_tee`

Task3：

- Checkpoint：`outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt`
- Final submission source：`outputs/submissions/submit_v15_task3_thr0285/submission/t3_vid`

## 27. 建议改稿顺序

1. 先改 Abstract、Contribution、Figure 2，让论文主线从旧 SegResNet/3D U-Net 转成 nnU-Net 5fold。
2. 重写 Task1 和 Task2 Method 小节。
3. 修改 Table 1、Table 2、Table 3。
4. 更新 Task3 的 epoch、threshold、结果。
5. 把 Results 从 local validation/TBD 改成 v19 online evaluation。
6. 最后改 Discussion/Conclusion，强调 nnU-Net 带来的提升和当前剩余问题。
