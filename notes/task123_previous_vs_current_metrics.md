# Task1/2/3 Previous Paper Method vs Current Best Method

本文档整理 `notes/main.pdf` 中原论文方法与当前最优 v19 方案的指标对比。主表采用线上/online evaluation 结果，保证同一评测口径下可比较。

当前最优提交：

`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission.zip`

## 主对比表：线上指标

DSC 越高越好；HD 和 ASD 越低越好。Delta 计算方式为：

- `Delta DSC = Current DSC - Previous DSC`
- `Delta HD = Current HD - Previous HD`
- `Delta ASD = Current ASD - Previous ASD`

| Task | Previous paper method | Previous DSC | Previous HD | Previous ASD | Current best method | Current DSC | Current HD | Current ASD | Delta DSC | Delta HD | Delta ASD |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Task1 CT | SegResNet-base bestcfg | 0.810562 | 6.902283 | 0.396594 | nnU-Net v2 3d_fullres 5fold `checkpoint_best.pth` ensemble | 0.857539 | 4.631175 | 0.279265 | +0.046977 | -2.271108 | -0.117329 |
| Task2 3D TEE | 3D U-Net large ROI192 + postprocess | 0.814276 | 10.904111 | 0.692376 | nnU-Net v2 3d_fullres 5fold `checkpoint_best.pth` ensemble | 0.846325 | 11.196889 | 0.640635 | +0.032049 | +0.292778 | -0.051741 |
| Task3 Video | UNet++ ResNet34, TTA, threshold 0.25 | 0.768802 | 104.463167 | 15.432556 | UNet++ ResNet34, TTA, threshold 0.285 | 0.770197 | 95.627329 | 14.537786 | +0.001395 | -8.835838 | -0.894770 |

## 简要结论

| Task | 结论 |
|---|---|
| Task1 CT | nnU-Net 5fold 相比旧 SegResNet 明显提升，DSC 提高约 `+0.0470`，HD/ASD 同时下降。论文主方法应改为 nnU-Net。 |
| Task2 3D TEE | nnU-Net 5fold 显著刷新 DSC，ASD 也更好；HD 比旧 v14 略差 `+0.2928`，但整体按 DSC 优先应采用 nnU-Net。 |
| Task3 Video | 模型框架不变，主要变化是 checkpoint/阈值策略；threshold `0.285` 相比旧设置使三项线上指标整体更均衡。 |

## 当前最优方法细节

| Task | Current best version | Key implementation details |
|---|---|---|
| Task1 CT | v17, reused in v19 | nnU-Net v2 `Dataset101_MVAA_Task1`, `3d_fullres`, PlainConvUNet, patch `[112,128,160]`, spacing `[0.5,0.357421875,0.357421875]`, batch size 2, folds 0-4, `checkpoint_best.pth`, no postprocess |
| Task2 3D TEE | v19 | nnU-Net v2 `Dataset102_MVAA_Task2`, `3d_fullres`, PlainConvUNet, patch `[96,128,160]`, spacing `[0.5395808816,0.2322079986,0.3726583719]`, batch size 2, folds 0-4, `checkpoint_best.pth`, no postprocess |
| Task3 Video | v15 | UNet++ ResNet34 ImageNet, input `448 x 800`, checkpoint `best.pt`, TTA enabled, threshold `0.285` |

## 论文原文 local validation 指标说明

`notes/main.pdf` 原文中的 Table 4 报告的是 local validation，不是线上评测结果，不能和 v19 online results 直接做数值对比。原文 local validation 如下：

| Task | Previous paper local method | Local DSC/Dice | Local HD | Local ASD | 说明 |
|---|---|---:|---:|---:|---|
| Task1 CT | `task1_segresnet_base_e150_b1_lr3e4_u03_thr08` | 0.8303 | 10.8895 | 0.8180 | local validation |
| Task2 3D TEE | `task2_large_e150_b1_roi160_c2_lr3e4_score525` | 0.7796 | 42.4294 | 2.8090 | local validation |
| Task3 Video | `task3_unetpp_res34_imagenet_e100_bs2` | 0.7785 | 58.2551 | 10.7754 | foreground-only internal validation |

建议论文中把这些 local validation 旧表降级为 “previous local model-selection results”，主结果表改用上面的 online comparison 表。

## 推荐论文表述

可以在论文 Results 中加入如下总结：

```text
Replacing the previous manually tuned volumetric models with nnU-Net v2 five-fold ensembles substantially improved the online performance of Task 1 and Task 2. For Task 1, DSC increased from 0.8106 to 0.8575, while HD and ASD decreased from 6.9023/0.3966 to 4.6312/0.2793. For Task 2, DSC increased from 0.8143 to 0.8463 and ASD decreased from 0.6924 to 0.6406, although HD slightly increased from 10.9041 to 11.1969. Task 3 retained the UNet++ ResNet34 framework, and threshold tuning from 0.25 to 0.285 improved the online DSC/HD/ASD from 0.7688/104.4632/15.4326 to 0.7702/95.6273/14.5378.
```
