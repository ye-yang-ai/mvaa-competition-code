# MVAA 三个 Task 最优配置与结果汇总

更新时间：2026-07-29

本文件汇总当前已记录的线上最优/最稳结果。当前按综合均衡策略选择 `v39` 作为主提交候选：Task1 采用 v25，Task2 采用 v19，Task3 采用 v39。v38 的 Task3 DSC 更高，但 HD/ASD 明显差于 v39，因此只作为 DSC 激进候选。

## 总览

线上指标中，DSC 越高越好，HD/ASD 越低越好。

| Task | 推荐版本 | 配置摘要 | DSC | HD | ASD | 判断 |
|---|---|---|---:|---:|---:|---|
| task1_ct | v25 | Dataset111 pseudo-top100 nnU-Net v2 5fold ensemble | 0.857813 | 4.671629 | 0.274123 | 当前 Task1 DSC/ASD 最优；v16/v37 HD 更好 |
| task2_tee | v19 | nnU-Net v2 5fold ensemble | 0.846325 | 11.196889 | 0.640635 | 当前 Task2 DSC/ASD 最佳；v14 HD 略好 |
| task3_vid | v39 | v35 五模型 ensemble + LemonFM gated branch, threshold 0.36 | 0.806144 | 72.862321 | 12.207524 | 当前 Task3 均衡最优；v38 DSC 更高但距离退化 |

## Task1 CT

- 最优线上版本：`v17`
- 模型/框架：`nnU-Net v2`
- 配置：Dataset101_MVAA_Task1，`3d_fullres`
- backbone/network：nnU-Net 自动规划的 `PlainConvUNet`
- ensemble：fold `0-4` 5fold ensemble
- v16 checkpoint：各 fold 的 `checkpoint_final.pth`
- v17 checkpoint：各 fold 的 `checkpoint_best.pth`
- patch size：`112 128 160`
- batch size：`2`
- target spacing：`0.5, 0.357421875, 0.357421875`
- 推理输入：`outputs/nnunet_task1_fold0/nnUNet_raw/Dataset101_MVAA_Task1/imagesTs`
- 推理输出：`outputs/submissions/submit_v17_task1_nnunet5fold_bestckpt_task2_v14_task3_v15/submission/t1_ct`

线上结果：

| DSC | HD | ASD |
|---:|---:|---:|
| 0.857539 | 4.631175 | 0.279265 |

Task1 线上对比：

| Metric | v15 | v16 final | v17 best | 当前最好 |
|---|---:|---:|---:|---|
| DSC | 0.810562 | 0.857340 | 0.857539 | v17 |
| HD | 6.902283 | 4.583182 | 4.631175 | v16 |
| ASD | 0.396594 | 0.278065 | 0.279265 | v16 |

v18 后处理结果：DSC `0.857534`，HD `4.603581`，ASD `0.280489`。相对 v17，HD 小幅改善但 DSC/ASD 变差；相对 v16，HD/ASD 仍更差，因此不推荐。

本地验证记录：

| score | DSC | HD | ASD |
|---:|---:|---:|---:|
| 0.7597828495 | 0.8297671676 | 11.4183769226 | 0.8392195106 |

旧 Task1 最优记录：

- 版本：`v10-v15`
- 模型：`SegResNet base`
- checkpoint：`outputs/exp/task1_segresnet_base_e150_b1_lr3e4_u03_thr08_bestcfg/checkpoints/best_model.pt`
- 线上结果：DSC `0.810562`，HD `6.902283`，ASD `0.396594`

## Task2 TEE

- 最优线上版本：`v19`
- 模型/框架：`nnU-Net v2`
- 配置：Dataset102_MVAA_Task2，`3d_fullres`
- backbone/network：nnU-Net 自动规划的 `PlainConvUNet`
- ensemble：fold `0-4` 5fold ensemble
- checkpoint：各 fold 的 `checkpoint_best.pth`
- patch size：`96 128 160`
- batch size：`2`
- target spacing：`0.5395808815956116, 0.2322079986333847, 0.372658371925354`
- 推理输入：`outputs/nnunet_task2_fold0/nnUNet_raw/Dataset102_MVAA_Task2/imagesTs`
- 推理输出：`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission/t2_tee`

线上结果：

| DSC | HD | ASD |
|---:|---:|---:|
| 0.846325 | 11.196889 | 0.640635 |

v19 相对 v14/v17/v18 的 Task2 变化：

| Metric | v14/v17/v18 | v19 | Delta |
|---|---:|---:|---:|
| DSC | 0.814276 | 0.846325 | +0.032049 |
| HD | 10.904111 | 11.196889 | +0.292778 |
| ASD | 0.692376 | 0.640635 | -0.051741 |

判断：v19 明确刷新 Task2 DSC，ASD 也更好；HD 略逊于 v14。因此当前按 DSC 优先采用 v19，若单独追求 Task2 HD 可保留 v14 作为备选。

旧 Task2 最优记录：

- 版本：`v14`
- 模型：`3D UNet large`
- 训练配置：200 epochs 配置，实际训练到 epoch 121 后停止，batch size 1，learning rate `2e-4`，weight decay `1e-5`
- ROI：`192 192 160`
- 类别数：3
- train crops：`2`
- sliding-window batch size：`1`
- 评分权重：DSC `0.5`，HD `0.25`，ASD `0.25`
- seed：`42`
- best epoch：`86`
- checkpoint：`outputs/exp/task2_unet_large_e200_b1_roi192x192x160_c2_lr2e4_score525_v13/checkpoints/best_model.pt`
- 后处理：`min_size=100, keep_components=1, fill_holes=True, close_iters=0`

线上结果：

| DSC | HD | ASD |
|---:|---:|---:|
| 0.814276 | 10.904111 | 0.692376 |

本地验证记录：

| score | DSC | HD | ASD |
|---:|---:|---:|---:|
| - | 0.8060218692 | 27.6713314056 | 1.9451344013 |

相对 v10 Task2 线上结果：

| Metric | v10 online | v14 online | Delta |
|---|---:|---:|---:|
| DSC | 0.805315 | 0.814276 | +0.008961 |
| HD | 16.629738 | 10.904111 | -5.725627 |
| ASD | 0.926641 | 0.692376 | -0.234265 |

### Task2 v13 本地候选

本轮训练用于尝试更大 ROI 和更长训练，本地验证优于 v10 Task2，并已通过 `v14` 在线上验证刷新 Task2。

- run dir：`outputs/exp/task2_unet_large_e200_b1_roi192x192x160_c2_lr2e4_score525_v13`
- 模型：`3D UNet large`
- 训练配置：200 epochs，batch size 1，learning rate `2e-4`，weight decay `1e-5`
- ROI：`192 192 160`
- 类别数：3
- train crops：`2`
- sliding-window batch size：`1`
- val count：20
- val interval：2
- 评分权重：DSC `0.5`，HD `0.25`，ASD `0.25`
- seed：`42`
- 实际训练进度：日志记录到 epoch `121/200` 后停止
- best epoch：`86`
- checkpoint：`outputs/exp/task2_unet_large_e200_b1_roi192x192x160_c2_lr2e4_score525_v13/checkpoints/best_model.pt`

本地验证结果：

| score | DSC | HD | ASD |
|---:|---:|---:|---:|
| 0.6986707229 | 0.8052513003 | 31.4862308502 | 1.9926220179 |

相对 v10 Task2 bestcfg 的本地验证变化：

| Metric | v10 local | v13 local | Delta |
|---|---:|---:|---:|
| score | 0.6853937357 | 0.6986707229 | +0.0132769871 |
| DSC | 0.7810249329 | 0.8052513003 | +0.0242263675 |
| HD | 46.1612014771 | 31.4862308502 | -14.6749706268 |
| ASD | 2.4988479614 | 1.9926220179 | -0.5062259436 |

判断：v13 在本地验证上是明确全面改善，尤其 HD 改善很大，DSC 也有明显提升。但它尚未线上验证，不能直接替代 v10 的线上结论。下一步适合先用 v13 checkpoint 生成 Task2 提交，或与 v10 做小权重 ensemble 再比较。

#### Task2 v13 后处理验证

评估脚本：`task2/evaluate_task2_postprocess.py`

验证 checkpoint：`outputs/exp/task2_unet_large_e200_b1_roi192x192x160_c2_lr2e4_score525_v13/checkpoints/best_model.pt`

本地验证 raw baseline：

| DSC | HD | ASD |
|---:|---:|---:|
| 0.8052433133 | 31.4974613190 | 1.9927623272 |

后处理对比：

| 配置 | DSC | HD | ASD | 相对 raw 判断 |
|---|---:|---:|---:|---|
| `min_size=100, keep_components=1, fill_holes=True` | 0.8060218692 | 27.6713314056 | 1.9451344013 | 最好，DSC/HD/ASD 全部改善 |
| `min_size=100, keep_components=1, fill_holes=False` | 0.8060169220 | 27.6713314056 | 1.9454380274 | 几乎等同 fill holes，说明主要收益来自 keep largest |
| `min_size=100, keep_components=2, fill_holes=True` | 0.8052610159 | 30.5735836029 | 1.9928576946 | HD 小幅改善，但 DSC/ASD 基本不变 |

推荐默认后处理：每个前景类别保留最大连通域，删除小于 `100` voxel 的连通域；`fill_holes` 可以开启，但当前验证显示它不是主要收益来源。

最佳后处理相对 raw 的变化：

| Metric | Raw | Post | Delta |
|---|---:|---:|---:|
| DSC | 0.8052433133 | 0.8060218692 | +0.0007785559 |
| HD | 31.4974613190 | 27.6713314056 | -3.8261299133 |
| ASD | 1.9927623272 | 1.9451344013 | -0.0476279259 |

判断：Task2 后处理确实有效，主要改善 HD，且没有牺牲 DSC。v14 线上结果曾确认 v13+post 是 Task2 最优方案；当前 v19 已进一步刷新 Task2 DSC/ASD，但 v14 的 HD 仍略好。

#### Task2 v13+post 提交 v14

- 提交版本：`v14`
- 提交目录：`outputs/submissions/submit_v14_task2_v13_post/submission`
- 提交压缩包：`outputs/submissions/submit_v14_task2_v13_post/submission.zip`
- 记录文件：`outputs/submissions/submit_v14_task2_v13_post/checkpoint_record.md`
- 组成：Task1 复制 v10；Task2 使用 v13 batch size 1 权重并开启后处理；Task3 复制 v10
- Task2 checkpoint：`outputs/exp/task2_unet_large_e200_b1_roi192x192x160_c2_lr2e4_score525_v13/checkpoints/best_model.pt`
- Task2 后处理参数：`min_size=100, keep_components=1, fill_holes=True, close_iters=0`
- 本地 Task2 post 结果：DSC `0.8060218692`，HD `27.6713314056`，ASD `1.9451344013`
- 线上结果：DSC `0.8142756501175983`，HD `10.904111011547963`，ASD `0.6923764603409446`
- 状态：已完成线上测评，Task2 明确刷新线上最佳

## Task3 VID

- 推荐线上版本：`v15`
- 模型：`Unet++`
- encoder：`ResNet34`
- encoder weights：`ImageNet`
- 输入尺寸：`448 800`
- 训练配置：100 epochs，batch size 2，learning rate `2e-4`，weight decay `1e-5`
- loss：`dice_focal`
- loss 权重：dice `0.7`，BCE `0.3`，focal `0.3`
- focal 参数：gamma `2.0`，alpha `0.75`
- 半监督配置：semi warmup 20 epochs，unsup weight `0.6`，unsup ramp 30 epochs，EMA decay `0.99`
- 伪标签参数：positive threshold `0.7`，negative threshold `0.1`，min area `80`
- v10 原始验证/推理阈值：`0.25`
- v15 推理阈值：`0.285`
- TTA：enabled
- AMP：enabled
- seed：`42`
- best epoch：`31`
- checkpoint：`outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt`

v10 线上结果：

| DSC | HD | ASD |
|---:|---:|---:|
| 0.768802 | 104.463167 | 15.432556 |

v15 线上结果：

| DSC | HD | ASD |
|---:|---:|---:|
| 0.770197 | 95.627329 | 14.537786 |

本地验证记录：

| score | DSC | HD | ASD | threshold |
|---:|---:|---:|---:|---:|
| 0.5571122781 | 0.7787975669 | 64.9372940063 | 11.0383148193 | 0.25 |

Task3 额外说明：

| 版本 | 配置 | DSC | HD | ASD | 结论 |
|---|---|---:|---:|---:|---|
| v10 | ResNet34 单模型 bestcfg | 0.768802 | 104.463167 | 15.432556 | 综合最均衡，ASD 最好 |
| v11 | v10 加保守二值后处理 `min_area=400, keep_top=2, close_iters=1` | 0.754698 | 99.258927 | 17.491449 | HD 最好，但 DSC/ASD 下降 |
| v12 | ResNet34 + EfficientNet-B4 概率 ensemble `0.5/0.5, threshold=0.15` | 0.790226 | 138.577390 | 16.059344 | DSC 最高，但 HD 明显变差 |
| v15 | v10 checkpoint + TTA + `threshold=0.285` | 0.770197 | 95.627329 | 14.537786 | 相对 v10/v14 三项全部改善，当前最均衡 |

因此如果只追 Task3 DSC，可考虑 `v12`；如果看三项指标的平衡，使用 `v15`。

### Task3 v10 阈值/后处理搜索

评估脚本：`task3/evaluate_task3_postprocess_grid.py`

验证 checkpoint：`outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt`

验证 split：`seed=42, val_video_count=2`，验证视频为 `REC_20250205_102353_979A`、`REC_20250322_101917_746A`，共 60 frames，其中 39 foreground frames、21 empty frames。指标在 448x800 验证尺寸上计算。

第一轮聚焦搜索：

- 目录：`outputs/analysis/task3_postprocess_grid/v10_focused`
- 搜索：threshold `0.20-0.35`，`min_area=0/50/80/120/200/500`，`keep_top=0/1/2/3`，`fill_holes=0/1`，`close_iters=0/1`

第二轮细阈值搜索：

- 目录：`outputs/analysis/task3_postprocess_grid/v10_thr_fine`
- 搜索：threshold `0.275-0.325`，步长 `0.005`，`min_area=0/50/80/120`，不做连通域保留、fill holes 或 close

本地验证最佳：

| 配置 | score | DSC fg | DSC all | HD | ASD | empty FP rate |
|---|---:|---:|---:|---:|---:|---:|
| baseline `threshold=0.25` | 0.596360 | 0.823322 | 0.585159 | 58.718449 | 8.638582 | 0.857143 |
| best score `threshold=0.285` | 0.600255 | 0.825315 | 0.603122 | 56.539052 | 8.362578 | 0.809524 |
| best Dice `threshold=0.275` | 0.598494 | 0.825421 | 0.603191 | 58.734287 | 8.442252 | 0.809524 |
| simple robust `threshold=0.30` | 0.600224 | 0.825068 | 0.602961 | 56.622837 | 8.325177 | 0.809524 |

判断：当前 split 上，连通域/填洞/close 没有超过单纯调高阈值；`threshold=0.285` 综合分最高，`threshold=0.30` 的 ASD 略好且更保守。v15 线上验证确认 `threshold=0.285` 有效，Task3 DSC、HD、ASD 相对 v10/v14 全部改善。

## 推荐提交

当前推荐主提交：`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission.zip`

距离指标备选：`outputs/submissions/submit_v16_task1_nnunet5fold_task2_v14_task3_thr0285/submission.zip`

不推荐候选：`outputs/submissions/submit_v18_task1_v17_post_min100_task2_v14_task3_v15/submission.zip`

对应三项线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7701966979705538, HD 95.62732850814587, ASD 14.537786225800227
```

历史已验证基线：`outputs/submissions/submit_v10_bestcfg_raw/submission.zip`

v17 相比 v16 只把 Task1 从 `checkpoint_final.pth` ensemble 换成 `checkpoint_best.pth` ensemble，DSC 略高，但 HD/ASD 略差。v18 在 v17 上删除小连通域，未带来有效提升。v19 在 v17 基础上只替换 Task2 为 nnU-Net 5fold ensemble，Task2 DSC 大幅提高、ASD 下降，HD 略差于 v14；按当前 DSC 优先策略采用 v19。
