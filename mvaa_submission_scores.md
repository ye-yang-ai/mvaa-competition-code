# MVAA 2026 Submission Scores v1-v26

本文档记录已知线上评测结果。v1-v3 如果没有明确保存到对话中的完整结果，暂不补猜；从有明确结果的版本开始记录。

## 版本说明

| Version | 主要配置 | 备注 |
|---|---|---|
| v4 | Task1 best + Task2 best 当时版本 + Task3 medium 后处理 | Task2 较早版本，Task3 后处理改善 HD/ASD |
| v5-light | Task1 best + Task2 UNet seed43 + Task3 light 后处理 | Task2 大幅提升，Task3 light 弱于 medium |
| v5-medium | Task1 best + Task2 UNet seed43 + Task3 medium 后处理 | 用户报告结果与 v5-light 完全一致；本地文件与 light 不同，疑似上传/缓存/复制问题 |
| v6 | Task1 best + Task2 UNet seed43 0.5 / SegResNet base 0.5 + Task3 medium | 旧最佳，已被 v9 小幅超过 |
| v7 | Task1 best + Task2 UNet seed43 0.7 / SegResNet base 0.3 + Task3 medium | 偏 UNet 后下降 |
| v8 | Task1 best + Task2 UNet seed43 0.4 / SegResNet base 0.6 + Task3 medium | 用户报告与 v7 完全一致；本地文件与 v7 不同，疑似异常 |
| v9 | Task1 best + Task2 UNet seed43 0.5 / SegResNet base 0.25 / SegResNet large 0.25 + Task3 medium | 曾为线上 Task2 最佳，已被 v10 超过 |
| v10 | best_task_configs 单模型提交：Task1 SegResNet bestcfg + Task2 large ROI160 bestcfg + Task3 Unet++ ResNet34 bestcfg | 三个 task 均刷新当前线上最佳 |
| v11 | v10 + Task3 保守二值后处理 `min_area=400, keep_top=2, close_iters=1` | Task3 HD 小幅优于 v10，但 DSC/ASD 下降 |
| v12 | v10 Task1/Task2 + Task3 ResNet34/EfficientNet-B4 概率 ensemble `0.5/0.5, threshold=0.15` | Task3 DSC 当前最高，但 HD 明显变差 |
| v14 | v10 Task1/Task3 + Task2 v13 batch size 1 ROI192 权重 + Task2 后处理 | Task2 明确刷新线上最佳；曾为推荐主提交 |
| v15 | v14 Task1/Task2 + Task3 v10 checkpoint TTA `threshold=0.285` | Task3 DSC/HD/ASD 相对 v10/v14 全部改善；曾为推荐主提交 |
| v16 | Task1 nnU-Net 5fold final checkpoint + Task2 v14 + Task3 v15 | Task1 大幅刷新，Task2/Task3 保持当前最强配置；距离指标最优 |
| v17 | Task1 nnU-Net 5fold best checkpoint + Task2 v14 + Task3 v15 | Task1 DSC 最高；曾为主提交，现作为 Task1 稳定基线 |
| v18 | v17 Task1 + 轻量小连通域删除，Task2 v14 + Task3 v15 | 后处理收益不明显；相对 v17，HD 小幅改善但 DSC/ASD 变差，不推荐 |
| v19 | Task1 v17 + Task2 nnU-Net 5fold best checkpoint ensemble + Task3 v15 | Task2 DSC 和 ASD 刷新历史最佳；HD 略差于 v14 |
| v20 | Task1 v17 + Task2 v19 + Task3 v15 `threshold=0.28, min_area=20` | Task3 轻量小连通域后处理试验；线上三项均略差于 v15/v19，不推荐 |
| v21 | Task1 v17 + Task2 v19 + Task3 v15 `threshold=0.285, min_total_area=1500` | Task3 总面积门控试验；线上结果与 v15/v19 完全一致，说明该门控未命中线上有效误检 |
| v22 | Task1 v17 + Task2 v19 + Task3 supervised-only UNet++ EfficientNet-B4 `threshold=0.25` | Task3 DSC 刷新当前最高，但 HD/ASD 明显退化；适合作为高召回模型来源，不推荐直接替代 v15 |
| v23 | Task1 v17 + Task2 v19 + Task3 v15/v22 probability ensemble `0.5/0.5, threshold=0.4` | Task3 DSC 再次刷新且 ASD 基本回到 v15 水平；HD 明显优于 v22/v12 但仍差于 v15，当前按 DSC 优先的主力候选 |
| v24 | Task1 v17 + Task2 v19 + Task3 v15/v22 probability ensemble `0.55/0.45, threshold=0.42` | v23 的轻微保守版本；DSC 小降，HD/ASD 小幅改善，但没有带来实质性 HD 突破 |
| v25 | Task1 Dataset111 pseudo-top100 self-training 5fold + Task2 v19 + Task3 v15 | Task1 自训练增量试验；Task1 DSC 和 ASD 刷新当前最好，HD 略差于 v17/v19，证明 pseudo-top100 有正增量 |
| v26 | Task1 v25 + Task2 v19 + Task3 v23 | 当前组件最优合并包；线上待测，预期为 v25 Task1 + v19 Task2 + v23 Task3 |

## 总表

数值均为线上评测结果。DSC 越高越好，HD/ASD 越低越好。

| Version | Task | DSC | HD | ASD | Notes |
|---|---|---:|---:|---:|---|
| v4 | task1_ct | 0.788095 | 7.392809 | 0.448069 | 与后续版本一致 |
| v4 | task2_tee | 0.721137 | 28.747381 | 2.264272 | 较早 Task2 best |
| v4 | task3_vid | 0.751156 | 214.696681 | 28.867822 | medium 后处理 |
| v5-light | task1_ct | 0.788095 | 7.392809 | 0.448069 | 与 v4 一致 |
| v5-light | task2_tee | 0.757979 | 24.674981 | 1.660142 | UNet seed43 |
| v5-light | task3_vid | 0.749944 | 234.375817 | 31.135561 | light 后处理，弱于 medium |
| v5-medium | task1_ct | 0.788095 | 7.392809 | 0.448069 | 用户报告 |
| v5-medium | task2_tee | 0.757979 | 24.674981 | 1.660142 | 用户报告 |
| v5-medium | task3_vid | 0.749944 | 234.375817 | 31.135561 | 用户报告与 light 完全一致，疑似异常 |
| v6 | task1_ct | 0.788095 | 7.392809 | 0.448069 | 与 v4 一致 |
| v6 | task2_tee | 0.770411 | 22.691037 | 1.489703 | UNet + SegResNet base 0.5/0.5，旧最佳 Task2 |
| v6 | task3_vid | 0.751156 | 214.696681 | 28.867822 | medium 后处理 |
| v7 | task1_ct | 0.788095 | 7.392809 | 0.448069 | 与 v6 一致 |
| v7 | task2_tee | 0.764245 | 23.753866 | 1.618962 | UNet 0.7 / SegResNet base 0.3，弱于 v6 |
| v7 | task3_vid | 0.751156 | 214.696681 | 28.867822 | 与 v6 一致 |
| v8 | task1_ct | 0.788095 | 7.392809 | 0.448069 | 用户报告 |
| v8 | task2_tee | 0.764245 | 23.753866 | 1.618962 | 用户报告与 v7 完全一致，疑似异常 |
| v8 | task3_vid | 0.751156 | 214.696681 | 28.867822 | 与 v7 一致 |
| v9 | task1_ct | 0.788095 | 7.392809 | 0.448069 | 与 v4-v8 一致 |
| v9 | task2_tee | 0.771490 | 21.974976 | 1.429489 | 三模型 ensemble，曾为最佳 Task2 |
| v9 | task3_vid | 0.751156 | 214.696681 | 28.867823 | 与 v6-v8 一致 |
| v10 | task1_ct | 0.810562 | 6.902283 | 0.396594 | bestcfg 单模型，当前最佳 Task1 |
| v10 | task2_tee | 0.805315 | 16.629738 | 0.926641 | bestcfg large ROI160 单模型，当前最佳 Task2 |
| v10 | task3_vid | 0.768802 | 104.463167 | 15.432556 | Unet++ ResNet34 bestcfg，Task3 当前最均衡 |
| v11 | task1_ct | 0.810562 | 6.902283 | 0.396594 | 与 v10 相同 |
| v11 | task2_tee | 0.805315 | 16.629738 | 0.926641 | 与 v10 相同 |
| v11 | task3_vid | 0.754698 | 99.258927 | 17.491449 | v10 Task3 保守后处理；HD 当前最好，但 DSC/ASD 下降 |
| v12 | task1_ct | 0.810562 | 6.902283 | 0.396594 | 与 v10 相同 |
| v12 | task2_tee | 0.805315 | 16.629738 | 0.926641 | 与 v10 相同 |
| v12 | task3_vid | 0.790226 | 138.577390 | 16.059344 | Task3 概率 ensemble；DSC 当前最高，但 HD 变差 |
| v14 | task1_ct | 0.810562 | 6.902283 | 0.396594 | 与 v10 相同，num_cases 30，missing_cases 0 |
| v14 | task2_tee | 0.814276 | 10.904111 | 0.692376 | v13 batch size 1 ROI192 checkpoint + 后处理，曾为线上最佳 Task2；当前 Task2 HD 仍略优于 v19，num_cases 20，missing_cases 0 |
| v14 | task3_vid | 0.768802 | 104.463167 | 15.432556 | 与 v10 相同，num_cases 48，missing_cases 0 |
| v15 | task1_ct | 0.810562 | 6.902283 | 0.396594 | 与 v14 相同，num_cases 30，missing_cases 0 |
| v15 | task2_tee | 0.814276 | 10.904111 | 0.692376 | 与 v14 相同，num_cases 20，missing_cases 0 |
| v15 | task3_vid | 0.770197 | 95.627329 | 14.537786 | v10 checkpoint + TTA + `threshold=0.285`，Task3 三项相对 v10/v14 全部改善，num_cases 48，missing_cases 0 |
| v16 | task1_ct | 0.857340 | 4.583182 | 0.278065 | nnU-Net 5fold ensemble，Task1 大幅刷新，num_cases 30，missing_cases 0 |
| v16 | task2_tee | 0.814276 | 10.904111 | 0.692376 | 与 v14/v15 相同，num_cases 20，missing_cases 0 |
| v16 | task3_vid | 0.770197 | 95.627329 | 14.537786 | 与 v15 相同，num_cases 48，missing_cases 0 |
| v17 | task1_ct | 0.857539 | 4.631175 | 0.279265 | nnU-Net 5fold `checkpoint_best.pth` ensemble，DSC 略高于 v16，HD/ASD 略差，num_cases 30，missing_cases 0 |
| v17 | task2_tee | 0.814276 | 10.904111 | 0.692376 | 与 v14/v15/v16 相同，num_cases 20，missing_cases 0 |
| v17 | task3_vid | 0.770197 | 95.627329 | 14.537786 | 与 v15/v16 相同，num_cases 48，missing_cases 0 |
| v18 | task1_ct | 0.857534 | 4.603581 | 0.280489 | v17 + `min_size=100` 小连通域删除；相对 v17 HD 略好，但 DSC/ASD 变差，num_cases 30，missing_cases 0 |
| v18 | task2_tee | 0.814276 | 10.904111 | 0.692376 | 与 v14/v15/v16/v17 相同，num_cases 20，missing_cases 0 |
| v18 | task3_vid | 0.770197 | 95.627329 | 14.537786 | 与 v15/v16/v17 相同，num_cases 48，missing_cases 0 |
| v19 | task1_ct | 0.857539 | 4.631175 | 0.279265 | 与 v17 相同，num_cases 30，missing_cases 0 |
| v19 | task2_tee | 0.846325 | 11.196889 | 0.640635 | nnU-Net Dataset102 3d_fullres 5fold `checkpoint_best.pth` ensemble；DSC/ASD 当前 Task2 最佳，HD 略差于 v14，num_cases 20，missing_cases 0 |
| v19 | task3_vid | 0.770197 | 95.627329 | 14.537786 | 与 v15/v17 相同，num_cases 48，missing_cases 0 |
| v20 | task1_ct | 0.857539 | 4.631175 | 0.279265 | 与 v19 相同，num_cases 30，missing_cases 0 |
| v20 | task2_tee | 0.846325 | 11.196889 | 0.640635 | 与 v19 相同，num_cases 20，missing_cases 0 |
| v20 | task3_vid | 0.770188 | 95.846126 | 14.617547 | v15 checkpoint + TTA/AMP，`threshold=0.28, min_area=20`；相对 v15/v19 轻微变差，num_cases 48，missing_cases 0 |
| v21 | task1_ct | 0.857539 | 4.631175 | 0.279265 | 与 v19 相同，num_cases 30，missing_cases 0 |
| v21 | task2_tee | 0.846325 | 11.196889 | 0.640635 | 与 v19 相同，num_cases 20，missing_cases 0 |
| v21 | task3_vid | 0.770197 | 95.627329 | 14.537786 | v15 checkpoint + TTA/AMP，`threshold=0.285, min_total_area=1500`；线上与 v15/v19 完全一致，num_cases 48，missing_cases 0 |
| v22 | task1_ct | 0.857539 | 4.631175 | 0.279265 | 与 v19 相同，num_cases 30，missing_cases 0 |
| v22 | task2_tee | 0.846325 | 11.196889 | 0.640635 | 与 v19 相同，num_cases 20，missing_cases 0 |
| v22 | task3_vid | 0.790266 | 216.022344 | 22.657026 | supervised-only UNet++ EfficientNet-B4，`512x896`，all-frame validation，TTA/AMP，`threshold=0.25`；Task3 DSC 当前最高，但 HD/ASD 明显差于 v15/v19/v12，num_cases 48，missing_cases 0 |
| v23 | task1_ct | 0.857539 | 4.631175 | 0.279265 | 与 v19 相同，num_cases 30，missing_cases 0 |
| v23 | task2_tee | 0.846325 | 11.196889 | 0.640635 | 与 v19 相同，num_cases 20，missing_cases 0 |
| v23 | task3_vid | 0.798404 | 122.469457 | 14.554252 | v15/v22 probability ensemble `0.5/0.5, threshold=0.4`；Task3 DSC 当前最高，ASD 接近 v15，HD 显著优于 v22/v12 但仍差于 v15，num_cases 48，missing_cases 0 |
| v24 | task1_ct | 0.857539 | 4.631175 | 0.279265 | 与 v19/v23 相同，num_cases 30，missing_cases 0 |
| v24 | task2_tee | 0.846325 | 11.196889 | 0.640635 | 与 v19/v23 相同，num_cases 20，missing_cases 0 |
| v24 | task3_vid | 0.797286 | 118.965305 | 14.092329 | v15/v22 probability ensemble `0.55/0.45, threshold=0.42`；相对 v23，DSC 小降、HD/ASD 小幅改善，num_cases 48，missing_cases 0 |
| v25 | task1_ct | 0.857813 | 4.671629 | 0.274123 | Dataset111 pseudo-top100 self-training nnU-Net 5fold `checkpoint_best.pth` ensemble；DSC/ASD 当前 Task1 最好，HD 略差于 v17/v19，num_cases 30，missing_cases 0 |
| v25 | task2_tee | 0.846325 | 11.196889 | 0.640635 | 沿用 v19，结果完全一致，num_cases 20，missing_cases 0 |
| v25 | task3_vid | 0.770197 | 95.627329 | 14.537786 | 沿用 v15/v19，结果完全一致，num_cases 48，missing_cases 0 |
| v26 | task1_ct | - | - | - | 沿用 v25；线上待测，预期 DSC 0.857813 / HD 4.671629 / ASD 0.274123 |
| v26 | task2_tee | - | - | - | 沿用 v19；线上待测，预期 DSC 0.846325 / HD 11.196889 / ASD 0.640635 |
| v26 | task3_vid | - | - | - | 沿用 v23；线上待测，预期 DSC 0.798404 / HD 122.469457 / ASD 14.554252 |

## Task1 对比

| Version | Task1 配置 | DSC | HD | ASD | 相对判断 |
|---|---|---:|---:|---:|---|
| v10-v15 | SegResNet base bestcfg | 0.810562 | 6.902283 | 0.396594 | 旧线上最佳 Task1 |
| v16 | nnU-Net 5fold `checkpoint_final.pth` ensemble | 0.857340 | 4.583182 | 0.278065 | 明确大幅刷新 Task1，HD/ASD 当前最好 |
| v17 | nnU-Net 5fold `checkpoint_best.pth` ensemble | 0.857539 | 4.631175 | 0.279265 | 旧 DSC 最高，但 HD/ASD 略逊于 v16 |
| v18 | v17 + min_size=100 postprocess | 0.857534 | 4.603581 | 0.280489 | 后处理收益不明显；不推荐替代 v17/v16 |
| v25 | Dataset111 pseudo-top100 self-training 5fold `checkpoint_best.pth` ensemble | 0.857813 | 4.671629 | 0.274123 | DSC/ASD 当前最好；HD 比 v17/v19 略差，比 v16 差更多 |

## Task2 对比

| Version | Task2 配置 | DSC | HD | ASD | 相对判断 |
|---|---|---:|---:|---:|---|
| v4 | 早期 Task2 best | 0.721137 | 28.747381 | 2.264272 | 旧基线 |
| v5-light / reported v5-medium | UNet seed43 | 0.757979 | 24.674981 | 1.660142 | 明显提升 |
| v6 | UNet 0.5 + SegResNet base 0.5 | 0.770411 | 22.691037 | 1.489703 | 旧最佳，仍很强 |
| v7 | UNet 0.7 + SegResNet base 0.3 | 0.764245 | 23.753866 | 1.618962 | 偏 UNet 下降 |
| v8 | UNet 0.4 + SegResNet base 0.6 | 0.764245 | 23.753866 | 1.618962 | 用户报告与 v7 完全一致，需谨慎解释 |
| v9 | UNet 0.5 + base 0.25 + large 0.25 | 0.771490 | 21.974976 | 1.429489 | 曾为线上最佳，large 分支带来小幅稳定收益 |
| v10 | large ROI160 bestcfg 单模型 | 0.805315 | 16.629738 | 0.926641 | 明显刷新 Task2，说明该单模型泛化强于 v9 ensemble |
| v14 | v13 large ROI192 batch size 1 + 后处理 | 0.814276 | 10.904111 | 0.692376 | 明确刷新线上 Task2；相对 v10 三项指标全部改善 |
| v19 | nnU-Net Dataset102 3d_fullres 5fold `checkpoint_best.pth` ensemble | 0.846325 | 11.196889 | 0.640635 | DSC 大幅刷新 Task2 历史最佳，ASD 也最佳；HD 略差于 v14 |

## Task3 对比

| Version | Task3 配置 | DSC | HD | ASD | 相对判断 |
|---|---|---:|---:|---:|---|
| v9 | 旧 medium 后处理 | 0.751156 | 214.696681 | 28.867823 | 旧基线 |
| v10 | Unet++ ResNet34 bestcfg 单模型 | 0.768802 | 104.463167 | 15.432556 | 当前最均衡，ASD 最好 |
| v11 | v10 二值后处理 `min_area=400, keep_top=2, close_iters=1` | 0.754698 | 99.258927 | 17.491449 | HD 当前最好，但后处理伤了 DSC 和 ASD |
| v12 | v10 ResNet34 + EfficientNet-B4 概率 ensemble `0.5/0.5, thr=0.15` | 0.790226 | 138.577390 | 16.059344 | DSC 当前最高，但远端误差变多，HD 明显差于 v10/v11 |
| v15 | v10 checkpoint + TTA + `threshold=0.285` | 0.770197 | 95.627329 | 14.537786 | 相对 v10/v14 三项全部改善；当前 Task3 最均衡 |
| v20 | v15 checkpoint + TTA/AMP + `threshold=0.28, min_area=20` | 0.770188 | 95.846126 | 14.617547 | 小连通域过滤没有线上收益，DSC/HD/ASD 均略差 |
| v21 | v15 checkpoint + TTA/AMP + `threshold=0.285, min_total_area=1500` | 0.770197 | 95.627329 | 14.537786 | 总面积门控线上未改变结果，等同 v15/v19 |
| v22 | supervised-only UNet++ EfficientNet-B4 `512x896` + TTA/AMP + `threshold=0.25` | 0.790266 | 216.022344 | 22.657026 | DSC 当前最高，但远端错误显著增加；说明新模型召回强、边界/假阳性控制不足 |
| v23 | v15/v22 probability ensemble `0.5/0.5, threshold=0.4` | 0.798404 | 122.469457 | 14.554252 | DSC 当前最高；ensemble 成功压回 v22 的 HD/ASD，大幅优于 v12，但 HD 仍不如 v15 |
| v24 | v15/v22 probability ensemble `0.55/0.45, threshold=0.42` | 0.797286 | 118.965305 | 14.092329 | 比 v23 更保守，HD/ASD 小幅改善但 DSC 小降；证明微调有效但收益有限 |

## 当前结论

当前按组件最优策略，**v26 = v25 Task1 + v19 Task2 + v23 Task3** 是新的主提交候选，线上待测。v25 证明 Task1 自训练有效：相对 v17/v19，Task1 DSC 提高 `+0.000274`，ASD 改善 `-0.005141`，但 HD 增加 `+0.040455`；相对 v16，DSC 提高 `+0.000473`，ASD 改善 `-0.003942`，但 HD 增加 `+0.088447`。因此如果按 DSC/ASD 优先，Task1 应切到 v25；如果极端重视 HD，Task1 仍可保留 v16。

当前已提交的完整包里，v23 仍是 Task3 DSC 优先主候选，v24 是更保守的 Task3 备选，v25 是 Task1 自训练验证包。v26 已将 v25 Task1 合并进 v23 这条 Task3 ensemble 分支；若 v26 结果符合预期，它会成为按 DSC 优先的当前主提交。

当前 v23 完整包线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7984039938855672, HD 122.46945721998627, ASD 14.554251785067644
```

v20 相比 v19 只改变 Task3：`threshold=0.28` 并删除小于 20 像素的小连通域。Task3 DSC 降低 `-0.000009`，HD 增加 `+0.218798`，ASD 增加 `+0.079761`，线上表现略差。

v21 相比 v19 只改变 Task3：保留 `threshold=0.285`，增加 `min_total_area=1500` 总面积门控。Task3 线上结果与 v19 完全一致；本地逐 PNG 比较也确认 v21 Task3 与 v19 完全相同，48 张预测中最小前景面积为 `15982`，因此该门控没有触发。

v22 相比 v19 只改变 Task3：使用 supervised-only UNet++ EfficientNet-B4 `512x896` checkpoint。Task3 DSC 提高 `+0.020069`，并比 v12 也高 `+0.000040`，刷新当前最高 DSC；但 HD 增加 `+120.395016`，ASD 增加 `+8.119239`。这说明 v22 的召回/覆盖更强，但预测边界或远端假阳性明显失控，不能作为均衡主提交。

v23 相比 v22 只改变 Task3 推理策略：把 v22 与稳定的 v15 做 `0.5/0.5` 概率 ensemble，并使用 `threshold=0.4`。Task3 DSC 继续提高 `+0.008138`，HD 改善 `-93.552887`，ASD 改善 `-8.102774`。相对 v15/v19，v23 的 Task3 DSC 提高 `+0.028207`，ASD 只增加 `+0.016466`，但 HD 增加 `+26.842129`。这说明 ensemble 方向成立：v15 成功压住了 v22 的大部分距离错误，同时保留并增强了 v22 的召回优势。

v24 相比 v23 只做轻微保守化：v15/v22 权重从 `0.5/0.5` 改为 `0.55/0.45`，threshold 从 `0.4` 提到 `0.42`。Task3 DSC 降低 `-0.001118`，HD 改善 `-3.504152`，ASD 改善 `-0.461923`。这说明小幅保守化方向是对的，但 HD 改善幅度不大，继续靠类似微调很难获得实质突破。

v16 相比 v15 只改变 Task1，Task2/Task3 结果保持一致；Task1 的 DSC、HD、ASD 三项全部大幅改善。

v17 相比 v16 只把 Task1 从 `checkpoint_final.pth` ensemble 换成 `checkpoint_best.pth` ensemble。Task1 DSC 提高 `+0.000198`，但 HD 增加 `+0.047993`，ASD 增加 `+0.001200`。

v18 相比 v17 增加了 Task1 轻量小连通域删除：DSC 降低 `-0.000005`，HD 改善 `-0.027594`，ASD 变差 `+0.001224`。整体收益不明显，不建议替代 v17；若看距离指标，v16 仍更好。

v25 相比 v17/v19 只改变 Task1：使用 Dataset111 pseudo-top100 self-training 5fold `checkpoint_best.pth` ensemble。Task1 DSC 提高 `+0.000274`，ASD 改善 `-0.005141`，但 HD 增加 `+0.040455`。这说明 top100 高置信伪标签带来了真实泛化增益，尤其是平均表面距离改善明显；但它没有改善最坏边界距离。下一步应优先把 v25 Task1 合并到 v23/v24 的 Task3 ensemble 包中，而不是继续单独提交 v25 这种 Task3 回退到 v15 的组合。

| Task1 Metric | v15 | v16 final | v17 best | v18 post | v25 pseudo-top100 | 当前最好 |
|---|---:|---:|---:|---:|---:|---|
| DSC | 0.810562 | 0.857340 | 0.857539 | 0.857534 | 0.857813 | v25 |
| HD | 6.902283 | 4.583182 | 4.631175 | 4.603581 | 4.671629 | v16 |
| ASD | 0.396594 | 0.278065 | 0.279265 | 0.280489 | 0.274123 | v25 |

| Task2 Metric | v10 | v14/v15 | Delta |
|---|---:|---:|---:|
| DSC | 0.805315 | 0.814276 | +0.008961 |
| HD | 16.629738 | 10.904111 | -5.725627 |
| ASD | 0.926641 | 0.692376 | -0.234265 |

| Task2 Metric | v14/v17/v18 | v19 nnU-Net 5fold | Delta |
|---|---:|---:|---:|
| DSC | 0.814276 | 0.846325 | +0.032049 |
| HD | 10.904111 | 11.196889 | +0.292778 |
| ASD | 0.692376 | 0.640635 | -0.051741 |

v11 / v12 / v15 主要只改变 Task3：

| Metric | v10 | v11 | v12 | v15 | 当前最好 |
|---|---:|---:|---:|---:|---|
| Task3 DSC | 0.768802 | 0.754698 | 0.790226 | 0.770197 | v12 |
| Task3 HD | 104.463167 | 99.258927 | 138.577390 | 95.627329 | v15 |
| Task3 ASD | 15.432556 | 17.491449 | 16.059344 | 14.537786 | v15 |

判断：v12 仍是 Task3 DSC 最高，但 HD 明显变差。v15 的阈值搜索结果在线上成立，虽然 DSC 只小幅超过 v10/v14，但 HD 和 ASD 明显改善。v16 在 v15 基础上用 nnU-Net 5fold 大幅刷新 Task1；v17 进一步小幅提高 Task1 DSC，但牺牲少量 HD/ASD。v18 的小连通域删除没有带来明确收益。v19 在 v17 基础上只替换 Task2 为 nnU-Net 5fold ensemble，Task2 DSC 大幅刷新且 ASD 降低，虽然 HD 略差于 v14，但按 DSC 优先应替代 v17 成为当前主提交。

v10 相比 v9 的改善：

| Task | Metric | v9 | v10 | Delta |
|---|---|---:|---:|---:|
| task1_ct | DSC | 0.788095 | 0.810562 | +0.022467 |
| task1_ct | HD | 7.392809 | 6.902283 | -0.490526 |
| task1_ct | ASD | 0.448069 | 0.396594 | -0.051475 |
| task2_tee | DSC | 0.771490 | 0.805315 | +0.033825 |
| task2_tee | HD | 21.974976 | 16.629738 | -5.345238 |
| task2_tee | ASD | 1.429489 | 0.926641 | -0.502849 |
| task3_vid | DSC | 0.751156 | 0.768802 | +0.017646 |
| task3_vid | HD | 214.696681 | 104.463167 | -110.233514 |
| task3_vid | ASD | 28.867823 | 15.432556 | -13.435266 |

判断：v10 是一次明确的大幅提升，不是随机小波动。三个 task 的 DSC 都升高，同时 HD/ASD 都下降，说明体素重叠和边界质量一起改善。Task2 提升最稳定，Task3 的距离指标改善最大。

## v14 提交记录

- 提交目录：`outputs/submissions/submit_v14_task2_v13_post/submission`
- 提交压缩包：`outputs/submissions/submit_v14_task2_v13_post/submission.zip`
- 记录文件：`outputs/submissions/submit_v14_task2_v13_post/checkpoint_record.md`
- 生成时间：2026-07-07
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v10 | 直接复制 `submit_v10_bestcfg_raw/submission/t1_ct` |
| task2_tee | v13 + post | 使用 `outputs/exp/task2_unet_large_e200_b1_roi192x192x160_c2_lr2e4_score525_v13/checkpoints/best_model.pt` 重新生成 |
| task3_vid | v10 | 直接复制 `submit_v10_bestcfg_raw/submission/t3_vid` |

Task2 v14 后处理参数：

| 参数 | 值 |
|---|---|
| `postprocess` | enabled |
| `post_min_size` | 100 |
| `post_keep_components` | 1 |
| `post_fill_holes` | true |
| `post_close_iters` | 0 |

v14 当前状态：已完成线上测评，Task2 线上结果为 DSC `0.8142756501175983`，HD `10.904111011547963`，ASD `0.6923764603409446`。v15 已在此基础上刷新 Task3，v16/v17 又进一步刷新 Task1，v19 进一步刷新 Task2 DSC/ASD。若单独看 Task2 HD，v14 仍略好。

## v15 提交记录

- 提交目录：`outputs/submissions/submit_v15_task3_thr0285/submission`
- 提交压缩包：`outputs/submissions/submit_v15_task3_thr0285/submission.zip`
- 记录文件：`outputs/submissions/submit_v15_task3_thr0285/checkpoint_record.md`
- 生成时间：2026-07-08
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v14 / v10 | 直接复制 `submit_v14_task2_v13_post/submission/t1_ct` |
| task2_tee | v14 | 直接复制 `submit_v14_task2_v13_post/submission/t2_tee` |
| task3_vid | v10 checkpoint 重新生成 | TTA enabled，`threshold=0.285`，不启用后处理 |

线上结果：

```text
task1_ct: DSC 0.8105616221627818, HD 6.902283124940843, ASD 0.3965936972734582
task2_tee: DSC 0.8142756501175983, HD 10.904111011547963, ASD 0.6923764603409446
task3_vid: DSC 0.7701966979705538, HD 95.62732850814587, ASD 14.537786225800227
```

## v18 提交记录

- 提交目录：`outputs/submissions/submit_v18_task1_v17_post_min100_task2_v14_task3_v15/submission`
- 提交压缩包：`outputs/submissions/submit_v18_task1_v17_post_min100_task2_v14_task3_v15/submission.zip`
- 记录文件：`outputs/submissions/submit_v18_task1_v17_post_min100_task2_v14_task3_v15/checkpoint_record.md`
- 生成时间：2026-07-08
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v17 + 轻量后处理 | 删除小于 100 voxels 的连通域，`keep_components=0`，不填洞，不 closing |
| task2_tee | v14 | Task2 v13 ROI192 + 后处理 |
| task3_vid | v15 | v10 checkpoint + TTA + `threshold=0.285` |

线上结果：

```text
task1_ct: DSC 0.8575338882053642, HD 4.603580536897585, ASD 0.2804887549429751
task2_tee: DSC 0.8142756501175983, HD 10.904111011547963, ASD 0.6923764603409446
task3_vid: DSC 0.7701966979705538, HD 95.62732850814587, ASD 14.537786225800227
```

判断：v18 不是有效改进。它相对 v17 只小幅改善 HD，但 DSC 与 ASD 下降；相对 v16，HD/ASD 仍更差。

## v19 提交记录

- 提交目录：`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission`
- 提交压缩包：`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/submission.zip`
- 记录文件：`outputs/submissions/submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15/checkpoint_record.md`
- 生成时间：2026-07-09
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v17 | Task1 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task2_tee | nnU-Net v2 5fold ensemble | Dataset102_MVAA_Task2，3d_fullres，fold 0-4，`checkpoint_best.pth` |
| task3_vid | v15 | v10 checkpoint + TTA + `threshold=0.285` |

线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7701966979705538, HD 95.62732850814587, ASD 14.537786225800227
```

判断：v19 是 Task2 的明确 DSC 跃升版本。相对 v14/v17/v18，Task2 DSC 提高约 `+0.032049`，ASD 降低约 `-0.051741`，但 HD 增加约 `+0.292778`。因此按 DSC 或 DSC+ASD 优先，v19 是当前 Task2 最优；如果单独追求 Task2 HD，v14 仍略好。

## v26 提交记录

- 提交目录：`outputs/submissions/submit_v26_task1_v25_task2_v19_task3_v23/submission`
- 提交压缩包：`outputs/submissions/submit_v26_task1_v25_task2_v19_task3_v23/submission.zip`
- 记录文件：`outputs/submissions/submit_v26_task1_v25_task2_v19_task3_v23/checkpoint_record.md`
- 生成时间：2026-07-24
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：待提交/待测评

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v25 | Dataset111 pseudo-top100 self-training nnU-Net v2 3d_fullres，5fold `checkpoint_best.pth` ensemble |
| task2_tee | v19 | Task2 nnU-Net v2 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task3_vid | v23 | v15/v22 probability ensemble，权重 `0.5/0.5`，threshold `0.4` |

预期线上结果：

```text
task1_ct: DSC 0.8578127264071896, HD 4.671629372756556, ASD 0.27412349094227223
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7984039938855672, HD 122.46945721998627, ASD 14.554251785067644
```

判断：v26 是当前组件最优合并包。它把 v25 的 Task1 自训练增量接到 v23 的 Task3 DSC-best ensemble 上，同时保留 v19 的 Task2 最强结果。由于三个任务目录都是直接复制已验证组件，若平台不出现缓存或上传异常，v26 应该只体现这些组件分数的组合。

## v25 提交记录

- 提交目录：`outputs/submissions/submit_v25_task1_pseudo_top100_best_task2_v19_task3_v15/submission`
- 提交压缩包：`outputs/submissions/submit_v25_task1_pseudo_top100_best_task2_v19_task3_v15/submission.zip`
- 记录文件：`outputs/submissions/submit_v25_task1_pseudo_top100_best_task2_v19_task3_v15/checkpoint_record.md`
- 生成时间：2026-07-23
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | Dataset111 pseudo-top100 self-training | nnU-Net v2 3d_fullres，5fold `checkpoint_best.pth` ensemble |
| task2_tee | v19 | Task2 nnU-Net v2 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task3_vid | v15 | v10 checkpoint + TTA + `threshold=0.285` |

线上结果：

```text
task1_ct: DSC 0.8578127264071896, HD 4.671629372756556, ASD 0.27412349094227223
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7701966979705538, HD 95.62732850814587, ASD 14.537786225800227
```

判断：v25 验证了 Task1 自训练方向有效。相对 v17/v19，Task1 DSC 提高约 `+0.000274`，ASD 改善约 `-0.005141`，但 HD 变差约 `+0.040455`。由于 Task2 与 v19 完全一致、Task3 与 v15/v19 完全一致，本次线上差异可以归因于 Task1 pseudo-top100 自训练。按 DSC/ASD 优先，v25 应替代 v17/v19 作为新的 Task1 组件；按 HD 优先，v16 仍是 Task1 距离指标备选。

## v24 提交记录

- 提交目录：`outputs/submissions/submit_v24_task3_v15_v22_ens_w55_thr042_task1_v17_task2_v19/submission`
- 提交压缩包：`outputs/submissions/submit_v24_task3_v15_v22_ens_w55_thr042_task1_v17_task2_v19/submission.zip`
- 记录文件：`outputs/submissions/submit_v24_task3_v15_v22_ens_w55_thr042_task1_v17_task2_v19/checkpoint_record.md`
- 生成时间：2026-07-23
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v17 / v19 | Task1 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task2_tee | v19 | Task2 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task3_vid | v15/v22 probability ensemble | v15 weight `0.55`，v22 weight `0.45`，threshold `0.42`，TTA/AMP，无后处理 |

线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7972857440407505, HD 118.96530512123586, ASD 14.092328937421556
```

判断：v24 相对 v23 做了轻微保守化，确实带来 HD/ASD 小幅改善，但改善有限：HD 只下降约 `3.504152`，ASD 下降约 `0.461923`，同时 DSC 下降约 `0.001118`。这说明权重/阈值微调已经接近当前 ensemble 的收益上限；若要进一步实质性降低 HD，需要转向更针对 outlier 的方法，例如 frame-level 错误定位、时序一致性、边界/距离约束训练或按视频自适应策略。

## v23 提交记录

- 提交目录：`outputs/submissions/submit_v23_task3_v15_v22_ens_w50_thr04_task1_v17_task2_v19/submission`
- 提交压缩包：`outputs/submissions/submit_v23_task3_v15_v22_ens_w50_thr04_task1_v17_task2_v19/submission.zip`
- 记录文件：`outputs/submissions/submit_v23_task3_v15_v22_ens_w50_thr04_task1_v17_task2_v19/checkpoint_record.md`
- 生成时间：2026-07-23
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v17 / v19 | Task1 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task2_tee | v19 | Task2 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task3_vid | v15/v22 probability ensemble | v15 weight `0.5`，v22 weight `0.5`，threshold `0.4`，TTA/AMP，无后处理 |

线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7984039938855672, HD 122.46945721998627, ASD 14.554251785067644
```

判断：v23 是当前 Task3 DSC 最高版本，且显著修复了 v22 的距离指标退化。相对 v22，Task3 DSC 继续提高，HD/ASD 大幅下降；相对 v15/v19，DSC 大幅提高、ASD 基本持平，但 HD 仍高约 `+26.842129`。因此 v23 是当前按 DSC 优先的主力候选；如果平台或论文更重视 HD，v15/v19 仍应作为距离指标备选。

## v22 提交记录

- 提交目录：`outputs/submissions/submit_v22_task3_effb4_sup_allframe_s42_task1_v17_task2_v19/submission`
- 提交压缩包：`outputs/submissions/submit_v22_task3_effb4_sup_allframe_s42_task1_v17_task2_v19/submission.zip`
- 记录文件：`outputs/submissions/submit_v22_task3_effb4_sup_allframe_s42_task1_v17_task2_v19/checkpoint_record.md`
- 生成时间：2026-07-23
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v17 / v19 | Task1 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task2_tee | v19 | Task2 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task3_vid | supervised-only EfficientNet-B4 | UNet++ EfficientNet-B4 ImageNet，`512x896`，all-frame validation，TTA/AMP，`threshold=0.25`，无后处理 |

线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7902659360112319, HD 216.0223443104334, ASD 22.657025617217844
```

判断：v22 是 Task3 的高召回/高 DSC 版本。它相对 v15/v19 的 Task3 DSC 提高约 `+0.020069`，也以极小幅度超过 v12 的 `0.790226`，但 HD/ASD 明显退化，甚至 HD 接近旧 v9/v10 之前的高错误区间。因此 v22 不应直接替代 v15 作为均衡版本，但值得保留为后续概率 ensemble、蒸馏、或更强距离约束训练的候选分支。

## v21 提交记录

- 提交目录：`outputs/submissions/submit_v21_task3_v15_thr0285_area1500_task1_v17_task2_v19/submission`
- 提交压缩包：`outputs/submissions/submit_v21_task3_v15_thr0285_area1500_task1_v17_task2_v19/submission.zip`
- 记录文件：`outputs/submissions/submit_v21_task3_v15_thr0285_area1500_task1_v17_task2_v19/checkpoint_record.md`
- 生成时间：2026-07-23
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v17 / v19 | Task1 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task2_tee | v19 | Task2 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task3_vid | v15 + 总面积门控 | v15 checkpoint + TTA/AMP + `threshold=0.285`；预测总前景面积 `<1500` pixels 时整帧清空 |

线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7701966979705538, HD 95.62732850814587, ASD 14.537786225800227
```

判断：v21 与 v19/v15 线上结果完全一致。本地逐 PNG 比较确认 v21 Task3 与 v19 完全相同，48 张预测中最小前景面积为 `15982`，所以 `min_total_area=1500` 根本没有触发。该方向不建议继续加大阈值盲试，因为需要把阈值提高到接近真实前景面积区间才会改变结果，风险会明显增大。

## v20 提交记录

- 提交目录：`outputs/submissions/submit_v20_task3_v15_thr028_min20_task1_v17_task2_v19/submission`
- 提交压缩包：`outputs/submissions/submit_v20_task3_v15_thr028_min20_task1_v17_task2_v19/submission.zip`
- 记录文件：`outputs/submissions/submit_v20_task3_v15_thr028_min20_task1_v17_task2_v19/checkpoint_record.md`
- 生成时间：2026-07-23
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | v17 / v19 | Task1 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task2_tee | v19 | Task2 nnU-Net 5fold `checkpoint_best.pth` ensemble，无后处理 |
| task3_vid | v15 + 小连通域过滤 | v15 checkpoint + TTA/AMP + `threshold=0.28` + `min_area=20` |

线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8463245904808815, HD 11.196888629485855, ASD 0.6406354094949988
task3_vid: DSC 0.7701881121613079, HD 95.84612622263508, ASD 14.61754742221453
```

判断：v20 相对 v19/v15 略差。Task3 DSC 下降约 `-0.000009`，HD 增加约 `+0.218798`，ASD 增加约 `+0.079761`。这说明本地轻量 threshold/postprocess 搜索的极小优势没有线上泛化，不能作为下一步主方向。

## v17 提交记录

- 提交目录：`outputs/submissions/submit_v17_task1_nnunet5fold_bestckpt_task2_v14_task3_v15/submission`
- 提交压缩包：`outputs/submissions/submit_v17_task1_nnunet5fold_bestckpt_task2_v14_task3_v15/submission.zip`
- 记录文件：`outputs/submissions/submit_v17_task1_nnunet5fold_bestckpt_task2_v14_task3_v15/checkpoint_record.md`
- 生成时间：2026-07-08
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | nnU-Net v2 5fold ensemble | Dataset101_MVAA_Task1，3d_fullres，fold 0-4，`checkpoint_best.pth` |
| task2_tee | v14 | Task2 v13 ROI192 + 后处理 |
| task3_vid | v15 | v10 checkpoint + TTA + `threshold=0.285` |

线上结果：

```text
task1_ct: DSC 0.8575385873310171, HD 4.6311747736863405, ASD 0.2792648483145077
task2_tee: DSC 0.8142756501175983, HD 10.904111011547963, ASD 0.6923764603409446
task3_vid: DSC 0.7701966979705538, HD 95.62732850814587, ASD 14.537786225800227
```

## v16 提交记录

- 提交目录：`outputs/submissions/submit_v16_task1_nnunet5fold_task2_v14_task3_thr0285/submission`
- 提交压缩包：`outputs/submissions/submit_v16_task1_nnunet5fold_task2_v14_task3_thr0285/submission.zip`
- 记录文件：`outputs/submissions/submit_v16_task1_nnunet5fold_task2_v14_task3_thr0285/checkpoint_record.md`
- 生成时间：2026-07-08
- 文件检查：submission 目录共 `101` 个文件；zip 根目录为 `t1_ct/`、`t2_tee/`、`t3_vid/`
- 线上状态：已测评通过，三个 task 均 `ok`，missing_cases 均为 `0`

组成：

| Task | 来源 | 说明 |
|---|---|---|
| task1_ct | nnU-Net v2 5fold ensemble | Dataset101_MVAA_Task1，3d_fullres，fold 0-4，`checkpoint_final.pth` |
| task2_tee | v14 | Task2 v13 ROI192 + 后处理 |
| task3_vid | v15 | v10 checkpoint + TTA + `threshold=0.285` |

线上结果：

```text
task1_ct: DSC 0.8573401364276171, HD 4.583182167432217, ASD 0.27806529165152766
task2_tee: DSC 0.8142756501175983, HD 10.904111011547963, ASD 0.6923764603409446
task3_vid: DSC 0.7701966979705538, HD 95.62732850814587, ASD 14.537786225800227
```

后续优先级：

1. 保留 v23 作为当前按 DSC 优先策略的主力候选；v19/v15 是 Task3 HD 指标备选；v16 作为 Task1 距离指标备选；不建议使用 v18/v20/v21。
2. 第一优先级继续优化 Task3：v23 证明 v15/v22 probability ensemble 有效；下一步做 ensemble 权重/阈值细搜、多视频 split 验证、空帧误检控制和时序一致性后处理。
3. 第二优先级优化 Task2：v19 的 nnU-Net 5fold 已显著刷新 DSC，可继续比较 `checkpoint_final.pth`、v14/v19 融合、以及轻量后处理对 HD 的影响。
4. 第三优先级优化 Task1：小连通域删除已经验证收益不明显；下一步不要继续单纯加大 `min_size`，应优先尝试概率阈值、v16/v17 概率融合、或 fold 权重平均。
5. v5-medium 和 v8 的线上结果与相邻版本完全一致，且本地 zip/Task2 文件不同，记录时应标注为疑似提交或平台缓存异常。
