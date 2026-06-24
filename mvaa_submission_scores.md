# MVAA 2026 Submission Scores v1-v10

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
| v10 | task3_vid | 0.768802 | 104.463167 | 15.432556 | Unet++ ResNet34 bestcfg，当前最佳 Task3 |

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

## 当前结论

当前已验证最强提交是 **v10**：

```text
task1_ct: DSC 0.810562, HD 6.902283, ASD 0.396594
task2_tee: DSC 0.805315, HD 16.629738, ASD 0.926641
task3_vid: DSC 0.768802, HD 104.463167, ASD 15.432556
```

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

后续优先级：

1. 保留 v10 作为当前主力提交和新基线。
2. 第一优先级优化 Task3：v10 虽然大幅改善 HD/ASD，但 Task3 的 HD 仍有 104.46，说明仍存在少量远端误检/漏检帧；优先做阈值、连通域、小区域过滤、时序一致性后处理。
3. 第二优先级优化 Task2：v10 单模型已经强于 v9 ensemble，下一步不要直接沿用 v9 权重；应以 v10 large ROI160 为主模型，尝试加入 v9 中互补模型的小权重 ensemble，例如 v10:old_UNet:old_base = 0.7:0.15:0.15 或 0.8:0.1:0.1。
4. 第三优先级优化 Task1：Task1 已经较稳，继续尝试阈值/后处理和少量 seed ensemble，目标是把 HD/ASD 再压低，而不是大改模型。
5. v5-medium 和 v8 的线上结果与相邻版本完全一致，且本地 zip/Task2 文件不同，记录时应标注为疑似提交或平台缓存异常。
