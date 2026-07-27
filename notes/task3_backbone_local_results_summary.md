# Task3 ResNet / EfficientNet 本地结果汇总

日期：2026-07-27

## 说明

本文档整理 Task3 当前 ResNet 系列和 EfficientNet-Bx 系列的本地 best checkpoint 结果，用于判断下一步 ensemble 候选。

注意：

- `all-frame` 验证更接近线上，因为同时包含前景帧和空帧。
- `fg-only` / 旧验证只看前景帧，容易高估模型稳定性，不能和 `all-frame` 结果直接横向比较。
- 最终仍要以上线提交结果为准，本地结果主要用于筛选候选。

## All-Frame 验证结果

| Backbone | 实验 | Seed | Best Epoch | Dice_fg | Dice_all | HD | ASD | Thr | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| EfficientNet-B4 | `t3_sup_unetpp_effb4_img512x896_allframe_s42` | 42 | 32 | 0.7893 | 0.8630 | 80.96 | 13.84 | 0.25 | v22 来源，高召回，但线上 HD 风险大 |
| EfficientNet-B4 EMA | `t3_v30_effb4_ema_semi_from_v22_s46` | 46 | 9 | 0.8509 | 0.9106 | 64.45 | 9.70 | 0.35 | 本地强，但线上 v27 不理想 |
| EfficientNet-B5 | `t3_v32_unetpp_effb5_sup_allframe_s42` | 42 | 14 | 0.8062 | 0.8740 | 72.55 | 12.85 | 0.25 | 老 B5，v29/v30 已证明 ensemble 有效 |
| EfficientNet-B5 | `t3_v34_unetpp_effb5_sup_allframe_s43` | 43 | 54 | 0.6391 | 0.7894 | 97.87 | 19.65 | 0.25 | 明显偏弱，不建议高权重 |
| EfficientNet-B5 | `t3_v35_unetpp_effb5_sup_allframe_s44` | 44 | 68 | 0.7675 | 0.8334 | 73.30 | 12.38 | 0.20 | 可用，适合和 B5_s42 做 seed ensemble |
| EfficientNet-B6 | `t3_v33_unetpp_effb6_sup_allframe_s42` | 42 | 34 | 0.8037 | 0.8724 | 72.77 | 11.91 | 0.175 | 和 B5_s42 接近，ASD 略好 |
| ResNet50 | `t3_v31_unetpp_resnet50_e100_s47` | 47 | 47 | 0.7383 | 0.8263 | 85.14 | 17.74 | 0.20 | 不如 B5/B6，不优先 |
| ResNet34 Multiclass | `t3_multiclass_res34_5class_s50` | 50 | 训练中 | 当前 best 0.6371 | 当前 best 0.4687 | 119.30 | 24.38 | 0.50 | 训练早期，暂不评价 |

## FG-Only / 旧验证结果

| Backbone | 实验 | Seed | Best Epoch | Dice_fg | HD | ASD | Thr | 判断 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| ResNet34 | `task3_unetpp_res34_imagenet_e100_bs2_bestcfg` | 42 | 31 | 0.7788 | 64.94 | 11.04 | 0.25 | v15 来源，线上很稳 |
| ResNet34 | `t3_v36_unetpp_res34_img448x800_s49` | 49 | 93 | 0.8121 | 39.56 | 6.65 | 0.55 | 新 ResNet34 seed，本地稳定性很好 |
| ResNet34 768x1344 | `res34_img768x1344_e160_bs2` | 42 | 42 | 0.7569 | 131.69 | 21.01 | 0.15 | 不如 v15 |
| ResNet34 768x1344 | `res34_img768x1344_e160_bs4` | 42 | 108 | 0.7655 | 128.87 | 20.17 | 0.25 | 不如 v15 |
| EfficientNet-B3 | `t3_sup_unetpp_effb3_img_e100_s43` | 42 | 30 | 0.7685 | 65.53 | 10.37 | 0.05 | 有参考价值，但不如 B5/B6 路线清晰 |
| EfficientNet-B4 640x1120 | `effb4_img640x1120_e140_bs2_sup_v12` | 42 | 127 | 0.8101 | 80.10 | 12.81 | 0.20 | 旧高分支，但线上风险仍偏大 |
| EfficientNet-B4 448x800 | `t3_sup_unetpp_effb4_img_e120_s42` | 42 | 35 | 0.7709 | 58.85 | 10.41 | 0.05 | 不如 all-frame B4/B5 路线清晰 |
| ResNet50 | 无更强结果 | - | - | - | - | - | - | 暂不推荐继续优先投入 |

## ResNet34_s49 是否值得单独线上测试

结论：有必要，但优先级低于 ensemble。

理由：

- ResNet34_s49 本地前景帧结果明显强于旧 v15：`Dice_fg 0.8121 / HD 39.56 / ASD 6.65`。
- 它可能是一个更稳的 ResNet34 分支，有机会降低 Task3 的 HD/ASD。
- 但它的验证方式是 `fg-only`，没有覆盖空帧误检风险，线上单模型 DSC 很可能不会超过 v29/v30 ensemble。

因此，推荐顺序是：

1. 先把 ResNet34_s49 放入 ensemble 作为稳定分支。
2. 如果 ensemble 结果显示它有正贡献，再生成一个 ResNet34_s49 单模型线上测试包。
3. 不建议直接把 ResNet34_s49 单模型作为主提交。

## 下一步 Ensemble 候选

当前最值得参与融合的模型：

| 角色 | 模型 | 原因 |
|---|---|---|
| 稳定基线 | v15 ResNet34_s42 | 线上已证明 HD/ASD 稳定 |
| 新稳定分支 | ResNet34_s49 | 本地 HD/ASD 很好，适合压远端错误 |
| 高召回分支 | v22 EfficientNet-B4_s42 | 提供 DSC/覆盖，但权重要控制 |
| 稳定 EfficientNet | B5_s42 | v29/v30 已证明有线上 ensemble 价值 |
| 新 B5 seed | B5_s44 | 本地表现健康，适合替代或补充 B5_s42 |
| 候选补充 | B6_s42 | 与 B5_s42 接近，ASD 略好，可做备选 |

暂不推荐：

- B5_s43：本地明显偏弱，容易拖低 ensemble。
- ResNet50_s47：没有超过 ResNet34/B5/B6 的证据。
- B4 EMA_s46：本地强，但线上 v27 已证明单模型距离风险仍大。

## 推荐第一版融合

推荐先做一个保守五模型候选：

```text
0.30 * v15 ResNet34_s42
0.20 * ResNet34_s49
0.20 * v22 EfficientNet-B4_s42
0.15 * B5_s42
0.15 * B5_s44
threshold = 0.40
```

目标：

- 保持 v29/v30 的 DSC 级别。
- 用 ResNet34_s49 和 B5_s44 增加多 seed 稳定性。
- 继续控制 v22/B4 权重，避免 HD/ASD 失控。

如果局部搜索允许，可以再测一个 B6 参与版本：

```text
0.30 * v15 ResNet34_s42
0.15 * ResNet34_s49
0.20 * v22 EfficientNet-B4_s42
0.15 * B5_s42
0.10 * B5_s44
0.10 * B6_s42
threshold = 0.40
```

推荐最终产出：

- 一个 DSC 优先版本。
- 一个 HD/ASD 优先版本。

是否提交线上，取决于本地局部搜索是否能在接近 v30 Dice 的情况下进一步降低 HD/ASD。
