# Task1 nnU-Net 自训练优化方案

更新时间：2026-07-23

本文档规划 Task1 CT 的下一阶段优化路线：基于当前最强的 nnU-Net v2 5-fold ensemble，为 unlabeled CT 生成高质量伪标签，筛选可信 case 加回训练，尝试突破当前 v17/v19 的 Task1 分数。

当前结论：Task1 已有 27 个 labeled case、1040 个 unlabeled case。现有 nnU-Net 5-fold 已充分训练，后处理收益很弱；真正可能带来新增信息的方向是自训练，但必须严格筛伪标签，不能直接把 1040 个伪标签全部加入训练。

## 1. 当前基线

当前 Task1 推荐基线：

| 项目 | 配置 |
|---|---|
| 线上版本 | v17，当前主提交 v19 中复用 v17 Task1 |
| 框架 | nnU-Net v2 |
| dataset | `Dataset101_MVAA_Task1` |
| config | `3d_fullres` |
| ensemble | fold `0-4` |
| checkpoint | `checkpoint_best.pth` |
| patch size | `112 128 160` |
| batch size | `2` |
| spacing | `0.5, 0.357421875, 0.357421875` |
| normalization | `CTNormalization` |

线上 Task1 结果：

| Version | Checkpoint | DSC | HD | ASD |
|---|---|---:|---:|---:|
| v16 | `checkpoint_final.pth` | 0.857340 | 4.583182 | 0.278065 |
| v17 | `checkpoint_best.pth` | 0.857539 | 4.631175 | 0.279265 |
| v18 | v17 + `min_size=100` | 0.857534 | 4.603581 | 0.280489 |

判断：

- v17 的 DSC 当前最高。
- v16 的 HD/ASD 略好。
- v18 小连通域后处理收益不明显。
- 单纯调后处理很难带来大幅提升。
- 单纯继续训练或微调 epoch/lr 也不是高收益方向，因为 5 个 fold 已完整跑到 1000 epochs，学习率后期接近 0。

## 2. 自训练目标

目标不是“扩大训练集数量”，而是“增加可靠监督信息”。

第一轮自训练目标：

- 从 1040 个 unlabeled CT 中筛出一批高可信 pseudo-labeled cases。
- 初始规模建议 `100` 或 `200`，不要超过 `300`。
- 构建新的 nnU-Net dataset，例如 `Dataset111_MVAA_Task1_PseudoTop100`。
- 先跑 fold 0 或少量 fold 验证方向，确认没有明显负收益。
- 如果内部验证可靠，再扩展到 5-fold ensemble 并准备线上提交。

## 3. 已完成的预分析

分析脚本：

```bash
/home/wuyongji/miniconda3/envs/mvaa/bin/python scripts/analyze_task1_mask_distributions.py
```

输出目录：

```text
outputs/analysis/task1_pseudo/
```

关键文件：

| 文件 | 作用 |
|---|---|
| `summary.md` | 当前 mask/image 分布摘要 |
| `mask_stats.csv` | labeled GT、v16/v17/v18 预测的 mask 统计 |
| `unlabeled_image_stats.csv` | 1040 个 unlabeled 图像 metadata |
| `unlabeled_metadata_screen.csv` | unlabeled metadata 初筛结果 |

当前 metadata 初筛结果：

| Priority | Count | 判断 |
|---|---:|---|
| high | 759 | shape/spacing/物理扫描范围接近 labeled/val，第一轮优先 |
| medium | 35 | 有轻微偏离，可作为第二轮补充 |
| low | 246 | 明显 outlier，第一轮不建议使用 |

重要观察：

- labeled GT 的 mask 全部是单连通域。
- v17 预测的物理体积分布和 labeled GT 很接近。
- v17 的主要问题不是整体偏大/偏小，而是少量小碎片或个别异常组件。
- unlabeled 中有不少 spacing 或扫描范围明显偏离的样本，直接全量自训练风险高。

## 4. 总体流程

推荐分 6 个阶段推进。

| 阶段 | 名称 | 目标 | 产物 |
|---|---|---|---|
| A | metadata 初筛 | 排除明显分布外 unlabeled | high/medium/low case list |
| B | 伪标签推理 | 用 v17 5-fold ensemble 预测 high-priority unlabeled | hard masks + softmax probabilities |
| C | 伪标签质量打分 | 根据体积、连通域、confidence、fold agreement 筛选 | pseudo quality ranking |
| D | 构建 pseudo dataset | 组合 real labels + selected pseudo labels | nnU-Net raw dataset |
| E | 小规模训练验证 | 先跑 fold 0 或 2-fold sanity | 判断是否值得 5-fold |
| F | 完整训练提交 | 训练 5-fold ensemble，生成提交 | 新 Task1 submission |

## 5. 阶段 A：metadata 初筛

已完成初版。

初筛依据：

- 图像 shape 是否接近 labeled/val。
- spacing 是否接近 labeled/val。
- 物理扫描体积是否接近 labeled/val。

推荐第一轮只使用：

```text
metadata_priority == high
```

候选列表来自：

```text
outputs/analysis/task1_pseudo/unlabeled_metadata_screen.csv
```

下一步需要新增一个小脚本导出 high-priority case list：

```text
outputs/analysis/task1_pseudo/unlabeled_high_priority_cases.txt
```

## 6. 阶段 B：伪标签推理

使用当前最强 Task1 teacher：

```text
Dataset101_MVAA_Task1
3d_fullres
fold 0-4
checkpoint_best.pth
```

推理输入：

```text
data/reference_data/t1_ct/train/unlabeled
```

推荐第一轮只推理 high-priority 759 个，而不是全量 1040 个。原因是 low-priority outlier 可能让伪标签质量很差，后续训练会放大错误。

推荐输出目录：

```text
outputs/pseudo/task1_v17_high759/
  imagesTs/
  pred/
  pred_prob/
  case_list.txt
  pseudo_inference_record.md
```

建议用 nnU-Net 保存概率：

```bash
CUDA_VISIBLE_DEVICES=2 \
LD_LIBRARY_PATH=/home/wuyongji/miniconda3/envs/mvaa/lib \
MPLCONFIGDIR=/tmp/mvaa_mpl_nnunet \
nnUNet_raw=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_fold0/nnUNet_raw \
nnUNet_preprocessed=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_fold0/nnUNet_preprocessed \
nnUNet_results=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_fold0/nnUNet_results \
/home/wuyongji/miniconda3/envs/mvaa/bin/nnUNetv2_predict \
  -i outputs/pseudo/task1_v17_high759/imagesTs \
  -o outputs/pseudo/task1_v17_high759/pred \
  -d 101 \
  -c 3d_fullres \
  -f 0 1 2 3 4 \
  -chk checkpoint_best.pth \
  -device cuda \
  --save_probabilities \
  --disable_progress_bar
```

需要注意：

- `imagesTs` 里仍然要用 nnU-Net 命名：`caseid_0000.nii.gz`。
- 不要直接修改原 `Dataset101_MVAA_Task1/imagesTs`，避免污染当前基线。
- 如果 `--save_probabilities` 输出很占空间，可以第一轮只跑 high-priority，确认磁盘容量后再扩展。

## 7. 阶段 C：伪标签质量打分

伪标签筛选是自训练成败关键。

建议给每个 pseudo case 计算以下指标：

| 指标 | 目的 | 推荐判断 |
|---|---|---|
| foreground physical volume | 排除异常大/小 mask | 接近 labeled GT p05-p95 或宽松 p01-p99 |
| foreground voxel ratio | 排除明显过分割 | 不要超过 labeled/val 明显范围 |
| component count | 排除碎片很多的伪标签 | 优先 `1-3` |
| largest component ratio | 判断主结构是否集中 | 优先 `>=0.999` |
| second component size | 判断是否有异常副结构 | 越小越好 |
| mean foreground probability | 判断前景置信度 | 越高越好 |
| low-confidence boundary ratio | 判断边界不确定性 | 越低越好 |
| fold agreement | 判断 5-fold 一致性 | 越高越好 |

第一轮如果只保存 ensemble 概率，而没有单 fold 概率，则至少做：

- volume filter
- component filter
- largest component ratio filter
- mean foreground probability filter

如果能承受额外推理成本，建议单独保存 fold 0-4 的 hard mask 或 probability，用于计算 fold agreement。这个指标比单纯 confidence 更可靠。

建议初版质量分数：

```text
quality_score =
  volume_score * 0.30
  + component_score * 0.20
  + largest_component_score * 0.20
  + foreground_confidence_score * 0.20
  + metadata_score * 0.10
```

如果有 fold agreement，则改为：

```text
quality_score =
  fold_agreement_score * 0.35
  + volume_score * 0.20
  + component_score * 0.15
  + largest_component_score * 0.15
  + foreground_confidence_score * 0.10
  + metadata_score * 0.05
```

推荐输出：

```text
outputs/analysis/task1_pseudo/pseudo_quality_v17_high759.csv
outputs/analysis/task1_pseudo/pseudo_quality_v17_high759_summary.md
outputs/analysis/task1_pseudo/pseudo_top100_cases.txt
outputs/analysis/task1_pseudo/pseudo_top200_cases.txt
```

## 8. 阶段 D：构建 pseudo nnU-Net dataset

不要覆盖 `Dataset101_MVAA_Task1`。新建 dataset id，建议：

| Dataset | 内容 |
|---|---|
| `Dataset111_MVAA_Task1_PseudoTop100` | 27 real + top100 pseudo |
| `Dataset112_MVAA_Task1_PseudoTop200` | 27 real + top200 pseudo |

目录建议：

```text
outputs/nnunet_task1_pseudo_top100/
  nnUNet_raw/
    Dataset111_MVAA_Task1_PseudoTop100/
      imagesTr/
      labelsTr/
      imagesTs/
      dataset.json
  nnUNet_preprocessed/
  nnUNet_results/
  task1_case_mapping.json
  pseudo_case_manifest.json
```

构建规则：

- 27 个真实 labeled case 必须全部保留。
- pseudo case 的 image 来自 `train/unlabeled`。
- pseudo case 的 label 来自阶段 B/C 筛选后的预测 mask。
- pseudo case id 建议加前缀，例如 `pseudo_0211`，避免和原始 labeled case id 冲突。
- `imagesTs` 仍然放竞赛 val images，用于最终推理提交。

`dataset.json` 仍然保持：

```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "target": 1},
  "numTraining": 127,
  "file_ending": ".nii.gz"
}
```

其中 `numTraining` 根据 real+pseudo 数量变化。

## 9. 阶段 E：小规模训练验证

第一轮不要直接 5-fold 全跑。建议先跑：

```text
Dataset111 top100, fold 0
```

目标：

- 看训练是否稳定。
- 看原 27 个 real validation fold 上的 Dice 是否下降。
- 看 pseudo 加入后是否出现过拟合或异常 prediction volume。

训练命令模板：

```bash
CUDA_VISIBLE_DEVICES=2 \
LD_LIBRARY_PATH=/home/wuyongji/miniconda3/envs/mvaa/lib \
MPLCONFIGDIR=/tmp/mvaa_mpl_nnunet \
nnUNet_raw=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_pseudo_top100/nnUNet_raw \
nnUNet_preprocessed=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_pseudo_top100/nnUNet_preprocessed \
nnUNet_results=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_pseudo_top100/nnUNet_results \
/home/wuyongji/miniconda3/envs/mvaa/bin/nnUNetv2_plan_and_preprocess \
  -d 111 \
  --verify_dataset_integrity
```

```bash
CUDA_VISIBLE_DEVICES=2 \
LD_LIBRARY_PATH=/home/wuyongji/miniconda3/envs/mvaa/lib \
MPLCONFIGDIR=/tmp/mvaa_mpl_nnunet \
nnUNet_raw=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_pseudo_top100/nnUNet_raw \
nnUNet_preprocessed=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_pseudo_top100/nnUNet_preprocessed \
nnUNet_results=/home/wuyongji/yangye_project/MVAA_v1/outputs/nnunet_task1_pseudo_top100/nnUNet_results \
/home/wuyongji/miniconda3/envs/mvaa/bin/nnUNetv2_train \
  111 3d_fullres 0
```

评价方式：

- 首先看 nnU-Net fold validation summary。
- 但注意：pseudo dataset 的 default split 会把 pseudo cases 混进 validation，不利于判断真实泛化。
- 因此更推荐生成自定义 `splits_final.json`，保证 validation 只来自原 27 个 real labeled cases，pseudo cases 永远只在 train。

推荐 split 策略：

```text
real labeled 27 cases: 沿用 Dataset101 原始 5-fold split
pseudo cases: 每个 fold 全部加入 train，不加入 val
```

这样每个 fold 的 validation 仍然是 real labeled case，和当前 v17 内部验证更可比。

## 10. 阶段 F：完整 5-fold 和提交

只有当阶段 E 满足以下条件，才进入完整 5-fold：

- fold 0 real-val Dice 不低于原 v17 fold 0。
- prediction volume 分布没有明显异常。
- validation 中没有出现空 mask、巨大 mask 或碎片爆炸。
- 对竞赛 val images 的预测体积分布仍接近 v17。

完整训练：

```bash
nnUNetv2_train 111 3d_fullres 0
nnUNetv2_train 111 3d_fullres 1
nnUNetv2_train 111 3d_fullres 2
nnUNetv2_train 111 3d_fullres 3
nnUNetv2_train 111 3d_fullres 4
```

推理：

```bash
nnUNetv2_predict \
  -i outputs/nnunet_task1_pseudo_top100/nnUNet_raw/Dataset111_MVAA_Task1_PseudoTop100/imagesTs \
  -o outputs/submissions/submit_task1_pseudo_top100/nnunet_task1_pred \
  -d 111 \
  -c 3d_fullres \
  -f 0 1 2 3 4 \
  -chk checkpoint_best.pth \
  -device cuda \
  --disable_progress_bar
```

然后用转换脚本生成提交格式：

```bash
/home/wuyongji/miniconda3/envs/mvaa/bin/python scripts/convert_task1_nnunet_predictions.py \
  --mapping-json outputs/nnunet_task1_pseudo_top100/task1_case_mapping.json \
  --pred-dir outputs/submissions/submit_task1_pseudo_top100/nnunet_task1_pred \
  --submission-task-dir outputs/submissions/submit_task1_pseudo_top100/submission/t1_ct \
  --overwrite
```

最终提交应复用当前最强 Task2/Task3：

```text
Task1: pseudo self-training result
Task2: v19 nnU-Net 5fold
Task3: v15
```

## 11. 推荐实验矩阵

第一轮只做最小可行验证：

| 实验 | Dataset | Pseudo 数量 | Fold | 目的 |
|---|---|---:|---|---|
| ST-A | Dataset111 | top100 | fold 0 | 判断自训练方向是否有效 |
| ST-B | Dataset112 | top200 | fold 0 | 如果 top100 稳定，测试更多 pseudo |
| ST-C | Dataset111 | top100 | fold 0-4 | 完整候选 |

不建议第一轮做：

- 1040 全量 pseudo。
- high+medium+low 全混合。
- 没有 confidence/volume 筛选就训练。
- 直接改原 Dataset101。

## 12. 风险点和止损条件

风险：

- 伪标签边界错误被 student 学进去。
- teacher 的系统性偏差被放大。
- outlier unlabeled 扰乱 nnU-Net 的 spacing/patch planning。
- pseudo cases 混入 validation，导致内部验证虚高。
- 加太多 pseudo 后真实 labeled 权重被稀释。

止损条件：

- fold 0 real-val Dice 明显低于 v17 fold 0。
- pseudo dataset 训练中 validation pseudo Dice 高、real Dice 低。
- 新模型对竞赛 val 的预测体积分布明显偏离 v17。
- 输出出现空 mask、大量碎片或异常大 mask。

## 13. 下一步执行清单

建议按以下顺序执行：

1. 从 `unlabeled_metadata_screen.csv` 导出 high-priority case list。
2. 准备 high-priority nnU-Net `imagesTs` 临时推理目录。
3. 用 v17 5-fold ensemble 对 high-priority unlabeled 推理，保存 hard mask 和 probabilities。
4. 写 pseudo quality scoring 脚本。
5. 选出 top100/top200 pseudo cases。
6. 写 pseudo dataset prepare 脚本，生成 `Dataset111` 和自定义 `splits_final.json`。
7. 先跑 `Dataset111 fold0`。
8. 对比原 v17 fold0 internal validation。
9. 如果通过，再扩展到 top200 或完整 5-fold。

当前推荐优先执行：

```text
阶段 B：只对 high-priority 759 个 unlabeled 做 v17 5-fold pseudo-label 推理并保存概率。
```
