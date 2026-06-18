# MVAA 三个 Task 代码结构与训练、检测、提交说明

本文档整理当前项目中 `task1`、`task2`、`task3` 的文件结构、脚本作用、关键函数、数据集结构，以及从训练到预测、打包提交的完整流程。

项目根目录：

```text
C:\Users\changba\Desktop\yy_files\competiton2
  README.md
  requirements.txt
  data/
  outputs/
  task1/
  task2/
  task3/
  notes/
  submission.zip
```

## 1. 三个 Task 的总体关系

三个任务都是医学图像/视频帧分割任务，但输入数据、监督方式和输出格式不同。

| Task | 任务内容 | 输入 | 输出 | 训练方式 |
| --- | --- | --- | --- | --- |
| Task1 | Cardiac CT 分割 | 3D CT NIfTI, `.nii.gz` | 3D 分割 mask, `.nii.gz` | 半监督：有标注 + 无标注 CT |
| Task2 | 3D TEE 分割 | 3D 超声 NIfTI, `.nii.gz` | 3D 分割 mask, `.nii.gz` | 全监督：图像 + 标签 |
| Task3 | 手术视频帧目标检测/分割 | 2D RGB 帧图像, `.png` | 二值 mask, `*_label_bin.png` | 半监督：有标注帧 + 无标注帧 |

注意：这里的“检测流程”在代码中实际表现为“推理/预测分割 mask”，即加载 checkpoint，对验证集或测试集图片生成提交所需的 mask 和 JSON。

## 2. Task 目录通用文件结构

每个 task 目录都有同一套核心脚本：

```text
taskX/
  dataset.py
  model_factory.py
  train.py
  utils.py
  generate_taskX_predictions.py
```

各文件作用：

| 文件 | 作用 |
| --- | --- |
| `dataset.py` | 发现数据文件、配对图像和标签、定义数据增强和 DataLoader。 |
| `model_factory.py` | 创建模型和 loss。Task1/Task2 使用 MONAI 3D UNet；Task3 使用 `segmentation_models_pytorch` 的 2D 网络。 |
| `train.py` | 训练入口。解析命令行参数、构建数据、模型、优化器、scheduler、训练循环、验证指标、保存 checkpoint。 |
| `utils.py` | 通用工具函数，例如随机种子、目录创建、JSON 保存、设备选择、指标打分、EMA 更新等。 |
| `generate_taskX_predictions.py` | 推理/预测入口。加载 best checkpoint，读取验证/测试图片，生成 mask 文件和 `taskX_predictions.json`。 |

## 3. Task1：Cardiac CT 半监督分割

### 3.1 文件结构

```text
task1/
  dataset.py
  model_factory.py
  train.py
  utils.py
  generate_task1_predictions.py
```

### 3.2 数据集结构

当前本地数据位于：

```text
data/reference_data/t1_ct/
  train/
    labeled/
      images/
        0001.nii.gz
        ...
      labels/
        0001-seg.nii.gz
        ...
    unlabeled/
      *.nii.gz
  val/
    images/
      *.nii.gz
```

当前统计：

```text
有标注训练图像: 27
有标注训练标签: 27
无标注训练图像: 1040
验证图像: 30
```

标签命名规则：`images/0001.nii.gz` 对应 `labels/0001-seg.nii.gz`。

### 3.3 脚本和关键函数

`task1/dataset.py`

| 函数/类 | 作用 |
| --- | --- |
| `discover_labeled_pairs(images_dir, labels_dir)` | 扫描有标注 CT 图像，并按 `case_id-seg.nii.gz` 找对应标签。 |
| `discover_unlabeled_files(images_dir)` | 扫描无标注 CT 图像，返回只包含 image 和 case_id 的样本列表。 |
| `get_labeled_train_transforms(...)` | 有标注训练增强：读取 NIfTI、归一化 HU 到 `[0,1]`、随机 crop、flip、rotate。 |
| `get_unlabeled_train_transforms(...)` | 无标注训练增强：只处理 image，随机空间 crop 和几何增强。 |
| `get_eval_transforms(...)` | 验证时的数据预处理，不做随机增强。 |
| `get_task1_dataloaders(...)` | 构建 labeled loader、unlabeled loader、内部 val loader，并保存 split 信息。 |
| `FixInvalidAffineD` | 在启用 spacing resample 时修复非法 affine，避免 MONAI 重采样报错。 |

`task1/model_factory.py`

| 函数 | 作用 |
| --- | --- |
| `get_model(...)` | 创建 3D UNet，支持 `small/base/large` 三种模型规模。 |
| `get_supervised_loss_fn()` | 返回 Dice + Cross Entropy 监督 loss。 |
| `get_unsupervised_loss_fn()` | 返回逐体素 Cross Entropy，用于 teacher 伪标签监督无标注数据。 |

`task1/train.py`

| 函数 | 作用 |
| --- | --- |
| `parse_args()` | 定义训练参数，如 `--data-root`、`--output-dir`、`--epochs`、`--roi-size`、半监督参数等。 |
| `_supervised_step(...)` | 对有标注 batch 前向计算 supervised loss。 |
| `_unsupervised_step(...)` | teacher 对无标注图像产生伪标签，student 对强扰动图像学习伪标签。 |
| `train_one_epoch(...)` | 一个 epoch 的训练逻辑，合并监督 loss 和无监督 consistency loss，并更新 EMA teacher。 |
| `evaluate(...)` | sliding window 推理验证集，计算 Dice、Hausdorff Distance、ASD。 |
| `main()` | 训练主流程：建 dataloader、student/teacher 模型、optimizer、scheduler、记录日志、保存 checkpoint。 |

`task1/utils.py`

| 函数/类 | 作用 |
| --- | --- |
| `MetricRefs` | 保存 HD/ASD 的参考值，用于把不同指标归一化成综合分数。 |
| `seed_everything(seed)` | 固定 Python、NumPy、PyTorch 随机种子。 |
| `ensure_dir(path)` | 创建输出目录。 |
| `metric_quality(...)` | 将 DSC、HD、ASD 转为统一 score，用于选择 best checkpoint。 |
| `get_current_unsup_weight(...)` | 根据 rampup 进度计算当前无监督 loss 权重。 |
| `update_ema_variables(student, teacher, ema_decay)` | 用 student 权重的 EMA 更新 teacher。 |
| `apply_strong_perturbation(...)` | 对无标注图像加入噪声和 dropout，构造强增强输入。 |

`task1/generate_task1_predictions.py`

| 函数 | 作用 |
| --- | --- |
| `discover_images(folder)` | 扫描待预测的 `.nii.gz` CT 图像。 |
| `get_infer_transforms(...)` | 构建推理预处理流程。 |
| `build_loader(...)` | 构建推理 DataLoader。 |
| `load_ckpt_config(ckpt_path)` | 读取 checkpoint 和训练参数。 |
| `_load_state_dict(model, ckpt_obj)` | 兼容不同 checkpoint 字段，加载模型权重。 |
| `resize_mask_to_shape(mask, out_shape)` | 将预测 mask resize 回原图 shape。 |
| `save_prediction_nifti(...)` | 用原图 affine/header 保存 `.nii.gz` 预测结果。 |
| `main()` | 完整预测流程，输出 `*-pred.nii.gz` 和 `task1_predictions.json`。 |

### 3.4 Task1 训练流程

1. 从 `data/reference_data/t1_ct/train/labeled` 读取有标注图像和标签。
2. 从 `data/reference_data/t1_ct/train/unlabeled` 读取无标注图像。
3. 有标注数据按 `val-ratio` 或 `val-count` 切一部分作为内部验证集。
4. 创建两个模型：`student` 和 `teacher`。teacher 初始复制 student，并且不直接反向传播。
5. 前 `--unsup-warmup-epochs` 个 epoch 只训练有标注数据。
6. 之后 teacher 给无标注图像生成伪标签，student 在强扰动图像上学习伪标签。
7. 每个优化 step 后用 EMA 更新 teacher。
8. 每隔 `--val-interval` 做 sliding window 验证，按综合 score 保存 `best_model.pt`。
9. 每个 epoch 保存 `latest_model.pt`。

常用本地训练命令：

```powershell
python .\task1\train.py `
  --data-root .\data\reference_data\t1_ct `
  --output-dir .\outputs\quick\task1 `
  --epochs 20 `
  --batch-size 1 `
  --roi-size 128 128 128 `
  --train-crops 1 `
  --sw-batch-size 1 `
  --num-workers 0 `
  --model-size small
```

### 3.5 Task1 检测/预测流程

1. 加载 `outputs/.../task1/checkpoints/best_model.pt`。
2. 读取 checkpoint 中保存的模型参数，例如 `num_classes`、`model_size`、`roi_size`。
3. 扫描 `data/reference_data/t1_ct/val/images` 下的 `.nii.gz`。
4. 用 MONAI `SlidingWindowInferer` 做 3D 滑窗推理。
5. 对 logits 做 `argmax` 得到类别 mask。
6. 将 mask resize 回原始 CT shape。
7. 保存为 `case_id-pred.nii.gz`。
8. 写出 `task1_predictions.json`，其中每条记录包含 `case_id` 和相对路径 `segmentation`。

命令示例：

```powershell
python .\task1\generate_task1_predictions.py `
  --ckpt-path .\outputs\quick\task1\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t1_ct\val\images `
  --submission-task-dir .\outputs\quick\submission\t1_ct `
  --num-workers 0
```

## 4. Task2：3D TEE 全监督分割

### 4.1 文件结构

```text
task2/
  dataset.py
  model_factory.py
  train.py
  utils.py
  generate_task2_predictions.py
```

### 4.2 数据集结构

当前本地数据位于：

```text
data/reference_data/t2_tee/
  train/
    train_001-US.nii.gz
    train_001-label.nii.gz
    train_002-US.nii.gz
    train_002-label.nii.gz
    ...
  val/
    images/
      val_001-US.nii.gz
      ...
```

当前统计：

```text
训练 US 图像: 105
训练标签: 105
验证 US 图像: 20
```

标签命名规则：`train_001-US.nii.gz` 对应 `train_001-label.nii.gz`。

### 4.3 脚本和关键函数

`task2/dataset.py`

| 函数 | 作用 |
| --- | --- |
| `discover_case_pairs(data_dir)` | 扫描 `*-US.nii.gz`，并查找对应 `*-label.nii.gz`。 |
| `split_train_val(files, val_count)` | 使用最后 `val_count` 个 case 作为内部验证集。 |
| `get_train_transforms(...)` | 训练增强：强度归一化、按类别 crop、仿射、flip、rotate、噪声、对比度、平滑等。 |
| `get_val_transforms(...)` | 验证预处理，不做随机增强。 |
| `get_dataloaders(...)` | 构建 train/val DataLoader，并返回 train/val 文件列表。 |

`task2/model_factory.py`

| 函数 | 作用 |
| --- | --- |
| `get_model(...)` | 创建 MONAI 3D UNet，支持 `small/base/large`。 |
| `get_loss_fn()` | 返回 Dice + Cross Entropy loss。 |

`task2/train.py`

| 函数 | 作用 |
| --- | --- |
| `parse_args()` | 定义训练参数，例如 `--data-dir`、`--val-count`、`--num-classes`、`--roi-size`。 |
| `train_one_epoch(...)` | 标准全监督训练，一个 batch 前向、计算 loss、反向传播、梯度裁剪、优化器更新。 |
| `validate(...)` | sliding window 验证，计算 Dice、HD、ASD。 |
| `main()` | 训练主流程，保存 `best_model.pt`、`latest_model.pt`、`history.csv`、`best_metrics.json`。 |

`task2/utils.py`

| 函数/类 | 作用 |
| --- | --- |
| `MetricRefs` | 保存 HD/ASD 参考值。 |
| `metric_quality_weighted(...)` | 按 DSC/HD/ASD 权重计算综合 score。 |
| `update_metric_refs(...)` | 根据当前验证指标平滑更新参考值。 |
| `seed_everything(...)`、`ensure_dir(...)`、`save_json(...)`、`get_device()` | 通用训练辅助函数。 |

`task2/generate_task2_predictions.py`

| 函数 | 作用 |
| --- | --- |
| `discover_images(folder)` | 扫描 `*-US.nii.gz` 验证/测试图像。 |
| `get_infer_transforms(...)` | 推理预处理。 |
| `build_loader(...)` | 构建推理 DataLoader。 |
| `load_ckpt_config(...)` | 加载 checkpoint 和训练参数。 |
| `resize_mask_to_shape(...)` | 将预测 mask resize 回原图 shape。 |
| `save_prediction_nifti(...)` | 保存 `.nii.gz` 预测 mask。 |
| `main()` | 输出 `*-pred.nii.gz` 和 `task2_predictions.json`。 |

### 4.4 Task2 训练流程

1. 扫描 `data/reference_data/t2_tee/train` 中的 US 图像和 label。
2. 按编号排序后，默认取最后 `--val-count 20` 个 case 做内部验证。
3. 构建 3D UNet 和 DiceCE loss。
4. 对训练 crop 做随机增强，使用 AdamW 优化。
5. 每隔 `--val-interval` 用 sliding window 做验证。
6. 根据 DSC/HD/ASD 的综合 score 保存 `best_model.pt`。
7. 每个 epoch 保存 `latest_model.pt`。

命令示例：

```powershell
python .\task2\train.py `
  --data-dir .\data\reference_data\t2_tee\train `
  --output-dir .\outputs\quick\task2 `
  --epochs 20 `
  --batch-size 1 `
  --roi-size 128 128 128 `
  --train-crops 1 `
  --sw-batch-size 1 `
  --val-count 20 `
  --num-workers 0 `
  --model-size small
```

### 4.5 Task2 检测/预测流程

1. 加载 `best_model.pt`。
2. 扫描 `data/reference_data/t2_tee/val/images` 中的 `*-US.nii.gz`。
3. 按训练参数恢复模型结构、ROI size、spacing 配置。
4. 用 sliding window 推理。
5. `argmax` 得到 3D 类别 mask。
6. 保存 `val_001-pred.nii.gz` 这类文件。
7. 生成 `task2_predictions.json`。

命令示例：

```powershell
python .\task2\generate_task2_predictions.py `
  --ckpt-path .\outputs\quick\task2\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t2_tee\val\images `
  --submission-task-dir .\outputs\quick\submission\t2_tee `
  --num-workers 0
```

## 5. Task3：手术视频帧半监督分割

### 5.1 文件结构

```text
task3/
  dataset.py
  model_factory.py
  train.py
  utils.py
  generate_task3_predictions.py
```

### 5.2 数据集结构

当前有标注数据位于：

```text
data/reference_data/t3_vid/
  train/
    REC_20250109_110158_583A/
      REC_20250109_110158_583A_000001.png
      REC_20250109_110158_583A_000001_png_Label.tar
      ...
    ...
  val/
    images/
      REC_.../
        REC_..._000001.png
        ...
```

无标注数据位于：

```text
data/images/
  PAVCB/
    *.png
  PAVCB2/
    *.png
  REC_.../
    *.png
```

当前统计：

```text
有标注训练视频文件夹: 6
有标注训练帧: 180
tar 标签: 180
验证视频文件夹: 2
验证帧: 48
无标注视频文件夹: 46
无标注帧: 1379
```

Task3 标注支持两种形式：

| 标签形式 | 说明 |
| --- | --- |
| `*_png_Label.tar` | tar 包中包含 `_Label.nii.gz`，代码读取其中像素值等于 `target_label` 的区域作为二值 mask。默认 `target_label=10`。 |
| `*_label_bin.png` | 已经转好的二值 PNG mask，像素大于 127 视为前景。 |

### 5.3 脚本和关键函数

`task3/dataset.py`

| 函数/类 | 作用 |
| --- | --- |
| `Sample` | 一个有标注样本的数据结构，包含 image、label、video_id、frame_idx。 |
| `read_binary_mask_from_label_tar(label_tar_path, target_label)` | 从 tar 中读取 NIfTI label，并转成指定 label 的二值 mask。 |
| `discover_samples(data_root)` | 递归扫描有标注帧，找到 `.png` 和对应 label。 |
| `split_train_val_by_video(samples, val_video_count, seed)` | 按视频维度切分训练/验证，避免同一视频帧同时出现在 train 和 val。 |
| `discover_unlabeled_images(unlabeled_root)` | 扫描无标注帧，过滤已有 label 或可视化文件。 |
| `sample_has_foreground(...)` | 判断样本 mask 是否有前景，用于验证集前景过滤。 |
| `build_fg_balanced_weights(...)` | 根据前景比例构造采样权重，缓解前景稀疏问题。 |
| `LabeledDataset` | 读取 RGB 图像和二值 mask，做几何/弱光照增强、resize、ImageNet normalization。 |
| `UnlabeledPairDataset` | 对同一无标注图像生成 weak 和 strong 两个增强版本，用于半监督一致性训练。 |

`task3/model_factory.py`

| 函数/类 | 作用 |
| --- | --- |
| `get_model(...)` | 创建 2D 分割模型，支持 `unet`、`unetplusplus`、`fpn`、`deeplabv3plus`。 |
| `DiceBCELoss` | Dice loss + BCE loss。 |
| `BinaryFocalWithLogitsLoss` | 二分类 focal loss，处理前景稀少。 |
| `DiceFocalLoss` | Dice loss + focal loss，默认使用。 |
| `get_loss_fn(...)` | 根据参数返回 `dice_bce` 或 `dice_focal` loss。 |

`task3/train.py`

| 函数 | 作用 |
| --- | --- |
| `parse_args()` | 定义训练参数，包括有标注路径、无标注路径、模型架构、半监督阈值、TTA、采样策略等。 |
| `dice_from_preds(...)` / `dice_from_logits(...)` | 计算二值分割 Dice。 |
| `predict_probs(...)` | 推理概率，可选水平/垂直翻转 TTA。 |
| `evaluate(...)` | 验证集评估：寻找最佳 threshold，计算 Dice、HD、ASD。 |
| `update_ema(teacher, student, decay)` | EMA 更新 teacher。 |
| `compute_unsup_weight(...)` | 根据 warmup/ramp 设置当前无监督 loss 权重。 |
| `main()` | 完整训练流程：按视频切分数据、构建 student/teacher、监督训练、无标注伪标签训练、验证、保存 checkpoint。 |

`task3/utils.py`

| 函数/类 | 作用 |
| --- | --- |
| `MetricRefs` | 保存 HD/ASD 参考值。 |
| `metric_quality_weighted(...)` | 按权重合成综合 score。 |
| `setup_logger(...)` | 同时输出终端和 `train.log`。 |
| `seed_everything(...)`、`ensure_dir(...)`、`save_json(...)`、`get_device()` | 通用工具函数。 |

`task3/generate_task3_predictions.py`

| 函数 | 作用 |
| --- | --- |
| `discover_images(folder, exts, video_folders)` | 扫描验证/测试图像，支持指定视频文件夹，也支持全量递归扫描。 |
| `load_ckpt_config(...)` | 读取 checkpoint 和训练参数。 |
| `pick_device(device_arg)` | 根据 `auto/cuda/cpu` 选择设备。 |
| `load_state_dict(...)` | 从 checkpoint 中加载模型权重。 |
| `predict_probs(...)` | 推理概率，可选 TTA。 |
| `main()` | 输出每帧二值 mask 和 `task3_predictions.json`。 |

### 5.4 Task3 训练流程

1. 从 `data/reference_data/t3_vid/train` 扫描有标注帧和 label。
2. 按视频切分 train/val，默认取若干视频作为内部验证集。
3. 可选择只保留有前景的验证样本，避免大量空 mask 影响 Dice。
4. 从 `data/images` 扫描无标注帧。
5. 创建 2D 分割模型，例如 `unetplusplus + efficientnet-b4` 或轻量的 `unet + resnet18`。
6. 创建 student 和 teacher，teacher 用 EMA 从 student 更新。
7. 有标注 batch 使用 Dice/Focal 或 Dice/BCE 监督训练。
8. 超过 `--semi-warmup-epochs` 后，无标注 batch 使用 teacher 对 weak 图像生成伪标签，student 对 strong 图像学习。
9. 用 `pseudo-pos-thr`、`pseudo-neg-thr`、`pseudo-min-area`、`pseudo-min-pos-ratio` 控制伪标签质量。
10. 验证时在多个 threshold 中选择 Dice 最好的阈值，并将该阈值写入 checkpoint 的 `val_metrics`。
11. 保存 `best.pt`、`last.pt`，并按 `--save-every` 保存 `epoch_XXX.pt`。

命令示例：

```powershell
python .\task3\train.py `
  --labeled-root .\data\reference_data\t3_vid\train `
  --unlabeled-root .\data\images `
  --output-dir .\outputs\quick\task3 `
  --epochs 20 `
  --batch-size 1 `
  --unlabeled-batch-size 1 `
  --image-size 448 800 `
  --val-video-count 1 `
  --num-workers 0 `
  --arch unet `
  --encoder-name resnet18 `
  --encoder-weights none `
  --no-val-tta
```

### 5.5 Task3 检测/预测流程

1. 加载 `outputs/.../task3/checkpoints/best.pt`。
2. 从 checkpoint 恢复模型架构、encoder、输入尺寸、ImageNet normalization、最佳 threshold。
3. 扫描 `data/reference_data/t3_vid/val/images` 下的所有图片。
4. 对每张图 resize 到训练尺寸。
5. 输出 sigmoid 概率，并可使用翻转 TTA。
6. 按 checkpoint 中保存的最佳 threshold 得到二值 mask。
7. resize 回原图大小。
8. 保存为 `原图stem_label_bin.png`。
9. 保留视频子目录结构，并生成 `task3_predictions.json`。

命令示例：

```powershell
python .\task3\generate_task3_predictions.py `
  --ckpt-path .\outputs\quick\task3\checkpoints\best.pt `
  --data-dir .\data\reference_data\t3_vid\val\images `
  --submission-task-dir .\outputs\quick\submission\t3_vid `
  --no-tta
```

## 6. 训练输出目录说明

建议所有训练结果放在 `outputs/` 下，例如：

```text
outputs/quick/
  task1/
    train.log
    history.csv
    split_info.json
    best_metrics.json
    checkpoints/
      best_model.pt
      latest_model.pt
  task2/
    train.log
    history.csv
    split_info.json
    best_metrics.json
    checkpoints/
      best_model.pt
      latest_model.pt
  task3/
    train.log
    history.csv
    split.json
    config.json
    final_metrics.json
    checkpoints/
      best.pt
      last.pt
      epoch_005.pt
      ...
```

`history.csv` 用来看每个 epoch 的 loss 和验证指标。`train.log` 用来看详细训练日志。`best_model.pt` 或 `best.pt` 是默认用于生成提交结果的 checkpoint。

## 7. 完整训练到提交流程

推荐流程：

1. 安装环境。
2. 分别训练 Task1、Task2、Task3。
3. 检查每个 task 的 checkpoint 是否生成。
4. 分别运行三个 `generate_task*_predictions.py`。
5. 检查提交目录中是否有 JSON 和预测 mask。
6. 将三个 task 文件夹打包成一个 `submission.zip`。

环境安装：

```powershell
conda activate mvaa
pip install -r .\requirements.txt
```

检查 checkpoint：

```powershell
dir .\outputs\quick\task1\checkpoints
dir .\outputs\quick\task2\checkpoints
dir .\outputs\quick\task3\checkpoints
```

生成三个 task 的预测后，推荐提交目录类似：

```text
outputs/quick/submission/
  t1_ct/
    task1_predictions.json
    0001-pred.nii.gz
    ...
  t2_tee/
    task2_predictions.json
    val_001-pred.nii.gz
    ...
  t3_vid/
    task3_predictions.json
    REC_.../
      REC_..._000001_label_bin.png
      ...
```

## 8. 提交什么东西，什么形式

最终提交一个 zip 文件，zip 根目录必须直接包含三个文件夹：

```text
submission.zip
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

不要把 `submission/` 或项目根目录本身包进 zip 里。也就是说，压缩包打开后第一层应该直接看到：

```text
t1_ct/
t2_tee/
t3_vid/
```

每个 task 的 JSON 最小格式如下：

```json
{
  "cases": [
    {
      "case_id": "0001",
      "segmentation": "0001-pred.nii.gz"
    }
  ]
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `case_id` | 样本 ID，通常来自输入文件名。 |
| `segmentation` | 预测 mask 的相对路径，必须相对于对应 task 文件夹。 |

Task3 的 `segmentation` 通常包含视频子目录，例如：

```json
{
  "case_id": "REC_20250507_123940_704A2_000003",
  "segmentation": "REC_20250507_123940_704A2/REC_20250507_123940_704A2_000003_label_bin.png"
}
```

PowerShell 打包命令：

```powershell
Compress-Archive `
  -Path .\outputs\quick\submission\t1_ct, .\outputs\quick\submission\t2_tee, .\outputs\quick\submission\t3_vid `
  -DestinationPath .\outputs\quick\submission.zip -Force
```

检查 zip 根目录：

```powershell
tar -tf .\outputs\quick\submission.zip | Select-Object -First 30
```

正确结果应该直接出现：

```text
t1_ct/
t2_tee/
t3_vid/
```

## 9. 本地已有 `submission.zip` 结构参考

当前项目根目录已有一个 `submission.zip`，其结构是正确的参考形式：

```text
submission.zip
  t1_ct/
    task1_predictions.json
    30 个 .nii.gz 预测文件
  t2_tee/
    task2_predictions.json
    20 个 .nii.gz 预测文件
  t3_vid/
    task3_predictions.json
    视频子目录/
      *_label_bin.png
```

如果后续重新训练并生成预测，只要保持同样结构即可提交。
