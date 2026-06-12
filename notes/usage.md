# MVAA 本地运行命令

本文档记录已经在本机跑通的 quick baseline 命令。

默认前提：

- 当前终端是 PowerShell
- 已经进入项目根目录
- 已经激活 `mvaa` conda 环境

```powershell
cd C:\Users\changba\Desktop\yy_files\competiton2
conda activate mvaa
```

## 通用检查

### 检查 CUDA 是否可用

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

如果 `torch.cuda.is_available()` 输出 `True`，并且能看到显卡名称，说明 GPU 环境可用。

## Task1 快速 baseline

### 1. 训练 Task1

```powershell
python .\task1\train.py `
  --data-root .\data\reference_data\t1_ct `
  --output-dir .\outputs\quick\task1 `
  --epochs 20 `
  --batch-size 1 `
  --steps-per-epoch 0 `
  --roi-size 128 128 128 `
  --train-crops 1 `
  --sw-batch-size 1 `
  --val-interval 2 `
  --num-workers 0 `
  --model-size small `
  --unsup-warmup-epochs 5 `
  --unsup-rampup-epochs 10
```

说明：

- `--data-root` 指向 Task1 的 CT 数据目录。
- `--output-dir` 是训练输出目录。
- `--epochs 20` 是先跑一版 quick baseline，不是最终最优训练。
- `--batch-size 1` 和 `--sw-batch-size 1` 更适合 8GB 显存。
- `--model-size small` 先保证能稳定跑通。

### 2. 查看 Task1 训练日志

持续查看日志：

```powershell
Get-Content .\outputs\quick\task1\train.log -Wait
```

只查看最后 30 行：

```powershell
Get-Content .\outputs\quick\task1\train.log -Tail 30
```

### 3. 检查 Task1 checkpoint

```powershell
dir .\outputs\quick\task1\checkpoints
```

正常情况下应该能看到：

```text
best_model.pt
last_model.pt
```

### 4. 生成 Task1 验证集预测

```powershell
python .\task1\generate_task1_predictions.py `
  --ckpt-path .\outputs\quick\task1\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t1_ct\val\images `
  --submission-task-dir .\outputs\quick\submission\t1_ct `
  --num-workers 0
```

### 5. 检查 Task1 提交文件

查看输出目录：

```powershell
dir .\outputs\quick\submission\t1_ct
```

检查 `.nii.gz` 预测文件数量：

```powershell
(dir .\outputs\quick\submission\t1_ct\*.nii.gz).Count
```

正常结果应该是：

```text
30
```

检查 JSON 是否存在：

```powershell
Test-Path .\outputs\quick\submission\t1_ct\task1_predictions.json
```

正常结果应该是：

```text
True
```

## Task2 快速 baseline

### 1. 训练 Task2

```powershell
python .\task2\train.py `
  --data-dir .\data\reference_data\t2_tee\train `
  --output-dir .\outputs\quick\task2 `
  --epochs 20 `
  --batch-size 1 `
  --roi-size 128 128 128 `
  --train-crops 1 `
  --sw-batch-size 1 `
  --val-interval 2 `
  --val-count 20 `
  --num-workers 0 `
  --model-size small
```

说明：

- `--data-dir` 指向 Task2 的 3D TEE 训练数据目录。
- `--val-count 20` 表示从训练集中取最后 20 个病例作为内部验证集。
- `--output-dir` 是 Task2 的训练输出目录。
- `--batch-size 1` 和 `--sw-batch-size 1` 更适合 8GB 显存。
- `--model-size small` 先保证能稳定跑通。

### 2. 查看 Task2 训练日志

持续查看日志：

```powershell
Get-Content .\outputs\quick\task2\train.log -Wait
```

只查看最后 30 行：

```powershell
Get-Content .\outputs\quick\task2\train.log -Tail 30
```

### 3. 检查 Task2 checkpoint

```powershell
dir .\outputs\quick\task2\checkpoints
```

正常情况下应该能看到：

```text
best_model.pt
last_model.pt
```

### 4. 生成 Task2 验证集预测

```powershell
python .\task2\generate_task2_predictions.py `
  --ckpt-path .\outputs\quick\task2\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t2_tee\val\images `
  --submission-task-dir .\outputs\quick\submission\t2_tee `
  --num-workers 0
```

### 5. 检查 Task2 提交文件

查看输出目录：

```powershell
dir .\outputs\quick\submission\t2_tee
```

检查 `.nii.gz` 预测文件数量：

```powershell
(dir .\outputs\quick\submission\t2_tee\*.nii.gz).Count
```

正常结果应该是：

```text
20
```

检查 JSON 是否存在：

```powershell
Test-Path .\outputs\quick\submission\t2_tee\task2_predictions.json
```

正常结果应该是：

```text
True
```

## Task3 快速 baseline

### 1. 训练 Task3

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
  --warmup-epochs 2 `
  --semi-warmup-epochs 5 `
  --unsup-ramp-epochs 10 `
  --save-every 5 `
  --early-stop-patience 0 `
  --no-val-tta
```

说明：

- `--labeled-root` 指向 Task3 的有标注训练帧。
- `--unlabeled-root` 指向 Task3 的无标注视频帧。
- `--image-size 448 800` 使用接近 baseline 默认的输入尺寸。
- `--arch unet` 和 `--encoder-name resnet18` 是较轻量的组合，适合先跑 quick baseline。
- `--encoder-weights none` 表示不下载 ImageNet 预训练权重。
- `--no-val-tta` 表示内部验证时不使用 TTA，加快训练。

### 2. 查看 Task3 训练日志

持续查看日志：

```powershell
Get-Content .\outputs\quick\task3\train.log -Wait
```

只查看最后 30 行：

```powershell
Get-Content .\outputs\quick\task3\train.log -Tail 30
```

### 3. 检查 Task3 checkpoint

```powershell
dir .\outputs\quick\task3\checkpoints
```

正常情况下应该能看到：

```text
best.pt
last.pt
epoch_005.pt
epoch_010.pt
...
```

### 4. 生成 Task3 验证集预测

```powershell
python .\task3\generate_task3_predictions.py `
  --ckpt-path .\outputs\quick\task3\checkpoints\best.pt `
  --data-dir .\data\reference_data\t3_vid\val\images `
  --submission-task-dir .\outputs\quick\submission\t3_vid `
  --no-tta
```

### 5. 检查 Task3 提交文件

查看输出目录：

```powershell
dir .\outputs\quick\submission\t3_vid
```

Task3 会保留视频子文件夹，所以用下面的命令统计预测 PNG 数量：

```powershell
(dir .\outputs\quick\submission\t3_vid -Recurse -Filter *_label_bin.png).Count
```

正常结果应该是：

```text
48
```

检查 JSON 是否存在：

```powershell
Test-Path .\outputs\quick\submission\t3_vid\task3_predictions.json
```

正常结果应该是：

```text
True
```

## 打包完整提交文件

当 Task1、Task2、Task3 都已经生成验证集预测后，需要把三个任务的结果打包成一个 `submission.zip` 上传到 CodaBench。

### 1. 检查三个任务目录是否存在

```powershell
dir .\outputs\quick\submission
```

正常情况下应该能看到：

```text
t1_ct
t2_tee
t3_vid
```

### 2. 检查三个任务的预测文件数量

```powershell
(dir .\outputs\quick\submission\t1_ct\*.nii.gz).Count
(dir .\outputs\quick\submission\t2_tee\*.nii.gz).Count
(dir .\outputs\quick\submission\t3_vid -Recurse -Filter *_label_bin.png).Count
```

正常结果应该依次是：

```text
30
20
48
```

### 3. 检查三个任务的 JSON 是否存在

```powershell
Test-Path .\outputs\quick\submission\t1_ct\task1_predictions.json
Test-Path .\outputs\quick\submission\t2_tee\task2_predictions.json
Test-Path .\outputs\quick\submission\t3_vid\task3_predictions.json
```

正常结果应该全部是：

```text
True
True
True
```

### 4. 压缩生成 submission.zip

注意：压缩包根目录必须直接包含 `t1_ct/`、`t2_tee/`、`t3_vid/`，不能多套一层 `submission/`。

不要使用 PowerShell 的 `Compress-Archive` 打包本项目的提交文件，因为它可能在 zip 内写入 Windows 反斜杠路径，导致 CodaBench 无法识别任务目录。

```powershell
python .\scripts\package_submission.py `
  --submission-dir .\outputs\quick\submission `
  --output-zip .\outputs\quick\submission.zip
```

生成的提交文件路径是：

```text
.\outputs\quick\submission.zip
```

### 5. 检查 zip 内部结构

```powershell
tar -tf .\outputs\quick\submission.zip | Select-Object -First 30
```

正常情况下应该看到类似：

```text
t1_ct/task1_predictions.json
t1_ct/0001-pred.nii.gz
t1_ct/0002-pred.nii.gz
...
t2_tee/task2_predictions.json
t2_tee/val_001-pred.nii.gz
...
t3_vid/task3_predictions.json
t3_vid/<video_folder>/*_label_bin.png
...
```

### 6. 上传到 CodaBench

上传文件：

```text
.\outputs\quick\submission.zip
```

上传后在 CodaBench validation phase 查看平台返回的指标：

```text
Task1_DSC / Task1_HD / Task1_ASD
Task2_DSC / Task2_HD / Task2_ASD
Task3_DSC / Task3_HD / Task3_ASD
```

第一版 quick baseline 的主要目标是确认提交格式和平台评测流程正常，不代表最终成绩。
