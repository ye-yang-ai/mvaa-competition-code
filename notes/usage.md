# MVAA 电脑运行命令 PowerShell

本文档只记录 Windows 电脑 PowerShell 写法。PowerShell 使用 ` .\path\file.py ` 路径和反引号 `` ` `` 换行。

默认前提：

- 已经进入项目根目录。
- 已经激活 `mvaa` conda 环境。

```powershell
cd C:\Users\changba\Desktop\yy_files\competiton2
conda activate mvaa
```

## 通用检查

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Task1

### 训练 quick baseline

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

### 正式实验建议

Task1 是 3D CT 半监督分割，20 epoch 主要用于流程验证。正式实验建议从 `epochs 100` 到 `200`、`model-size base/large`、`batch-size 1/2` 做对比。训练参数和结果统一记录到 `notes/training_runs.md`。

```powershell
python .\task1\train.py `
  --data-root .\data\reference_data\t1_ct `
  --output-dir .\outputs\exp\task1_base_e100_bs1_mb `
  --epochs 100 `
  --batch-size 1 `
  --steps-per-epoch 0 `
  --roi-size 128 128 128 `
  --train-crops 1 `
  --sw-batch-size 1 `
  --val-interval 2 `
  --num-workers 0 `
  --model-size base `
  --unsup-warmup-epochs 10 `
  --unsup-rampup-epochs 30
```

### 查看训练日志

```powershell
Get-Content .\outputs\quick\task1\train.log -Wait
Get-Content .\outputs\quick\task1\train.log -Tail 30
```

### 检查 checkpoint

```powershell
dir .\outputs\quick\task1\checkpoints
```

正常会看到 `best_model.pt` 和 `latest_model.pt`，部分旧说明里写作 `last_model.pt`。

### 生成验证集预测

当前推荐使用 `task1-exp-004` 的 large 模型：

```powershell
python .\task1\generate_task1_predictions.py `
  --ckpt-path .\outputs\exp\task1_base_e100_b1_ml\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t1_ct\val\images `
  --submission-task-dir .\outputs\quick\submission\t1_ct `
  --num-workers 0
```

### 检查提交文件

```powershell
dir .\outputs\quick\submission\t1_ct
(dir .\outputs\quick\submission\t1_ct\*.nii.gz).Count
Test-Path .\outputs\quick\submission\t1_ct\task1_predictions.json
```

正常数量是 `30`，JSON 检查应为 `True`。

## Task2

### 训练 quick baseline

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

### 查看日志和 checkpoint

```powershell
Get-Content .\outputs\quick\task2\train.log -Wait
Get-Content .\outputs\quick\task2\train.log -Tail 30
dir .\outputs\quick\task2\checkpoints
```

### 生成验证集预测

```powershell
python .\task2\generate_task2_predictions.py `
  --ckpt-path .\outputs\quick\task2\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t2_tee\val\images `
  --submission-task-dir .\outputs\quick\submission\t2_tee `
  --num-workers 0
```

### 检查提交文件

```powershell
dir .\outputs\quick\submission\t2_tee
(dir .\outputs\quick\submission\t2_tee\*.nii.gz).Count
Test-Path .\outputs\quick\submission\t2_tee\task2_predictions.json
```

正常数量是 `20`，JSON 检查应为 `True`。

## Task3

### 训练 quick baseline

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

### 查看日志和 checkpoint

```powershell
Get-Content .\outputs\quick\task3\train.log -Wait
Get-Content .\outputs\quick\task3\train.log -Tail 30
dir .\outputs\quick\task3\checkpoints
```

### 生成验证集预测

```powershell
python .\task3\generate_task3_predictions.py `
  --ckpt-path .\outputs\quick\task3\checkpoints\best.pt `
  --data-dir .\data\reference_data\t3_vid\val\images `
  --submission-task-dir .\outputs\quick\submission\t3_vid `
  --no-tta
```

### 检查提交文件

```powershell
dir .\outputs\quick\submission\t3_vid
(dir .\outputs\quick\submission\t3_vid -Recurse -Filter *_label_bin.png).Count
Test-Path .\outputs\quick\submission\t3_vid\task3_predictions.json
```

正常数量是 `48`，JSON 检查应为 `True`。

## 打包提交

压缩包根目录必须直接包含 `t1_ct/`、`t2_tee/`、`t3_vid/`，不能多套一层 `submission/`。不要用 PowerShell 的 `Compress-Archive` 打包本项目提交文件。

```powershell
python .\scripts\package_submission.py `
  --submission-dir .\outputs\quick\submission `
  --output-zip .\outputs\quick\submission.zip
```

检查 zip 结构：

```powershell
tar -tf .\outputs\quick\submission.zip | Select-Object -First 30
```

上传文件：

```text
.\outputs\quick\submission.zip
```
