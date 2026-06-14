# MVAA 服务器运行命令 Bash

本文档只记录 Linux 服务器 bash 写法。bash 使用 `/` 路径和反斜杠 `\` 换行，不要使用 PowerShell 的反引号 `` ` ``。

默认前提：

- 当前机器：`yangye@CVI-A6k90`
- 项目目录：`/data1/yangye/projects/MVAA`
- 已经激活 `mvaa` conda 环境。

```bash
cd /data1/yangye/projects/MVAA
conda activate mvaa
```

## 通用检查

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
nvidia-smi
```

指定显卡时使用 `CUDA_VISIBLE_DEVICES=0`、`CUDA_VISIBLE_DEVICES=1`、`CUDA_VISIBLE_DEVICES=2` 或 `CUDA_VISIBLE_DEVICES=3`。例如指定物理 3 号卡：

```bash
CUDA_VISIBLE_DEVICES=3 python task1/train.py --help
```

## Task1

### 训练 quick baseline

```bash
CUDA_VISIBLE_DEVICES=0 python task1/train.py \
  --data-root data/reference_data/t1_ct \
  --output-dir outputs/quick/task1 \
  --epochs 20 \
  --batch-size 1 \
  --steps-per-epoch 0 \
  --roi-size 128 128 128 \
  --train-crops 1 \
  --sw-batch-size 1 \
  --val-interval 2 \
  --num-workers 0 \
  --model-size small \
  --unsup-warmup-epochs 5 \
  --unsup-rampup-epochs 10
```

### 正式实验建议

Task1 是 3D CT 半监督分割，20 epoch 主要用于流程验证。正式实验建议从 `epochs 100` 到 `200`、`model-size base/large`、`batch-size 1/2` 做对比。训练参数和结果统一记录到 `notes/training_runs.md`。

当前已经完成的 4 次实验见 `notes/training_runs.md`。内部验证分数最高的是 `outputs/exp/task1_base_e100_b1_ml`，checkpoint 为：

```text
outputs/exp/task1_base_e100_b1_ml/checkpoints/best_model.pt
```

推荐继续对比时从下面命令改参数：

```bash
CUDA_VISIBLE_DEVICES=0 python task1/train.py \
  --data-root data/reference_data/t1_ct \
  --output-dir outputs/exp/task1_base_e100_bs1_mb \
  --epochs 100 \
  --batch-size 1 \
  --steps-per-epoch 0 \
  --roi-size 128 128 128 \
  --train-crops 1 \
  --sw-batch-size 1 \
  --val-interval 2 \
  --num-workers 0 \
  --model-size base \
  --unsup-warmup-epochs 10 \
  --unsup-rampup-epochs 30
```

### 查看训练日志

```bash
tail -f outputs/quick/task1/train.log
tail -n 30 outputs/quick/task1/train.log
```

### 检查 checkpoint

```bash
ls -la outputs/quick/task1/checkpoints
```

正常会看到 `best_model.pt` 和 `latest_model.pt`，部分旧说明里写作 `last_model.pt`。

### 生成验证集预测

当前推荐使用 `task1-exp-004` 的 large 模型：

```bash
CUDA_VISIBLE_DEVICES=0 python task1/generate_task1_predictions.py \
  --ckpt-path outputs/exp/task1_base_e100_b1_ml/checkpoints/best_model.pt \
  --data-dir data/reference_data/t1_ct/val/images \
  --submission-task-dir outputs/quick/submission/t1_ct \
  --num-workers 0
```

### 检查提交文件

```bash
ls -la outputs/quick/submission/t1_ct
find outputs/quick/submission/t1_ct -maxdepth 1 -name '*.nii.gz' | wc -l
test -f outputs/quick/submission/t1_ct/task1_predictions.json && echo True || echo False
```

正常数量是 `30`，JSON 检查应为 `True`。

## Task2

### 训练 quick baseline

```bash
CUDA_VISIBLE_DEVICES=0 python task2/train.py \
  --data-dir data/reference_data/t2_tee/train \
  --output-dir outputs/quick/task2 \
  --epochs 20 \
  --batch-size 1 \
  --roi-size 128 128 128 \
  --train-crops 1 \
  --sw-batch-size 1 \
  --val-interval 2 \
  --val-count 20 \
  --num-workers 0 \
  --model-size small
```

### 查看日志和 checkpoint

```bash
tail -f outputs/quick/task2/train.log
tail -n 30 outputs/quick/task2/train.log
ls -la outputs/quick/task2/checkpoints
```

### 生成验证集预测

```bash
CUDA_VISIBLE_DEVICES=0 python task2/generate_task2_predictions.py \
  --ckpt-path outputs/quick/task2/checkpoints/best_model.pt \
  --data-dir data/reference_data/t2_tee/val/images \
  --submission-task-dir outputs/quick/submission/t2_tee \
  --num-workers 0
```

### 检查提交文件

```bash
ls -la outputs/quick/submission/t2_tee
find outputs/quick/submission/t2_tee -maxdepth 1 -name '*.nii.gz' | wc -l
test -f outputs/quick/submission/t2_tee/task2_predictions.json && echo True || echo False
```

正常数量是 `20`，JSON 检查应为 `True`。

## Task3

### 训练 quick baseline

```bash
CUDA_VISIBLE_DEVICES=0 python task3/train.py \
  --labeled-root data/reference_data/t3_vid/train \
  --unlabeled-root data/images \
  --output-dir outputs/quick/task3 \
  --epochs 20 \
  --batch-size 1 \
  --unlabeled-batch-size 1 \
  --image-size 448 800 \
  --val-video-count 1 \
  --num-workers 0 \
  --arch unet \
  --encoder-name resnet18 \
  --encoder-weights none \
  --warmup-epochs 2 \
  --semi-warmup-epochs 5 \
  --unsup-ramp-epochs 10 \
  --save-every 5 \
  --early-stop-patience 0 \
  --no-val-tta
```

### 查看日志和 checkpoint

```bash
tail -f outputs/quick/task3/train.log
tail -n 30 outputs/quick/task3/train.log
ls -la outputs/quick/task3/checkpoints
```

### 生成验证集预测

```bash
CUDA_VISIBLE_DEVICES=0 python task3/generate_task3_predictions.py \
  --ckpt-path outputs/quick/task3/checkpoints/best.pt \
  --data-dir data/reference_data/t3_vid/val/images \
  --submission-task-dir outputs/quick/submission/t3_vid \
  --no-tta
```

### 检查提交文件

```bash
ls -la outputs/quick/submission/t3_vid
find outputs/quick/submission/t3_vid -name '*_label_bin.png' | wc -l
test -f outputs/quick/submission/t3_vid/task3_predictions.json && echo True || echo False
```

正常数量是 `48`，JSON 检查应为 `True`。

## 打包提交

压缩包根目录必须直接包含 `t1_ct/`、`t2_tee/`、`t3_vid/`，不能多套一层 `submission/`。

```bash
python scripts/package_submission.py \
  --submission-dir outputs/quick/submission \
  --output-zip outputs/quick/submission.zip
```

检查 zip 结构：

```bash
tar -tf outputs/quick/submission.zip | head -30
```

上传文件：

```text
outputs/quick/submission.zip
```
