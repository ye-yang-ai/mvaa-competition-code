# MVAA Baseline Runbook

This repository imports the official MVAA baseline and keeps local data, outputs,
checkpoints, and generated submissions out of git.

## Environment

Create the local environment:

```powershell
conda create -y -n mvaa python=3.10
conda run -n mvaa python -m pip install --upgrade pip
conda run -n mvaa python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
conda run -n mvaa python -m pip install -r .\requirements.txt
```

Verify CUDA and core packages:

```powershell
conda run -n mvaa python -c "import torch, monai; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(monai.__version__)"
```

## Data Layout

The downloaded archives were extracted under `.\data`:

```text
data/
  MVAA_Data.zip
  task3_unlabeled.zip
  reference_data/
    t1_ct/
    t2_tee/
    t3_vid/
  images/
```

Expected discovered counts:

```text
Task1 labeled: 27
Task1 unlabeled: 1040
Task1 val: 30
Task2 train pairs: 105
Task2 val: 20
Task3 labeled samples: 180
Task3 unlabeled images: 1379
Task3 val images: 48
```

## Smoke Tests

These commands only prove that data loading, training, checkpointing, prediction,
and submission packaging work. They are not meaningful for leaderboard quality.

### Task 1

```powershell
conda run -n mvaa python .\task1\train.py `
  --data-root .\data\reference_data\t1_ct `
  --output-dir .\outputs\smoke\task1 `
  --epochs 1 --batch-size 1 --steps-per-epoch 1 `
  --roi-size 32 32 32 --train-crops 1 --sw-batch-size 1 `
  --val-interval 1 --max-labeled-cases 2 --max-unlabeled-cases 1 `
  --max-val-cases 1 --val-count 1 --num-workers 0 `
  --model-size small --unsup-warmup-epochs 1 --log-name smoke.log

conda run -n mvaa python .\task1\generate_task1_predictions.py `
  --ckpt-path .\outputs\smoke\task1\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t1_ct\val\images `
  --submission-task-dir .\outputs\smoke\submission\t1_ct `
  --num-workers 0
```

### Task 2

```powershell
conda run -n mvaa python .\task2\train.py `
  --data-dir .\data\reference_data\t2_tee\train `
  --output-dir .\outputs\smoke\task2 `
  --epochs 1 --batch-size 1 --roi-size 32 32 32 `
  --train-crops 1 --sw-batch-size 1 --val-interval 1 `
  --val-count 1 --max-train-cases 1 --max-val-cases 1 `
  --num-workers 0 --model-size small --log-name smoke.log

conda run -n mvaa python .\task2\generate_task2_predictions.py `
  --ckpt-path .\outputs\smoke\task2\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t2_tee\val\images `
  --submission-task-dir .\outputs\smoke\submission\t2_tee `
  --num-workers 0
```

### Task 3

```powershell
conda run -n mvaa python .\task3\train.py `
  --labeled-root .\data\reference_data\t3_vid\train `
  --unlabeled-root .\data\images `
  --output-dir .\outputs\smoke\task3 `
  --epochs 1 --batch-size 1 --unlabeled-batch-size 1 `
  --image-size 128 224 --max-train-samples 4 --max-val-samples 2 `
  --max-unlabeled-samples 2 --val-video-count 1 --num-workers 0 `
  --arch unet --encoder-name resnet18 --encoder-weights none `
  --warmup-epochs 0 --semi-warmup-epochs 1 --save-every 1 `
  --early-stop-patience 0 --no-val-tta --no-cache-masks

conda run -n mvaa python .\task3\generate_task3_predictions.py `
  --ckpt-path .\outputs\smoke\task3\checkpoints\best.pt `
  --data-dir .\data\reference_data\t3_vid\val\images `
  --submission-task-dir .\outputs\smoke\submission\t3_vid `
  --no-tta
```

Package a smoke submission:

```powershell
Compress-Archive `
  -Path .\outputs\smoke\submission\t1_ct, .\outputs\smoke\submission\t2_tee, .\outputs\smoke\submission\t3_vid `
  -DestinationPath .\outputs\smoke\submission.zip -Force

tar -tf .\outputs\smoke\submission.zip | Select-Object -First 30
```

The zip root should directly contain `t1_ct/`, `t2_tee/`, and `t3_vid/`.

## Full Baseline Entry Points

Use larger settings for real training. Keep outputs under `.\outputs` so they
remain ignored by git.

```powershell
conda run -n mvaa python .\task1\train.py --data-root .\data\reference_data\t1_ct --output-dir .\outputs\task1
conda run -n mvaa python .\task2\train.py --data-dir .\data\reference_data\t2_tee\train --output-dir .\outputs\task2
conda run -n mvaa python .\task3\train.py --labeled-root .\data\reference_data\t3_vid\train --unlabeled-root .\data\images --output-dir .\outputs\task3
```

Prediction commands should point at the best checkpoint produced by each task.

## Git Workflow

The first commit is the imported official baseline plus local ignore rules:

```powershell
git log --oneline
git diff HEAD
```

Use small commits for local changes, for example:

```powershell
git add requirements.txt task1/generate_task1_predictions.py task2/generate_task2_predictions.py task3/generate_task3_predictions.py notes/baseline_runbook.md
git commit -m "Add local baseline run support"
```
