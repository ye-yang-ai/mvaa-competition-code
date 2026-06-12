# MVAA Local Usage

These commands assume PowerShell is already in the project root and the `mvaa`
conda environment is active:

```powershell
cd C:\Users\changba\Desktop\yy_files\competiton2
conda activate mvaa
```

## Task1 Quick Baseline

### 1. Check CUDA

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 2. Train Task1

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

### 3. Watch Logs

```powershell
Get-Content .\outputs\quick\task1\train.log -Wait
```

Or inspect the latest lines:

```powershell
Get-Content .\outputs\quick\task1\train.log -Tail 30
```

### 4. Check Checkpoints

```powershell
dir .\outputs\quick\task1\checkpoints
```

Expected files include:

```text
best_model.pt
last_model.pt
```

### 5. Generate Task1 Validation Predictions

```powershell
python .\task1\generate_task1_predictions.py `
  --ckpt-path .\outputs\quick\task1\checkpoints\best_model.pt `
  --data-dir .\data\reference_data\t1_ct\val\images `
  --submission-task-dir .\outputs\quick\submission\t1_ct `
  --num-workers 0
```

### 6. Check Task1 Submission Files

```powershell
dir .\outputs\quick\submission\t1_ct
```

```powershell
(dir .\outputs\quick\submission\t1_ct\*.nii.gz).Count
```

Expected count:

```text
30
```

```powershell
Test-Path .\outputs\quick\submission\t1_ct\task1_predictions.json
```

Expected result:

```text
True
```
