# Task3 MedSAM2 Stage C Semi-Supervised Training

## Goal

Stage C tests whether unlabeled Task3 video frames can improve the MedSAM2 encoder route.

Previous Stage A/B results showed:

```text
Stage A frozen v1: Dice 0.6655 | HD 104.76 | ASD 17.10
Stage A decoder v2: Dice 0.6646 | HD 90.83  | ASD 17.93
Stage B neck v1:   Dice 0.6570 | HD 84.63  | ASD 16.82
```

The website target to beat remains:

```text
Task3 DSC 0.7617500535 | HD 91.8473543097 | ASD 15.1721390878
```

## Implementation

Implemented optional EMA teacher semi-supervised training in:

```text
task3/train_medsam2_encoder.py
```

Added:

```text
--unlabeled-root
--unlabeled-batch-size
--max-unlabeled-samples
--semi-warmup-epochs
--unsup-weight
--unsup-ramp-epochs
--ema-decay
--pseudo-pos-thr
--pseudo-neg-thr
--pseudo-min-area
--pseudo-min-pos-ratio
--unsup-neg-weight
```

Training logic:

```text
student = MedSAM2 encoder + LightFPNDecoder
teacher = EMA copy of student

labeled batch:
  supervised loss on labeled masks

unlabeled batch:
  teacher(weak augmentation) -> pseudo labels
  student(strong augmentation) -> unsupervised BCE

total loss:
  supervised_loss + lambda_u * unsupervised_loss
```

Checkpoints now save:

```text
model_state
teacher_state
optimizer_state
scheduler_state
args
val_metrics
```

History now records:

```text
train_loss
sup_loss
unsup_loss
lambda_u
pseudo_pos_ratio
pseudo_conf_ratio
```

## Debug

Compilation passed:

```text
python -m py_compile task3/train_medsam2_encoder.py
```

Small GPU debug passed:

```text
outputs/medsam2_stageC/task3_frozen_v1_semi_debug
```

The semi branch was also tested with low pseudo thresholds to confirm nonzero unsupervised loss:

```text
outputs/medsam2_stageC/task3_frozen_v1_semi_debug_thr02
```

That run confirmed the unsupervised branch works, but also showed low thresholds can produce all-foreground pseudo masks.

## Stage C v1

Command:

```text
CUDA_VISIBLE_DEVICES=2 python task3/train_medsam2_encoder.py \
  --labeled-root data/t3_vid/train \
  --unlabeled-root data/images \
  --output-dir outputs/medsam2_stageC/task3_frozen_v1_semi \
  --epochs 80 \
  --batch-size 8 \
  --unlabeled-batch-size 8 \
  --image-size 512 512 \
  --decoder-version v1 \
  --encoder-train-mode frozen \
  --lr 1e-3 \
  --semi-warmup-epochs 6 \
  --unsup-weight 0.3 \
  --unsup-ramp-epochs 12 \
  --ema-decay 0.99 \
  --pseudo-pos-thr 0.60 \
  --pseudo-neg-thr 0.10 \
  --pseudo-min-area 100 \
  --pseudo-min-pos-ratio 0.0005
```

Stopped manually after epoch 17.

Best result was still the supervised warmup epoch:

```text
best epoch: 6
Dice: 0.6661
HD: 98.83
ASD: 16.97
threshold: 0.40
```

After semi-supervision started:

```text
pseudo_pos_ratio: about 0.03-0.04
pseudo_conf_ratio: about 0.92-0.97
val_pred_pos_ratio dropped below GT ratio
val Dice did not recover above warmup best
```

Interpretation:

```text
Hard pseudo-label BCE over-weights background pixels.
Teacher pseudo masks are too conservative.
The unsupervised loss pushes the student toward smaller masks.
```

## Stage C v1b

Added negative-pixel downweighting:

```text
--unsup-neg-weight 0.2
```

Command:

```text
CUDA_VISIBLE_DEVICES=2 python task3/train_medsam2_encoder.py \
  --labeled-root data/t3_vid/train \
  --unlabeled-root data/images \
  --output-dir outputs/medsam2_stageC/task3_frozen_v1_semi_v1b \
  --epochs 80 \
  --batch-size 8 \
  --unlabeled-batch-size 8 \
  --image-size 512 512 \
  --decoder-version v1 \
  --encoder-train-mode frozen \
  --lr 1e-3 \
  --semi-warmup-epochs 8 \
  --unsup-weight 0.1 \
  --unsup-ramp-epochs 20 \
  --ema-decay 0.99 \
  --pseudo-pos-thr 0.60 \
  --pseudo-neg-thr 0.10 \
  --pseudo-min-area 100 \
  --pseudo-min-pos-ratio 0.0005 \
  --unsup-neg-weight 0.2
```

Stopped manually after epoch 15.

Best result was still the supervised warmup epoch:

```text
best epoch: 6
Dice: 0.6658
HD: 101.78
ASD: 17.13
threshold: 0.40
```

Semi-supervised epochs still showed conservative pseudo masks:

```text
pseudo_pos_ratio: about 0.03-0.04
pseudo_conf_ratio: about 0.94-0.97
val_pred_pos_ratio remained below GT ratio
```

## Conclusion

The Stage C framework is implemented and technically works.

However, direct Mean Teacher hard pseudo-label BCE is not yet useful for this MedSAM2 setup. The teacher pseudo labels are too small, and confident background pixels dominate the unsupervised signal.

Do not continue tuning only `unsup_weight` and warmup. The failure mode is structural.

## Recommended Next Direction

Stage C v2 should change the pseudo-label objective:

```text
1. Positive-only pseudo supervision:
   use only t_probs >= pseudo_pos_thr pixels, ignore negative pseudo pixels.

2. Soft consistency:
   minimize student probability against teacher probability only on confident foreground-ish regions.

3. Area-ratio gate:
   skip unlabeled samples where teacher positive area is too small or too large.

4. Use warmup best to generate offline pseudo labels:
   inspect and filter pseudo masks before training on them.
```

The most conservative next experiment is:

```text
Stage C v2: positive-only pseudo labels
lambda_u <= 0.1
warmup from epoch 8
skip pseudo masks with area ratio < 0.02
skip pseudo masks with area ratio > 0.25
```
