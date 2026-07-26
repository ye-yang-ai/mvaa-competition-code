# Task3 v24 Diagnosis and Retraining Direction

Date: 2026-07-23

## Context

Online v24:

| Metric | Value |
|---|---:|
| DSC | 0.7972857440 |
| HD | 118.9653051212 |
| ASD | 14.0923289374 |

v24 is a conservative version of v23:

- v23: v15/v22 ensemble weight `0.5/0.5`, threshold `0.4`
- v24: v15/v22 ensemble weight `0.55/0.45`, threshold `0.42`

Online v24 slightly reduced HD/ASD compared with v23, but the HD improvement was small. This suggests local threshold/weight tuning is close to saturation.

## Diagnostic Runs

New diagnostic script:

```text
task3/analyze_task3_ensemble_frame_errors.py
```

Outputs:

```text
outputs/analysis/task3_frame_errors/v23_ens_w50_thr04_s42_vc2
outputs/analysis/task3_frame_errors/v24_ens_w55_thr042_s42_vc2
```

Internal 2-video validation summary:

| Config | Dice all | Dice fg | HD | ASD | Status counts |
|---|---:|---:|---:|---:|---|
| v15 raw `thr=0.285` | 0.603120 | 0.825313 | 56.539 | 8.361 | 39 fg_hit, 17 empty_fp, 4 empty_tn |
| v23 ens `0.5/0.5 thr=0.4` | 0.885956 | 0.824548 | 66.464 | 9.323 | 39 fg_hit, 21 empty_tn |
| v24 ens `0.55/0.45 thr=0.42` | 0.869944 | 0.825555 | 65.651 | 9.212 | 39 fg_hit, 20 empty_tn, 1 empty_fp |

## Key Findings

1. v24 did not change the worst-HD frame pattern.

The same frames dominate worst HD in v23 and v24. The top error is:

```text
REC_20250205_102353_979A_000090
v23 HD: 264.9
v24 HD: 266.3
```

v24 reduces some small components but does not fix the main failure mode.

2. HD is not mainly caused by empty-frame false positives.

On internal validation:

- v23 has 0 empty false positives.
- v24 has only 1 empty false positive, with area 9 pixels.

The large HD frames are foreground frames, not empty frames.

3. Main error type is foreground-frame outlier behavior.

Worst v24 frame:

```text
case: REC_20250205_102353_979A_000090
dice: 0.581
HD: 266.3
ASD: 43.7
GT area: 69618
Pred area: 41116
area ratio: 0.591
pred components: 2
largest component: 38404
no-GT-overlap component area: 2712
```

This is not a tiny-noise-only problem. It combines:

- large under-segmentation
- a spatially wrong extra component
- strong frame-to-frame area and centroid jumps

4. v22-style high-recall training improved DSC but damaged distance stability.

v22/v23/v24 recover empty-frame behavior compared with v15 on the internal split, but introduce worse foreground-frame HD outliers. The current model family lacks direct supervision for temporal consistency, boundary distance, and hard transition frames.

## Retraining Direction

The next retrain should not simply repeat supervised Dice/Focal training. It should target the actual failure mode:

### Direction A: hard-frame focused supervised fine-tuning

Start from the best high-recall branch or ensemble-teacher setting, then oversample frames with:

- high local validation HD/ASD
- low Dice
- large predicted/GT area mismatch
- large frame-to-frame area or centroid changes
- multiple components or no-GT-overlap components

This directly attacks the frames that dominate HD.

### Direction B: boundary/distance-aware loss

Add a lightweight boundary term on top of Dice/Focal:

- boundary Dice or boundary BCE
- distance-transform weighted BCE near object edges
- optional Hausdorff-style surrogate if stable enough

Goal: keep v22/v23 DSC gain while reducing surface outliers.

### Direction C: temporal consistency regularization

Use adjacent labeled frames or unlabeled video frames to discourage unstable predictions:

- prediction consistency between neighboring frames after weak augmentation
- penalty on abrupt foreground area jumps
- penalty on abrupt centroid/shape jumps
- short-window pseudo-label smoothing from teacher predictions

This is especially relevant because worst frames show large frame-to-frame area/centroid changes.

### Direction D: distill v23/v24 into a single stable student

Use v23/v24 as high-DSC teachers and v15 as a stability teacher:

- positive target from ensemble probability
- suppress components that only appear in v22-like branch
- train one student to learn high recall plus stable boundaries

This may reduce inference complexity and improve generalization.

## Recommended Next Experiment

First substantial retrain candidate:

```text
v25_train: hard-frame focused fine-tuning
base: v22 EfficientNet-B4 supervised all-frame checkpoint
loss: Dice/Focal + boundary weighted BCE
sampler: foreground-balanced + hard-frame oversampling
validation: all-frame Dice for DSC, plus HD/ASD and empty-frame presence
postprocess: keep disabled during training selection; evaluate after checkpoint
```

Expected purpose:

- keep DSC near v23/v24
- reduce foreground-frame HD outliers
- avoid relying on threshold micro-tuning

If this improves local worst-HD frames without dropping Dice, then generate a new online candidate by ensembling v25 with v15, similar to v23/v24.

## 2026-07-23 Short-Run Result

Implemented:

- `task3/build_task3_hard_frame_csv.py`
- `task3/analyze_task3_ensemble_frame_errors.py`
- `task3/train.py` support for:
  - `--hard-frame-csv`
  - `--hard-frame-weight`
  - `--boundary-loss-weight`
  - `--init-ckpt`

Hard-frame CSV:

```text
outputs/analysis/task3_frame_errors/v24_ens_w55_thr042_s42_vc2/hard_frames_top40.csv
```

It contains 26 hard frames selected from v24 frame diagnostics. Main reasons include high HD, low Dice, no-GT-overlap components, area-ratio outliers, temporal centroid jumps, and many components.

Short training run:

```text
outputs/opt/task3/t3_v25_short_hard_boundary_effb4_s43
```

Configuration:

- base checkpoint: `outputs/opt/task3/t3_sup_unetpp_effb4_img512x896_allframe_s42/checkpoints/best.pt`
- split seed: `43`
- train videos: `REC_20250109_110158_583A`, `REC_20250205_102353_979A`, `REC_20250211_102216_914A`, `REC_20250322_101917_746A`
- val videos: `REC_20250414_104310_305A`, `REC_20250418_104439_675A`
- hard-frame hits: 26 in train, 0 in val
- hard-frame weight: `4.0`
- boundary loss weight: `0.15`
- epochs: `10`
- LR: `5e-5`
- validation TTA: disabled for speed

Same-split grid comparison without TTA:

| Model | Best threshold | Dice all | Dice fg | HD | ASD | Empty FP |
|---|---:|---:|---:|---:|---:|---:|
| v22 original | 0.4 | 0.957357 | 0.926897 | 40.826625 | 4.168312 | 0 |
| v25 short hard/boundary | 0.5 | 0.953540 | 0.920354 | 43.530649 | 4.465011 | 0 |

Conclusion:

This exact v25 short-run recipe is not effective. It makes the model slightly worse on a clean held-out split:

- Dice all: `-0.003817`
- Dice fg: `-0.006543`
- HD: `+2.704024`
- ASD: `+0.296699`

Interpretation:

- Directly fine-tuning v22 with hard-frame oversampling plus a simple boundary BCE is too aggressive or not aligned with the online failure mode.
- The original v22 checkpoint is already strong on the seed 43 validation videos, so naive fine-tuning can erode a good optimum.
- Do not generate an online submission from this checkpoint.

Next direction should change the formulation rather than extend this run:

1. Use hard-frame diagnostics for validation/selection and postprocess design, not heavy oversampling only.
2. Try softer distillation from v15/v23 probabilities instead of direct supervised fine-tuning.
3. Add a temporal/component consistency postprocess before another training-heavy attempt.
4. If retraining again, use lower hard-frame weight such as `1.5-2.0`, lower boundary weight such as `0.03-0.05`, and compare against v22 on multiple split seeds before considering online submission.

## 2026-07-23 Temporal Postprocess Check

Implemented local validation script:

```text
task3/evaluate_task3_temporal_postprocess.py
```

Validation target:

- v24-style ensemble
- v15 weight `0.55`
- v22 weight `0.45`
- seed `42`
- val videos: `REC_20250205_102353_979A`, `REC_20250322_101917_746A`

Two mask-level temporal postprocess families were tested:

1. `largest` / `temporal`
   - keep largest component
   - optionally keep extra components only when supported by previous/next frame main component

2. `remove_unsupported_small`
   - keep largest component
   - keep large extra components
   - remove only small unsupported extra components

Main result:

| Candidate | Threshold | Dice all | Dice fg | HD | ASD | Empty FP |
|---|---:|---:|---:|---:|---:|---:|
| raw best score | 0.46 | 0.885939 | 0.824521 | 65.823844 | 9.244223 | 0.000 |
| raw best HD in tested grid | 0.40 | 0.871425 | 0.827833 | 65.560129 | 9.109854 | 0.0476 |
| best conservative temporal HD | 0.40 | 0.871411 | 0.827812 | 65.319527 | 9.293653 | 0.0476 |
| best temporal by score | 0.46 | 0.885852 | 0.824388 | 67.331667 | 9.478302 | 0.000 |

Conclusion:

Mask-level temporal connected-component postprocess is not strong enough in its current form.

- `largest` is harmful: it removes true secondary structures and worsens HD to about `85-88`.
- Simple temporal support filtering is also harmful or neutral.
- The conservative variant gives only a tiny HD improvement (`65.56 -> 65.32`) under a lower-threshold setting, while ASD worsens and score remains below raw.

Do not generate an online submission from this mask-level temporal postprocess.

Recommended next temporal direction:

1. Probability-level temporal smoothing with confidence gating, not binary component deletion.
2. Training-level temporal consistency using adjacent frame pairs/triplets.
3. If postprocess is revisited, it should be target-specific and uncertainty-aware rather than largest-component based.

## 2026-07-23 Temporal Consistency Short Training

Implemented:

- `TemporalPairDataset` in `task3/dataset.py`
- temporal consistency options in `task3/train.py`:
  - `--temporal-consistency-weight`
  - `--temporal-batch-size`
  - `--temporal-max-frame-gap`
  - `--temporal-conf-thr`
  - `--temporal-area-jump-thr`

Training idea:

- Build adjacent labeled frame pairs from the same training videos.
- Use the EMA teacher to predict both frames.
- Let the student match the teacher's probability difference between adjacent frames.
- Use confidence gating and area-jump gating so normal motion is not forced to be identical.

Short training run:

```text
outputs/opt/task3/t3_v26_short_temporal_consistency_effb4_s43
```

Configuration:

- base checkpoint: `outputs/opt/task3/t3_sup_unetpp_effb4_img512x896_allframe_s42/checkpoints/best.pt`
- seed: `43`
- temporal pairs: 116
- temporal consistency weight: `0.03`
- temporal max frame gap: `1000`
- temporal confidence threshold: `0.7`
- temporal area jump threshold: `0.6`
- no hard-frame oversampling
- no boundary loss
- epochs: `10`
- LR: `5e-5`
- validation TTA: disabled for speed

Same-split grid comparison without TTA:

| Model | Best threshold | Dice all | Dice fg | HD | ASD | Empty FP |
|---|---:|---:|---:|---:|---:|---:|
| v22 original | 0.4 | 0.957357 | 0.926897 | 40.826625 | 4.168312 | 0 |
| v25 hard/boundary | 0.5 | 0.953540 | 0.920354 | 43.530649 | 4.465011 | 0 |
| v26 temporal consistency | 0.35 | 0.955593 | 0.923874 | 43.314234 | 4.434184 | 0 |

Conclusion:

Temporal consistency training is technically feasible and stable, but the first short-run recipe does not beat the original v22 checkpoint on the seed 43 held-out videos.

Compared with v25 hard/boundary, v26 is slightly better:

- Dice all: `+0.002053`
- Dice fg: `+0.003520`
- HD: `-0.216416`
- ASD: `-0.030828`

Compared with v22 original, v26 is still worse:

- Dice all: `-0.001763`
- Dice fg: `-0.003023`
- HD: `+2.487609`
- ASD: `+0.265872`

Do not generate an online submission from this v26 checkpoint yet.

Next temporal retraining options:

1. Reduce temporal consistency weight to `0.01` or `0.015`.
2. Use temporal consistency only after a warmup/freeze period, so the v22 optimum is not disturbed early.
3. Use v23/v24 ensemble as a fixed teacher instead of EMA teacher.
4. Combine temporal consistency with probability-level distillation, not hard-frame oversampling.

## 2026-07-23 v27 Lower-Weight Delayed Temporal Training

Implemented `--temporal-start-epoch` in `task3/train.py`.

Short training run:

```text
outputs/opt/task3/t3_v27_short_temporal_w001_start4_effb4_s43
```

Configuration:

- base checkpoint: `outputs/opt/task3/t3_sup_unetpp_effb4_img512x896_allframe_s42/checkpoints/best.pt`
- seed: `43`
- temporal pairs: 116
- temporal consistency weight: `0.01`
- temporal start epoch: `4`
- no hard-frame oversampling
- no boundary loss
- epochs: `10`
- LR: `5e-5`
- validation TTA: disabled for speed

Same-split grid comparison without TTA:

| Model | Best threshold | Dice all | Dice fg | HD | ASD | Empty FP |
|---|---:|---:|---:|---:|---:|---:|
| v22 original | 0.4 | 0.957357 | 0.926897 | 40.826625 | 4.168312 | 0 |
| v26 temporal weight 0.03 | 0.35 | 0.955593 | 0.923874 | 43.314234 | 4.434184 | 0 |
| v27 temporal weight 0.01 start epoch 4 | 0.7 | 0.956058 | 0.924670 | 43.151841 | 4.394639 | 0 |
| v27 best HD setting | 0.5 | 0.955792 | 0.924215 | 42.984806 | 4.427954 | 0 |

Conclusion:

The lower-weight delayed strategy is better than v26, but still does not beat the original v22 checkpoint.

Compared with v26:

- Dice all: `+0.000464`
- Dice fg: `+0.000796`
- HD: `-0.162393`
- ASD: `-0.039545`

Compared with v22 original:

- Dice all: `-0.001299`
- Dice fg: `-0.002227`
- HD: `+2.325216`
- ASD: `+0.226327`

This suggests the temporal consistency implementation is usable, and reducing/delaying the temporal loss is the right direction, but EMA-teacher temporal consistency alone is not strong enough for a real breakthrough.

Recommended next step:

- Stop submitting EMA-temporal checkpoints directly.
- Try fixed-teacher distillation from v23/v24 probability ensemble, optionally with a very small temporal term.
- The target should be to preserve v23/v24 DSC behavior while learning a single smoother student, rather than fine-tuning v22 with EMA temporal consistency only.
