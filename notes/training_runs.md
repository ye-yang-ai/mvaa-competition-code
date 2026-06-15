# MVAA 训练记录

本文档专门记录每一次训练的参数、输出目录、checkpoint 和结果。输出目录可以保持简洁，例如 `outputs/quick/task1`、`outputs/exp/task1_base_e100_bs1`；具体参数写在这里。

## 记录规则

- 每次开始训练前新增一行，先填写任务、时间、命令和关键参数。
- 训练完成后补充 best checkpoint、验证指标、提交 zip 或备注。
- `outputs/quick/submission/t1_ct`、`outputs/quick/submission/t2_tee`、`outputs/quick/submission/t3_vid` 是比赛提交结构，不要改名。

## 训练记录表

| Run ID | 日期 | Task | 输出目录 | GPU | Epochs | Batch | Train Crops | Model/Arch | ROI/Image Size | 半监督参数 | Checkpoint | 验证结果 | 备注 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |
| task1-exp-001 | 2026-06-14 | Task1 | `outputs/exp/task1_base_e100_bs1_mb` | `CUDA_VISIBLE_DEVICES=0` | 100 | 1 | 1 | `base` | `128 128 128` | `warmup=10, rampup=30` | `outputs/exp/task1_base_e100_bs1_mb/checkpoints/best_model.pt` | best_epoch=50, score=0.6616, DSC=0.7613, HD=34.0414, ASD=1.4742 | base, batch 1 |
| task1-exp-002 | 2026-06-14 | Task1 | `outputs/exp/task1_base_e100_bs2_mb` | `CUDA_VISIBLE_DEVICES=0` | 100 | 2 | 2 | `base` | `128 128 128` | `warmup=10, rampup=30` | `outputs/exp/task1_base_e100_bs2_mb/checkpoints/best_model.pt` | best_epoch=54, score=0.6879, DSC=0.8044, HD=27.9058, ASD=1.0250 | base, batch 2 |
| task1-exp-003 | 2026-06-14 | Task1 | `outputs/exp/task1_base_e200_bs2_mb` | `CUDA_VISIBLE_DEVICES=2` | 200 | 2 | 2 | `base` | `128 128 128` | `warmup=10, rampup=30` | `outputs/exp/task1_base_e200_bs2_mb/checkpoints/best_model.pt` | best_epoch=58, score=0.7382, DSC=0.7928, HD=12.3973, ASD=0.9805 | base, batch 2, 200 epochs |
| task1-exp-004 | 2026-06-14 | Task1 | `outputs/exp/task1_base_e100_b1_ml` | `CUDA_VISIBLE_DEVICES=3` | 100 | 1 | 2 | `large` | `128 128 128` | `warmup=10, rampup=30` | `outputs/exp/task1_base_e100_b1_ml/checkpoints/best_model.pt` | best_epoch=54, score=0.7388, DSC=0.8348, HD=17.1357, ASD=0.8173 | large, batch 1 |

## 已完成实验命令

### task1-exp-001

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

### task1-exp-002

```bash
CUDA_VISIBLE_DEVICES=0 python task1/train.py \
  --data-root data/reference_data/t1_ct \
  --output-dir outputs/exp/task1_base_e100_bs2_mb \
  --epochs 100 \
  --batch-size 2 \
  --steps-per-epoch 0 \
  --roi-size 128 128 128 \
  --train-crops 2 \
  --sw-batch-size 1 \
  --val-interval 2 \
  --num-workers 0 \
  --model-size base \
  --unsup-warmup-epochs 10 \
  --unsup-rampup-epochs 30
```

### task1-exp-003

```bash
CUDA_VISIBLE_DEVICES=2 python task1/train.py \
  --data-root data/reference_data/t1_ct \
  --output-dir outputs/exp/task1_base_e200_bs2_mb \
  --epochs 200 \
  --batch-size 2 \
  --steps-per-epoch 0 \
  --roi-size 128 128 128 \
  --train-crops 2 \
  --sw-batch-size 1 \
  --val-interval 2 \
  --num-workers 0 \
  --model-size base \
  --unsup-warmup-epochs 10 \
  --unsup-rampup-epochs 30
```

### task1-exp-004

```bash
CUDA_VISIBLE_DEVICES=3 python task1/train.py \
  --data-root data/reference_data/t1_ct \
  --output-dir outputs/exp/task1_base_e100_b1_ml \
  --epochs 100 \
  --batch-size 1 \
  --steps-per-epoch 0 \
  --roi-size 128 128 128 \
  --train-crops 2 \
  --sw-batch-size 1 \
  --val-interval 2 \
  --num-workers 0 \
  --model-size large \
  --unsup-warmup-epochs 10 \
  --unsup-rampup-epochs 30
```

## 结果补充模板

```text
Run ID:
训练完成时间:
Best checkpoint:
内部验证指标:
CodaBench 指标:
观察:
下一步:
```


```bash
CUDA_VISIBLE_DEVICES=0 python task2/train.py \
  --data-dir data/reference_data/t2_tee/train \
  --output-dir outputs/exp/task2_large_e100_b2 \
  --epochs 100 \
  --batch-size 2 \
  --roi-size 128 128 128 \
  --train-crops 1 \
  --sw-batch-size 1 \
  --val-interval 2 \
  --val-count 20 \
  --num-workers 0 \
  --model-size large
```



```bash
CUDA_VISIBLE_DEVICES=2 python task1/train.py \
  --data-root data/reference_data/t1_ct \
  --output-dir outputs/exp/task1_base_e100_bs2_ml \
  --epochs 100 \
  --batch-size 2 \
  --steps-per-epoch 0 \
  --roi-size 128 128 128 \
  --train-crops 1 \
  --sw-batch-size 1 \
  --val-interval 2 \
  --num-workers 0 \
  --model-size large \
  --unsup-warmup-epochs 10 \
  --unsup-rampup-epochs 30
```

```bash
  CUDA_VISIBLE_DEVICES=3 python task3/train.py \
    --labeled-root data/reference_data/t3_vid/train \
    --unlabeled-root data/images \
    --output-dir outputs/exp/task3_unetpp_effb4_none_e100_bs2 \
    --epochs 100 \
    --batch-size 2 \
    --unlabeled-batch-size 1 \
    --image-size 448 800 \
    --val-video-count 1 \
    --num-workers 0 \
    --arch unetplusplus \
    --encoder-name efficientnet-b4 \
    --encoder-weights none \
    --warmup-epochs 5 \
    --semi-warmup-epochs 15 \
    --unsup-ramp-epochs 30 \
    --save-every 5 \
    --early-stop-patience 30
```

```bash
  CUDA_VISIBLE_DEVICES=0 python task1/train.py \
    --data-root data/reference_data/t1_ct \
    --output-dir outputs/exp/task1_large_e100_b1_c2_u05_thr07 \
    --epochs 100 \
    --batch-size 1 \
    --roi-size 128 128 128 \
    --train-crops 2 \
    --sw-batch-size 1 \
    --val-interval 2 \
    --num-workers 0 \
    --model-size large \
    --unsup-warmup-epochs 20 \
    --unsup-rampup-epochs 30 \
    --unsup-weight 0.5 \
    --pseudo-threshold 0.7
```


  CUDA_VISIBLE_DEVICES=3 python task2/train.py \
    --data-dir data/reference_data/t2_tee/train \
    --output-dir outputs/exp/task2_large_e150_b1_roi160_c2_lr3e4_score525 \
    --epochs 150 \
    --batch-size 1 \
    --roi-size 160 160 160 \
    --train-crops 2 \
    --sw-batch-size 1 \
    --val-interval 2 \
    --val-count 20 \
    --num-workers 0 \
    --model-size large \
    --lr 3e-4 \
    --weight-decay 1e-5 \
    --score-dsc-weight 0.5 \
    --score-hd-weight 0.25 \
    --score-asd-weight 0.25