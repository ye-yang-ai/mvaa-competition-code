# MVAA v39-safe Final Docker

This Docker package wraps the current validated best online combination:

- Task1: v25, nnU-Net v2 Dataset111 pseudo-top100, folds 0-4, `checkpoint_best.pth`.
- Task2: v19, nnU-Net v2 Dataset102, folds 0-4, `checkpoint_best.pth`.
- Task3: v39 six-branch ensemble, v35 five-model body plus LemonFM gated branch, weights `0.27/0.19/0.17/0.1275/0.1425/0.10`, threshold `0.36`.

Build from the repository root:

```bash
docker build -f docker/final/Dockerfile -t mvaa-v39-safe:final .
```

Local run contract:

```bash
docker run --rm --gpus all \
  --network none \
  -v /path/to/test_input:/input:ro \
  -v /path/to/output:/output:rw \
  mvaa-v39-safe:final
```

Generate the CodaBench descriptor zip after pushing the image:

```bash
python scripts/create_final_docker_submission.py \
  --image YOUR_REGISTRY/mvaa-v39-safe@sha256:YOUR_DIGEST \
  --output-zip outputs/final_docker_submission/submission.zip
```
