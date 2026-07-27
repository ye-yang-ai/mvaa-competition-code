# MVAA v35 Final Docker

This Docker package wraps the current validated best online combination:

- Task1: v25, nnU-Net v2 Dataset111 pseudo-top100, folds 0-4, `checkpoint_best.pth`.
- Task2: v19, nnU-Net v2 Dataset102, folds 0-4, `checkpoint_best.pth`.
- Task3: v35 five-model probability ensemble, threshold `0.40`.

Build from the repository root:

```bash
docker build -f docker/final/Dockerfile -t mvaa-v35:final .
```

Local run contract:

```bash
docker run --rm --gpus all \
  --network none \
  -v /path/to/test_input:/input:ro \
  -v /path/to/output:/output:rw \
  mvaa-v35:final
```

Generate the CodaBench descriptor zip after pushing the image:

```bash
python scripts/create_final_docker_submission.py \
  --image YOUR_REGISTRY/mvaa-v35@sha256:YOUR_DIGEST \
  --output-zip outputs/final_docker_submission/submission.zip
```
