# MVAA v39-safe Final Docker Runbook

记录日期：2026-07-31

## 最终配置

最终 Docker-safe 配置采用：

- Task1：v25，Dataset111 pseudo-top100 nnU-Net 5-fold `checkpoint_best.pth`
- Task2：v19，Dataset102 nnU-Net 5-fold `checkpoint_best.pth`
- Task3：v39，v35 五模型主体 + LemonFM gated branch

对应预测 zip：

```text
outputs/submissions/final_submission_safe_task1_5fold_v25_task2_v19_task3_v39/submission.zip
```

## 为什么停止继续优化

Task1：

- top50 fold0/fold1 本地验证低于 top100，同方向继续训练性价比低。
- top200 线上没有超过 top100 v25。
- ResEnc-L top100 已完成 fold0/fold1/fold2，均低于普通 top100。
- 15-fold ensemble 线上 Task1 更高，但最终 Docker 构建和复现风险更大，已按最终策略回退到 5-fold v25。

Task3：

- v39 是当前综合最稳配置：相对 v35，DSC/HD/ASD 同时小幅改善。
- v38 虽然 DSC 更高，但 HD/ASD 明显更差，不适合作最终稳健 Docker。
- 继续做阈值/后处理/新模型搜索，预期收益很小，且更容易引入 Docker 复现风险。

## 已完成的 Docker 前置改动

- 新增 `scripts/final_infer_v39_safe.py`
- `docker/final/Dockerfile` 改为打包 v39 Task3 六分支权重
- `docker/final/run.sh` 改为调用 v39-safe 入口
- `docker/final/README.md` 更新为 v39-safe
- `.dockerignore` 放行 v39 需要的两个 LemonFM checkpoint
- Task3 推理在 `MVAA_NO_PRETRAINED_INIT=1` 下跳过 LemonFM 预训练文件，只用已训练 checkpoint 严格加载参数

验证：

```text
missing_copy_sources []
py_compile passed
LemonFM segmentation checkpoint loaded without pretrained init
presence gate checkpoint loaded without pretrained init, threshold=0.54
```

## Build context

服务器端已生成最终 Docker build context：

```text
outputs/final_docker_context_mvaa_v39_safe_20260731_102131
```

大小约：

```text
14G
```

该目录应整体下载到本地 Docker 机器构建。

## 构建命令

```bash
docker build -f docker/final/Dockerfile -t mvaa-v39-safe:final .
```

如果 pip 慢：

```bash
docker build -f docker/final/Dockerfile -t mvaa-v39-safe:final --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

推送后使用 digest 生成 CodaBench Docker descriptor：

```bash
python scripts/create_final_docker_submission.py \
  --image YOUR_DOCKERHUB_NAME/mvaa-v39-safe@sha256:YOUR_DIGEST \
  --output-zip outputs/final_docker_submission/submission.zip
```

## 当前服务器限制

本服务器当前用户无法访问 Docker daemon：

```text
permission denied while trying to connect to /var/run/docker.sock
```

因此本次只在服务器上完成 build context 准备和静态/加载验证，未能直接执行 `docker build`。
