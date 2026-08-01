# MVAA v35 Final Docker Submission Runbook

记录日期：2026-07-28

本文记录 MVAA 2026 最终 Docker 提交的完整打包流程。最终提交不是预测 mask zip，也不是直接上传权重，而是上传一个包含 `submission.json` 的 `submission.zip`，其中写入 Docker Hub 镜像地址。Codabench 会拉取该镜像，在官方环境中读取 `/input`，并把推理结果写到 `/output`。

## 1. 最终采用的最优组合

最终 Docker 使用当前已验证的综合最优版本：`v35`。

- Task1：`v25`
  - 模型：nnU-Net v2 `3d_fullres`
  - 数据集名：`Dataset111_MVAA_Task1_PseudoTop100`
  - folds：`fold_0` 到 `fold_4`
  - checkpoint：`checkpoint_best.pth`
- Task2：`v19`
  - 模型：nnU-Net v2 `3d_fullres`
  - 数据集名：`Dataset102_MVAA_Task2`
  - folds：`fold_0` 到 `fold_4`
  - checkpoint：`checkpoint_best.pth`
- Task3：`v35`
  - 方法：5 模型 probability ensemble
  - threshold：`0.40`
  - TTA：开启
  - AMP：开启

Task3 ensemble 权重：

| 权重 | 模型 | checkpoint |
|---:|---|---|
| 0.30 | ResNet34 seed42 | `outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt` |
| 0.20 | ResNet34 seed49 | `outputs/opt/task3/t3_v36_unetpp_res34_img448x800_s49/checkpoints/best.pt` |
| 0.20 | EfficientNet-B4 seed42 | `outputs/opt/task3/t3_sup_unetpp_effb4_img512x896_allframe_s42/checkpoints/best.pt` |
| 0.15 | EfficientNet-B5 seed42 | `outputs/opt/task3/t3_v32_unetpp_effb5_sup_allframe_s42/checkpoints/best.pt` |
| 0.15 | EfficientNet-B5 seed44 | `outputs/opt/task3/t3_v35_unetpp_effb5_sup_allframe_s44/checkpoints/best.pt` |

## 2. 服务器端准备的 Docker 上下文

服务器不能运行 Docker，因此只在服务器上整理 Docker build context，然后用 XFTP 下载到本地 Windows 机器构建。

服务器端最终导出目录：

```text
/home/wuyongji/yangye_project/MVAA_v1/outputs/final_docker_context_mvaa_v35_20260727_205845
```

该目录大小约 `6.7G`，包含：

- `docker/final/Dockerfile`
- `docker/final/requirements.txt`
- `docker/final/run.sh`
- `scripts/final_infer_v35.py`
- `scripts/create_final_docker_submission.py`
- `task3/`
- Task1 nnU-Net 5 folds 权重
- Task2 nnU-Net 5 folds 权重
- Task3 5 个 ensemble checkpoint

构建前已完成的检查：

```bash
python -m py_compile scripts/final_infer_v35.py scripts/create_final_docker_submission.py task3/generate_task3_ensemble_predictions.py task3/generate_task3_multi_ensemble_predictions.py
```

```bash
python -c "from pathlib import Path; root=Path('outputs/final_docker_context_mvaa_v35_20260727_205845'); missing=[]; [missing.append(line.split()[1]) for line in (root/'docker/final/Dockerfile').read_text().splitlines() if line.startswith('COPY ') and not (root/line.split()[1]).exists()]; print('missing_copy_sources', missing)"
```

预期输出：

```text
missing_copy_sources []
```

## 3. 本地下载路径

使用 XFTP 将整个目录下载到 Windows 本地：

```text
D:\project\MVAA_2026\submissions\docker\final_docker_context_mvaa_v35_20260727_205845
```

注意：需要下载整个目录，不只下载 Dockerfile。因为 Docker build context 中包含所有权重和推理脚本。

## 4. 本地 Docker 构建

进入本地 Docker build context：

```powershell
cd D:\project\MVAA_2026\submissions\docker\final_docker_context_mvaa_v35_20260727_205845
```

构建镜像：

```powershell
docker build -f docker/final/Dockerfile -t mvaa-v35:final .
```

实际已成功构建的本地镜像名：

```text
mvaa-v35:final
```

构建成功日志中出现：

```text
naming to docker.io/library/mvaa-v35:final
```

## 5. 构建失败处理记录

第一次构建时失败在：

```text
RUN apt-get update && apt-get install ...
```

错误核心：

```text
502 Bad Gateway
archive.ubuntu.com
security.ubuntu.com
```

原因是 Docker 容器内部访问 Ubuntu apt 软件源失败，不是模型、权重或 Docker Desktop 的问题。

处理方式：

- 删除 `Dockerfile` 中非必要的 `apt-get update/install` 步骤。
- 当前推理代码不依赖 `cv2`，因此 `git libglib2.0-0 libgl1` 不是最终运行的硬依赖。
- 给 pip 增加可选 `PIP_INDEX_URL` 参数，默认使用 PyPI。

如果以后 pip 下载慢，可以用国内镜像源构建：

```powershell
docker build -f docker/final/Dockerfile -t mvaa-v35:final --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
```

## 6. Docker Hub 标记和推送

Docker Hub 用户名：

```text
yeyang2004
```

先登录：

```powershell
docker login
```

给本地镜像打 Docker Hub tag：

```powershell
docker tag mvaa-v35:final yeyang2004/mvaa-v35:final
```

推送到 Docker Hub：

```powershell
docker push yeyang2004/mvaa-v35:final
```

## 7. 获取镜像 digest

推送完成后获取 digest：

```powershell
docker inspect --format='{{index .RepoDigests 0}}' yeyang2004/mvaa-v35:final
```

本次成功使用的镜像地址：

```text
yeyang2004/mvaa-v35@sha256:27717b98bdef3df5dd64e23a8a6c0d9c0d39f6969f6594f306f0660b3e2c0635
```

## 8. 生成 Codabench submission.zip

本地使用的 Python 命令：

```powershell
& 'C:\Users\24001\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts/create_final_docker_submission.py --image yeyang2004/mvaa-v35@sha256:27717b98bdef3df5dd64e23a8a6c0d9c0d39f6969f6594f306f0660b3e2c0635 --output-zip submission.zip
```

生成文件位置：

```text
D:\project\MVAA_2026\submissions\docker\final_docker_context_mvaa_v35_20260727_205845\submission.zip
```

Codabench 最终上传文件：

```text
submission.zip
```

不要上传：

- Docker build context 目录
- 权重文件
- 预测 mask
- Docker image tar

## 9. 最终提交内容说明

最终上传的 `submission.zip` 内部只需要包含官方要求的 `submission.json`。其中 `submission.json` 指向 Docker Hub 上的固定 digest 镜像：

```text
yeyang2004/mvaa-v35@sha256:27717b98bdef3df5dd64e23a8a6c0d9c0d39f6969f6594f306f0660b3e2c0635
```

使用 digest 而不是普通 tag 的原因是：digest 是不可变镜像引用，可以确保 Codabench 拉取到的就是本次构建成功的镜像。

## 10. 官方运行逻辑

Docker 里不训练，只推理。

官方评测时会：

1. 拉取 `submission.json` 中指定的 Docker 镜像。
2. 启动容器。
3. 把测试数据挂载到 `/input`。
4. 容器入口脚本自动执行推理。
5. 推理脚本写出结果到 `/output`。
6. 官方读取 `/output` 进行评分。

镜像内的入口：

```text
/workspace/run.sh
```

主推理脚本：

```text
/workspace/scripts/final_infer_v35.py
```

预期输出结构：

```text
/output/t1_ct/task1_predictions.json
/output/t1_ct/*.nii.gz
/output/t2_tee/task2_predictions.json
/output/t2_tee/*.nii.gz
/output/t3_vid/task3_predictions.json
/output/t3_vid/*.png
```
