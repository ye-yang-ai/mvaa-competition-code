# MVAA_v1 项目管理与实验归档建议

生成时间：2026-07-23

本文档基于当前工程 `/home/wuyongji/yangye_project/MVAA_v1` 的实际目录、Git 状态、输出规模和已有记录方式整理。目标不是一次性重构整个项目，而是建立一套以后能持续执行的文件安放、实验记录、结果管理和清理规范。

## 1. 当前工程的总体判断

这个工程的问题不是代码结构混乱，而是已经进入竞赛项目后期常见状态：代码体积很小，实验产物、提交包、伪标签、nnU-Net 中间目录和分析文件成为主体。

当前观察到的体量大致如下：

| 路径 | 规模 | 判断 |
|---|---:|---|
| 项目总目录 | 约 78G | 主要由数据和输出构成 |
| `outputs/` | 约 60G | 当前最需要管理的目录 |
| `data/` | 约 19G | 原始/参考数据，不应进 Git |
| `outputs/exp/` | 约 27G | 正式训练实验集中地 |
| `outputs/opt/` | 约 12G | 优化实验目录，但语义和 `exp/` 有重叠 |
| `outputs/final_docker_context_mvaa_v19_20260719_112045/` | 约 5.1G | 应归为最终包或交付构建目录 |
| `outputs/submissions/` | 约 2.2G | 线上提交记录，价值很高，应严格保留记录 |
| 代码目录 `task1/ task2/ task3/ scripts/` | MB 级 | 不是主要混乱来源 |

当前 `.gitignore` 已经忽略了 `data/`、`checkpoints/`、`outputs/`、`runs/`、模型权重和日志文件，这是正确的。也就是说，大文件没有进入 Git，这一点做得很好。

真正的问题是：

- 输出目录很多，但生命周期不清楚。
- 实验记录有好习惯，但格式不统一。
- 提交结果记录较完整，但训练 run 记录不够一致。
- README 仍像原始 baseline 文档，不能代表当前工程状态。
- 有些脚本和记录存在绝对路径，换机器或换目录后容易失效。
- Git 工作区存在未提交的代码和新增脚本，如果不及时提交，以后难以追溯某次结果对应的代码版本。

## 2. 需要优先改正的习惯

### 2.1 不要让目录名承担全部信息

当前很多目录名已经很长，例如：

```text
submit_v19_task2_nnunet5fold_best_task1_v17_task3_v15
task2_unet_large_e200_b1_roi192x192x160_c2_lr2e4_score525_v13
```

这种命名有帮助，但不能只靠目录名记录实验。目录名适合快速识别，完整信息应该放在目录内部的元数据文件中。

以后每个正式训练 run 建议至少有：

```text
run.json
command.sh
metrics.json
notes.md
git_commit.txt
env.txt
```

含义如下：

| 文件 | 内容 |
|---|---|
| `run.json` | 机器可读配置，包括 task、模型、输入路径、输出路径、seed、重要超参数 |
| `command.sh` | 原始训练或推理命令，保证能复现 |
| `metrics.json` | best epoch、本地验证指标、阈值、后处理参数 |
| `notes.md` | 人写的目的、实验假设、结论、是否推荐继续 |
| `git_commit.txt` | 实验开始时的 Git commit hash 和 dirty 状态 |
| `env.txt` | Python、PyTorch、CUDA、nnU-Net、MONAI、SMP 等关键版本 |

其中 `notes.md` 可以很短，但必须回答三个问题：

```text
Purpose:
Conclusion:
Keep or discard:
```

### 2.2 README 应该成为当前项目入口

当前 `README.md` 仍然是 `MVAA Baseline` 风格，里面还有 `cd baseline`、默认路径和旧式说明。这个会误导未来的自己。

建议把 README 改成当前项目入口，只回答这些问题：

1. 当前项目解决什么任务。
2. 当前最优提交是哪一个。
3. 当前最优三任务分别来自哪些模型或版本。
4. 如何准备环境。
5. 如何复现当前主提交。
6. 目录结构应该怎么看。
7. 结果记录在哪里查。

README 不应该塞入所有实验细节。详细实验路线可以留在 `notes/`，线上结果表可以留在独立结果文档。

### 2.3 结果总表只能有一个权威来源

当前已经有两个很有价值的文档：

- `mvaa_submission_scores.md`
- `best_task_results_summary.md`

这说明你已经意识到结果记录很重要。但是未来要避免多个文件分别更新，最后互相不一致。

建议保留一个主结果表作为唯一权威来源，例如：

```text
notes/results/submissions.md
```

或者如果想更方便程序读取：

```text
notes/results/submissions.csv
```

每次线上提交后，只允许这个主表记录最终结果。其他文档可以引用它，但不要复制一份完整表格。

推荐字段：

```text
version
date
task1_source
task2_source
task3_source
task1_dsc
task1_hd
task1_asd
task2_dsc
task2_hd
task2_asd
task3_dsc
task3_hd
task3_asd
num_cases
missing_cases
zip_path
record_path
status
conclusion
```

这样以后查一个问题会非常直接：

- v22 为什么不作为主提交？
- v19 的 Task2 来源是什么？
- 哪个版本 Task3 DSC 最高？
- 哪个版本 Task3 HD/ASD 最均衡？
- 某个 zip 是否已经线上测过？

### 2.4 提交包记录方式值得保留，但应模板化

当前 `outputs/submissions/*/checkpoint_record.md` 做得比训练记录更完整。很多版本已经记录了 composition、命令、检查项和线上结果，这是非常好的习惯。

建议以后所有提交目录固定为：

```text
outputs/submissions/submit_vXX_short_description/
  checkpoint_record.md
  submission/
    t1_ct/
    t2_tee/
    t3_vid/
  submission.zip
  checks.json
```

`checkpoint_record.md` 固定包含：

```text
# submit_vXX_xxx

Generated:
Purpose:

## Composition

| Task | Source | Notes |

## Commands

## Checks

## Online Result

## Conclusion
```

其中 `Conclusion` 很重要。不要只记录数值，还要写一句决策：

```text
Not recommended as main submission: improves Task3 DSC but worsens HD/ASD heavily.
```

这种一句话会在几周后救你很多时间。

## 3. 建议的目录结构

当前目录可以逐步演化为下面这种结构，不建议现在立刻大规模搬动已有 60G 输出。

```text
MVAA_v1/
  README.md
  requirements.txt
  task1/
  task2/
  task3/
  scripts/
  assets/
  notes/
    project_management_recommendations.md
    results/
      submissions.md
      best_current.md
    runbooks/
      reproduce_v19.md
      package_submission.md
    opt/
      task1_route.md
      task2_route.md
      task3_route.md
  data/
    reference_data/
  checkpoints/
    pretrained/
  external/
  outputs/
    runs/
    smoke/
    analysis/
    pseudo/
    submissions/
    packages/
    tmp/
```

### 3.1 `outputs/runs/`

正式训练 run 放这里。建议按 task 再分一层：

```text
outputs/runs/task1/
outputs/runs/task2/
outputs/runs/task3/
```

正式 run 命名建议：

```text
YYYYMMDD_task_model_keyparams_seed
```

示例：

```text
20260723_task3_unetpp_effb4_512x896_sup_s42
```

如果仍想保留 v 编号，也可以：

```text
v22_task3_unetpp_effb4_512x896_sup_s42
```

关键是保持一致，不要一会儿用 `exp`，一会儿用 `opt`，一会儿用 `quick`。

### 3.2 `outputs/smoke/`

冒烟测试、短跑、导入测试、单 batch 测试都放这里。这个目录默认可删除。

要求：

- 不作为最终结果来源。
- 不进入结果总表。
- 如果 smoke run 后来变成正式 run，应重新跑或复制到 `outputs/runs/` 并补齐元数据。

### 3.3 `outputs/analysis/`

分析产物放这里，例如：

- mask distribution
- postprocess grid
- ensemble grid
- frame error analysis
- pseudo-label quality score

建议每个分析目录也有一个 `analysis_record.md`：

```text
Purpose:
Inputs:
Command:
Key result:
Next action:
```

### 3.4 `outputs/pseudo/`

伪标签数据放这里。伪标签尤其容易失控，因为它既像数据，又像模型输出。

每个伪标签目录必须有：

```text
manifest.json
case_list.txt
pseudo_inference_record.md
quality_summary.md
```

必须记录：

- 来源模型。
- 使用哪些 unlabeled case。
- 阈值或 ensemble 策略。
- 是否保存概率。
- 质量筛选规则。
- 最终用于哪个训练 run。

### 3.5 `outputs/submissions/`

线上提交包放这里。这个目录价值最高，不要随便删除。

建议保留：

- 所有已经线上测过的 `checkpoint_record.md`。
- 所有主提交或关键对照提交的 `submission.zip`。
- 可以删除重复的中间预测目录，但删除前必须确认 zip 和 record 都完整。

### 3.6 `outputs/packages/`

Docker context、最终交付包、论文复现实验包放这里。

例如当前：

```text
outputs/final_docker_context_mvaa_v19_20260719_112045/
```

语义上更适合放入：

```text
outputs/packages/final_docker_context_mvaa_v19_20260719_112045/
```

这个不急着移动，但以后新建时建议按这个规则。

### 3.7 `outputs/tmp/`

临时目录统一放这里，不要散落成：

```text
outputs/tmp_task1_postprocess_sanity
outputs/quick
outputs/debug_xxx
```

建议：

```text
outputs/tmp/task1_postprocess_sanity_20260723/
```

`tmp/` 默认可删除，里面不保存唯一结果。

## 4. 文件生命周期分级

每个文件或目录都应该属于一个生命周期等级。

### 4.1 永久保留

这些文件不要删除：

- 最终采用过的 checkpoint。
- 线上提交过并有结果的 `submission.zip`。
- 每次提交的 `checkpoint_record.md`。
- 结果总表。
- 最优方案总结。
- 生成最终结果所需的转换脚本。

### 4.2 阶段性保留

这些文件在论文、报告或复现实验完成前保留：

- 分析 CSV。
- 可视化图。
- grid search JSON。
- frame error report。
- pseudo-label quality report。

### 4.3 可重建

这些可以在空间紧张时清理，但清理前要确认有命令和配置可重建：

- nnU-Net preprocessed data。
- 中间预测概率图。
- 临时 inference output。
- 非主提交的展开 `submission/` 目录。
- smoke run checkpoint。

### 4.4 可随时删除

这些默认不应该长期保留：

- `__pycache__/`
- `.tmp`
- 失败的 smoke run。
- 空日志。
- 重复 zip。
- 没有记录、没有结论、没有被引用的临时目录。

## 5. Git 管理建议

当前 Git 管理的方向是对的：代码和文档进 Git，大数据和输出不进 Git。

但需要更频繁地提交小而清晰的变更。尤其当一次线上结果依赖某些脚本改动时，必须让结果和代码版本能对应起来。

推荐提交粒度：

```text
feat(task3): add all-frame validation option
feat(task1): add pseudo-label quality scoring
docs(results): record v22 online metrics
docs(runbook): document v19 reproduction workflow
fix(task3): correct postprocess threshold handling
```

不要把很多天的实验脚本、结果文档、训练代码混成一个大提交。竞赛后期最容易出的问题是：

```text
我知道 v22 是这个结果，但我不知道 v22 当时到底用了哪版代码。
```

每个正式实验开始时建议保存：

```bash
git rev-parse HEAD > outputs/runs/.../git_commit.txt
git status --short >> outputs/runs/.../git_commit.txt
```

如果工作区是 dirty，也要记录。dirty 不是绝对不能跑实验，但必须知道 dirty 在哪里。

## 6. 路径管理建议

脚本中尽量避免写死个人绝对路径，例如：

```text
/home/wuyongji/yangye_project/MVAA_v1
```

实验记录里保存绝对路径可以接受，因为它记录的是当时环境。但可复用脚本里建议使用相对路径或自动推导项目根目录。

Bash 脚本建议：

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
```

Python 脚本建议：

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
```

然后所有默认路径基于 `REPO_ROOT` 构造。

对于训练和推理入口，建议优先通过 CLI 参数传路径，而不是要求用户改源码里的常量。

## 7. 命名规范建议

### 7.1 正式训练 run

推荐格式：

```text
YYYYMMDD_task_model_keyparams_seed
```

示例：

```text
20260723_task3_unetpp_effb4_512x896_sup_s42
20260723_task2_nnunet_3dfullres_5fold_best
```

如果要保留版本号：

```text
v22_task3_unetpp_effb4_512x896_sup_s42
```

### 7.2 提交版本

推荐格式：

```text
submit_vXX_short_change
```

示例：

```text
submit_v24_task3_v15_v22_ens_w60_thr045
```

提交目录名应该表达“这次相对上一版改变了什么”，不要试图塞入所有细节。完整细节放 `checkpoint_record.md`。

### 7.3 分析目录

推荐格式：

```text
analysis_name/source_version/date_or_short_tag
```

示例：

```text
outputs/analysis/task3_ensemble/v15_v22_s42_vc2_allframe
outputs/analysis/task1_pseudo/v17_high759
```

## 8. 建议补充的模板

### 8.1 训练 run 模板

```markdown
# Run Record

Run:
Date:
Task:
Owner:

## Purpose

## Hypothesis

## Command

```bash
```

## Data

## Model

## Key Params

## Metrics

| Metric | Value |
|---|---:|

## Artifacts

## Conclusion

## Next Action
```

### 8.2 提交记录模板

```markdown
# submit_vXX_xxx

Generated:

## Purpose

## Composition

| Task | Source | Notes |
|---|---|---|

## Commands

## Checks

| Check | Value |
|---|---:|

## Online Result

| Task | DSC | HD | ASD | num_cases | missing_cases |
|---|---:|---:|---:|---:|---:|

## Conclusion
```

### 8.3 分析记录模板

```markdown
# Analysis Record

Date:
Analysis:

## Purpose

## Inputs

## Command

## Key Findings

## Decision

## Follow-up
```

## 9. 后续整理顺序

不建议现在立刻搬动所有大目录。更现实的顺序如下。

### 第一步：修文档入口

优先改：

- `README.md`
- 一个唯一结果总表
- 一个当前最优方案说明

目标是让未来的自己打开项目后，10 分钟内知道当前主线。

### 第二步：建立模板

新增：

```text
notes/templates/run_record_template.md
notes/templates/submission_record_template.md
notes/templates/analysis_record_template.md
```

后续所有新实验按模板写。旧实验不用一次性补完，只补最重要的 v19、v22、v23 等。

### 第三步：统一新输出目录

从下一次实验开始使用新结构：

```text
outputs/runs/
outputs/smoke/
outputs/packages/
outputs/tmp/
```

旧的 `outputs/exp/`、`outputs/opt/` 暂时保留，避免移动后脚本和记录失效。

### 第四步：清理可删除产物

先只清理确定无价值的内容：

- `__pycache__/`
- 空目录。
- 明确失败的 smoke run。
- 没有被任何记录引用的临时输出。

不要先删除 submission、checkpoint 和 analysis。先建立索引，再清理。

### 第五步：把关键实验补齐记录

优先补：

- 当前主提交。
- 当前 Task1 最优来源。
- 当前 Task2 最优来源。
- 当前 Task3 最优来源。
- 最近几次失败但有启发的提交，例如 v20、v21、v22、v23。

失败实验也要写结论，因为它们防止你以后重复踩坑。

## 10. 每次新实验前后的检查清单

### 实验前

- 确认当前 Git 状态。
- 写清楚实验目的。
- 确认输出目录不会覆盖旧结果。
- 确认数据路径和 checkpoint 路径。
- 保存命令。
- 保存 Git commit 和 dirty 状态。

### 实验中

- 日志写到 run 目录。
- 指标写成机器可读 JSON 或 CSV。
- 不要只依赖终端输出。

### 实验后

- 写 `Conclusion`。
- 判断 `keep / discard / needs online test`。
- 如果生成提交包，补 `checkpoint_record.md`。
- 如果得到线上结果，立刻更新唯一结果总表。
- 如果代码有变化，提交 Git。

## 11. 最重要的原则

可以把项目管理压缩成四句话：

1. 代码进 Git，大数据和输出不进 Git。
2. 目录名只做索引，完整信息写入记录文件。
3. 每个正式结果必须能追溯到命令、代码版本、checkpoint 和数据来源。
4. 不要等项目结束再整理；每次实验结束花 3 分钟写结论。

如果只能改一个习惯，就改第 4 条。很多实验项目最后变乱，不是因为目录一开始设计不好，而是因为每次实验结束时没有留下足够明确的判断。
