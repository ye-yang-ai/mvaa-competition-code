# MVAA Optimization Log

本文档只记录阶段性优化决策和结论，避免把完整训练日志复制进来。详细 stdout、checkpoint、mask、可视化和指标文件放在 `outputs/` 对应 run 目录。

## 记录格式

每完成一个阶段，追加一条：

```text
## YYYY-MM-DD - <stage name>

Goal:
- 本阶段要验证的单一问题。

Changes:
- 关键代码/参数/数据/后处理变化。

Run:
- output_dir:
- checkpoint:
- data split:
- command or script:

Metrics:
- Task:
- baseline:
- candidate:
- delta:
- validation scope:

Decision:
- accept / reject / retry
- 原因。

Next:
- 下一步最小动作。
```

## 2026-06-20 - Strategy Refinement

Goal:
- 在 Task3 MedSAM2 MVP 前，先固定策略边界、工程风险和阶段推进规则。

Changes:
- 将 MedSAM2 主线明确为 offline refine / quality filter / pseudo distillation。
- 将 PEFT / LoRA 保持为后置实验。
- 将 Task3 设为第一优先级，Task1 第二优先级，Task2 先做小规模 ablation。
- 增加 Task3 foreground-only 与 all-frame 双评估要求。
- 增加 refined + baseline fallback 作为真正比较对象。
- 增加每阶段 accept / reject / retry 的推进标准。
- 修复默认 repo root 和数据路径风险。

Decision:
- accept。
- 下一步先做 Task3 one-video MVP，不进入批量伪标签或 LoRA。

Next:
- 选择一个 Task3 baseline checkpoint。
- 生成一个 internal validation video 的 baseline coarse mask。
- 接入 MedSAM2 mask prompt propagation。
- 输出 baseline / refined / fallback / GT 指标和可视化。
