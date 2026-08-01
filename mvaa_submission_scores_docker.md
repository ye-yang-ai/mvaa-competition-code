# MVAA Docker Submission Scores

记录 Docker final-submission 阶段的官方评测结果。

## v19

Task-level mean metrics:

| Task | mean DSC | mean HD | mean ASD |
|---|---:|---:|---:|
| Task 1 (CT) | 0.837991 | 6.272406 | 0.562456 |
| Task 2 (TEE) | 0.858118 | 9.297447 | 0.494249 |
| Task 3 (Video/Image) | 0.709688 | 508.268189 | 346.275617 |

## v35

Task-level mean metrics:

| Task | mean DSC | mean HD | mean ASD |
|---|---:|---:|---:|
| Task 1 (CT) | 0.838233 | 6.243079 | 0.557513 |
| Task 2 (TEE) | 0.858118 | 9.297447 | 0.494240 |
| Task 3 (Video/Image) | 0.760200 | 494.150328 | 381.538430 |

## v35 vs v19

Delta is computed as `v35 - v19`. For DSC, higher is better. For HD and ASD, lower is better.

| Task | DSC delta | HD delta | ASD delta |
|---|---:|---:|---:|
| Task 1 (CT) | +0.000242 | -0.029327 | -0.004943 |
| Task 2 (TEE) | +0.000000 | +0.000000 | -0.000009 |
| Task 3 (Video/Image) | +0.050512 | -14.117861 | +35.262813 |

Summary:

- v35 improves Task 1 slightly across DSC, HD, and ASD.
- v35 keeps Task 2 effectively unchanged.
- v35 improves Task 3 DSC and HD, but Task 3 ASD is worse than v19.
