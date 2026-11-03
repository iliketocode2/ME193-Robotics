**1 known-count trials** with the all-sensor model (the assignment asks for at least 10: record more with `collect.py --trial-steps N`).

| Trial | Actual steps | Predicted | Error | % error |
| --- | --- | --- | --- | --- |
| trial50_20261102-223545 | 50 | 27 | -23 | -46.0% |

| Statistic | Value |
| --- | --- |
| Mean absolute error | 23.0 steps |
| Mean percent error (signed) | -46.0% |
| Mean absolute percent error | 46.0% |
| Standard deviation of percent error | n/a% |
| 95% confidence interval on mean percent error | n/a% to n/a% |

**False positives vs missed steps** on held-out test sessions (step matched within 200 ms):

| Activity | True steps | Detected | False positives | Missed |
| --- | --- | --- | --- | --- |

- False positives while standing / shaking / turning: **0**
- False positives while walking (extra steps): **0**
- Missed steps: **0**
- Step-event F1: **0.000**

Ablation: run `python train.py --channels accel` and re-run this script.

![Predicted vs actual](plots/predicted_vs_actual.png)

![Error distribution](plots/error_distribution.png)

![Sample trace](plots/sample_trace.png)

![Training curves](plots/training_curves_all.png)
