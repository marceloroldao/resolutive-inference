# Methodology

The research program evaluates compact sequential inference under explicit resource budgets. Each hypothesis must identify the task, data-generating assumptions, competing methods, fitted parameter or maintained-statistic count, compute budget, and falsification criterion before aggregate results are inspected.

Compact-Pro targets an operating point of roughly 119 parameters/statistics in its reference configuration. Because the count changes with dimensionality and state count, every result must provide an exact accounting. Compact-Robust is a future line for heavy tails, outliers, missingness, and distribution shift; it must not be described as effective before comparative evidence exists.

Primary comparisons include Gaussian HMM and Student-t-emission HMM baselines. Evaluation should distinguish predictive quality, calibration, detection delay, robustness, memory, and runtime rather than collapsing all behavior into a single score. Conclusions are limited to the tested datasets, splits, budgets, and uncertainty intervals and do not establish general superiority over neural networks.

## Evidence labels

**Synthetic results** are measurements from controlled generators in this repository. They test
implementation hypotheses under declared assumptions and are not real-world validation.

**Empirical results** are reserved here for measurements from independent external datasets with
recorded provenance. No current Edge streaming result has that label.

The LUT/backtrace benchmark is exploratory synthetic evaluation. Its memory accounting reports the explicitly declared deployment-format model estimate, LUT, and
runtime buffers independently. The current three-state decoder is not the earlier 119-value
Compact-Robust configuration, which is not present in this checkout.

## Future physical experiment (not implemented)

A later experiment may connect `ADC voltage → Resolutive controller → PWM/frequency command →
physical system → ADC voltage` and test whether the controller can find and track
`u* = argmax V(u)`. That study should compare against conventional search or adaptive-control
methods on physical measurements. Hardware code and hardware claims are intentionally outside the
current stage.
