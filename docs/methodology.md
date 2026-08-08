# Methodology

The research program evaluates compact sequential inference under explicit resource budgets. Each hypothesis must identify the task, data-generating assumptions, competing methods, fitted parameter or maintained-statistic count, compute budget, and falsification criterion before aggregate results are inspected.

Compact-Pro targets an operating point of roughly 119 parameters/statistics in its reference configuration. Because the count changes with dimensionality and state count, every result must provide an exact accounting. Compact-Robust is a future line for heavy tails, outliers, missingness, and distribution shift; it must not be described as effective before comparative evidence exists.

Primary comparisons include Gaussian HMM and Student-t-emission HMM baselines. Evaluation should distinguish predictive quality, calibration, detection delay, robustness, memory, and runtime rather than collapsing all behavior into a single score. Conclusions are limited to the tested datasets, splits, budgets, and uncertainty intervals and do not establish general superiority over neural networks.

The first baseline implementation is a known-parameter filtering control. It is
intentionally separate from Compact-Pro so agreement can expose regressions, but
Gaussian agreement is not independent evidence of performance. Parameter fitting,
hyperparameter selection, and held-out evaluation belong to a later protocol stage.
