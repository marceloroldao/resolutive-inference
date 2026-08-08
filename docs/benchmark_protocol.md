# Benchmark Protocol

1. Freeze the dataset version, provenance, checksum, preprocessing, and split-generation code.
2. Declare primary metrics, resource budgets, parameter/statistic accounting, and exclusion rules before evaluation.
3. Compare Compact-Pro with Gaussian HMM and Student-t HMM baselines on identical splits. Add other baselines only with documented tuning parity.
4. Run multiple predetermined seeds and retain raw per-run outputs. Report central estimates, dispersion, and confidence or bootstrap intervals.
5. Separate hyperparameter selection from final evaluation. Never tune on the test sequence.
6. Record the code commit, configuration, Python and dependency versions, hardware, wall time, and peak memory.
7. Include failure cases and sensitivity analyses for sequence length, latent-state mismatch, noise, outliers, and distribution shift.
8. Treat generated tables and figures as derived artifacts; they must identify the script and raw measurements used to create them.

Claims must be scoped to observed evidence. Statistical significance, when reported, should be accompanied by effect sizes, uncertainty, and correction for multiple comparisons where applicable.
