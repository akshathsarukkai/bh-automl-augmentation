# Primary confirmatory result (Phase 15)

Preregistered in `PRIMARY_EXPERIMENT.md` at commit `13fbfde`, before any confirmatory outer-test outcome existed. Every threshold below is quoted from that document, not chosen after seeing these numbers.

## Verdict

**NULL** — The interval lies entirely inside the practical-equivalence band, so a practically meaningful benefit and a practically meaningful harm are both excluded.

Computed mechanically by rule `section_6_interpretation_table` from the numbers below.

## The question

Does `anonymous_transfer_without_ae` reduce outer-test RMSE relative to the matched real-only `direct_xgboost` control at training fraction 0.2? Positive deltas favour augmentation.

## Paired per-seed deltas

| Seed | Comparator RMSE | Augmented RMSE | Delta (reduction) | Accepted synthetic rows | Status |
| --- | --- | --- | --- | --- | --- |
| 5 | 9.761513 | 10.370256 | -0.608743 | 345 | included |
| 6 | 10.649638 | 11.516282 | -0.866644 | 357 | included |
| 7 | 9.973668 | 10.600020 | -0.626352 | 338 | included |
| 8 | 11.701744 | 12.329004 | -0.627260 | 346 | included |
| 9 | 9.925247 | 10.238692 | -0.313445 | 350 | included |
| 10 | 11.680788 | 12.021013 | -0.340225 | 346 | included |
| 11 | 11.212575 | 11.418690 | -0.206115 | 362 | included |
| 12 | 10.605043 | 11.726199 | -1.121156 | 341 | included |
| 13 | 9.165247 | 10.021845 | -0.856598 | 335 | included |
| 14 | 10.411220 | 11.454156 | -1.042936 | 350 | included |

## Primary estimand

- Mean paired RMSE reduction: **-0.660947**
- 95% percentile bootstrap over seeds as clusters (10000 replicates, fixed seed 1501): **[-0.844599, -0.477761]**
- Practical minimum effect: 1.0 RMSE
- Units improved: 0; worsened: 10; tied: 0

## Secondary test

- Two-sided Wilcoxon signed-rank p-value: 0.001953 (statistic 0.0) — **secondary**. Significance alone never establishes success.

## Unit accounting

- Planned units: 10
- Included: 10 (seeds [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
- Excluded, degenerate transfer pool: 0 (seeds [])
- Failed or missing: 0 (seeds [])
- All planned seeds accounted for: True

## Protocol provenance

- Run directory: `results/autonomous_execution/phase_15/corrected-20260729-13fbfde-phase15-confirmation-v1`
- Run manifest hash: `e4be0ed026b9d3bae8852ffe5e85e62e48f6a54bf29f3b2e874d703016370318`
- Plan hash: `f998c7354f228c1c5eb6c61cf7437873d9264d734449e75083464c307edfcf08`
- Dataset hash: `df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365`
- Canonical split hash: `9ddb713ace55842aa8794a0d30a457dea92c3338d479b51c7d7a48eafed9b50c`
- Per-family search budget: `{'anonymous_transfer_without_ae': 3, 'direct_xgboost': 3, 'linear_autoencoder': 6, 'matched_direct_mlp': 10, 'redesigned_supervised_ae': 11, 'truncated_svd': 6, 'typed_transfer_without_ae': 3}`
- The two preregistered arms received equal search budgets: True
- The run also computed a Phase 14 AE retention decision (`secondary_ablation`). That is Phase 14 machinery and forms no part of this analysis.
- Analysis hash: `72ace136b23d70d23b105430d2c753daf9571b2f881e37378848b755be0fd3a5`

## Scope

Random-split behaviour is the primary regime. The Phase 9 chemical OOD regimes are not part of this confirmation. The five other method families present in the run are descriptive only and enter no preregistered claim.
