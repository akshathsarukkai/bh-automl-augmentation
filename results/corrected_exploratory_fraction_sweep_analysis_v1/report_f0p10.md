# Exploratory result (post hoc, no verdict): observed-only condition transfer at training fraction 0.1

Declared in `EXPLORATORY_FRACTION_SWEEP.md` at commit `2f7c74f` before any of its outer tests was read. **This is not a preregistered confirmatory experiment.** No interpretation table is applied and no verdict is produced; the interval below is descriptive and may not be promoted to a claim.

## No verdict

This design is exploratory and post hoc; it applies no preregistered interpretation table and yields no verdict. The interval is reported descriptively.

## The question

Does anonymous condition-transfer augmentation under the `observed_only_low_data` eligibility rule (`exploratory_observed_only_condition_transfer`) reduce outer-test RMSE relative to the matched real-only XGBoost control (`exploratory_matched_real_only_control`) at training fraction 0.1 on seeds 6–14? Positive deltas favour augmentation.

## 1. Paired per-seed deltas (all planned seeds)

| Seed | Comparator RMSE | Augmented RMSE | Delta (reduction) | Accepted rows, observed-only | Accepted rows, globally-unmeasured | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 12.399597 | 13.261042 | -0.861445 | 212 | 3 | included |
| 7 | 11.786925 | 11.619022 | +0.167903 | 201 | 0 | included |
| 8 | 13.549088 | 13.557230 | -0.008141 | 203 | 0 | included |
| 9 | 12.144143 | 12.695154 | -0.551011 | 193 | 1 | included |
| 10 | 12.737334 | 13.603013 | -0.865679 | 196 | 0 | included |
| 11 | 12.626682 | 13.232337 | -0.605655 | 185 | 0 | included |
| 12 | 12.877572 | 13.478580 | -0.601008 | 194 | 0 | included |
| 13 | 11.343931 | 12.034963 | -0.691031 | 204 | 0 | included |
| 14 | 12.708498 | 13.265907 | -0.557409 | 209 | 1 | included |

## 2. Primary estimand and interval

- Mean paired RMSE reduction: **-0.508164**
- 95% percentile bootstrap over seeds as clusters (10000 replicates, fixed seed 1701): **[-0.707061, -0.272953]**
- Practical minimum effect: 1.0 RMSE

## 3. Consistency

- Units improved: 1; worsened: 8; tied: 0 (of 9 included)

## 4. Secondary test

- Two-sided Wilcoxon signed-rank p-value: 0.011719 (statistic 2.0) — **secondary**. Significance alone never establishes success.

## 5. Size of the treatment under both eligibility rules

Source: validation-only pool accounting that regenerates each unit's candidate pool from labeled_train under each eligibility rule; the 2026-09-03 protocol run recorded no per-unit counts.

| Rule | Transfer kind | Generated | Accepted | Rejected: observed in labeled train | Rejected: already measured | Quarantined (validation/test identity) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| globally_unmeasured_prospective | anonymous | 2853 | 5 | 0 | 2848 | 0 |
| globally_unmeasured_prospective | typed | 2853 | 7 | 0 | 2846 | 0 |
| observed_only_low_data | anonymous | 2853 | 1797 | 250 | 0 | 557 |
| observed_only_low_data | typed | 2853 | 1789 | 252 | 0 | 565 |

## 6. Degenerate, failed and missing units

- Planned units per arm: 9
- Primary pair included: 9 (seeds [6, 7, 8, 9, 10, 11, 12, 13, 14])
- Primary pair excluded, degenerate pool: 0 (seeds [])
- Primary pair failed or missing: 0 (seeds [])
- All planned seeds accounted for: True

No secondary control arm is part of this design; the globally-unmeasured accepted-row counts in section 5 come from the pool accounting alone.

## 7. Withheld-cell oracle (secondary, non-selecting)

Measures how well frozen pseudo-labels reconstruct hidden outer-training cells. It never influenced generation, ranking, teacher fitting, policy selection, weights or student fitting, and it is not substituted for the primary outcome.

| Transfer kind | Seeds | Hidden-cell coverage | Pseudo-label MAE | Pseudo-label RMSE | Spearman | Bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| anonymous | 9 | 0.0699 | 10.322 | 15.401 | 0.782 | +1.296 |
| typed | 9 | 0.0695 | 9.990 | 15.030 | 0.807 | +0.902 |

## Protocol provenance

- Run root: `results/corrected_exploratory_fraction_sweep_f0p10`
- Dataset: `data/processed/bh_canonical_roles_v1.csv` (`df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365`)
- Split aggregate hash: `9ddb713ace55842aa8794a0d30a457dea92c3338d479b51c7d7a48eafed9b50c`
- Feature metadata hash: `d935e2d9cb2ffcdb080a9ee23d157f653217252629b2ec4f268208855cdaeebe`
- Code commits bound by the arms: `2f7c74f0c0bf84cd2d2a74c2cb7b70134c3cee59`
- The two preregistered arms received equal search budgets: True
- Pool accounting bundle: `results/corrected_exploratory_fraction_sweep_pool_accounting_v1` (manifest `20c2e88e986253e59d67228dd495ff81dd8ca11dffc30b5c5a59527755dca87c`)
- Analysis hash: `df2e372cade8dc51198f50cf4846097f021d4bb95a1b0b8c2b3667b74cd933f8`

## Scope

Training fraction 0.1 only; canonical grouped random splits only; one dataset; not OOD evidence; exploratory and post hoc. Neither confirmatory result (Phase 15 at fraction 0.2, observed-only transfer at 0.05) is re-opened, re-cut or re-interpreted by this analysis.
