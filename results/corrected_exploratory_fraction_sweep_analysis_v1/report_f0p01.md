# Exploratory result (post hoc, no verdict): observed-only condition transfer at training fraction 0.01

Declared in `EXPLORATORY_FRACTION_SWEEP.md` at commit `2f7c74f` before any of its outer tests was read. **This is not a preregistered confirmatory experiment.** No interpretation table is applied and no verdict is produced; the interval below is descriptive and may not be promoted to a claim.

## No verdict

This design is exploratory and post hoc; it applies no preregistered interpretation table and yields no verdict. The interval is reported descriptively.

## The question

Does anonymous condition-transfer augmentation under the `observed_only_low_data` eligibility rule (`exploratory_observed_only_condition_transfer`) reduce outer-test RMSE relative to the matched real-only XGBoost control (`exploratory_matched_real_only_control`) at training fraction 0.01 on seeds 6–14? Positive deltas favour augmentation.

## 1. Paired per-seed deltas (all planned seeds)

| Seed | Comparator RMSE | Augmented RMSE | Delta (reduction) | Accepted rows, observed-only | Accepted rows, globally-unmeasured | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 21.793043 | 22.416484 | -0.623441 | 22 | 0 | included |
| 7 | 21.249725 | 21.393260 | -0.143535 | 24 | 0 | included |
| 8 | 26.194766 | 26.036890 | +0.157876 | 27 | 0 | included |
| 9 | 21.520826 | 22.428600 | -0.907774 | 24 | 0 | included |
| 10 | 21.904335 | 22.834711 | -0.930375 | 23 | 0 | included |
| 11 | 20.516951 | 19.989506 | +0.527445 | 25 | 0 | included |
| 12 | 24.095302 | 24.873457 | -0.778156 | 27 | 0 | included |
| 13 | 21.458911 | 22.844327 | -1.385416 | 26 | 0 | included |
| 14 | 25.799805 | 25.452505 | +0.347300 | 25 | 0 | included |

## 2. Primary estimand and interval

- Mean paired RMSE reduction: **-0.415120**
- 95% percentile bootstrap over seeds as clusters (10000 replicates, fixed seed 1701): **[-0.806036, -0.009683]**
- Practical minimum effect: 1.0 RMSE

## 3. Consistency

- Units improved: 3; worsened: 6; tied: 0 (of 9 included)

## 4. Secondary test

- Two-sided Wilcoxon signed-rank p-value: 0.128906 (statistic 9.0) — **secondary**. Significance alone never establishes success.

## 5. Size of the treatment under both eligibility rules

Source: validation-only pool accounting that regenerates each unit's candidate pool from labeled_train under each eligibility rule; the 2026-09-03 protocol run recorded no per-unit counts.

| Rule | Transfer kind | Generated | Accepted | Rejected: observed in labeled train | Rejected: already measured | Quarantined (validation/test identity) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| globally_unmeasured_prospective | anonymous | 288 | 0 | 0 | 288 | 0 |
| globally_unmeasured_prospective | typed | 192 | 0 | 0 | 192 | 0 |
| observed_only_low_data | anonymous | 288 | 223 | 11 | 0 | 53 |
| observed_only_low_data | typed | 192 | 152 | 11 | 0 | 28 |

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
| anonymous | 9 | 0.0079 | 14.875 | 19.675 | 0.644 | +1.164 |
| typed | 9 | 0.0054 | 14.436 | 19.106 | 0.672 | +1.859 |

## Protocol provenance

- Run root: `results/corrected_exploratory_fraction_sweep_f0p01`
- Dataset: `data/processed/bh_canonical_roles_v1.csv` (`df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365`)
- Split aggregate hash: `9ddb713ace55842aa8794a0d30a457dea92c3338d479b51c7d7a48eafed9b50c`
- Feature metadata hash: `d935e2d9cb2ffcdb080a9ee23d157f653217252629b2ec4f268208855cdaeebe`
- Code commits bound by the arms: `2f7c74f0c0bf84cd2d2a74c2cb7b70134c3cee59`
- The two preregistered arms received equal search budgets: True
- Pool accounting bundle: `results/corrected_exploratory_fraction_sweep_pool_accounting_v1` (manifest `20c2e88e986253e59d67228dd495ff81dd8ca11dffc30b5c5a59527755dca87c`)
- Analysis hash: `77ca51af62fe99e58bbc8caf38d7bb396534ce3a83860430b38e82e194e595d5`

## Scope

Training fraction 0.01 only; canonical grouped random splits only; one dataset; not OOD evidence; exploratory and post hoc. Neither confirmatory result (Phase 15 at fraction 0.2, observed-only transfer at 0.05) is re-opened, re-cut or re-interpreted by this analysis.
