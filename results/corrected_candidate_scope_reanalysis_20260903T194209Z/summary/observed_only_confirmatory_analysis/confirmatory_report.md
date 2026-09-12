# Confirmatory result: observed-only condition transfer

Preregistered in `PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md` at commit `c6aab9e`, before any confirmatory outer-test outcome for this experiment existed. Every threshold below is quoted from that document, not chosen after seeing these numbers.

## Verdict

**NULL** — The interval lies entirely inside the practical-equivalence band, so a practically meaningful benefit and a practically meaningful harm are both excluded.

Computed mechanically by rule `section_7_interpretation_table` from the numbers below.

## The question

Does anonymous condition-transfer augmentation under the `observed_only_low_data` eligibility rule (`observed_only_condition_transfer`) reduce outer-test RMSE relative to the matched real-only XGBoost control (`matched_real_only_control`) at training fraction 0.05 on seeds 6–14? Positive deltas favour augmentation.

## 1. Paired per-seed deltas (all planned seeds)

| Seed | Comparator RMSE | Augmented RMSE | Delta (reduction) | Accepted rows, observed-only | Accepted rows, globally-unmeasured | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 16.789005 | 16.203107 | +0.585899 | 125 | 2 | included |
| 7 | 13.374946 | 13.735474 | -0.360527 | 109 | 0 | included |
| 8 | 14.792385 | 14.689141 | +0.103245 | 119 | 0 | included |
| 9 | 13.201940 | 13.569399 | -0.367459 | 101 | 0 | included |
| 10 | 12.687842 | 13.508136 | -0.820294 | 108 | 0 | included |
| 11 | 13.499380 | 14.971391 | -1.472011 | 110 | 0 | included |
| 12 | 15.972314 | 16.342043 | -0.369729 | 104 | 0 | included |
| 13 | 13.620135 | 13.836321 | -0.216187 | 113 | 0 | included |
| 14 | 15.390352 | 15.865209 | -0.474857 | 110 | 0 | included |

## 2. Primary estimand and interval

- Mean paired RMSE reduction: **-0.376880**
- 95% percentile bootstrap over seeds as clusters (10000 replicates, fixed seed 1601): **[-0.746037, -0.034749]**
- Practical minimum effect: 1.0 RMSE

## 3. Consistency

- Units improved: 2; worsened: 7; tied: 0 (of 9 included)

## 4. Secondary test

- Two-sided Wilcoxon signed-rank p-value: 0.097656 (statistic 8.0) — **secondary**. Significance alone never establishes success.

## 5. Size of the treatment under both eligibility rules

Source: validation-only pool accounting that regenerates each unit's candidate pool from labeled_train under each eligibility rule; the 2026-09-03 protocol run recorded no per-unit counts.

| Rule | Transfer kind | Generated | Accepted | Rejected: observed in labeled train | Rejected: already measured | Quarantined (validation/test identity) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| globally_unmeasured_prospective | anonymous | 1431 | 2 | 0 | 1429 | 0 |
| globally_unmeasured_prospective | typed | 1419 | 3 | 0 | 1416 | 0 |
| observed_only_low_data | anonymous | 1431 | 999 | 95 | 0 | 271 |
| observed_only_low_data | typed | 1419 | 982 | 97 | 0 | 271 |

## 6. Degenerate, failed and missing units

- Planned units per arm: 9
- Primary pair included: 9 (seeds [6, 7, 8, 9, 10, 11, 12, 13, 14])
- Primary pair excluded, degenerate pool: 0 (seeds [])
- Primary pair failed or missing: 0 (seeds [])
- All planned seeds accounted for: True

Secondary control arm `globally_unmeasured_condition_transfer` (`globally_unmeasured_prospective`): prospective-novelty contrast; reported with accepted-row counts, never substituted into the primary comparison.

| Seed | Status | Control RMSE | Comparator RMSE | Generated | Accepted | Rejected: already measured |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 6 | complete | 16.760657 | 16.789005 | 159 | 2 | 157 |
| 7 | degenerate | n/a | 13.374946 | 159 | 0 | 159 |
| 8 | degenerate | n/a | 14.792385 | 159 | 0 | 159 |
| 9 | degenerate | n/a | 13.201940 | 159 | 0 | 159 |
| 10 | degenerate | n/a | 12.687842 | 159 | 0 | 159 |
| 11 | degenerate | n/a | 13.499380 | 159 | 0 | 159 |
| 12 | degenerate | n/a | 15.972314 | 159 | 0 | 159 |
| 13 | degenerate | n/a | 13.620135 | 159 | 0 | 159 |
| 14 | degenerate | n/a | 15.390352 | 159 | 0 | 159 |

Control arm units: 1 complete, 8 degenerate, 0 missing. Degeneracy here is the control's result: it measures how completely the historical eligibility rule suppressed the treatment.

## 7. Withheld-cell oracle (secondary, non-selecting)

Measures how well frozen pseudo-labels reconstruct hidden outer-training cells. It never influenced generation, ranking, teacher fitting, policy selection, weights or student fitting, and it is not substituted for the primary outcome.

| Transfer kind | Seeds | Hidden-cell coverage | Pseudo-label MAE | Pseudo-label RMSE | Spearman | Bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| anonymous | 9 | 0.0369 | 11.654 | 16.274 | 0.761 | +1.655 |
| typed | 9 | 0.0362 | 11.329 | 15.871 | 0.791 | +0.711 |

## Protocol provenance

- Run root: `results/corrected_candidate_scope_reanalysis_20260903T194209Z`
- Dataset: `data/processed/bh_canonical_roles_v1.csv` (`df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365`)
- Split aggregate hash: `9ddb713ace55842aa8794a0d30a457dea92c3338d479b51c7d7a48eafed9b50c`
- Feature metadata hash: `d935e2d9cb2ffcdb080a9ee23d157f653217252629b2ec4f268208855cdaeebe`
- Code commits bound by the arms: `08f91d92da8053df3d8fe6904f6478bd4a648056`, `46d9b90b28d3628d2c5abf6246d8bb29ee6cf9c0`
- The two preregistered arms received equal search budgets: True
- Pool accounting bundle: `results/corrected_observed_only_transfer_pool_accounting_v1` (manifest `7d4908830292e4678c5577d49755b9d96a32034f24be989079bed5a02407a830`)
- Analysis hash: `134a3dc01fdec16d31b3a63d93e6b722766238cb494e0c60ebee96860f23e0bb`

## Scope

Training fraction 0.05 only; canonical grouped random splits only; one dataset; not OOD evidence. The Phase 15 confirmatory result at fraction 0.2 stands under its own protocol and is neither superseded nor re-cut by this experiment.
