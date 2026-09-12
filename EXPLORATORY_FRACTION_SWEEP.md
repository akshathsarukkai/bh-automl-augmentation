# Exploratory training-fraction sweep (post hoc, declared before running)

**Status: exploratory. This is not a preregistered confirmatory experiment and
produces no verdict.** It is declared here, before any of its outer tests is
read, so that what was planned and what was reported can be compared.

## 1. Why this exists

Two preregistered confirmatory experiments have been executed and both are null
(`PRIMARY_EXPERIMENT.md` at training fraction 0.2;
`PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md` at fraction 0.05). They leave an
obvious gap: the effect of observed-only condition transfer as a *function of
training fraction*. Development evidence (seeds 0–4, validation only) predicted
that the gap between the two eligibility rules is largest where the learner has
seen the least, and that the augmented arm is mildly worse than real-only at
every fraction. Whether that pattern holds on outer tests, and whether the
direction changes at the extremes, has not been measured.

This sweep measures it descriptively. It is **post hoc** in the sense of
preregistration section 12: it was designed after both confirmatory results
were known, and it may not be promoted to a confirmatory claim.

## 2. Design (frozen at the commit that adds this file)

| Item | Value |
| --- | --- |
| Arms | `exploratory_observed_only_condition_transfer` (anonymous condition transfer, `candidate_scope.mode: observed_only_low_data`) and `exploratory_matched_real_only_control` (real-only XGBoost). Same three-policy grids, same transfer configuration and same selection procedure as the confirmatory configs in `configs/corrected_observed_only_transfer/`; only the training fraction, the family name and the output paths differ. |
| Training fractions | **0.01** (32 labeled rows) and **0.10** (317 labeled rows). 0.05 is the confirmatory point and is not re-run; 0.2 and 1.0 are excluded because Phase 15 consumed the 0.2 outer tests on these seeds. |
| Seeds | 6 through 14 on `results/corrected_canonical_splits_phase15`. Verified against the repository-global evaluation registry: no record touches these seeds at these fractions. |
| Evaluation units | 9 seeds × 2 fractions × 2 arms = 36 (unit, arm) pairs, each evaluated on its outer test exactly once through the registry. |
| Metric and estimand | Outer-test RMSE; per-seed paired reduction, real-only minus augmented, positive favours augmentation. Reported per fraction. |
| Interval | 95% percentile bootstrap over seeds as clusters, 10,000 replicates, fixed seed **1701** (a fresh seed, distinct from 1501 and 1601). |
| Degenerate units | Recorded with `--record-degenerate` and reported with counts, excluded from the paired estimate, exactly as in the confirmatory protocol. |
| Treatment size | Regenerated per unit under both eligibility rules by the validation-only pool accounting (`configs/corrected_exploratory_fraction_sweep_pool_accounting.yaml`). |
| Not computed | No interpretation table, no verdict, no significance test. The Wilcoxon statistic is reported only as a descriptive number. |

## 3. What will be reported

For each fraction: every paired delta, the mean and its interval, units
improved/worsened, accepted synthetic rows per unit under both rules, degenerate
and failed units, and the withheld-cell oracle metrics. Then one table placing
the two fractions next to the confirmatory 0.05 result, clearly labelled as
exploratory against confirmatory.

## 4. Commitments

- The units are consumed through the registry; the sweep cannot be re-run on
  them. A result that looks positive at one fraction is reported as an
  exploratory observation that would need its own preregistration and fresh
  units to become a claim.
- Nothing here re-opens, re-cuts or re-interprets either confirmatory result.
- This file is not edited after the sweep runs; any change is recorded
  separately and labelled post-run.
