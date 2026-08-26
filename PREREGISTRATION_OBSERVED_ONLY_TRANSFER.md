# Observed-only condition transfer (preregistration)

**Status: preregistered. No confirmatory outer-test result for this experiment
existed when this document was committed.**

This document is frozen before its confirmatory evaluation runs. It is not to be
edited after any confirmatory outer-test outcome is generated. Any change forced
by circumstance must be recorded separately, in a distinct file, explicitly
labelled post-preregistration, with its reason and its timestamp.

This is a **new experiment**, not a re-analysis of the Phase 15 confirmatory
result. Phase 15 stands under its exact executed protocol and is not superseded,
retracted, or re-cut by anything here. See §10.

---

## 1. The question

Does anonymous condition-transfer augmentation improve low-data
Buchwald–Hartwig yield prediction, relative to a matched real-only baseline,
when candidate eligibility is decided from **the labeled low-data subset the
simulated learner actually observed** rather than from the complete historical
dataset?

## 2. Why this experiment exists, and why it is not Phase 15 again

The canonical Buchwald–Hartwig dataset is a near-complete factorial: 3,955 of
3,960 grid cells are measured. Two very different quantities follow from that,
and the repository had been conflating them.

- **Global discovery headroom is almost zero.** Exactly five eligible unmeasured
  reactions exist. Any condition transfer assembled from the observed role
  vocabulary is essentially guaranteed to exist *somewhere* in the dataset.
- **Low-data transfer headroom is large.** At a 5% training fraction the
  simulated learner has observed 159 of 3,955 reactions. The other ~96% of the
  matrix is unknown to it, and reconstructing those cells is exactly what
  condition transfer is supposed to do.

An eligibility rule that rejects a candidate because it exists somewhere in the
complete dataset therefore answers the *global* question while being applied to
the *low-data* one. The repository-wide audit
(`scripts/audit_candidate_scope.py`) established which runners did this. The
measured consequence is severe: in the Phase 11 production bundle, 11,682 of
11,727 chemical candidates were rejected as `already_measured`, and **93.6% of
those collided only with rows the simulated learner had never seen**. The
chemical augmentation arms carried 0.4 to 1.2 effective added rows against a
nominal budget of 159 — they were numerically identical to the real-only arm.

This experiment asks the low-data question under the low-data rule.

## 3. Candidate-scope semantics (frozen)

The primary arm runs under `candidate_scope.mode: observed_only_low_data`, whose
contract is:

| Rule | Frozen behaviour |
| --- | --- |
| Reject if identity occurs in `labeled_train` | Yes — reason `observed_in_labeled_train` |
| Reject duplicate generated identities | Yes — reasons `duplicate_synthetic`, `feature_duplicate_synthetic` |
| Reject if identity occurs elsewhere in the complete dataset | **No.** Complete-dataset membership is structurally unavailable at generation time: an observed-only `CandidateScopePolicy` raises if handed global identity keys. |
| Candidates matching `hidden_outer_train` | Legitimate pseudo-labelled training examples. |
| Candidates matching `validation` or `test` identities | Quarantined — reason `quarantined_held_out_identity`, excluded from student training, retained in the candidate audit. |
| Canonical chemistry, exact role change, feature compatibility, train-only parents | Unchanged from the existing contracts. |
| Hidden `yield` values during generation, pseudo-labelling, policy search, student fitting | Never read. Hidden outer-training outcomes are reachable only through `LowDataEvaluationUnit.hidden_outer_train_oracle()`, after the candidate pool and its pseudo-labels are frozen and hash-verified. |

The comparator arm and the prospective control run under the same protocol with
the eligibility rule as the only difference.

## 4. Specification

| Item | Value |
| --- | --- |
| Primary method | Anonymous condition-transfer augmentation + XGBoost, `candidate_scope.mode: observed_only_low_data` |
| Comparator | Matched real-only XGBoost — same model, same grid, same splits, same selection procedure, same search budget; differing only in whether accepted synthetic rows are added to training |
| Secondary control | The same augmented arm under `candidate_scope.mode: globally_unmeasured_prospective`, reported as the prospective-novelty contrast, **not** as a second primary |
| Dataset | `data/processed/bh_canonical_roles_v1.csv`, sha256 `df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365` |
| Split regime | Canonical grouped random splits, complete group separation on `canonical_reaction_key`, directory `results/corrected_canonical_splits_phase15` |
| Evaluation units | **Seeds 6 through 14 at training fraction 0.05** — nine units whose outer-test rows have never been read. Seed 5 at fraction 0.05 is deliberately excluded: it was consumed by a pre-preregistration technical dry run (§8a). Verified unconsumed against the repository-global evaluation registry before launch. |
| Registry family | `observed_only_condition_transfer` |
| Training fraction | **0.05** (primary, 159 labeled rows). No other fraction is primary. |
| Primary metric | Outer-test RMSE (lower is better) |
| Primary estimand | Mean paired RMSE reduction: comparator RMSE minus augmented RMSE, per seed. Positive favours augmentation. |
| Practical minimum effect | **1.0 RMSE**, the same margin used by Phase 14 retention and by the Phase 15 preregistration |
| Paired units | 9 (one per seed, 6 through 14) |
| Confidence interval | 95% percentile bootstrap over the 9 paired per-seed deltas, resampling seeds as clusters, 10,000 replicates, fixed bootstrap seed **1601** |
| Statistical test | Two-sided Wilcoxon signed-rank on the paired deltas, reported as **secondary**. The interval and the practical margin decide the outcome. |

### Why fraction 0.05 and not 0.2

Fraction 0.2 is where the *previous* confirmatory experiment ran, and all ten of
its units at that fraction are consumed. Independently of availability, 0.05 is
the better test of this specific question: it is where the gap between the two
eligibility rules is largest, because the learner has observed the least. A
bounded development run at fraction 0.05 (§8b) recorded 1 accepted candidate
under the global rule against 580 under the observed-only rule, from the same
795 generated across five seeds. Fraction 0.2 is not analysed as part of this
confirmation.

## 5. Matched search budgets

Identical for both arms and confined to training-derived data:

- Both arms receive the **same number of candidate policies** from the same
  grid. Neither arm may receive a larger search budget than the other.
- Hyperparameters are selected on the validation partition only.
- Teacher fitting for pseudo-labels, donor selection, similarity computation and
  transfer-pool construction use `labeled_train` rows only.
- The declared `candidate_scope.mode` is bound into the scientific config hash,
  and therefore into the scientific binding, the search manifest and the frozen
  policy. A frozen policy from a differently-scoped search cannot be replayed
  against this one.
- The winning policy per arm per unit is frozen, with its hash recorded, before
  any outer-test outcome is read.
- Each (unit, arm) pair is evaluated on the outer test **exactly once**, through
  the repository-global evaluation registry, which reserves the identity before
  the refit and marks it `failed_or_uncertain` if the refit raises.

## 6. Exclusion and failure handling

- A unit whose transfer pool is **degenerate** (zero accepted synthetic rows)
  makes the augmented arm identical to its comparator by construction. Such a
  unit is **excluded from the primary paired analysis** and reported explicitly
  with its count.
- If more than 2 of 9 units are degenerate under the observed-only rule, the
  primary analysis is reported as **inconclusive on this dataset** and the
  degeneracy is itself reported as the finding.
- The globally-unmeasured control arm is **expected to be degenerate on most or
  all units** — development evidence records 1 accepted candidate in 2,540. That
  degeneracy is the control's result, not a failure of it: it is the direct
  measurement of how completely the historical eligibility rule suppressed the
  treatment. Its units are reported with their accepted-row counts and are never
  substituted into the primary comparison.
- Any unit that fails for any other reason is reported with its failure reason
  and its registry record marked `failed_or_uncertain`. It is not silently
  dropped and is not retried against the same outer test.
- All 9 planned seeds are accounted for, whether they completed, were excluded,
  or failed.
- If fewer than 7 units remain after exclusions, the primary analysis is
  reported as underpowered and no positive claim is made.

## 7. Predefined interpretation of every outcome

Let *d* be the mean paired RMSE reduction (comparator minus augmented, positive
favours augmentation), with 95% bootstrap interval [*L*, *U*], over *n* included
units.

| Outcome | Condition | Conclusion |
| --- | --- | --- |
| **Positive** | *d* ≥ 1.0 **and** *L* > 0 **and** at least 7 of *n* units improve | Observed-only condition transfer provides a practically meaningful and reasonably consistent benefit at 5% training data. |
| **Null** | [*L*, *U*] lies entirely within (−1.0, +1.0) | No practically meaningful effect in either direction. Both a ≥1.0 benefit and a ≥1.0 harm are excluded. |
| **Negative** | *d* ≤ −1.0 **and** *U* < 0 | Observed-only condition transfer causes practically meaningful harm. |
| **Inconclusive** | The interval admits both a ≥1.0 benefit and a ≥1.0 harm, or *n* < 7 | The experiment does not resolve the question at this sample size. |

Statistical significance alone never establishes success. An interval excluding
zero but lying entirely inside (−1.0, +1.0) is a **null** result.

## 8. The oracle diagnostic is secondary and non-selecting

The `withheld_cell_transfer_oracle` family measures how well the frozen
pseudo-labels reconstruct hidden outer-training cells: overlap count and
coverage, MAE, RMSE, Spearman, bias, high-yield precision/recall/enrichment and
top-k enrichment, distance to the nearest labeled training reaction under its
declared metric, and the same metrics stratified by transferred role.

It is **evidence about the mechanism, not about the primary outcome**. It is
declared here as secondary and non-selecting, and it may not:

- influence candidate generation, candidate ranking, or teacher fitting;
- influence policy selection or hyperparameter selection;
- influence synthetic weights or student fitting;
- be substituted for the primary outcome if the primary comes out null.

A good oracle result alongside a null primary is a coherent and reportable
finding: it would mean the transfer reconstructs withheld chemistry accurately
while that accuracy does not translate into test-set predictive gain.

### 8a. Pre-preregistration technical dry run, and why seed 5 is excluded

Before this document was written, the confirmatory chain was executed once
end to end on **seed 5 at fraction 0.05** to prove the code path works: policy
search selected `anonymous-xgb-1` at validation RMSE 13.370, and final
evaluation produced outer-test RMSE 13.994 on 396 test rows from 159 refit
rows. That run wrote to a scratch directory and claimed no registry identity,
but it did read that unit's outer test.

Rather than pretend otherwise, seed 5 at fraction 0.05 is **excluded from the
confirmatory set**. The confirmatory experiment runs on seeds 6 through 14,
giving nine paired units, above the seven-unit power floor of §6. The dry-run
numbers above are recorded here as engineering evidence and are not part of any
primary or secondary analysis.

### 8b. Development evidence

Recorded before this preregistration, on **seeds 0 through 4** (disjoint from
the confirmatory seeds) at fractions 0.01, 0.05 and 0.10, **validation only**;
no outer test was materialized and no registry identity was claimed. Bundle:
`results/corrected_candidate_scope_development_v1/`.

**Candidate pool, summed over the five development seeds**

| Fraction | Eligibility rule | Generated | Accepted | Rejected observed | Quarantined | Rejected already-measured |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | globally unmeasured | 160 | **0** | 0 | 0 | 160 |
| 0.01 | observed only | 160 | **125** | 2 | 32 | 0 |
| 0.05 | globally unmeasured | 795 | **1** | 0 | 0 | 794 |
| 0.05 | observed only | 795 | **580** | 45 | 134 | 0 |
| 0.10 | globally unmeasured | 1585 | **0** | 0 | 0 | 1585 |
| 0.10 | observed only | 1585 | **999** | 143 | 317 | 0 |

(anonymous transfer; typed transfer is within a few percent at every row.)

Under the historical rule the treatment does not exist: 1 accepted candidate in
2,540 generated across all fractions. Under the observed-only rule the pool is
roughly 1,700x larger. **This is the central quantitative finding of the audit,
and it is a statement about candidate availability, not about accuracy.**

**Withheld-cell oracle, observed-only rule, anonymous / typed**

| Fraction | Overlap with hidden cells | Hidden-cell coverage | MAE | RMSE | Spearman | Bias |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 100% / 100% of candidates | 0.8% / 0.6% | 13.92 / 12.47 | 19.99 / 17.83 | 0.672 / 0.729 | +2.69 / +0.20 |
| 0.05 | 99.8% / 99.8% | 3.9% / 3.9% | 11.54 / 10.89 | 16.33 / 14.97 | 0.767 / 0.805 | +0.07 / −0.01 |
| 0.10 | 100% / 100% | 7.0% / 7.1% | 9.83 / 9.93 | 14.15 / 14.04 | 0.817 / 0.825 | +0.20 / −0.19 |

Essentially every candidate the observed-only rule admits lands on a reaction
that was measured but hidden from the simulated learner — which is precisely
what the historical rule was suppressing. Pseudo-label rank correlation with the
true withheld yield is 0.67 to 0.83 and improves with training fraction.

**Validation RMSE, mean over the five development seeds**

| Fraction | Real-only control | Observed-only, anonymous | Observed-only, typed | Globally-unmeasured, anonymous |
| --- | ---: | ---: | ---: | ---: |
| 0.01 | 21.890 | 22.413 | 22.291 | 21.890 |
| 0.05 | 14.016 | 14.718 | 14.535 | 14.003 |
| 0.10 | 12.091 | 12.369 | 12.389 | 12.091 |

**Development evidence therefore predicts a null or mildly negative primary
result.** The augmented arms are 0.3 to 0.7 validation RMSE *worse* than the
matched real-only control at every fraction, while the globally-unmeasured arms
are numerically identical to that control because their pools are empty.

This is recorded here, before the confirmatory run, so that a null or negative
outcome cannot later be presented as a surprise and a positive outcome cannot be
presented as expected. The two findings are separable and both will be reported:
the candidate-scope correction makes the treatment *exist* (a large, real, and
previously hidden effect on pool size), and it may nonetheless fail to improve
outer-test prediction.

## 9. What will be reported

- Paired per-seed deltas, all 9, individually.
- The mean paired effect with its 95% interval, and the verdict from §7.
- Consistency: how many units improved, how many worsened.
- The Wilcoxon *p*-value, labelled secondary.
- Accepted-synthetic-row count per unit under **both** eligibility rules, so the
  size of the treatment is visible alongside its effect.
- Every degenerate and failed unit, with counts.
- The oracle metrics of §8, clearly labelled secondary and non-selecting.

## 10. Relationship to the existing Phase 15 result

The Phase 15 confirmatory experiment is **not invalidated and not re-run** by
this experiment. Its candidate eligibility was already scoped to the rows each
phase could see — inner policy-fit rows during search (507 identities) and saved
training rows during placement and final evaluation (633 identities), never the
3,955-row canonical universe, with a hash guard that hard-fails if a pool
consumes anything else. Its recorded null therefore already answers a low-data
question.

What the audit narrows is the *stated reason* for that null, not the null. Phase
15 documentation attributed it in part to the near-complete factorial leaving
"almost no novel chemistry to propose" — a global-discovery-headroom argument
that does not apply to an observed-only pool with 335 to 362 accepted synthetic
rows per unit. The scope note added to `RESULT_STATUS.md` and
`PRIMARY_EXPERIMENT.md` records this. Phase 15's numbers, verdict and registry
records are untouched.

## 11. Contracts this experiment inherits

Canonical seven-role identity; RDKit featurization pinned to the dataset's
canonicalization vocabulary, verified at run time by
`assert_stored_identities_match_roles`; saved grouped splits with complete group
separation; fixed validation and test membership; training-only augmentation,
teacher fitting and representation learning; validation-only policy selection;
policies frozen before outer-test evaluation; exactly one outer-test evaluation
per declared unit through the repository-global registry; immutable
hash-verifiable outputs; and explicit reporting of negative and null results.

## 12. Anti-gaming commitments

- The primary metric, margin, fraction, seed set, exclusion rule, interval
  method and interpretation table above are fixed as of this commit.
- The confirmatory run writes to fresh output directories and reserves its
  outer-test claims through the repository-global registry, so a second attempt
  on the same units is refused by construction rather than by discipline.
- If the result is null or negative, it is reported as the finding. It is not
  re-cut by fraction, metric, subgroup, or seed subset in search of a positive.
- The prospective control and the oracle diagnostic may not be promoted to
  primary after the fact.
- Any post hoc analysis performed after seeing the result is labelled post hoc
  and reported separately.
