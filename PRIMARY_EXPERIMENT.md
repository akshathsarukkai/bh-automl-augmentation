# Primary confirmatory experiment (preregistration)

**Status: preregistered. No confirmatory outer-test result existed when this
document was committed.**

This document is frozen before the confirmatory evaluation runs. It is not to be
edited after any confirmatory outer-test outcome is generated. Any change forced
by circumstance must be recorded separately, in a distinct file, explicitly
labelled post-preregistration, with its reason and its timestamp.

Preregistered at commit `0405f2b` (the Phase 14 ledger checkpoint). The
confirmatory run must be launched from a commit that has this file in its
history.

---

## 1. The question

Does anonymous condition-transfer augmentation improve low-data
Buchwald–Hartwig yield prediction, relative to a matched real-only baseline,
under the corrected canonical protocol?

This is the project's central thesis. Every earlier phase was infrastructure or
development evidence for it. It is being asked once, on evaluation units that
have never been touched.

## 2. Why this comparison, and not a better-looking one

Development evidence, not preference, selects this comparison.

- Phase 14 evaluated seven method families on ten canonical random units. At
  training fraction 0.2 the outer-test RMSEs were: anonymous transfer 10.461,
  direct XGBoost 10.467, matched direct MLP 10.509, typed transfer 10.571,
  supervised AE 10.758, truncated SVD 16.058, linear autoencoder 16.067.
  Anonymous transfer and its matched real-only XGBoost control differ by
  **0.006 RMSE**. That is the number this experiment is designed to confirm or
  refute, and development evidence predicts a **null** result.
- The supervised autoencoder **failed its Phase 14 criterion** and is therefore
  ineligible for confirmation. It appears nowhere in this experiment.
- Truncated SVD and the linear autoencoder are excluded because Phases 13 and 14
  both show them to be far worse than every other family; confirming a method
  already known to be poor would be uninformative.
- Training fraction 0.2 is chosen because it is where augmentation has room to
  act. Phase 14's pool audit records 507 parents yielding 340 accepted synthetic
  rows against 633 measured rows at fraction 0.2 (a 54% increase), but only 224
  accepted from 2532 parents at fraction 1.0 (8.8%), because the canonical
  dataset is a near-complete factorial. Testing at fraction 1.0 as the primary
  would test a treatment that barely applies.
- The comparator is the *matched real-only* arm — the identical XGBoost model,
  the identical hyperparameter grid, the identical splits, the identical
  selection procedure — differing only in whether accepted synthetic rows are
  added to training. Nothing else may differ.

I am deliberately preregistering a comparison that development evidence expects
to come out null. A well-powered null on the central thesis is a real result and
is more useful than confirming a hand-picked favourable contrast.

## 3. Specification

| Item | Value |
| --- | --- |
| Primary method | Anonymous condition-transfer augmentation + XGBoost |
| Comparator | Matched real-only XGBoost (same model, grid, splits, selection; no synthetic rows) |
| Dataset | `data/processed/bh_canonical_roles_v1.csv`, sha256 `df61cb657747e7f7715ff93060b869550a154a435f6e325d84448477f161d365` |
| Split regime | Canonical grouped random splits, complete group separation on `canonical_reaction_key` |
| Evaluation units | **10 fresh seeds, 5 through 14**, in a new immutable split directory. The existing five-seed artifacts under `results/corrected_canonical_splits/` are not mutated, not extended, and not re-read for these seeds. |
| Training fraction | **0.2** (primary). Fraction 1.0 is secondary and descriptive only. |
| Primary metric | Outer-test RMSE (lower is better) |
| Primary estimand | Mean paired RMSE reduction: comparator RMSE minus augmented RMSE, per seed. Positive means augmentation helps. |
| Practical minimum effect | **1.0 RMSE**, the same margin used as Phase 14's retention threshold, roughly 10% of the ~10.5 baseline |
| Paired units | 10 (one per fresh seed) |
| Confidence interval | 95% percentile bootstrap over the 10 paired per-seed deltas, resampling seeds as clusters, 10 000 replicates, fixed seed 1501 |
| Statistical test | Two-sided Wilcoxon signed-rank on the paired deltas, reported as **secondary**. The interval and the practical margin decide the outcome; significance alone does not. |

## 4. Policy selection

Identical for both arms and confined to training-derived data:

- Hyperparameters are selected within each arm on an inner validation split
  carved from the saved training rows only, group-disjoint from the inner fit
  rows.
- Both arms receive the **same number of candidate policies** from the same
  grid. No arm may receive a larger search budget than the other.
- Teacher fitting for pseudo-labels, transfer-pool construction, donor
  selection, and target scaling use training rows only.
- The winning policy per arm per unit is frozen, with its hash recorded, before
  any outer-test outcome is read.
- Each (unit, arm) pair is evaluated on the outer test **exactly once**, through
  the repository-global evaluation registry. No outer-test metric may influence
  any selection.

## 5. Exclusion and failure handling

- A unit whose transfer pool is **degenerate** (zero accepted synthetic rows)
  makes the augmented arm identical to its comparator by construction. Such a
  unit is **excluded from the primary paired analysis** and reported explicitly
  with its count. If more than 3 of 10 units are degenerate, the primary
  analysis is reported as **inconclusive on this dataset** and the degeneracy is
  reported as the finding.
- Any unit that fails to complete for any other reason is reported with its
  failure reason and its registry record marked `failed_or_uncertain`. It is not
  silently dropped and is not retried against the same outer test.
- All 10 planned seeds are accounted for in the report, whether they completed,
  were excluded, or failed. The report states the count in each category.
- If fewer than 7 units remain after exclusions, the primary analysis is
  reported as underpowered and no positive claim is made.

## 6. Predefined interpretation of every outcome

Let *d* be the mean paired RMSE reduction (comparator minus augmented, positive
favours augmentation), with 95% bootstrap interval [*L*, *U*], over *n* included
units.

| Outcome | Condition | Conclusion |
| --- | --- | --- |
| **Positive** | *d* ≥ 1.0 **and** *L* > 0 **and** at least 7 of *n* units improve | Augmentation provides a practically meaningful and reasonably consistent benefit. |
| **Null** | [*L*, *U*] lies entirely within (−1.0, +1.0) | No practically meaningful effect in either direction. A practically meaningful benefit **and** a practically meaningful harm are both excluded. |
| **Negative** | *d* ≤ −1.0 **and** *U* < 0 | Augmentation causes practically meaningful harm. |
| **Inconclusive** | Interval admits both a ≥1.0 benefit and a ≥1.0 harm, or *n* < 7 | The experiment does not resolve the question at this sample size. |

Statistical significance alone never establishes success. An interval excluding
zero but lying entirely inside (−1.0, +1.0) is a **null** result, not a positive
one.

## 7. What will be reported

- Paired per-seed deltas, all 10, individually.
- The mean paired effect with its 95% interval, and the practical-significance
  verdict from the table above.
- Consistency: how many units improved, how many worsened.
- The Wilcoxon *p*-value, labelled secondary.
- Random-split behaviour is the primary regime. Behaviour under the Phase 9
  chemical OOD regimes is **not** part of this confirmation and will be
  described only as pre-existing development evidence, clearly separated.
- Every degenerate and failed unit, with counts.
- The accepted-synthetic-row count per unit, so the size of the treatment is
  visible alongside its effect.

## 8. Contracts this experiment inherits

Canonical seven-role identity; RDKit featurization; saved grouped splits with
complete group separation; fixed validation and test membership; training-only
augmentation, teacher fitting and representation learning; validation-only
policy selection; policies frozen before outer-test evaluation; exactly one
outer-test evaluation per declared unit; immutable hash-verifiable outputs; and
explicit reporting of negative and null results.

Historical role-aware-v2 and historical unverified LOGO results remain excluded
and are not used here.

## 8a. Candidate-scope scope note (added post hoc, after the result)

**Added 2026-08-25, after this experiment completed. Nothing above was changed;
no number, verdict, or protocol statement is edited. This section records what a
later repository-wide audit established about the *scope* of the question this
experiment answered.**

This experiment's candidate eligibility was `observed_only_low_data`: a
generated reaction was rejected only when its canonical identity occurred in the
rows the phase could actually see — the inner policy-fit subset during search
(507 identities) and the saved training subset during placement and final
evaluation (633 identities) — never the 3,955-row canonical universe. A hash
guard in `redesigned_ae_benchmark.py` hard-fails if a pool consumes any other
key set. The audit confirmed this from the executed bundle's `pool_summary.csv`,
whose `measured_identity_key_scope` column reads only `inner_policy_fit_rows`
and `saved_training_rows`.

**The verdict is therefore unaffected and is not retracted.** The treatment
genuinely applied: every unit had 335 to 362 accepted synthetic rows against 633
measured rows, roughly 55% augmentation.

What the audit does narrow is the *stated reason* offered for the null in §2 and
in the completion report. That reasoning invoked the near-complete factorial —
"only five eligible unmeasured reactions exist" — to explain why augmentation
had little room to act. **That is a global-discovery-headroom argument, and it
does not apply to an observed-only pool.** A near-complete matrix can leave
almost no globally novel chemistry while still leaving most of the matrix hidden
from a low-data learner, which is exactly the regime this experiment ran in and
is why its pools were large rather than empty. The correct reading of the null
is: *anonymous condition transfer supplied a substantial, legitimate, non-empty
low-data augmentation and still did not shift outer-test RMSE by a practically
meaningful amount at training fraction 0.2.*

The distinction, the taxonomy, and the families it does affect are recorded in
`RESULT_STATUS.md` under *Candidate-scope semantics*. A separate new experiment
at a lower training fraction is preregistered in
`PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md`; it does not re-analyse this result.

## 9. Anti-gaming commitments

- The primary metric, margin, seed count, exclusion rule, interval method and
  interpretation table above are fixed as of this commit.
- The confirmatory run writes to a fresh output directory and reserves its
  outer-test claims through the repository-global registry, so a second attempt
  on the same units is refused by construction rather than by discipline.
- If the result is null or negative, it is reported as the finding. It is not
  re-cut by fraction, metric, subgroup, or seed subset in search of a positive.
- Any post hoc analysis performed after seeing the result is labelled post hoc
  and reported separately from the primary analysis.
