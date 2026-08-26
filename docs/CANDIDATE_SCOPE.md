# Candidate scope: which reactions are eligible, and why it depends on the question

Generating a synthetic reaction and deciding whether that reaction is an
*eligible* synthetic target are two different steps. The second one has no
single correct answer: it depends entirely on which scientific question the
experiment is asking. This document is the methodology reference for that
choice.

---

## 1. The two protocols

### Low-data augmentation / retrospective knowledge transfer

Simulate a learner that has observed only a nested low-data labeled subset of a
historical dataset. Ask whether augmenting that learner's training set with
generated, pseudo-labelled reactions improves its predictions.

Under this protocol the learner **cannot know** whether a generated reaction
exists elsewhere in the historical file. That file is the experimenter's
apparatus, not the learner's evidence. Rejecting a candidate because the
experimenter can see it somewhere is a leak of experimenter knowledge into the
simulated learner's decision procedure — and, on a densely measured dataset, it
suppresses essentially the entire treatment.

### Prospective discovery

Propose reactions that have genuinely never been measured, so that running one
in a laboratory would produce new information. Here the complete measured
dataset *is* the relevant universe, and a candidate that duplicates any measured
reaction is worthless by definition.

These two protocols must never share one eligibility rule.

---

## 2. The four partitions of a low-data evaluation unit

Defined once, in `bh_augmentation.evaluation.low_data_partitions`, and never
redefined downstream.

| Partition | Definition | Visible to the learner? |
| --- | --- | --- |
| `labeled_train` | `outer_split == "train"` **and** `included_in_training_subset` | Yes — this is the learner's entire evidence |
| `hidden_outer_train` | `outer_split == "train"` **and not** `included_in_training_subset` | No |
| `validation` | `outer_split == "valid"` | Labels visible to policy selection only |
| `test` | `outer_split == "test"` | No, until one final evaluation |

Candidate generation, donor selection, teacher fitting, similarity computation
and policy construction may use **`labeled_train` information only**.

`hidden_outer_train` is deliberately absent from the data model that reaches
generation: `CandidateScopePolicy` has no field that can carry it. That makes
"the algorithm never consults hidden outer-training membership" a structural
property rather than a convention someone has to remember.

---

## 3. The taxonomy

Declared in `bh_augmentation.augmentation.candidate_scope`, assigned per module
in `candidate_scope_registry.py`, and verified against the code by
`scripts/audit_candidate_scope.py`.

| Label | Rejects a candidate when | Used by |
| --- | --- | --- |
| `observed_only_low_data` | its canonical identity occurs in `labeled_train`, or duplicates another generated candidate | every low-data augmentation-benefit runner |
| `globally_unmeasured_prospective` | its canonical identity occurs **anywhere** in the complete measured dataset | Phase 18 prospective package, recommendation simulation, the Phase 1–3 identity smokes |
| `representation_augmentation` | never — the chemistry is deliberately unchanged, or the generated object has no chemical identity | SMILES randomization, order permutation, latent interpolation, feature GAN, feature-space controls |
| `not_applicable` | no eligibility semantics involved | conduits, diagnostics, protocol infrastructure |

### Quarantine is not eligibility

A candidate whose identity matches a `validation` or `test` reaction is
**quarantined**: recorded in the candidate audit with reason
`quarantined_held_out_identity` and excluded from student training, but never
counted as chemistry the learner has observed. Partition membership is not label
information, and the inductive benchmark requires the exclusion; conflating it
with "already measured" would make the two counts uninterpretable.

Under a group-safe split, `labeled_train` and the quarantine set are disjoint by
construction, and `LowDataEvaluationUnit` asserts that. The older row-level
splitters can place one identity in both; there, observed chemistry outranks
quarantine, the overlap is subtracted, and its size is reported as
`quarantine_overlap_with_observed_count`.

---

## 4. Per-family semantic rules

| Family | Generates new chemistry? | Rule |
| --- | --- | --- |
| Anonymous condition transfer | Yes | caller's declared scope |
| Typed / role-aware condition transfer | Yes | caller's declared scope |
| Condition recombination (+ ensemble filter) | Yes | caller's declared scope; the scope also bounds the generation budget |
| External-family typed transfer | Yes | caller's declared scope |
| SMILES randomization | **No** — RDKit atom renumbering, same molecule, label copied | no identity gate. Canonicalization is an involution over this transform, so an identity gate would reject 100% of the output by definition |
| Reaction/role-order permutation | **No**, while the caller passes same-role columns | no identity gate. The module does not infer chemical exchangeability and refuses an empty column list rather than guessing |
| Exact duplication, oversampling, reweighting | **No** — verbatim measured rows | not applicable; duplicating measured chemistry is the declared intent |
| Nearest-neighbour pseudo-labelling, self-training, feature mixup | **No** — feature-space coordinates | no identity gate; recorded as `chemical_identity_applicable=False` |
| Latent interpolation | **No** — latent coordinate | identity recorded as **null**, not false: the question is not answerable from a latent vector, which is a different statement from answering it "no" |
| Utility-guided feature GAN | **No** — generated feature coordinate | as above; feature-hash deduplication applies, chemical-identity rejection does not |

---

## 5. Why a near-complete dataset still leaves augmentation headroom

The canonical Buchwald–Hartwig dataset is a saturated factorial: 4 ligands × 3
bases × 22 solvents/additives × 15 aryl halides × 1 amine ≈ 3,960 cells, of
which 3,955 are measured.

- **Global discovery headroom ≈ 0.** Exactly five eligible unmeasured reactions
  exist. Any condition transfer assembled from the observed role vocabulary is
  essentially guaranteed to exist somewhere in the file.
- **Low-data transfer headroom is large.** At a 5% training fraction the learner
  has observed 159 of 3,955 reactions. About 96% of the matrix is unknown to it.

The first number says nothing about the second. Applying the global rule to a
low-data experiment collapses the treatment to nothing:

| Fraction | Rule | Generated | Accepted |
| --- | --- | ---: | ---: |
| 0.01 | globally unmeasured | 160 | **0** |
| 0.01 | observed only | 160 | **125** |
| 0.05 | globally unmeasured | 795 | **1** |
| 0.05 | observed only | 795 | **580** |
| 0.10 | globally unmeasured | 1,585 | **0** |
| 0.10 | observed only | 1,585 | **999** |

(anonymous transfer, summed over five development seeds;
`results/corrected_candidate_scope_development_v1`.)

This failure mode is invisible on a sparse dataset, where global rejection
removes few candidates. It is only obvious on a saturated design matrix, where
the gate stops filtering bad chemistry and starts filtering *all* chemistry.

---

## 6. Retrospective withheld-cell reconstruction

A third question, distinct from both protocols above:

> Can condition transfer reconstruct measured reactions that were unavailable to
> the simulated low-data learner?

`bh_augmentation.evaluation.withheld_cell_oracle` answers it with a two-stage
contract:

1. `FrozenCandidateBundle.freeze()` seals the candidate identities and their
   pseudo-labels after generation, stores a hash of each, and **refuses a frame
   that already carries a measured-outcome column**.
2. `withheld_cell_oracle_diagnostic()` is the only function that reads hidden
   outer-training yields. It re-verifies both hashes before the join, returns
   metrics, and mutates nothing.

Because the bundle is sealed and hash-verified first, an oracle metric cannot
become an input to candidate generation, candidate ranking, teacher fitting,
policy selection, hyperparameter selection, synthetic weights, or student
fitting: by the time any hidden yield is legible, everything those stages
consume is already frozen.

Reported per evaluation unit and training fraction: generated and unique
candidate counts, overlap with hidden outer-training rows, overlap fraction and
hidden-cell coverage, pseudo-label MAE, RMSE, Spearman, bias, high-yield
precision/recall/enrichment and top-k enrichment, distance to the nearest
labeled training reaction, and the same metrics stratified by transferred role.

That last one carries a caveat worth stating: the two generators do not share a
distance metric. Anonymous transfer records a cosine distance in [0, 2]; typed
transfer records an unbounded Euclidean distance. The oracle therefore reports
`support_distance_metric` alongside `nearest_labeled_training_distance_*`, and
derives a bounded `nearest_labeled_training_similarity_*` only where the
conversion `1 - distance` is defined. Averaging the two as one "similarity"
produces a number outside [0, 1] and means nothing.

**This diagnostic is secondary and non-selecting.** It measures the mechanism,
not the benchmark, and may not be substituted for a null primary outcome.

---

## 7. Identity vocabulary: a failure mode worth naming

Candidate identities are computed live from roles; a partition's identities are
normally read from the stored `canonical_reaction_key` column. If the installed
RDKit canonicalizes differently from the version that built the dataset, those
two vocabularies stop intersecting — and an eligibility gate comparing them then
rejects *nothing*, including candidates identical to observed training
reactions, with no visible symptom in any metric.

`assert_stored_identities_match_roles` makes that loud. It runs when a low-data
unit is built, when a scope is derived from split frames, and as a preflight
check in `scripts/run_lowdata_candidate_scope_reanalysis.sh`. The canonical
dataset was built with `rdkit==2023.9.6`; the run manifests record the exact
dependency set, `.github/workflows/tests.yml` pins it in both jobs, and
`requirements.txt` documents why.

Pinning CI has a visible cost worth knowing about: the identity-dependent test
modules previously errored at fixture setup and finished instantly. Under the
pin they actually execute, and the Phase 14/15 benchmark module is genuinely
expensive — a single test in it takes over nine minutes, because every isolated
fit spawns a fresh interpreter that re-imports torch, RDKit and scikit-learn.
That is real coverage being paid for, not a regression.

---

## 8. Auditing the repository

```bash
python scripts/audit_candidate_scope.py          # writes the machine-readable audit
python scripts/audit_candidate_scope.py --check  # verify only; nonzero exit on drift
```

The audit walks every call that can reach the canonical identity gate, records
how each supplies its eligibility rule, and fails when a module declared
`observed_only_low_data` is found handing a complete-dataset key set to a
generator. Adding a new generation call site to an undeclared module also fails,
which is what keeps the taxonomy from rotting.
