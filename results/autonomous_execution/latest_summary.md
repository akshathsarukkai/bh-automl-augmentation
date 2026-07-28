# Autonomous Execution Summary

Updated: 2026-07-27T07:04:26Z

The resumable 18-phase ledger was initialized from verified Batch 3 state.
Batch 3 and Phases 1–3 are checkpointed locally; Phase 4 has passed its gate
and is awaiting its local checkpoint.

Current status: Phase 1 passed and was checkpointed locally at
`5268df37a39ba443fdf26ee15dc0a09d3bcb59ae`.

Phase 1 validation:

- Focused tests: 108 passed (3 deprecation warnings)
- Full suite: 365 passed (34 warnings)
- Ruff and `git diff --check`: passed
- Canonical integration smoke: 288 candidates deterministically regenerated;
  286 rejected as measured and 2 as source-identical; zero accepted
- Artifact hashes: validated

The zero-accepted smoke result is retained as a scientifically relevant
property of the densely crossed canonical HTE subset. Separate focused tests
exercise valid accepted candidates, chemical duplicates, and feature
collisions.

Current status: Phase 2 passed and was checkpointed locally at
`4c9f9c7497d0cc19927a3689841de91b5525d0ca`.

Phase 2 validation:

- Focused tests: 77 passed (1 pandas warning)
- Full suite: 384 passed (35 warnings)
- Ruff and `git diff --check`: passed
- Canonical integration smoke: strict anonymous transfer correctly produced
  zero candidates because catalyst is invariant; strict typed `reject` produced
  zero; declared typed `random` fallback produced 2 auditable candidates
- Exact all/any role diffs, no-unrequested-change assertions, and single-step
  fallback behavior are covered in shared and runner-level regression tests
- Artifact hashes: validated

Scientific limitation: a strict anonymous full-condition policy cannot operate
on this canonical HTE subset while catalyst is a requested invariant role. This
is retained as a null eligibility result, not converted into a hidden fallback.

Current status: Phase 3 passed and was checkpointed locally at
`1bbfa0ebd08f514e2e1844687f944f3c46352adb`.

Phase 3 validation:

- Focused tests: 131 passed (5 warnings)
- Full suite: 414 passed (35 warnings)
- Ruff and `git diff --check`: passed
- Canonical integration smoke: 192 proposals, exact maximum of 2 per source,
  input-permutation-invariant audit records, and zero proposals under an
  impossible configured threshold
- Artifact hashes: validated

The dense canonical HTE grid yielded zero accepted chemical candidates after
identity rejection; focused fixtures cover nonempty ranking and selection.
Calibrated uncertainty is deliberately null pending Phase 12, while transparent
deterministic proxy ranks are labeled accordingly. Nonchemical interpolation
and GAN controls report component chemical similarities as null.

Current status: Phase 4 passed and was checkpointed locally at
`e601b1445cf08394d4ec2408c82ec3ac10bd3f04`.

Phase 4 validation:

- Focused tests: 85 passed
- Full suite: 442 passed (35 warnings)
- Ruff and `git diff --check`: passed
- Canonical integration smoke: real-only, anonymous, role-aware, and hybrid
  runners loaded seed 0 at 1% training data from the same saved assignments
- All four split-audit CSVs are byte-identical and record aggregate hash
  `cb1d85ab...`, seed hash `3c004950...`, and fixed validation/test ID hashes
- All 44 runner output hashes recorded by the smoke manifest were revalidated

Scientific result: strict anonymous and typed transfer had no eligible nonzero
policy in this dense canonical slice, so the runners recorded exclusions and
did not evaluate augmentation on the outer test. The hybrid completed its
real-only AE control and selected no hybrid policy. This null eligibility result
was not replaced by fallback generation.

Current status: Phase 5 passed and was checkpointed locally at
`0e3013887c9701fece8e052ecb1f3f6624f4367a`.

Phase 5 validation:

- Focused tests: 89 passed
- Full suite: 463 passed (35 warnings)
- Ruff and `git diff --check`: passed
- Production integration smoke: 8 inner-validation metric rows, one frozen
  policy, zero outer-test accesses during search, 4 final metric rows from one
  outer-test prediction batch
- All 5 scientific smoke outputs plus the exactly-once evaluation claim were
  independently hash-validated
- Anonymous and role-aware transfer both have end-to-end regression tests
  proving validation-only search, frozen refit, one test prediction batch, and
  propagation of the full measured canonical-key exclusion set

The smoke is a protocol check, not evidence that transfer improves yield
prediction. AE hybrid selection remains development-only until Phase 6 repairs
its internal validation and joint selection.

Current status: Phase 6 passed and was checkpointed locally at
`fc5b59aaae917e20594dff27abf7f48d58644248`.

Phase 6 validation:

- Focused tests: 78 passed
- Full suite: 492 passed (35 warnings)
- Ruff and `git diff --check`: passed
- Canonical integration smoke: 32 eligible measured training rows were split
  once into 26 AE-fit and 6 untouched real internal-validation rows
- Three full joint tuples were declared; two strict anonymous tuples had zero
  accepted synthetic rows and were explicitly excluded
- The eligible real-only tuple was frozen, refit on all 32 eligible measured
  rows with no validation reuse, and evaluated through one globally claimed
  outer-test attempt and one prediction batch
- No measured-refit, synthetic-parent, canonical validation, or test overlap
  was detected; all 11 scientific outputs were independently rehashed

The Phase 6 smoke is a leakage/refit protocol result, not evidence of
augmentation efficacy or AE superiority. Accepted-synthetic production
fixtures exercise the complete transfer→AE→downstream path and exact synthetic
identity manifests. The legacy combined runner remains development-only, and
its test-derived oracle is explicitly marked
`post_hoc_test_oracle_not_for_selection`.

Current status: Phase 7 passed and was checkpointed locally at
`f4134eff438f3465656aef4dbcea30ec202deba2`.

Phase 7 validation:

- Focused tests: 66 passed (2 expected/deprecation warnings)
- Full suite: 531 passed (36 warnings)
- Ruff and `git diff --check`: passed
- Canonical product LOGO: all 5 product groups held out exactly once, 10 fixed
  Ridge metric rows, aggregate assignment hash `705a4474...`
- Canonical reactant LOGO: all 15 substrate groups held out exactly once, 30
  fixed Ridge metric rows, aggregate assignment hash `5d6ad098...`
- Every fold covers all 3,955 rows exactly once as train or test; source,
  canonical group, and canonical reaction-key overlaps are zero
- Both target manifests and all 12 completion-bound outputs rehash exactly;
  the root completion manifest was written only after both targets completed
- Tracked historical stress LOGO outputs now reject default loading and remain
  available only through explicit invalid-result inspection

The logical targets use canonical identity columns:
`product_key -> canonical_product_key` and
`reactant_key -> canonical_substrate_key`. No model or representation is
selected from LOGO test metrics in Phase 7.

Canonical dependencies detected:

- `data/processed/bh_canonical_roles_v1.csv`
- `results/corrected_canonical_splits/split_manifest.json`

No external blockers are currently active.

Current status: Phase 8 is in progress from checkpoint
`f4134eff438f3465656aef4dbcea30ec202deba2`.

Phase 8 plan:

- Build deterministic nested leave-one-group-out contracts for each outer fold.
- Evaluate every declared policy across every inner OOD fold using inner labels only.
- Support explicit sample-weighted and group-weighted policy summaries.
- Freeze the deterministic inner-validation winner, refit on all outer-training
  groups, and evaluate the outer group once.
- Save assignment hashes and fold-level manifests sufficient to reproduce every
  outer and inner partition exactly.

Phase 8 repair status:

- The first canonical product-group search produced five frozen outer units,
  but remains preliminary and excluded from gate evidence.
- Adversarial review demonstrated that consistently rehashed artifacts could
  switch frozen selection semantics or truthfully report forbidden outer-label
  access without final rejection.
- The first exact-once registry was tied to the search artifact directory and
  could be bypassed by copying that bundle; multi-fold interruption was also
  not safely distinguishable from completion.
- Repairs are enforcing search-data roles, structurally isolating each outer
  search, replaying the configured policy budget and aggregation protocol,
  semantically validating compact artifacts, and using a repository-global
  identity-addressed per-unit evaluation registry.

Current status: Phase 8 passed and was checkpointed locally at
`fbbe2bf0811f6e8bd58b7e443ea68a52f2aef517`.

Phase 8 validation:

- Focused tests: 111 passed (1 expected singleton-R2 warning)
- Full suite: 581 passed (36 warnings)
- Ruff and `git diff --check`: passed
- Canonical product nested OOD: 5 outer groups, 20 group-disjoint inner folds,
  80 complete inner metric rows, and 40 explicit group/sample weighted rows
- Frozen model-policy selection was replayed from persisted inner rows before
  any outer access; every compact table and fold manifest was semantically
  rebuilt from canonical identities
- Final evaluation produced one prediction batch per outer group and 10 metric
  rows; a second final invocation reused all five repository-global records
  with zero new prediction batches
- All 10 recorded root artifacts, every fold manifest, and every registry
  record rehash exactly

Scope limitation: this gate establishes nested OOD protocol and real-only
model-policy selection evidence on canonical product groups. It does not claim
typed-transfer OOD benefit, reactant-key Phase 8 empirical execution, or
unbiased confirmatory evidence.

Current status: Phase 9 is in progress from checkpoint
`fbbe2bf0811f6e8bd58b7e443ea68a52f2aef517`.

Phase 9 gate validation:

- Focused tests: 44 passed
- Full suite: 625 passed (36 warnings)
- Ruff and `git diff --check`: passed
- Authoritative canonical output:
  `results/autonomous_execution/phase_09/corrected-20260726-fbbe2bf-phase9-v2`
- Seven ordered split families produced 296 fold records: 289 viable and
  seven explicitly excluded. Every viable fold has complete nearest-training
  similarity evidence under one canonical seven-role fingerprint contract.
- The maximum-similarity-bounded split satisfies its configured 0.95 ceiling.
- Independent validation rebuilds canonical source/key mappings, scaffold,
  cluster, condition, ligand, base, bounded-similarity assignments,
  fingerprints, nearest neighbors, lexical ties, and similarity quantiles.
- Six undersupported reaction-fingerprint clusters are excluded with their
  true attempted train/test sizes. Amine scaffold OOD is excluded because only
  one direct scaffold exists.

Scientific limitation: electrophile scaffold OOD contains only two broad
cores (benzene and pyridine), so it is retained as limited descriptive
fixed-policy evidence and not represented as broad or nested scaffold
generalization evidence.

Current status: Phase 9 passed and was checkpointed locally at
`596e65801414c5e77fc63cd59b048bd9e1a12a83`.

Current status: Phase 10 passed and was checkpointed locally at
`5c2262fef524ae633e00baf9bb224eb4c0800490`.

Phase 10 gate validation:

- Focused tests: 105 passed (8 expected legacy-alias warnings)
- Full suite: 657 passed (36 warnings)
- Ruff and `git diff --check`: passed
- Small canonical integration smoke:
  `results/autonomous_execution/phase_10/corrected-20260726-596e658-phase10-v4-smoke`
- Authoritative production evidence:
  `results/autonomous_execution/phase_10/corrected-20260726-596e658-phase10-production-v3`
- Production uses 2048-bit RDKit role fingerprints, fixed 100-tree XGBoost,
  five random seeds at each of five nested fractions, all five product LOGO
  folds, all 15 reactant LOGO folds, and the primary maximum-similarity OOD
  split.
- All eight representations are paired over 46 immutable units, yielding
  148,808 source-level predictions and 736 metric rows. No representation
  selection occurs.
- Planning reads identity columns only and freezes split, feature, model, and
  config hashes before labels load. Saved Phase 7 LOGO and Phase 9 chemical
  OOD assignments are independently replayed.
- Validation requires exact test membership and recomputes every metric from
  canonical outcomes and persisted predictions, then rebuilds every summary.
  Production-v2 and production-v3 scientific artifacts are byte-identical.

Scientific result: the complete reaction representations are close and their
small RMSE differences are inconsistent across random and OOD evidence. There
is no supported representation-superiority conclusion. Product-free is
slightly better on the random means but slightly worse on product/reactant
LOGO and bounded-similarity OOD. Substrate-only and condition-only are
substantially worse. `product_aware` is an intentional exact duplicate control
of seven-role blocks and is not counted as independent evidence.

Current status: Phase 11 passed and was checkpointed locally at
`a39a3ebfc391b2d5f9bdf50979ccc5b5bc9ce51e`.

Phase 11 gate validation:

- Focused tests: 139 passed (1 pandas warning)
- Full suite: 724 passed (36 warnings)
- Ruff and `git diff --check`: passed
- Small canonical integration smoke:
  `results/autonomous_execution/phase_11/corrected-20260726-5c2262f-phase11-smoke-v11`
- Authoritative production evidence:
  `results/autonomous_execution/phase_11/corrected-20260726-5c2262f-phase11-production-v2`
- The production matrix pairs all 13 predefined controls over the same five
  canonical saved-split units, fixed 5% training fraction, 2048-bit RDKit
  features, 100-tree XGBoost protocol, and nominal one-for-one augmentation
  budget. It persists 25,740 source-level predictions and 130 metric rows.
- Exact duplication, random oversampling, yield-stratified oversampling,
  sample reweighting, nearest-neighbor pseudo-labeling, self-training, and
  feature mixup consume their declared budgets exactly.
- Strict chemical identity/change rules permit only 2 anonymous, 3 random
  typed, and 6 context-matched typed candidates across 795 nominal additions;
  the remaining budget is explicitly underfilled without fallback or backfill.

Scientific result: at this single development fraction, no generic control
improves mean RMSE over real-only. Anonymous transfer is 0.025 RMSE better on
average but adds only two rows and wins two of five seeds; this is not evidence
of transfer benefit. Typed controls are budget-underfilled and are not
credited with gains.

Independent adversarial review found no Phase 11 gate blocker. Every accepted
candidate is canonical, role-exact `all`/`reject`, fallback-free,
train-parent-only, and free of measured, source, canonical-key, and feature
duplicates. The filtered and unfiltered typed controls share exact prefilter
pools and accepted sets.

Current status: Phase 12 passed and was checkpointed locally at
`3fff5baf284d130b09557064f65a0f99b68a01ca`.

Phase 12 validation:

- Focused tests: 177 passed (7 known warnings)
- Full suite: 793 passed (36 known warnings)
- Ruff and `git diff --check`: passed
- Two fresh end-to-end smokes independently validated and produced identical
  hashes for all 14 scientific artifacts
- Production calibration: 30 saved split/fraction/condition units, six
  uncertainty methods, 70,860 calibration predictions, 224/224 exactly
  reconstructed hidden unit-targets, and 1,344 hidden method predictions
- Deterministic validator refitting grounded estimator audits and all persisted
  prediction fields; five fully rehashed oracle/provenance attacks are covered
  by regression tests
- Frozen validation policies selected bootstrap ExtraTrees in 26 units and
  heterogeneous disagreement in four units

Scientific result: validation-selected methods achieved hidden-measured RMSE
9.067, MAE 6.251, and 90.6% marginal coverage for nominal 90% intervals. The
validation-frozen filter accepted 114/224 targets; accepted targets had
descriptive RMSE 5.349 versus 11.737 for rejected targets. This supports use of
validation-calibrated uncertainty as a filtering signal in this exploratory
retrospective experiment, not a causal or prospective claim.

Limitations: accepted-target interval coverage was 84.2%, so marginal interval
calibration does not transfer conditionally after filtering. The 224 unit-target
records contain 163 unique measured rows, per-unit target counts are 2–15, and
role/donor/context cohorts overlap and are descriptive only. Prospective
laboratory validation has not been performed.

Authoritative outputs:

- `results/autonomous_execution/phase_12/corrected-20260726-a39a3eb-phase12-smoke-v12-a`
- `results/autonomous_execution/phase_12/corrected-20260726-a39a3eb-phase12-smoke-v12-b`
- `results/autonomous_execution/phase_12/corrected-20260726-a39a3eb-phase12-production-v1`

Current status: Phase 13 is in progress from checkpoint
`3fff5baf284d130b09557064f65a0f99b68a01ca`.

Phase 13 plan:

- Compare SVD, a linear autoencoder, PLS, sparse feature selection, a compact
  bottleneck MLP, and a parameter-matched direct MLP under the shared frozen
  search/final-evaluation protocol.
- Fit representations only on eligible outer-training rows and select
  hyperparameters from inner validation without outer-test access.
- Record exact parameter counts, latent width, wall time, peak memory,
  selection metrics, test metrics, split hashes, and feature metadata.
- Use the canonical saved nested assignments and refuse incompatible hashes.

Phase 13 gate validation:

- Focused tests: 46 passed
- Full suite: 839 passed (36 known warnings)
- Ruff and `git diff --check`: passed
- Two fresh end-to-end smokes independently validated and produced
  byte-identical hashes for all 12 deterministic scientific artifacts
- Production benchmark: 10 saved split/fraction units, six method families,
  120 validation candidates, 60 frozen within-family policies, 23,760 final
  predictions, and 240 final metric rows
- All 120 search fits and 60 final refits used exactly their saved training
  subsets, with zero saved-validation or outer-test overlap
- Every unit/method was claimed before test access and evaluated in exactly
  one final test-prediction batch
- An independent full validator replay reproduced every fit state,
  prediction, selection, metric, and output hash

Development result: at 20% training, direct and bottleneck MLP mean RMSEs were
13.596 and 13.647; at full training, they were 10.196 and 9.977. The linear
autoencoder, SVD, PLS-latent-plus-Ridge, and sparse-selection baselines were
worse in this paired random-split benchmark. Every method remains reported;
outer-test metrics did not select a representation or setting. Phase 14 must
decide supervised-AE benchmark status under predefined validation-selected
comparisons.

Limitations: this phase covers five seeds at fractions 0.2 and 1.0 on random
canonical splits only, and the width search is limited to 8 and 16. Results
are descriptive development evidence. Resource measurements are observational
upper bounds over isolated fit/transform/predict workers, and evaluation
claims are run-local.

Authoritative outputs:

- `results/autonomous_execution/phase_13/corrected-20260726-3fff5ba-phase13-smoke-v1-a`
- `results/autonomous_execution/phase_13/corrected-20260726-3fff5ba-phase13-smoke-v1-b`
- `results/autonomous_execution/phase_13/corrected-20260726-3fff5ba-phase13-production-v1`

Current status: Phase 13 passed and was checkpointed locally at
`58613dd0907602ce295958ca81aa70b9ed618d54`.

## Handoff reconciliation

Phase 13 passed its gate and was committed at
`58613dd0907602ce295958ca81aa70b9ed618d54`, but the ledger updates recording
that fact were never checkpointed. Two bookkeeping defects were found and
repaired without rewriting any historical event:

1. `phases[13].git_commit_at_end` was `null` even though `events.jsonl`
   already recorded the `phase_passed` event with the correct ending commit.
2. Phases 9 through 13 wrote `targeted_tests` and `full_suite_result` without
   the `status` key that `_passed_phase_issues` requires. Under the repository's
   own verifier this silently reopened every phase from 9 onward. The recorded
   `phase_gate_validated` events establish that all ten results did pass, so
   the missing key was restored rather than the phases being rerun.

Reconciliation evidence:

- `verify_passed_phases` now reports zero issues across phases 1-13.
- All five canonical dependency hashes still match `state.json`
  (`dataset`, `split_manifest`, `outer_assignments`, `low_data_assignments`,
  and the aggregate `split_hash`).
- Every phase 1-13 ending commit exists in `git log` on `main`.
- The Phase 13 production bundle was independently replayed with
  `validate_low_complexity_benchmark`, not merely checksum-compared.

Phase 14 was already started by the previous session and left mid-flight. Its
implementation modules and configs exist but were never committed, and its
three output bundles are retained on disk and marked invalid in place:

- `phase_14/corrected-20260726-58613dd-phase14-smoke-v1-a` and `-b` are
  pre-hardening smokes that both evaluated the same outer-test unit.
- `phase_14/corrected-20260726-58613dd-phase14-production-v1` was interrupted
  before any outer-test access and has no manifest.

None of the three are Phase 14 evidence. Phase 14 restarts from fresh output
directories against the hardened implementation.

Current status: Phase 14 is in progress from checkpoint
`58613dd0907602ce295958ca81aa70b9ed618d54`.

Phase 14 plan:

- Implement compact supervised-autoencoder architectures for canonical input
  widths with exact parameter accounting and measured-only target scaling.
- Add synthetic supervised and reconstruction weighting, fingerprint-aware
  reconstruction objectives, role balancing, masking/denoising, and separate
  positive-bit and zero-bit reconstruction diagnostics.
- Compare the redesigned AE against predefined SVD, linear bottleneck, direct
  XGBoost, direct MLP, anonymous transfer without AE, and typed transfer
  without AE controls using validation-only policy selection.
- Retain the AE as a primary benchmark only if its validation-selected policy
  consistently beats the predefined simple controls; otherwise classify it
  as a secondary ablation without weakening the gate.
