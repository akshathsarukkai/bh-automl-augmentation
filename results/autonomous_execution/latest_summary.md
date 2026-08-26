# Autonomous Execution Summary

Updated: 2026-07-29T05:10:00Z

The resumable 18-phase ledger was initialized from verified Batch 3 state.
All 18 phases are passed and checkpointed locally.
Phase 16's empirical arm is
externally blocked; see `blockers.md`.

This file is append-oriented: each phase's section reflects what was known when
that phase closed. The most recent sections are at the end.

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

Phase 14 pre-run review (no outer-test access has occurred):

The inherited implementation was reviewed adversarially before being run.
The protocol backbone is sound — search, then placement on the untouched
saved validation split, then freeze, then retention, then exactly one
outer-test read per unit and family; non-transductivity is enforced by
refitting for the test batch and requiring an identical state hash; AE epoch
selection and target scaling are measured-only on a group-disjoint
train-derived split; and the selection, partition, and metric replays
re-derive their quantities rather than trusting written flags.

Four defects would have compromised the scientific claim and are being
repaired before the run:

1. The AE family received 11 candidate policies per unit while every control
   received 2. Because within-family selection is a minimum over inner
   validation RMSE, this favours the AE by winner's curse alone.
2. Only the AE family could choose its data protocol. The parameter-matched
   direct MLP was forced to real-only data, so any AE win would conflate
   architecture with access to the teacher-labelled synthetic pool.
3. The outer-test evaluation identity embedded the git commit and float
   placement metrics, so an unrelated commit or a one-ULP metric difference
   would have minted a fresh claim and permitted a second outer-test read.
4. The runner could not redirect the shared evaluation registry, so the same
   config could never be run twice and determinism was not demonstrable.

Measured cost: one AE fit takes 21.7 s at fraction 0.2 and 286.3 s at
fraction 1.0 on real-only rows at hidden width 128; synthetic augmentation
roughly doubles the training rows.

Current status: Phase 14 implementation hardening in progress from commit
`71b80ca`. No Phase 14 outer-test outcome has been read.

## Phase 16 — external typed-reaction dataset adapter

Passed and checkpointed locally at
`a173ea12a52f78c126d461ea4e188e876dbb9b58`.

    implementation passed
    external empirical validation blocked

Committed ahead of Phase 14 because its empirical arm is externally blocked and
its implementation is independent of the Phase 14 outcome, which the roadmap's
ordering rule permits. Phase 14's production benchmark was running concurrently.

Scientific question: can the canonical typed-reaction, transfer and evaluation
framework be applied to a chemically distinct reaction family without
redesigning the system around that dataset?

Implementation answer: yes, and the seam is one object.
`ReactionFamilyAdapter` holds the family id, the ordered typed role names, the
role-to-column mapping, the substrate/condition/product partition, the
transferable condition subset, the family-scoped canonicalization and
reaction-key schema versions, the reaction-SMILES assembly order, a required
`DatasetProvenance` record, and a family-specific eligibility hook. Everything
else is reused unchanged: RDKit canonicalization, stable identity encodings,
grouped split construction with complete group separation, validation-only
policy search with single-shot outer-test evaluation, metrics, hashing and
manifests. Two tests assert the reuse positively — the splitter is asserted to
be the same object the Buchwald-Hartwig path uses, and the policy-protocol
source is asserted to contain no family vocabulary.

The Buchwald-Hartwig adapter is constructed from the existing constants and is
asserted to reproduce the current canonicalization key-for-key and hash-for-hash,
so it is definitionally today's behaviour. No existing file was modified.

Typed Suzuki-Miyaura roles: `organohalide`, `organoboron`, `catalyst`, `ligand`,
`base`, `solvent_or_additive`, `product`. Buchwald-Hartwig terminology was not
reused where the chemistry differs — there is no nitrogen nucleophile, and
"aryl halide" would wrongly exclude heteroaryl and alkenyl electrophiles and
triflate/tosylate/mesylate pseudohalides. The boron reagent is classified as a
substrate rather than a transferable condition because it contributes the carbon
fragment that ends up in the product skeleton, so transferring it would fabricate
a different reaction and attach a donor's yield to it.

Gate: 75 focused tests, full suite 1049 passed, ruff clean, and an end-to-end
search plus final-evaluation smoke in a temporary directory asserting no
validation or test leakage into fitting, policies frozen before test access, one
test evaluation per unit, and hash-verifiable outputs. On the fixture, typed
transfer accepted 33 candidates and rejected 37 as already-measured, so the
duplicate-rejection path is genuinely exercised.

Blocked-arm statement: no Suzuki-Miyaura dataset exists anywhere in this
repository and nothing here downloads one. Every Suzuki artifact is exercised
against a clearly labelled synthetic fixture whose yields are a deterministic
function of the row index. No empirical Suzuki result is claimed, produced, or
implied. `docs/EXTERNAL_DATASETS.md` records candidate datasets, acquisition
steps, required columns, mandatory provenance fields and exact run commands,
with every URL, DOI and license marked verify-before-use rather than asserted.

Authoritative outputs:

- `docs/EXTERNAL_DATASETS.md`
- `configs/suzuki_external_validation.yaml`
- `src/bh_augmentation/data/reaction_family.py`

## Phase 18 — sequential experiment selection

Passed and checkpointed locally at `085649cb428a` (see state.json for the full
hash). Committed ahead of Phases 14, 15 and 17 because it does not depend on
the Phase 14 outcome and deliberately excludes the supervised autoencoder,
whose status was still being decided by a concurrently running benchmark.

Scientific question: does the validated prediction system select useful
experiments more efficiently than realistic baselines?

Model, chosen from evidence rather than preference: `bootstrap_extra_trees`
with split-conformal intervals at 0.9 coverage. Phase 12 recorded only two
estimators passing calibrated selection and chose this one in 26 of its 30
frozen policies. The config validator hard-rejects any other estimator, and the
supervised autoencoder is never imported.

Protocol: at round t an acquisition function sees only the seed pool and labels
acquired strictly before t; pool outcomes are loaded only after every campaign
completes. Seed pool, candidate pool, rounds, batch size and total budget are
identical across all seven strategies. The random baseline is a complete
256-replicate distribution, not one trajectory. The manifest records
`future_label_access_events = 0`.

Result: informed acquisition beats random by a practically meaningful margin,
but the margin lives in how much useful chemistry a fixed budget buys, not in
whether the single best reaction is eventually found. Over 320 experiments,
diversity-aware and greedy returned about 4.3 times the high-yield hits (229.8
and 222.2 against 51.6), about 2.2 times the mean acquired yield (73.6 and 72.3
against 33.2), and recovered roughly 80% of the true top-50 against 13% for
random — at the 100th percentile of the random distribution in every evaluation
unit, and the 0th percentile for cumulative regret.

Null and negative findings, retained rather than suppressed:

- `best_yield_discovered` improves only 99.9 against 98.4 and reaches only the
  76th to 93rd percentile. Random with 320 draws from a dense 3164-cell grid
  already finds near-maximal yield. This metric is a weak discriminator on this
  dataset and must not be quoted as evidence.
- `unique_substrate_keys` is uninformative: only 15 substrate pairs exist and
  every strategy saturates at 15.
- Condition diversity is a negative result. Every informed strategy acquires
  fewer unique condition blocks than random (139-197 against 207), at the 0th
  percentile in all units. Exploiting predicted yield narrows the explored
  condition space. Only the support-distance policy beats random on support
  coverage, and it pays for that with much worse yield outcomes.
- Conformal coverage degrades on acquisition-selected batches: 0.82 empirical
  against 0.90 nominal for the exploitative policies, because a selected batch
  is not exchangeable.

Structural finding that constrains the whole project: the eligible prospective
candidate space contains exactly five reactions. The canonical dataset is a
near-complete factorial — 15 canonical substrate groups times 264 canonical
condition groups is 3960 cells, of which 3955 are measured. This was confirmed
independently. It also explains the recurring zero-accepted-candidate null
results recorded in Phases 1 through 4: there is almost no novel chemistry to
propose by recombining measured role values. The candidate list was not padded,
because lengthening it would require inventing molecules outside the measured
inventory or relaxing chemical eligibility.

The prospective package carries seven explicit canonical roles per candidate,
per-role provenance, calibrated intervals reported unclipped so the coverage
guarantee is preserved, support distances, 24 held-out measured controls whose
observed interval coverage was 0.918 against a nominal 0.900, diversity and
uncertainty rationales, and a protocol draft that lists what the canonical
dataset cannot supply rather than inventing it: temperature, time,
stoichiometry, concentration, scale, atmosphere, work-up and analytical method.

    Prospective laboratory validation has not been performed.

Every candidate's practical accessibility is recorded as unknown. No claim is
made about synthesis feasibility, safety, cost or likelihood of laboratory
success.

Limitation: the 256-bit representation matches Phase 12, whose calibration
evidence this reuses, rather than Phase 13's 2048-bit benchmark. The acquisition
policies are therefore driven by a weaker predictor than the repository's best
validated one, which makes the reported margins conservative rather than
inflated. Whether they would grow with a stronger model is untested.

Authoritative outputs:

- `results/autonomous_execution/phase_18/corrected-20260727-phase18-production-v1`
- `results/autonomous_execution/phase_18/corrected-20260728-phase18-prospective-package-v1`

## Phase 17 — reproducibility and tamper-evidence

Passed and checkpointed locally at `07479aeb9d09` (full hash in state.json).
Committed ahead of Phases 14 and 15 because it is infrastructure, asserts no
Phase 14 outcome, and the Phase 14 production benchmark was running.

The phase consolidated the existing manifest, config, claim and hashing
machinery rather than layering a second system beside it.

Security finding, real and fixed: the outer-test evaluation registry root was
declared as a working-directory-relative path in both `nested_ood_final.py`
and `redesigned_ae_benchmark.py`. Running a scientific runner from any other
directory therefore created a fresh, empty registry and silently permitted a
completed outer-test identity to be evaluated a second time — defeating the
repository-global property the registry exists to provide. The root is now
anchored to the checkout containing `pyproject.toml`, with no
environment-variable override, because an override would itself be a bypass.
`redesigned_ae_benchmark.py` still carries the old constant because its
benchmark was executing; that run was launched with an explicit absolute
registry directory, and the one-line fix must be applied once Phase 14 lands.

Tests prove that relocating or deleting a result directory does not release a
claim, that completed and failed identities cannot be re-reserved, and that the
registry exposes no delete, release, reset or unclaim operation. One residual
limit is documented rather than papered over: deleting a record file from the
filesystem does release its claim, which is not addressable in-process and is
why `state.json` remains the root of trust.

Clean-checkout reproduction, actually executed: two `git worktree` checkouts
and one `git archive` export at `851d64e`, nothing installed globally, running
the real canonical audit, the real group-safe split builder and the real Phase
13 benchmark on committed fixture rows. Twelve of fifteen benchmark artifacts
were byte-identical between independent checkouts, including every plan,
policy, claim, prediction and metric file. The three that differed are resource
timings and the manifests that hash them. No predicted value, metric, split,
policy or claim differed.

What cannot be reproduced is stated plainly rather than glossed: the canonical
dataset and the canonical split directory are gitignored and absent from a
clean checkout, so the full-scale Phase 10-18 bundles are not directly
reproducible. `docs/REPRODUCIBILITY.md` gives the regeneration commands and
pins the expected SHA-256 of each from `state.json`, and records that the
full-scale regeneration was not executed rather than implying it was observed.
Bit-identical results across different hardware are explicitly not claimed.

The reproduction exercise found three real defects, all fixed: the shared
config validator imported an uncommitted module at import scope and so could
not be imported from a clean checkout at all; absolute paths leaked into
`config_hash` and `plan_hash`; and `git archive` exports cannot run the
scientific runners, which correctly refuse without a resolvable git commit.

The Phase 13 production bundle was re-validated by full replay against the
consolidated runner and still validates at manifest hash
`f58ffad9c9d974ac4539b5bba515e3cc42cf6fd2b7f1ad941ec8b43630e033fa`.

`RESULT_STATUS.md` now classifies every method and result family. The
confirmatory-evidence class is explicitly empty because Phase 15 has not run,
and Phase 14 is marked in progress with no result asserted in either direction;
a test enforces that.

Gate: 85 focused tests, full suite 1164 passed, ruff clean, 28 covered
config-validation rejections, and a CI job that proves a one-byte mutation of a
completed artifact is detected.

Authoritative outputs:

- `docs/REPRODUCIBILITY.md`
- `RESULT_STATUS.md`
- `src/bh_augmentation/utils/strict_config.py`
- `src/bh_augmentation/utils/scientific_manifest.py`
- `.github/workflows/tests.yml`

## Phase 14 — supervised autoencoder reassessment

Passed and checkpointed locally at `6a618ed139c8` (full hash in state.json).

Scientific question: under the corrected split, scaling, selection and
evaluation protocols, does a compact supervised autoencoder add reproducible
value beyond simpler representation-learning and prediction baselines?

**Answer: no. The AE is classified as a secondary ablation and is not carried
into the primary confirmatory hypothesis.**

The retention decision used placement-validation RMSE only, against a criterion
fixed in the config before any outer-test outcome existed, and was frozen before
the first outer-test read. All five criteria failed:

| Fraction | Mean delta | Median delta | Margin wins | Practical losses |
| --- | --- | --- | --- | --- |
| 0.2 | -0.999 | -1.099 | 0 of 5 | 4 of 5 |
| 1.0 | -0.453 | -0.237 | 0 of 5 | 1 of 5 |

Delta is best-control RMSE minus AE RMSE, so negative means the AE is worse.
The pooled seed-cluster bootstrap gives mean -0.726 with 95% CI
[-1.101, -0.395], excluding zero on the wrong side. The AE lost to the best
simple control in 9 of the 10 evaluation units.

The negative result is conservative because the protocol was tilted toward the
AE. Per-unit search budgets were AE 11, matched direct MLP 10, each transfer
control 6, truncated SVD 6, linear autoencoder 6, direct XGBoost 3. Within-family
selection is a minimum over inner-validation RMSE, so the family with the largest
budget benefits most from winner's curse. The AE had the largest budget and still
lost. The same asymmetry would have made a positive result untrustworthy.

Diagnostics beyond aggregate loss explain the outcome. Aggregate reconstruction
MSE looks excellent — 0.0034 at fraction 0.2 and 0.0006 at 1.0 — but it is
dominated by 5.6 million zero bits against only 65 thousand positive bits. On
the chemically meaningful set bits, reconstruction is 13 to 32 times worse:
positive-bit MSE 0.0799 against zero-bit MSE 0.0025 at fraction 0.2, and 0.0066
against 0.0005 at fraction 1.0. The decoder is largely learning to emit zeros.
An aggregate-loss-only view would have made this representation look healthy.

Validation selection was unstable: six different AE candidates won across the
ten units, with no configuration reliably best.

The AE is also the most expensive family. Mean final-fit cost was 136.7 s and
1.24 GB peak RSS increment, against 57.3 s and 0.52 GB for the parameter-matched
direct MLP and 12.0 s and 0.81 GB for direct XGBoost.

Outer-test metrics, reported for completeness and used for none of the above:
at fraction 0.2 the AE ranks fifth of seven families (10.758 against 10.461 for
the best); at fraction 1.0 it ranks second behind the matched direct MLP (6.096
against 5.994). The paired AE-minus-MLP gap is small and changes sign across
seeds (+0.249 and +0.103 mean). Truncated SVD and the linear autoencoder are far
worse everywhere at roughly 15.7 to 16.1 RMSE, consistent with Phase 13.

Fairness and protocol repairs made before the run, after adversarial review:
controls received real hyperparameter grids; the parameter-matched MLP now
selects its data protocol exactly like the AE, so an AE win could not be an
artifact of the AE alone reaching the synthetic pool; the outer-test evaluation
identity no longer embeds the git commit, float placement metrics, or which
policy won the search; transfer pools no longer consume canonical identity keys
from validation or test rows; the retention bootstrap now tests against the
practical margin rather than zero; a zero-row transfer pool is a loud failure
rather than a silent degradation to plain XGBoost; and structural validation
runs before the outer test is consumed.

Gate: 480 candidate policies over the 10 Phase 13 canonical random units, two
determinism smokes with 18 of 20 byte-identical artifacts, independent replay
validation of the production bundle, full suite 1164 passed, ruff clean, and
`git diff --check` clean. Exactly one outer-test evaluation per unit and family,
70 evaluation claims all created after the retention freeze, and
`test_used_for_selection_or_retention` recorded false.

Authoritative outputs:

- `results/autonomous_execution/phase_14/corrected-20260727-71b80ca-phase14-smoke-v2-a`
- `results/autonomous_execution/phase_14/corrected-20260727-71b80ca-phase14-smoke-v2-b`
- `results/autonomous_execution/phase_14/corrected-20260728-851d64e-phase14-production-v3`

Current status: Phase 14 passed. Phase 15 may now define its primary
confirmatory hypothesis. The supervised autoencoder failed its Phase 14
criterion and is therefore ineligible for confirmation.

## Phase 15 — primary confirmatory experiment

Preregistered at `13fbfdef7ac0cfd05dda2c1c33bb6a59b4d64fff` and executed at
`253e70e601e2`. The preregistration was committed before any confirmatory
outer-test result existed, and was not modified afterwards. No amendment was
required.

Question: does anonymous condition-transfer augmentation improve low-data
Buchwald–Hartwig yield prediction relative to a matched real-only XGBoost
baseline, under the corrected canonical protocol? This is the project's central
thesis.

**Result: NULL.**

| Quantity | Value |
| --- | --- |
| Mean paired effect | −0.6609 RMSE (comparator minus augmented; negative means augmentation is worse) |
| 95% bootstrap CI | [−0.8446, −0.4778] |
| Units improved / worsened | 0 of 10 / 10 of 10 |
| Wilcoxon two-sided p (secondary) | 0.001953 |
| Units included / degenerate / failed | 10 / 0 / 0 |

Per-seed paired deltas (seeds 5–14): −0.6087, −0.8666, −0.6264, −0.6273,
−0.3134, −0.3402, −0.2061, −1.1212, −0.8566, −1.0429.

The verdict is null under section 6 of the preregistration — not negative, and
not positive. The interval excludes zero and the Wilcoxon p is 0.002, so a
results-first reading would report significant harm. But the whole interval lies
inside the ±1.0 RMSE practical-equivalence band fixed in advance, which section 6
defines as null: a practically meaningful benefit and a practically meaningful
harm are both excluded. This is precisely the case the preregistration was
written to adjudicate, and it is why the rule was pinned before any data was
seen. Statistical significance alone does not establish an effect in either
direction.

The null is not an artifact of the treatment failing to apply. Every unit had a
healthy pool — 335 to 362 accepted synthetic rows against 633 measured rows,
roughly 55% augmentation, consistent with Phase 14's 340.

A protocol fact worth recording. The confirmation is directionally worse than
Phase 14's near-tie of −0.006, and the reason is search-budget asymmetry rather
than anything discovered after the fact. In Phase 14 the augmented arm
enumerated 6 candidate policies against the comparator's 3, and that extra
budget included a synthetic-downweighting knob for which the real-only arm has
no analogue. Section 4 requires equal budgets, so under a genuinely matched
comparison the augmented arm is pinned to a single weight and loses on all ten
seeds. Part of what looked like augmentation parity in development evidence was
the augmented arm being allowed to search harder.

Equalizing the budgets required one optional config key, because it was
impossible through configuration alone: the AE grid is hard-coded to a
one-factor-at-a-time design that forces at least two augmented supervised
weights. The change is backward compatible — Phase 14's resolved config hash
recomputes bit-identically — and both arms enumerated 3 policies here.

Ten fresh seeds, 5 through 14, were built into
`results/corrected_canonical_splits_phase15/` with aggregate hash
`9ddb713ace55842aa8794a0d30a457dea92c3338d479b51c7d7a48eafed9b50c`. Zero
train/validation/test group overlap on `canonical_reaction_key` in every seed.
The existing five-seed artifacts were not mutated, not extended and not re-read;
Phase 14 had already consumed the outer-test claims on seeds 0 through 4.

Scope limits: training fraction 0.2 only, canonical grouped random splits only.
This is not OOD evidence, and fraction 1.0 was preregistered as secondary and
descriptive only and was not run.

Gate: 19 of 19 scientific artifacts byte-identical across two determinism smokes
against temporary registries, independent replay validation, the analysis
re-deriving to an identical hash, 70 of 70 registry claims metrics_complete with
exactly one outer-test batch each, `test_used_for_selection_or_retention` false,
full suite 1185 passed, ruff clean, `git diff --check` clean.

Authoritative outputs:

- `PRIMARY_EXPERIMENT.md`
- `results/corrected_canonical_splits_phase15`
- `results/autonomous_execution/phase_15/corrected-20260729-13fbfde-phase15-confirmation-v1`

## Candidate-scope audit (2026-08-25, post-roadmap)

A repository-wide audit of candidate-eligibility semantics ran after the roadmap
closed. It changes no historical number and re-runs nothing. Three findings
change how entries above should be read.

**1. Phase 15 is unaffected.** Its candidate eligibility was already scoped to
the rows each phase could observe (507 identities during search, 633 during
placement and final; never the 3,955-row canonical universe), enforced by a hash
guard. The confirmatory null stands.

**2. "Exactly five eligible unmeasured reactions" is a global-discovery claim
and must stop doing low-data explanatory work.** Wherever this summary uses the
near-complete factorial to explain a zero-accepted-candidate result or an
augmentation null, that reasoning applies only to runs whose eligibility rule
was the complete measured universe — the Phase 1-3 smokes and Phase 11. It does
not apply to Phase 14/15, whose pools were large precisely because their rule
was observed-only. A saturated matrix can leave almost no *global* discovery
headroom while leaving most of the matrix hidden from a low-data learner.

**3. The Phase 11 chemical-augmentation conclusions are withdrawn as
uninformative.** Under its executed global rule, 11,682 of 11,727 chemical
candidates were rejected as `already_measured`, and 93.6% of those collided only
with rows the simulated learner had never seen. The five chemical arms carried
0.4 to 1.2 effective added rows against a nominal budget of 159, so the recorded
"only 2 anonymous, 3 random typed, and 6 context-matched typed candidates across
795 nominal additions" and "anonymous transfer is 0.025 RMSE better on average
but adds only two rows" compare *(real-only)* against *(real-only + 159 rows)*.
The run is retained unaltered as a record of the executed protocol; its
conclusion about chemical augmentation is not.

See `RESULT_STATUS.md` § *Candidate-scope semantics* and
`docs/CANDIDATE_SCOPE.md`. A new experiment is preregistered in
`PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md`; development evidence for it is in
`results/corrected_candidate_scope_development_v1`.

## Roadmap complete

All 18 phases are passed and checkpointed locally at `432c1a5`. Nothing has been
pushed. See `AUTONOMOUS_COMPLETION_REPORT.md` for the full audit, the scientific
findings, the retained null and negative results, external and prospective
status, remaining debt, and reproduction and resume instructions.

Final gate: full suite 1191 passed, ruff clean, `git diff --check` clean.
`first_resumable_phase` returns `None`, the terminal state.

Two headline results, both obtained under protocols frozen before the outcomes
were visible, and both negative or null:

- the supervised autoencoder adds no reproducible value and is a secondary
  ablation;
- condition-transfer augmentation produces no practically meaningful change,
  with a preregistered verdict of null.

A structural finding bounds both: the canonical dataset is a near-complete
factorial, 3955 of 3960 cells measured, leaving exactly five eligible unmeasured
reactions.
