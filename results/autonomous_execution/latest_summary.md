# Autonomous Execution Summary

Updated: 2026-07-27T00:29:42Z

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

Current status: Phase 7 has passed its scientific and engineering gate and is
awaiting its local checkpoint commit.

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
