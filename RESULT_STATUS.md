# Result Status

This registry freezes the interpretation of existing experiment outputs. Code
repairs do not retroactively validate metrics produced by an invalid feature
path. Corrected experiments must use fresh output directories after review.

## Classification vocabulary

Every method and result family below carries **exactly one** status.

| Status | Meaning |
| --- | --- |
| **Supported** | Infrastructure or a contract that other work is entitled to rely on. Not itself a scientific claim. |
| **Development evidence** | A real, leakage-audited result that informs design decisions. It is *not* a confirmatory claim: the hypothesis was not pre-registered before the result was seen. |
| **Confirmatory evidence** | A result from a hypothesis frozen before any outer-test outcome was loaded, evaluated exactly once. |
| **Experimental** | A method retained for exploration or as a control. Its outputs must not be cited as evidence about method quality. |
| **Invalidated** | Known-wrong outputs. Default readers refuse to load them. They must never appear in a benchmark claim. |
| **Deprecated** | Superseded and not to be used for new work. Not necessarily wrong; simply replaced. |
| **Externally blocked** | Cannot progress without something a human must supply (licensed data, network access). No result is claimed either way. |

**The confirmatory-evidence class contains exactly one result, and it is a
null.** Phase 15 preregistered and executed one primary confirmatory hypothesis:
anonymous condition-transfer augmentation against a matched real-only XGBoost
baseline. The verdict was null — no practically meaningful change in either
direction. Everything else in this repository is development evidence,
experimental, invalidated, deprecated, or externally blocked.

## Infrastructure and contracts

| Family | Status |
| --- | --- |
| Canonical seven-role RDKit data audit (Batch 3) | Supported |
| Canonical grouped (group-safe) split assignments | Supported |
| Canonical reaction-key identity and duplicate/replicate audit | Supported |
| Feature compatibility and representation contract (`features/compatibility.py`) | Supported |
| Repository-global outer-test evaluation registry (`evaluation/evaluation_registry.py`) | Supported |
| Invalid-result gating (`results/status.py`) | Supported |
| Isolated fit worker and resource accounting | Supported |
| Shared strict config validation (`utils/strict_config.py`, Phase 17) | Supported |
| Shared scientific manifest and `verify_manifest` (`utils/scientific_manifest.py`, Phase 17) | Supported |
| Local reproducibility bundle tooling (`scripts/build_reproducibility_bundle.py`, Phase 17) | Supported |
| Bounded production-path smoke and CI lint/smoke jobs (Phase 17) | Supported |
| Read-only evaluation-registry inspector (`scripts/inspect_evaluation_registry.py`, Phase 17) | Supported |
| Autonomous roadmap state verifier (`autonomous_execution.py`) | Supported |

See `docs/REPRODUCIBILITY.md` for what a result is traceable to, what an actually
executed clean-checkout reproduction verified, and what a human must supply.

## Development evidence

| Family | Status |
| --- | --- |
| Corrected representation baselines | Development evidence |
| Corrected anonymous condition transfer | Development evidence |
| Corrected anonymous transfer + supervised autoencoder | Development evidence |
| Corrected anonymous-versus-role-aware comparison (corrected rows only) | Development evidence |
| Batch 2 corrected random-split outputs | Development evidence |
| Phase 5 separated policy search / final evaluation protocol runs | Development evidence |
| Phase 6 repaired AE internal validation and joint policy selection | Development evidence |
| Phase 7 rebuilt corrected product and reactant LOGO outputs | Development evidence |
| Phase 8 nested group-aware OOD validation | Development evidence |
| Phase 9 scaffold, cluster, similarity, and condition OOD splits | Development evidence |
| Phase 10 no-augmentation representation baselines | Development evidence |
| Phase 11 simple augmentation controls with matched budgets | Development evidence |
| Phase 12 pseudo-label accuracy and uncertainty calibration | Development evidence |
| Phase 13 low-complexity representation-learning baselines | Development evidence |
| Phase 18 recommendation simulation (`results/autonomous_execution/phase_18/corrected-20260727-phase18-production-v1`) | Development evidence |
| Phase 18 prospective package (`results/autonomous_execution/phase_18/corrected-20260728-phase18-prospective-package-v1`) | Development evidence |

Every entry here is a leakage-audited result with a frozen plan, a hash-verified
bundle, and a single outer-test evaluation per unit — but the hypothesis was not
pre-registered ahead of seeing the result. None of these is a confirmatory claim.

**The Phase 18 prospective package is a proposal, not a validated outcome.**
Prospective laboratory validation has **not** been performed. The package
records which experiments would be run and why; nothing in it constitutes
evidence that those experiments succeed.

## Experimental

| Family | Status |
| --- | --- |
| Utility-guided feature GAN augmentation | Experimental |
| Latent interpolation augmentation | Experimental |
| SMILES randomization augmentation | Experimental |
| Reaction-order permutation augmentation | Experimental |
| Condition recombination pseudo-label augmentation | Experimental |
| Teacher-ensemble uncertainty-filtered augmentation | Experimental |
| Chemical augmentation controls | Experimental |
| Synthetic identity / candidate-deduplication controls | Experimental |
| AutoML model-selection runner (`run_automl.py`) | Experimental |
| Hidden-condition probes | Experimental |
| Random-split baselines used as sanity checks | Experimental |

Latent-interpolation and GAN coordinates have deterministic feature hashes but no
fabricated chemical identity. They are development controls only and are
scientifically ineligible as molecular candidates.

## Invalidated

| Family | Status |
| --- | --- |
| Role-aware condition transfer v2 | **Invalidated** |
| Legacy (pre-canonical) role-aware condition transfer | **Invalidated** |
| Any table containing role-aware v2 rows (`condition_transfer_matched_comparison`) | **Invalidated** |
| Historical `results/stress/logo_product/` outputs | **Invalidated** |
| Historical `results/stress/logo_reactant/` outputs | **Invalidated** |
| Historical baseline `results/baseline/stress_logo_product_metrics.csv` | **Invalidated** |
| Historical baseline `results/baseline/stress_logo_reactant_metrics.csv` | **Invalidated** |

These are enforced in code by `INVALID_RESULT_FAMILIES` and
`INVALID_HISTORICAL_PATHS` in `src/bh_augmentation/results/status.py`. Default
readers refuse them; `allow_invalid=True` permits explicit historical inspection
only. They are excluded from every benchmark claim and from every comparison
table. Fresh directories whose names begin with `corrected_` are exempt from the
role-aware name match and are governed by the corrected-revalidation rules below.

## Deprecated

| Family | Status |
| --- | --- |
| Legacy feature aliases (`reaction_smiles`, `reaction_plus_components`, `reaction_combined_redundant`, `reaction_role_concat`, `reaction_role_concat_delta`, `role_separated_conditions`, `role_separated_conditions_delta`) | Deprecated |
| Removed feature kinds (`fp_concat`, `fp_plus_conditions`, `categorical_conditions`) | Deprecated |
| Pre-canonical (Batch 1) random-split experiment outputs | Deprecated |
| Pre-Phase-7 LOGO runner code paths | Deprecated |
| Per-runner bespoke config resolution helpers superseded by `utils/strict_config.py` | Deprecated |

Corrected scientific configs must name a canonical feature kind explicitly.
Alias support for old configuration names does not make old outputs valid.

## Externally blocked

| Family | Status |
| --- | --- |
| Phase 16 external typed-reaction dataset adapter | Externally blocked |
| Suzuki–Miyaura external-family validation | Externally blocked |

The adapter and its fixture-driven tests exist and pass, but no licensed
Suzuki–Miyaura dataset is present locally and acquisition requires a human with
network access. **No external-generalization result is claimed in either
direction.** See `docs/EXTERNAL_DATASETS.md`.

## Completed since this table was first written

| Phase | State |
| --- | --- |
| Phase 14 — reassess the supervised autoencoder | **Complete. Development evidence.** The supervised autoencoder **failed** its predefined retention criterion and is classified as a **secondary ablation**. It is not a supported method and was excluded from confirmation. Authoritative output: `results/autonomous_execution/phase_14/corrected-20260728-851d64e-phase14-production-v3`. |
| Phase 15 — primary confirmatory hypothesis | **Complete. Confirmatory evidence — the only entry in that class.** Verdict **null**: anonymous condition-transfer augmentation produces no practically meaningful change relative to a matched real-only XGBoost baseline. Preregistered at `13fbfde` before any result existed. Authoritative output: `results/autonomous_execution/phase_15/corrected-20260729-13fbfde-phase15-confirmation-v1`. |
| Candidate-scope reanalysis — observed-only condition transfer | **Preregistered, not yet executed.** `PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md` freezes a new confirmatory experiment on seeds 6–14 at training fraction 0.05, comparing observed-only condition transfer against a matched real-only control, with a globally-unmeasured prospective control and a withheld-cell oracle diagnostic as declared secondary evidence. Development evidence: `results/corrected_candidate_scope_development_v1`. |
| Phase 18 — prospective preparation | The recommendation simulation and the prospective package are both committed and classified as development evidence above. **Prospective laboratory validation has not been performed**, so no phase claims a wet-lab outcome. |

The confirmatory-evidence class contains exactly one result, and that result is
a null. Its scope is training fraction 0.2, canonical grouped random splits, one
dataset, and **candidate eligibility decided from the rows each phase could
observe** (§ *Candidate-scope semantics* below). It is **not** OOD evidence and
must not be quoted as one.

No prospective laboratory claim is made by any phase.

## Candidate-scope semantics

Every result family that generates synthetic reactions decides *candidate
eligibility* under exactly one of two scientifically different rules. The rule
is declared in `src/bh_augmentation/augmentation/candidate_scope_registry.py`
and verified against the code by `scripts/audit_candidate_scope.py`.

| Rule | Question it answers | A candidate is ineligible when |
| --- | --- | --- |
| `observed_only_low_data` | Does augmentation help a learner that has seen only a small labeled subset? | Its canonical identity occurs in `labeled_train`, or it duplicates another generated candidate. Complete-dataset membership is **not** consulted. |
| `globally_unmeasured_prospective` | Is this reaction genuinely new chemistry worth running? | Its canonical identity occurs **anywhere** in the complete measured dataset. |

Two further labels exist for the audit and construct no generation policy:
`representation_augmentation` (SMILES randomization, role/reaction-order
permutation, latent and feature-space methods — the chemistry is deliberately
unchanged, or the object generated has no chemical identity at all) and
`not_applicable`.

**Why the distinction matters here.** The canonical Buchwald–Hartwig dataset is
a near-complete factorial: 3,955 of 3,960 grid cells are measured, leaving five
globally novel reactions. **Global discovery headroom is therefore almost
zero.** But a learner restricted to a 5% training fraction has observed 159 of
3,955 reactions, so **low-data transfer headroom is very large**. Applying the
global rule to a low-data augmentation experiment suppresses almost the entire
treatment: development evidence at seeds 0–4 records **1 accepted candidate out
of 2,540 generated** under the global rule against **1,704** under the
observed-only rule. A near-complete matrix can have essentially no global
discovery headroom while still offering substantial low-data transfer headroom,
because most cells are hidden from the simulated learner even though they exist
in the file.

### Status of families under this distinction

| Family | Rule | Effect of the correction |
| --- | --- | --- |
| Phase 15 primary confirmatory (`redesigned_ae_benchmark`) | `observed_only_low_data` | **None.** Already scoped to each phase's visible rows — inner policy-fit rows during search (507 identities) and saved training rows during placement and final (633), never the 3,955-row universe, with a hash guard that hard-fails otherwise. The verdict, numbers and registry records are unchanged. What is narrowed is the *stated reason* for the null, not the null. |
| Phase 14 redesigned AE benchmark | `observed_only_low_data` | None, for the same reason. |
| Phase 18 prospective package | `globally_unmeasured_prospective` | None. Global novelty is the correct and intended rule for a prospective claim. |
| Phases 1–3 identity/role-change/ranking smokes | `globally_unmeasured_prospective` | None to the runs. Their zero-accepted results are **global-novelty statements** and must not be quoted as evidence about how much a low-data learner could generate. |
| Phase 11 matched-budget augmentation controls | `observed_only_low_data` | **Materially affected.** Under the executed global rule, 11,682 of 11,727 chemical candidates were rejected as `already_measured`, and 93.6% of those collided only with rows the simulated learner had never seen. The chemical arms carried 0.4–1.2 effective added rows against a nominal budget of 159, so their comparisons against the oversampling controls compared *(real-only)* with *(real-only + 159 rows)*. **The Phase 11 chemical-augmentation conclusions are withdrawn as uninformative about chemical augmentation**; the run itself is retained unaltered as a record of the executed protocol. |
| Corrected anonymous / role-aware condition transfer | `observed_only_low_data` | Materially affected; their candidate pools were suppressed by the same rule. Existing bundles are retained under their executed protocol and are not cited as low-data augmentation-benefit evidence. |
| Frozen-policy search / final evaluation (Phases 5, 6) | `observed_only_low_data` | Latent only. No completed final evaluation on that chain selected an augmenting method, so no published metric changed. |
| External reaction-family validation | `observed_only_low_data` | Semantics corrected in code, config and tests. The empirical datasets remain **externally blocked**; no run was fabricated. |

## Why role-aware v2 is invalid

Measured role-aware rows and synthetic role-aware rows were featurized with
inconsistent semantic block meanings. Measured rows used three reaction
sections while synthetic rows were manually assembled from individually named
chemical roles. Equal vector widths therefore did not guarantee equal feature
semantics. Existing role-aware v2 metrics and tables containing those metrics
must not be used in benchmark claims.

The old output directories are retained for historical inspection. Loading
them requires an explicit invalid-result override. Alias support for old feature
configuration names does not make these outputs valid.

## Representation terminology

`reaction_section_concat` means reactant section + agent section + product
section. It does not identify individual chemical roles.

`bh_role_separated` means reactant 1 + reactant 2 + catalyst + ligand + base +
solvent/additive + product in that exact order.

Equal matrix widths are insufficient for compatibility. Feature names,
fingerprint settings, role order, backend, and exact block slices must also
match before measured and synthetic matrices can be stacked.

Corrected revalidation outputs use fresh directories containing `corrected`,
a `run_manifest.json` with `historical_results_loaded=false`, and row-level
`result_status=corrected_revalidation`. A corrected name alone is insufficient:
comparison scripts validate manifests, feature hashes, split hashes, dataset
hashes, and matched real-only metrics.

## Why historical LOGO outputs are invalid

The retained product and reactant outputs under `results/stress/logo_product/`
and `results/stress/logo_reactant/`, together with the corresponding
`results/baseline/stress_logo_product_metrics.csv` and
`stress_logo_reactant_metrics.csv` files, do not contain sufficient evidence to
verify leave-one-group-out evaluation. In particular, the baseline tables do
not provide reproducible fold assignments, split hashes, and train/test group
overlap audits. Their labels alone cannot establish that the saved metrics came
from valid LOGO splits.

These artifacts remain on disk for provenance. Default result readers reject
them; `allow_invalid=True` permits explicit historical inspection only. Fresh
corrected Phase 7 directories remain eligible when their manifests establish
the declared held-out group, zero train/test group overlap, complete expected
fold coverage, and assignment hashes.

## Canonical identity and future benchmarks

Batch 2 corrected the measured/synthetic feature-semantics mismatch. Its
random-split outputs remain valid development evidence, but they are not final
benchmark evidence because canonical molecular duplicates and experimental
replicates had not yet been audited when those splits were constructed.

Batch 3 preserves Batch 2 outputs and introduces versioned RDKit canonical
identities plus group-safe split assignments. Future corrected benchmark runs
should use the canonical grouped assignments so rows representing the same
seven-role canonical reaction cannot cross train, validation, and test.

Canonicalization completing successfully does not establish that the dataset is
clean. The duplicate, replicate, yield-conflict, invalid-structure, and
fingerprint-collision reports require separate scientific review. Batch 3 does
not generate new model results.
