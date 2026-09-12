# Condition-Transfer Augmentation for Low-Data Buchwald-Hartwig Yield Prediction

This repository is a research-code package for evaluating whether
reaction-aware data augmentation can improve low-data Buchwald-Hartwig reaction
yield prediction. It is built around preregistration, one-shot outer-test
evaluation through a repository-global registry, and hash-addressed result
manifests, so that every headline number can be re-derived from committed files.

## Status and Headline Results

Two preregistered confirmatory experiments have been executed. **Both verdicts
are null**: anonymous condition-transfer augmentation with teacher pseudo-labels
produces no practically meaningful change in low-data yield prediction relative
to a matched real-only XGBoost baseline on this dataset.

| Experiment | Preregistered at | Regime | Mean paired RMSE reduction (95% CI) | Units improved | Verdict |
| --- | --- | --- | --- | --- | --- |
| Phase 15 primary confirmation | `13fbfde` (`PRIMARY_EXPERIMENT.md`) | fraction 0.2, seeds 5–14, 335–362 accepted synthetic rows per unit | −0.661 [−0.845, −0.478] | 0 of 10 | **null** |
| Observed-only condition transfer | `c6aab9e` (`PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md`) | fraction 0.05, seeds 6–14, 101–125 accepted synthetic rows per unit | −0.377 [−0.746, −0.035] | 2 of 9 | **null** |

Positive values would favour augmentation. Both intervals exclude zero on the
harmful side and both lie entirely inside the ±1.0 RMSE practical-equivalence
band the preregistrations fixed in advance, which those documents define as a
null: a practically meaningful benefit and a practically meaningful harm are
both excluded. Neither result is out-of-distribution evidence.

The second experiment also settles a methodological question. The canonical
dataset is a near-complete factorial, and the historical eligibility rule
rejected any synthetic reaction that existed *anywhere* in it, which suppressed
the treatment to 0–2 accepted rows per unit. Deciding eligibility from the rows
the low-data learner actually observed restores it to 101–125 rows per unit. The
verdict is null either way, so the earlier nulls were not artifacts of an
empty treatment. The control arm run under the historical rule was degenerate on
8 of 9 units, and that degeneracy is reported as its result.

Secondary, non-selecting evidence: pseudo-labels for the withheld cells that the
observed-only rule admits reconstruct the hidden measured yields with Spearman
0.76 and MAE 11.7 yield points, so the transfer is chemically informative even
though it does not translate into outer-test gain.

### Exploratory (post hoc, no verdict): effect versus training fraction

Declared in `EXPLORATORY_FRACTION_SWEEP.md` before its outer tests were read
and run on the same seeds at the two fractions the confirmatory work had not
touched. It is descriptive only and cannot be promoted to a claim.

| Label | Fraction | Labeled rows | Accepted synthetic rows per unit | Mean paired RMSE reduction (95% CI) | Units improved |
| --- | ---: | ---: | --- | --- | --- |
| exploratory | 0.01 | 32 | 22–27 | −0.415 [−0.806, −0.010] | 3 of 9 |
| confirmatory | 0.05 | 159 | 101–125 | −0.377 [−0.746, −0.035] | 2 of 9 |
| exploratory | 0.10 | 317 | 185–212 | −0.508 [−0.707, −0.273] | 1 of 9 |

At no fraction does the augmented arm beat the matched real-only control, and
every interval sits inside the ±1.0 RMSE band. The pseudo-labels get more
accurate as the fraction grows (Spearman 0.64 → 0.76 → 0.78 against the
withheld yields), while the outer-test effect does not improve. All 36 sweep
units completed; none was degenerate. Full outputs, one analysis per fraction
and the comparison table: `results/corrected_exploratory_fraction_sweep_analysis_v1/`.

Two further structural findings: the supervised autoencoder failed its
predefined retention criterion (Phase 14), and informed acquisition strategies
recover about 4× the high-yield hits of random selection over a 320-experiment
budget (Phase 18, development evidence). `RESULT_STATUS.md` classifies every
result family; `AUTONOMOUS_COMPLETION_REPORT.md` is the narrative record.

### How To Verify The Headline Results

Both confirmatory bundles are committed, unlike the rest of `results/`. From a
fresh clone with the `science` extra installed:

```bash
python -m pytest tests/test_committed_confirmatory_artifacts.py -q
```

That test recomputes each verdict from the committed per-unit outer-test
metrics, frozen policies, claims, degenerate-unit records and pool accounting,
compares the recomputed analysis hash with the committed one, and verifies the
scientific manifests. It does the same for the exploratory sweep. To read the
reports directly:

- `results/autonomous_execution/phase_15/corrected-20260729-13fbfde-phase15-confirmation-v1/confirmatory_report.md`
- `results/corrected_exploratory_fraction_sweep_analysis_v1/fraction_sweep_summary.md` (exploratory)
- `results/corrected_candidate_scope_reanalysis_20260903T194209Z/summary/observed_only_confirmatory_analysis/confirmatory_report.md`

The full-scale inputs those runs consumed (the 12 MB canonical dataset and the
split directories) are gitignored; `docs/REPRODUCIBILITY.md` gives the
regeneration commands and their pinned hashes.

## Historical Scope

The original v0 scope was deliberately conservative: reproducible data loading,
cleaning, splitting, featurization, classical regression baselines, safe
augmentation, decision-oriented evaluation, and lightweight reporting. The
sections below document that machinery and the experiments layered on it.
Generative synthetic reaction data remains out of scope.

## Project Overview

The core question is:

Can safe augmentation of measured reaction records improve yield prediction or
reaction recommendation when only a small number of Buchwald-Hartwig reactions
are available for training?

The MVP compares:

- no augmentation
- reaction-smiles-derived feature representations
- low-data training fractions
- random splits and held-out group splits
- simulated single-round reaction recommendation

## Canonical Representation

The processed TDC Buchwald-Hartwig dataset has populated `reaction_smiles`, but
its normalized `aryl_halide_smiles`, `amine_smiles`, `ligand_smiles`,
`base_smiles`, and `additive_smiles` columns are fully `UNKNOWN`. These legacy
columns are not required and are dropped when they contain no information.

The original `reaction_smiles` string remains preserved source data. Batch 3
adds a versioned RDKit-canonical identity for each of the seven recovered
Buchwald-Hartwig roles without rewriting the source strings. Future corrected
benchmarks should use the saved canonical group assignments rather than
row-level random splits.

Component-column augmentation is disabled. The active augmentation recombines
condition tokens parsed from `reaction_smiles`. Stress-test `product_key` and
`reactant_key` values are also derived from `reaction_smiles`.

## What The MVP Tests

- Loading local CSV reaction-yield data.
- Optional loading from TDC if `PyTDC` is installed by the user.
- Cleaning to a normalized Buchwald-Hartwig schema.
- Deterministic train/validation/test splits.
- Held-out group splits for ligand/base/additive/aryl-halide/amine columns.
- Role-aware reaction Morgan fingerprint ablations with RDKit.
- Ridge, Random Forest, ExtraTrees, and optional XGBoost/CatBoost regressors.
- Train-only condition recombination with teacher pseudo-labeling.
- Auditing that augmentation is train-only and changes model inputs.
- Regression metrics and decision metrics.
- Markdown report generation and simple matplotlib plots.
- A fixture-based end-to-end guardrail test requiring no real dataset.

## What The MVP Deliberately Does Not Test

- It does not test generative chemistry models.
- It does not create generative synthetic reaction data.
- It does not pseudo-label measured validation or test reactions.
- It does not use deep learning as a confirmatory method. Torch-based
  representation learners (a supervised autoencoder, matched MLPs, a
  feature-space GAN) exist as development controls; the autoencoder failed
  its retention criterion and none of them enters a confirmatory claim.
- It does not provide a web app or dashboard.
- It does not claim random split performance is real chemistry
  generalization.
- It does not replace external chemical validation or domain review.

Random split results should be treated as a software and modeling sanity check,
not as evidence of chemistry generalization. For a more realistic stress test,
prefer held-out group splits such as held-out ligands or aryl halides.

## Installation

From the repository root:

```bash
python -m pip install -e ".[dev,science]"
```

The `science` extra installs XGBoost and RDKit pinned to `2023.9.6`, the
version that produced the canonical dataset's stored reaction identities. A
different RDKit canonicalizes differently and silently breaks the candidate
eligibility gate (see `docs/CANDIDATE_SCOPE.md`, section 7), so the pin is part
of the scientific contract, not a convenience.

Run the test suite first (use `python -m pytest`, not the bare `pytest` script,
so the repository root is on `sys.path`):

```bash
python -m pytest
```

The unit tests use small synthetic fixtures and do not require internet access,
the real Buchwald-Hartwig dataset, TDC, CatBoost, or Optuna. Without RDKit or
XGBoost the identity-dependent and confirmatory-method tests skip themselves and
say so under `pytest -ra`. The full suite takes about 25 minutes because the
Phase 14/15 benchmark module spawns isolated fit workers.

## Code Quality And CI

Pytest and Ruff are configured in `pyproject.toml`.

Run tests locally:

```bash
pytest
```

Run Ruff locally after installing the dev extra:

```bash
python -m pip install -e ".[dev]"
ruff check src tests
```

GitHub Actions runs three jobs on push and pull requests:

- **lint** — `python -B -m ruff check .` over the whole repository.
- **tests** — the full `pytest` suite.
- **production_path_smoke** — a bounded end-to-end run of the real scientific
  production path (canonical data audit → canonical group-safe splits → the
  Phase 13 low-complexity benchmark) on a 300-row slice of the committed
  `data/processed/bh_clean_stress.csv` fixture, followed by a
  reproducibility-bundle build and a tamper-evidence assertion. It requires no
  gitignored artifact and no network access, and exists because unit tests alone
  cannot catch an end-to-end break in a scientific runner.

Run the smoke locally with:

```bash
python -B scripts/run_production_path_smoke.py --output-directory /tmp/bh-smoke --rows 300
```

## Reproducibility And Result Integrity

Every completed scientific bundle carries a hash-addressed manifest binding the
result to its code commit, dataset, split, features, configuration, plan,
outputs, dependency versions, and hardware/threading environment. Verify any
bundle with:

```bash
python -B - <<'PY'
from bh_augmentation.utils.scientific_manifest import verify_manifest
print(verify_manifest("results/<some-corrected-bundle>").to_dict())
PY
```

Build a local, non-publishing reproducibility bundle for a result:

```bash
python -B scripts/build_reproducibility_bundle.py \
    --result-directory results/<some-corrected-bundle> \
    --bundle-directory /tmp/my-bundle \
    --config configs/<the-config-used>.yaml
```

Inspect outer-test evaluation claims (read-only; there is deliberately no way to
delete or release a claim):

```bash
python -B scripts/inspect_evaluation_registry.py list
```

`docs/REPRODUCIBILITY.md` records an actually executed clean-checkout
reproduction, exactly what it verified, and what a human must supply for the
full-scale canonical inputs (which are gitignored). `RESULT_STATUS.md` classifies
every method and result family.

## Dataset Setup Options

The expected cleaned schema is:

```text
reaction_id
aryl_halide_smiles
amine_smiles
ligand_smiles
base_smiles
additive_smiles
solvent
temperature
reaction_smiles
yield
```

### Option 1: Local CSV

Place a CSV locally and point the config at it:

```yaml
dataset:
  name: buchwald_hartwig
  path: data/processed/bh_clean.csv
```

The loader and cleaner can be used directly:

```python
from bh_augmentation.data.load_data import load_reaction_csv
from bh_augmentation.data.clean_data import clean_buchwald_hartwig

raw = load_reaction_csv("data/raw/buchwald_hartwig.csv")
clean = clean_buchwald_hartwig(raw)
```

### Option 2: Optional TDC Loader

TDC is not a hard dependency. Install it only if you want to fetch the
Buchwald-Hartwig yield dataset through TDC:

```bash
python -m pip install PyTDC
```

```python
from bh_augmentation.data.load_data import save_tdc_buchwald_hartwig

save_tdc_buchwald_hartwig("data/raw/tdc_buchwald_hartwig.csv")
```

### Inspecting Raw Reaction Strings

If a TDC or local CSV does not expose separated component columns, inspect the
raw reaction string format before changing preprocessing:

```bash
python -m bh_augmentation.data.inspect_reactions --input data/raw/buchwald_hartwig_tdc.csv
```

The command prints columns, shape, likely reaction-string columns, and counts
for strings containing `>`, exactly two `>` characters, `.`, halogens, and
nitrogen.

## Baseline Run

After `configs/baseline.yaml` points to a real cleaned CSV:

```bash
python -m bh_augmentation.run_baseline --config configs/baseline.yaml
```

Default output:

```text
results/baseline/baseline_metrics.csv
results/baseline/split_metadata.csv
```

### Feature Ablations

The supported named feature modes are:

- `reaction_morgan_sum`: sums fingerprints for all reaction molecules. It is
  compact but loses reactant/agent/product role information.
- `reaction_section_concat`: concatenates summed reactant-section,
  agent-section, and product-section fingerprints. It does not resolve
  individual chemical roles.
- `reaction_section_concat_delta`: adds the numeric product-minus-reactant
  fingerprint difference.
- `bh_role_separated`: concatenates seven dataset-specific molecular roles in
  the fixed order reactant 1, reactant 2, catalyst, ligand, base,
  solvent/additive, and product.
- `bh_role_separated_delta`: appends product-minus-reactant-1,
  product-minus-reactant-2, and product-minus-reactant-pair blocks to the seven
  role blocks.

Product features are acceptable for this reaction-yield task because the
intended product structure is known. Results require more careful
interpretation for prospective settings where product identity is uncertain.
`reaction_section_concat_delta` is the default MVP representation. Component
columns such as ligand, base, and additive may remain `UNKNOWN` in TDC and
should not be primary features. The metrics CSV records `feature_kind` and
`n_features` for each ablation.

Migration from older configs:

- `features.kind: reaction_smiles` is deprecated and maps to
  `reaction_morgan_sum`.
- `features.kind: reaction_plus_components` is deprecated and maps to
  `reaction_section_concat_delta`.
- `reaction_role_concat` and `reaction_role_concat_delta` are deprecated aliases
  for the corresponding `reaction_section_*` representations.
- `role_separated_conditions` and `role_separated_conditions_delta` are
  deprecated aliases for `bh_role_separated` and `bh_role_separated_delta`.
- `fp_concat`, `fp_plus_conditions`, and `categorical_conditions` are removed
  named modes. Use one of the three supported reaction modes instead.
- `reaction_combined_redundant` is deprecated and maps to
  `reaction_section_concat_delta`.

Equal feature widths do not establish compatibility. Any measured/synthetic
stack must also match canonical feature names, radius, bit count, resolved
fingerprint backend, role ordering, and exact half-open block slices. See
`RESULT_STATUS.md` for invalid historical outputs produced before this check.

### Canonical Data Audit And Grouped Splits

Canonicalize and audit the seven molecular roles:

```bash
python -m bh_augmentation.data.audit_canonical_dataset \
  --config configs/canonical_data_audit.yaml
```

The source CSV is not modified. The command preserves each raw role string,
adds isomeric RDKit canonical SMILES, constructs a schema-versioned reaction
key, and reports invalid structures, duplicate rows, experimental replicates,
yield conflicts, and feature-vector equality separately. It performs no
tautomer standardization, neutralization, salt stripping, fragment removal, or
protonation normalization.

Build deterministic canonical-group-safe splits:

```bash
python -m bh_augmentation.data.canonical_splits \
  --config configs/canonical_grouped_splits.yaml
```

All rows with one canonical seven-role reaction identity remain in one outer
split. The 1%, 5%, 10%, and 20% training sets are cumulative complete-group
prefixes of the same outer training pool.

These audit concepts are intentionally distinct:

- **Canonical molecular equivalence** means RDKit maps role strings to the same
  molecular graph identity under the documented canonicalization settings.
- **Feature-vector equality** means the selected finite fingerprint vectors are
  equal; collisions do not prove chemical equivalence.
- **Experimental replication** means multiple measured rows share one canonical
  seven-role reaction identity.
- **Yield conflict** means replicates of one canonical reaction report differing
  measured yields.

Canonicalization preserves stereochemistry, isotopes, formal charges,
aromaticity, disconnected fragments, and reactant-role order. It does not prove
that role assignment or measured metadata is scientifically correct.

## Augmentation Run

Run the condition-recombination augmentation comparison:

```bash
python -m bh_augmentation.run_augmentation --config configs/augmentation.yaml
```

Default output:

```text
results/augmentation/condition_recombine_metrics.csv
results/augmentation/condition_recombine_split_metadata.csv
```

Validation and test sets are split from real data before augmentation. Only the
training split may be augmented.

Audit whether configured augmentation actually changes model inputs before
trusting augmentation metrics:

```bash
python -m bh_augmentation.audit_augmentation \
  --config configs/augmentation.yaml
```

The audit checks train-only application, split leakage, changed columns,
feature-vector duplication, and copied labels. It writes:

```text
results/augmentation/audit/augmentation_audit.txt
results/augmentation/audit/augmentation_audit.json
```

### Condition Recombination Pseudo-Label Augmentation

The active method parses `A.B.C.D.E.F>>P` as two substrate tokens, condition
tokens, and a product. It keeps `A.B` and `P` from a source training row,
replaces the conditions with those from another training row, and rebuilds the
reaction string. Candidates too dissimilar to measured training reactions are
removed with a nearest-neighbor Tanimoto filter, and every candidate passes a
**canonical identity gate** whose eligibility rule is declared explicitly — see
*Candidate Scope: Three Different Questions* below.

A teacher model is fitted only on real training rows and predicts the synthetic
yields, which are clipped to `[0, 100]`. The student then trains on real plus
synthetic rows. Validation and test rows are never used by the teacher and are
never augmented. Fully `UNKNOWN` normalized component columns are not used.

Run the audit before the experiment:

```bash
python -m bh_augmentation.audit_augmentation --config configs/augmentation.yaml
python -m bh_augmentation.run_augmentation --config configs/augmentation.yaml
```

### Validation-Selected Augmentation Policy Search

The active augmentation config searches synthetic multipliers, minimum
nearest-neighbor similarities, teacher models, and augmentation random seeds.
Every policy is trained on the same real training split and ranked using
validation RMSE only. The test split is untouched during search and is evaluated
only after one policy has been selected for each train fraction, model, feature
configuration, and fold.

Run the audit and search:

```bash
python -m bh_augmentation.audit_augmentation --config configs/augmentation.yaml
python -m bh_augmentation.run_augmentation --config configs/augmentation.yaml
```

Inspect the selected policies:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv("results/augmentation/selected_policies.csv").to_string(index=False))
PY
```

Search validation results, selected policies, and final selected-policy metrics
are written separately under `results/augmentation/`. Compare validation and
test rows in `selected_policy_metrics.csv` against the matching train fractions
in the low-data baseline. Test metrics must not be used to revise the policy
grid or selection rule.

For larger synthetic fractions, use:

```bash
python -m bh_augmentation.audit_augmentation --config configs/augmentation_large_search.yaml
python -m bh_augmentation.run_augmentation --config configs/augmentation_large_search.yaml
```

The large search excludes low-similarity policies at multipliers of 2 or 3.
Student models weight real rows at `1.0` and synthetic rows by clipped nearest-
training similarity. Models without `sample_weight` support emit a warning and
fall back to unweighted fitting. Policy selection still uses validation RMSE
only; test metrics are computed after selection.

### Teacher Ensemble Uncertainty-Filtered Augmentation

`condition_recombine_ensemble_filter` uses the same train-only condition-block
recombination, but labels each candidate with multiple teachers trained only on
real training rows. Teacher disagreement is summarized by prediction standard
deviation and range. Candidates must pass both nearest-training similarity and
configurable disagreement thresholds before the ensemble mean becomes their
pseudo-label.

Policies vary synthetic fraction, similarity, uncertainty thresholds, and
candidate seed. Selection uses validation RMSE only; the untouched test split is
evaluated after selection.

```bash
pytest
python -m bh_augmentation.audit_augmentation --config configs/augmentation_ensemble_filter.yaml
python -m bh_augmentation.run_augmentation --config configs/augmentation_ensemble_filter.yaml
```

Compare `results/augmentation_ensemble_filter/selected_policy_metrics.csv`
against `results/stress/lowdata/baseline_metrics.csv` and the current
`condition_recombine_pseudolabel` selected-policy results. Use matched train
fractions and seeds, and do not use test differences to revise policy selection.

### Utility-Guided Feature GAN Augmentation

`utility_guided_feature_gan` is not a plain GAN and it does not generate raw
reaction SMILES. It learns feature-space candidates from the train-only
`reaction_section_concat` representation, optionally through train-only
TruncatedSVD. A WGAN-GP-style generator and critic encourage realistic feature
points, while an inner reward-validation split scores whether candidate batches
improve downstream yield prediction.

The split roles are deliberately separate:

- generator train rows fit the feature transform, generator, critic, teachers,
  and nearest-neighbor filters
- reward-valid rows score inner utility rewards during candidate selection
- outer validation rows select the final policy
- test rows are untouched until the selected policy is evaluated once

Run:

```bash
pytest
python -m bh_augmentation.run_augmentation --config configs/augmentation_utility_guided_gan.yaml
```

The checked-in config is intentionally laptop-sized. Expand the search grid only
after the smoke run is passing, since each policy trains a generator, teacher
ensemble, and downstream student model.

To train the student directly in the SVD latent space where the generator
operates, run:

```bash
python -m bh_augmentation.run_augmentation \
  --config configs/augmentation_utility_guided_gan_latent_tiny.yaml
```

Outputs are written to:

- `results/augmentation_utility_guided_gan/policy_search_metrics.csv`
- `results/augmentation_utility_guided_gan/selected_policies.csv`
- `results/augmentation_utility_guided_gan/selected_policy_metrics.csv`

## Stress-Test Splits

Create deterministic product and order-invariant reactant group keys:

```bash
python -m bh_augmentation.data.make_stress_dataset \
  --input data/processed/bh_clean.csv \
  --output data/processed/bh_clean_stress.csv
```

Run a held-out product experiment:

```bash
python -m bh_augmentation.run_baseline \
  --config configs/stress_heldout_product.yaml
```

Run a held-out reactant experiment:

```bash
python -m bh_augmentation.run_baseline \
  --config configs/stress_heldout_reactant.yaml
```

The current TDC dataset has only about five unique product groups, so held-out
product results may have high variance. Each stress config uses a distinct
results directory. Reusing an existing `output.metrics_path` overwrites that
CSV rather than appending to it.

### Leave-One-Group-Out Stress Tests

A single held-out product split depends strongly on which of the five product
groups is selected. Leave-one-group-out evaluation holds each group out once,
then summarizes performance across all folds.

Run product LOGO:

```bash
python -m bh_augmentation.run_baseline \
  --config configs/stress_logo_product.yaml
```

Run reactant LOGO:

```bash
python -m bh_augmentation.run_baseline \
  --config configs/stress_logo_reactant.yaml
```

Each results directory contains fold-level metrics, split metadata, and summary
metrics. The summary reports the mean and standard deviation across held-out
groups for each feature, model, split, and metric combination. Use the mean as
overall performance and the standard deviation as sensitivity to which
chemical group was excluded.

## AutoML Run

The repository includes a compact Optuna-based AutoML runner:

```bash
python -m bh_augmentation.run_automl --config configs/automl.yaml
```

Install Optuna explicitly before using it:

```bash
python -m pip install optuna
```

Default outputs:

```text
results/automl/trials.csv
results/automl/best_config.json
results/automl/final_test_metrics.csv
```

The search space is intentionally small: Ridge, Random Forest, ExtraTrees, and
the supported reaction feature modes. The
default objective is validation top-k hit rate, with validation RMSE available
as a fallback objective.

Optional libraries such as XGBoost and CatBoost are supported only when
installed and requested.

## Recommendation Simulation

To simulate a single round of reaction optimization from a small measured seed
set:

```bash
python -m bh_augmentation.run_recommendation --config configs/augmentation.yaml
```

Default output:

```text
results/recommendation/topk_metrics.csv
```

The simulator trains on a small seed set, ranks remaining candidate reactions,
reveals the true yields of the top-k selected reactions from the dataset, and
compares random selection, a non-augmented model, and configured augmented
models.

## Report Generation

To summarize available results:

```bash
python -m bh_augmentation.run_report --results-dir results --output results/final_report.md
```

The report skips missing result files. When relevant data is available, plots
are written under:

```text
results/plots/
```

## Complete MVP Workflow

To run the fixture-data MVP workflow end to end:

```bash
python -m bh_augmentation.run_mvp --config configs/mvp.yaml
```

The default MVP config uses `tests/fixtures/sample_bh.csv`, expands the cleaned
fixture rows in memory, and writes all outputs under:

```text
results/mvp/
```

The workflow runs:

- baseline experiment
- low-data experiment
- safe augmentation experiment
- top-k recommendation simulation
- final Markdown report generation

It does not run Optuna by default. To use a provided cleaned Buchwald-Hartwig
CSV instead of fixture data, set `dataset.use_fixture: false` and
`dataset.path` in `configs/mvp.yaml`.

## Evaluation Metrics

Regression metrics:

- `rmse`: root mean squared error. Lower is better.
- `mae`: mean absolute error. Lower is better.
- `r2`: coefficient of determination. Higher is better.
- `pearson`: linear correlation between true and predicted yields.
- `spearman`: rank correlation between true and predicted yields.

Decision metrics:

- `top_k_hit_rate`: fraction of top-k predicted reactions whose true yield is
  above a threshold.
- `top_k_average_true_yield`: average true yield among top-k predicted
  reactions.
- `simple_regret`: true best candidate yield minus best true yield among the
  selected top-k candidates. Lower is better.
- `experiments_to_first_hit`: number of ranked experiments needed before the
  first reaction above the high-yield threshold is found.

## Why Validation And Test Data Must Stay Real

Augmentation is label-preserving only under specific assumptions. If augmented
rows appear in validation or test sets, the evaluation can become inflated or
misleading because the model may be tested on transformed versions of training
chemistry rather than independent measured reactions.

This repository therefore follows the rule:

1. Split real measured rows into train, validation, and test.
2. Apply augmentation only to the training split.
3. Evaluate only on untouched real validation and test rows.

This rule is especially important for low-data and recommendation-style
experiments, where a small amount of leakage can dominate the apparent benefit.

## Candidate Scope: Three Different Questions

Generating a synthetic reaction and deciding whether it is *eligible* are
separate steps, and the eligibility rule depends entirely on which scientific
question is being asked. This repository distinguishes three, and the
distinction is declared in code
(`src/bh_augmentation/augmentation/candidate_scope_registry.py`) and verified by
`python scripts/audit_candidate_scope.py`.

**1. Globally novel chemistry generation** (`globally_unmeasured_prospective`).
Propose reactions that have never been measured. A candidate whose canonical
identity occurs anywhere in the complete historical dataset is ineligible. This
is the correct rule for prospective discovery and for the Phase 18 package.

**2. Low-data pseudo-label augmentation** (`observed_only_low_data`). Simulate a
learner that has observed only a small labeled subset. A candidate is ineligible
only when its identity occurs in that observed subset, or duplicates another
generated candidate. Whether the reaction happens to exist elsewhere in the file
is *unknowable to that learner* and must not decide eligibility. Candidates
matching the validation or outer-test partitions are quarantined — recorded in
the candidate audit, excluded from student training — rather than silently
dropped.

**3. Retrospective withheld-cell reconstruction**
(`withheld_cell_transfer_oracle`). Ask how accurately the pseudo-labels
reconstruct reactions that *were* measured but were hidden from the simulated
learner. This is a diagnostic, not a benchmark. The hidden yields are oracle
labels: they are read only after the candidate pool and its pseudo-labels are
frozen and hash-verified, and they may never influence generation, ranking,
teacher fitting, policy selection, hyperparameters, synthetic weights, or
student fitting.

A fourth label, `representation_augmentation`, covers families that
re-*represent* existing chemistry rather than generating new chemistry — SMILES
randomization, reaction/role-order permutation, latent interpolation, and
feature-space GAN candidates. Chemical-identity rejection is inapplicable to
them: the molecule is deliberately unchanged, or the generated object is a
coordinate with no seven-role identity at all, which those modules record as a
null rather than as a false.

### Why a complete dataset can still leave room for augmentation

The canonical Buchwald–Hartwig matrix is a near-complete factorial: 3,955 of
3,960 cells are measured, so exactly five globally novel reactions exist.
**Global discovery headroom is therefore almost zero.** But a learner restricted
to a 5% training fraction has observed 159 of those 3,955 reactions, so **low-data
transfer headroom is very large** — about 96% of the matrix is unknown to it.

These two numbers describe different worlds, and using the first to reason about
the second suppresses the treatment entirely. Development evidence at seeds 0–4
records **1 accepted candidate out of 2,540 generated** under the global rule
against **1,704** under the observed-only rule. `RESULT_STATUS.md` records which
result families this affected and which it did not, `docs/CANDIDATE_SCOPE.md` is
the methodology reference, and
`PREREGISTRATION_OBSERVED_ONLY_TRANSFER.md` freezes the confirmatory
experiment that follows from it.

That experiment has been executed (see *Status and Headline Results*). On the
nine confirmatory units the observed-only rule accepted 999 of 1,431 generated
anonymous-transfer candidates and the global rule accepted 2. The outer-test
verdict under the observed-only rule is **null**. The full analysis, with every
unit and both arms accounted for, is committed at
`results/corrected_candidate_scope_reanalysis_20260903T194209Z/summary/observed_only_confirmatory_analysis/`.

## Corrected Revalidation

Corrected condition-transfer experiments use the explicit RDKit-backed
`bh_role_separated` representation for both measured and synthetic rows. They
refuse to overwrite non-empty output directories, select policies using
validation RMSE, and evaluate test data only for matched baselines and selected
policies. The corrected representation baseline reports every canonical
representation rather than selecting one by test performance.

Run the complete corrected workflow manually with (`caffeinate` keeps a Mac
awake; omit it on Linux):

```bash
caffeinate -dimsu bash scripts/run_corrected_revalidation.sh \
  2>&1 | tee results/corrected_revalidation_run.log
```

Historical role-aware v2 and matched-comparison outputs remain invalid. See
`RESULT_STATUS.md` for the evidence registry.

### Low-Data Candidate-Scope Reanalysis

The reanalysis that introduced the candidate-scope taxonomy runs as one
deterministic script. It audits every call site, runs the bounded tests,
regenerates only the calculations the semantics change affects, executes the
preregistered confirmatory experiment on fresh evaluation units, runs the
withheld-cell oracle diagnostic, and verifies hashes and leakage contracts. It
never pushes, never deletes or overwrites a historical output, and never
downloads a dataset.

```bash
caffeinate -i bash scripts/run_lowdata_candidate_scope_reanalysis.sh \
  2>&1 | tee results/corrected_candidate_scope_reanalysis_run.log
```

It requires an environment whose RDKit matches the one that built the canonical
dataset (`rdkit==2023.9.6`); a preflight check fails fast otherwise, because a
different canonicalization would make the stored and generated identity
vocabularies disjoint and silently accept every candidate.

The 2026-09-03 execution of that script completed every primary and comparator
unit and then aborted at the first globally-unmeasured control unit whose
candidate pool accepted zero rows, an outcome the preregistration expects for
that arm. `scripts/resume_observed_only_confirmatory.sh` finishes such a run in
place: it leaves complete units untouched, runs pending units with
`scripts/run_policy_search.py --record-degenerate` so a degenerate unit is
recorded rather than crashed on, and performs the closing hash-verification and
registry-snapshot steps. The per-unit treatment size the preregistration
requires comes from a validation-only regeneration of each unit's candidate
pool under both rules:

```bash
RUN_LABEL=20260903T194209Z bash scripts/resume_observed_only_confirmatory.sh
python -B scripts/run_candidate_scope_reanalysis.py \
  --config configs/corrected_observed_only_transfer_pool_accounting.yaml \
  --output-directory results/corrected_observed_only_transfer_pool_accounting_v1
python -B scripts/run_observed_only_confirmatory_analysis.py \
  --run-root results/corrected_candidate_scope_reanalysis_20260903T194209Z \
  --pool-accounting-directory results/corrected_observed_only_transfer_pool_accounting_v1
```

## Suggested Experiment Order

1. Run `pytest` to verify the fixture-data guardrail.
2. Prepare or load a local Buchwald-Hartwig CSV.
3. Install RDKit.
4. Run the four feature ablations.
5. Run random-split baselines as a software sanity check only.
6. Run held-out group splits for chemistry generalization stress tests.
7. Run low-data fractions.
8. Run the augmentation audit before enabling any future augmentation.
9. Run the recommendation simulation.
10. Generate the final Markdown report.

## Current Limitations

- The real dataset is not bundled.
- Random split performance should not be interpreted as real chemistry
  generalization.
- Morgan fingerprints and randomized SMILES require RDKit.
- XGBoost and CatBoost are optional and not required by tests.
- Optuna is optional and required only for `run_automl`.
- Recommendation simulation is single-round only.
- Reporting is intentionally lightweight and uses CSV summaries plus simple
  matplotlib plots.
- Condition recombination assumes the first two left-side tokens are substrates;
  this dataset-specific MVP convention requires chemical review.

## Troubleshooting

- If `No rows remain after cleaning`, inspect the raw columns and confirm that
  a yield column and reaction string column are present.
- If all component columns are `"UNKNOWN"`, run the reaction inspection command
  above. TDC may expose only a full `Drug`/`reaction_smiles` string and not
  separate ligand/base/additive/component fields.
- If the feature matrix is all zeros, run:

  ```bash
  python -m bh_augmentation.features.diagnostics --input data/processed/bh_clean.csv --column reaction_smiles
  ```

  Confirm that reaction strings are split into molecular components before
  RDKit fingerprinting.
- TDC Buchwald-Hartwig may store reactions as serialized dict-like records such
  as `{'product': '...', 'catalyst': '', 'reactant': '...'}` rather than
  canonical reaction SMILES. The featurizer extracts molecular values from
  those records before RDKit fingerprinting.
- If TDC does not expose separated components, use
  `reaction_role_concat_delta`, which derives roles from the parsed reaction
  record, or use a richer processed dataset with explicit components.

## License

This repository is released under the MIT License; see `LICENSE`.

## Citation

If you use this code or cite its preregistered results, please use the metadata
in `CITATION.cff`. Both confirmatory verdicts are nulls and should be cited as
such; no positive augmentation effect is claimed anywhere in this repository.

## Future Extensions

- Add broader AutoML search spaces and pruning once the MVP behavior is stable.
- Add repeated split evaluation with confidence intervals.
- Add richer chemical descriptors and ablation studies.
- Add more realistic hard splits, such as scaffold-like aryl halide groups.
- Add multi-round active-learning style recommendation simulation.
- Add model persistence and experiment manifests.
- Add richer report tables for comparing augmentation helped/hurt/neutral
  across split types and low-data fractions.
- Add optional Optuna pruning and repeated-study summaries.
