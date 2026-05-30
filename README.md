# AutoML-Guided Data Augmentation for Low-Data Buchwald-Hartwig Yield Prediction

This repository is an MVP research-code package for evaluating whether simple,
label-preserving data augmentation can improve low-data Buchwald-Hartwig
reaction yield prediction.

The v0 scope is deliberately conservative. It focuses on reproducible data
loading, cleaning, splitting, featurization, classical regression baselines,
safe augmentation, decision-oriented evaluation, and lightweight reporting.
Generative synthetic reaction data is out of scope for v0.

## Project Overview

The core question is:

Can safe augmentation of measured reaction records improve yield prediction or
reaction recommendation when only a small number of Buchwald-Hartwig reactions
are available for training?

The MVP compares:

- no augmentation
- randomized SMILES augmentation, when RDKit is available
- explicit reaction component order permutation, only for caller-approved
  exchangeable columns
- low-data training fractions
- random splits and held-out group splits
- simulated single-round reaction recommendation

## What The MVP Tests

- Loading local CSV reaction-yield data.
- Optional loading from TDC if `PyTDC` is installed by the user.
- Cleaning to a normalized Buchwald-Hartwig schema.
- Deterministic train/validation/test splits.
- Held-out group splits for ligand/base/additive/aryl-halide/amine columns.
- Morgan fingerprints with RDKit, plus categorical condition one-hot features.
- Ridge, Random Forest, ExtraTrees, and optional XGBoost/CatBoost regressors.
- Safe train-only augmentation.
- Regression metrics and decision metrics.
- Markdown report generation and simple matplotlib plots.
- A fixture-based end-to-end guardrail test requiring no real dataset.

## What The MVP Deliberately Does Not Test

- It does not test generative chemistry models.
- It does not create generative synthetic reaction data.
- It does not use pseudo-labeling.
- It does not train deep learning models.
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
python -m pip install -e .
python -m pip install -r requirements.txt
```

Run the fixture-data test suite first:

```bash
pytest
```

The tests use small synthetic fixtures and do not require internet access, the
real Buchwald-Hartwig dataset, TDC, XGBoost, CatBoost, or Optuna.

RDKit is optional for most of the repository, but required for Morgan
fingerprints and real randomized SMILES behavior:

```bash
python -m pip install rdkit
```

If RDKit is unavailable, non-RDKit tests and categorical-only workflows still
run.

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

GitHub Actions runs the fixture-based test suite on push and pull requests. The
base CI install does not require RDKit, TDC, XGBoost, CatBoost, or Optuna.

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

## Augmentation Run

To compare no augmentation against configured safe augmentation:

```bash
python -m bh_augmentation.run_augmentation --config configs/augmentation.yaml
```

Default output:

```text
results/augmentation/safe_aug_metrics.csv
results/augmentation/split_metadata.csv
```

Validation and test sets are split from real data before augmentation. Only the
training split may be augmented.

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

The search space is intentionally small: Ridge, Random Forest, ExtraTrees,
categorical condition features by default, and safe augmentation choices. The
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

## Suggested Experiment Order

1. Run `pytest` to verify the fixture-data guardrail.
2. Prepare or load a local Buchwald-Hartwig CSV.
3. Run the baseline with categorical-only features if RDKit is unavailable.
4. Install RDKit and enable molecular fingerprint features.
5. Run random-split baselines as a software sanity check only.
6. Run held-out group splits for chemistry generalization stress tests.
7. Run low-data fractions.
8. Run safe augmentation comparisons.
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
- Safe augmentation depends on caller-provided assumptions about which reaction
  components are exchangeable.

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
