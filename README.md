# AutoML-Guided Data Augmentation for Low-Data Buchwald-Hartwig Yield Prediction

This repository contains an MVP research-code package for studying safe data
augmentation strategies for low-data Buchwald-Hartwig reaction yield prediction.

The first MVP intentionally avoids generative models, deep learning, web apps,
dashboards, and pseudo-labeling. It focuses on simple, auditable augmentation
methods that preserve the original chemistry labels, such as SMILES
randomization and reaction component order permutation.

## MVP Phases

1. **Repository scaffold**
   - Clean Python package layout under `src/bh_augmentation`.
   - Minimal configuration files for baseline, augmentation, and AutoML phases.
   - Placeholder modules for data loading, cleaning, splitting, featurization,
     augmentation, model training, evaluation, and reporting.
   - Pytest smoke test to verify package imports.

2. **Data preparation**
   - Load raw Buchwald-Hartwig yield data when available.
   - Clean and standardize columns.
   - Create deterministic train/validation/test splits.
   - Include synthetic fixture data for tests so the real dataset is not
     required.

3. **Baseline modeling**
   - Implement simple scikit-learn baselines.
   - Track deterministic metrics on held-out splits.
   - Keep optional libraries such as XGBoost or CatBoost behind graceful
     fallbacks.

4. **Safe augmentation**
   - Add label-preserving augmentation methods:
     - SMILES randomization when RDKit is available.
     - Reaction component order permutation where chemically appropriate.
   - Compare augmented and non-augmented baselines.

5. **Evaluation and reporting**
   - Report regression metrics, top-k selection quality, and regret-style
     analysis.
   - Generate lightweight matplotlib plots and tabular summaries.

## Installation

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

## Data Loading

The repository supports two data-loading paths.

### Local CSV

Use a local CSV when you already have data downloaded or exported:

```python
from bh_augmentation.data.load_data import load_reaction_csv

df = load_reaction_csv("data/raw/buchwald_hartwig.csv")
```

### Optional TDC Loader

The TDC loader is optional and is not installed as a hard dependency. Install it
only when you want to fetch the Buchwald-Hartwig yield dataset through TDC:

```bash
python -m pip install PyTDC
```

```python
from bh_augmentation.data.load_data import (
    load_tdc_buchwald_hartwig,
    save_tdc_buchwald_hartwig,
)

df = load_tdc_buchwald_hartwig()
save_tdc_buchwald_hartwig("data/raw/tdc_buchwald_hartwig.csv")
```

If TDC is not installed, local CSV loading and the rest of the package still
work.

## Running Tests

```bash
pytest
```

## Running A Baseline

After creating a cleaned CSV at the configured dataset path, run:

```bash
python -m bh_augmentation.run_baseline --config configs/baseline.yaml
```

By default, metrics are written to
`results/baseline/baseline_metrics.csv`.

## Comparing Safe Augmentation

To compare the no-augmentation baseline against configured safe augmentation
strategies, run:

```bash
python -m bh_augmentation.run_augmentation --config configs/augmentation.yaml
```

The runner creates train/validation/test splits from real rows first, applies
augmentation only to the training split, and evaluates on untouched validation
and test rows. By default, metrics are written to
`results/augmentation/safe_aug_metrics.csv`.

## Notes

- The real dataset is not assumed to be downloaded.
- The code is intended to run on a laptop.
- Heavy ML implementations are out of scope for the initial scaffold.
