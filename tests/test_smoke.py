"""Smoke tests for the initial package scaffold."""

import importlib


def test_package_imports() -> None:
    """The top-level package should import successfully."""
    package = importlib.import_module("bh_augmentation")

    assert package.__version__ == "0.1.0"


def test_placeholder_modules_import() -> None:
    """All scaffolded placeholder modules should import successfully."""
    module_names = [
        "bh_augmentation.data.load_data",
        "bh_augmentation.data.clean_data",
        "bh_augmentation.data.split_data",
        "bh_augmentation.features.featurize",
        "bh_augmentation.features.descriptors",
        "bh_augmentation.augmentation.smiles_randomization",
        "bh_augmentation.augmentation.order_permutation",
        "bh_augmentation.models.baselines",
        "bh_augmentation.models.train",
        "bh_augmentation.models.predict",
        "bh_augmentation.evaluation.metrics",
        "bh_augmentation.evaluation.topk",
        "bh_augmentation.evaluation.regret",
        "bh_augmentation.reporting.make_report",
        "bh_augmentation.utils.config",
        "bh_augmentation.utils.seed",
    ]

    for module_name in module_names:
        importlib.import_module(module_name)
