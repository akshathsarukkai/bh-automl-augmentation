"""Tests for reproducibility utilities."""

import os

import numpy as np

from bh_augmentation.utils.seed import set_global_seed


def test_set_global_seed_makes_numpy_random_deterministic() -> None:
    """Repeated seeding should reproduce NumPy random numbers."""
    set_global_seed(123)
    first = np.random.random(5)

    set_global_seed(123)
    second = np.random.random(5)

    np.testing.assert_array_equal(first, second)


def test_set_global_seed_sets_pythonhashseed() -> None:
    """The hash seed environment variable should be set for child processes."""
    set_global_seed(456)

    assert os.environ["PYTHONHASHSEED"] == "456"
