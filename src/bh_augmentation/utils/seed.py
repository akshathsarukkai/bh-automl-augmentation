"""Reproducibility helpers."""

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed common Python randomness sources.

    This sets `PYTHONHASHSEED` for child processes and seeds Python's `random`
    module plus NumPy's global random generator.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
