"""Data loading helpers for reaction-yield CSV files."""

from pathlib import Path

import pandas as pd


def load_reaction_csv(path: str | Path) -> pd.DataFrame:
    """Load a reaction-yield CSV file into a DataFrame.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw reaction-yield records.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the path is not a file or pandas cannot parse the CSV.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Reaction CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Reaction CSV path is not a file: {csv_path}")

    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Reaction CSV is empty: {csv_path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse reaction CSV: {csv_path}") from exc
