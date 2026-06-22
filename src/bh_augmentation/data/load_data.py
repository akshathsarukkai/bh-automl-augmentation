"""Data loading helpers for reaction-yield datasets."""

from pathlib import Path

import pandas as pd

from bh_augmentation.data.clean_data import clean_buchwald_hartwig

TDC_BUCHWALD_HARTWIG_NAME = "Buchwald-Hartwig"


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


def load_tdc_buchwald_hartwig() -> pd.DataFrame:
    """Load the optional TDC Buchwald-Hartwig yield dataset.

    TDC is imported inside this function so the repository does not require it
    for local CSV workflows or tests. The first call may download data through
    TDC depending on the user's local cache.

    Returns
    -------
    pd.DataFrame
        Buchwald-Hartwig reaction-yield records from TDC.

    Raises
    ------
    ImportError
        If the optional TDC package is not installed.
    """
    try:
        from tdc.single_pred import Yields
    except ImportError as exc:
        raise ImportError(
            "TDC is required to load the Buchwald-Hartwig dataset from TDC. "
            "Install it explicitly with `python -m pip install PyTDC` or use "
            "`load_reaction_csv` for a local CSV file."
        ) from exc

    dataset = Yields(name=TDC_BUCHWALD_HARTWIG_NAME)
    data = dataset.get_data()
    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame(data)
    return data


def save_tdc_buchwald_hartwig(output_path: str | Path) -> pd.DataFrame:
    """Load the optional TDC Buchwald-Hartwig dataset and save it as CSV.

    Parameters
    ----------
    output_path:
        Destination CSV path. Parent directories are created if needed.

    Returns
    -------
    pd.DataFrame
        The dataset that was written to disk.
    """
    data = clean_buchwald_hartwig(load_tdc_buchwald_hartwig())
    diagnostics = data.attrs.get("tdc_adapter")
    if diagnostics:
        print(f"TDC preprocessing diagnostics: {diagnostics}")
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(csv_path, index=False)
    return data
