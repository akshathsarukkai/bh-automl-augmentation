"""Lightweight reporting helpers for experiment outputs."""

from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


def save_metrics_csv(
    metrics: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    """Save metric records to a CSV file and return the output path."""
    csv_path = Path(output_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(csv_path, index=False)
    return csv_path
