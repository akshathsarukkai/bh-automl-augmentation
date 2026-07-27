"""Build canonical scaffold, cluster, similarity, and condition OOD splits."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from bh_augmentation.evaluation.chemical_ood_artifacts import (
    ChemicalOODConfig,
    build_chemical_ood_artifacts,
)
from bh_augmentation.utils.corrected_runs import stable_hash


def _load_config(path: str | Path) -> tuple[Path, Path, ChemicalOODConfig]:
    value: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Chemical OOD config must be a mapping.")
    expected = {"dataset_path", "canonical_split_directory", "split_config"}
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(
            f"Chemical OOD config keys mismatch: missing={missing}, unknown={unknown}."
        )
    if not isinstance(value["split_config"], dict):
        raise ValueError("split_config must be a mapping.")
    return (
        Path(value["dataset_path"]),
        Path(value["canonical_split_directory"]),
        ChemicalOODConfig.from_mapping(value["split_config"]),
    )


def _default_output(config: ChemicalOODConfig) -> Path:
    commit = subprocess.run(
        ["git", "rev-parse", "--short=8", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or "unknown"
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    return Path(
        "results/corrected_chemical_ood_splits"
    ) / f"{date}-{commit}-{stable_hash(config.to_dict())[:8]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    dataset, split_directory, config = _load_config(args.config)
    output = (
        Path(args.output_directory)
        if args.output_directory
        else _default_output(config)
    )
    print("=" * 60)
    print("PHASE 09/18 SMOKE: CHEMISTRY-AWARE OOD SPLITS")
    print("=" * 60)
    paths = build_chemical_ood_artifacts(
        dataset_path=dataset,
        canonical_split_directory=split_directory,
        output_directory=output,
        config=config,
    )
    manifest = json.loads(paths["manifest"].read_text())
    print(json.dumps(manifest["family_summary"], indent=2, sort_keys=True))
    print(f"Validated manifest: {paths['manifest']}")


if __name__ == "__main__":
    main()
