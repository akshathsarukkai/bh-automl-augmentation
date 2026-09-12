#!/usr/bin/env python
"""Generate the exploratory training-fraction sweep configs from the confirmatory ones.

Each output is the confirmatory seed template with exactly four changes: the
training fraction, the registry family, the output directories, and the header
comment.  The transfer configuration, model grid, dataset, split directory and
candidate scope are copied verbatim so the sweep differs from the confirmatory
experiment only in the fraction (see EXPLORATORY_FRACTION_SWEEP.md).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "configs" / "corrected_observed_only_transfer"
TARGET_DIR = ROOT / "configs" / "exploratory_fraction_sweep"
FRACTIONS = ("0.01", "0.10")
SEEDS = tuple(range(6, 15))
FAMILIES = {
    "observed_only_condition_transfer": "exploratory_observed_only_condition_transfer",
    "matched_real_only_control": "exploratory_matched_real_only_control",
}
HEADER = """# Exploratory training-fraction sweep: NOT a preregistered confirmatory
# experiment. Declared by EXPLORATORY_FRACTION_SWEEP.md before any of its outer
# tests was read. Generated from the confirmatory seed template by
# scripts/generate_exploratory_fraction_sweep_configs.py; only the training
# fraction, the registry family and the output paths differ from
# configs/corrected_observed_only_transfer/{source}.
#
# One evaluation unit per file: the v1 frozen-policy protocol requires exactly
# one seed and one training fraction so that the scientific binding names a
# single outer test.
"""


def render(source_text: str, *, source_name: str, fraction: str, family: str, seed: int) -> str:
    body = source_text.split("\ndataset:\n", 1)[1]
    body = "dataset:\n" + body
    body, n = re.subn(r"^  train_fractions: \[0\.05\]$", f"  train_fractions: [{fraction}]", body, flags=re.M)
    assert n == 1, source_name
    body, n = re.subn(r"^  family: .*$", f"  family: {family}", body, flags=re.M)
    assert n == 1, source_name
    slug = fraction.replace(".", "p")
    body, n = re.subn(
        r"^  search_directory: .*$",
        f"  search_directory: results/corrected_exploratory_fraction_sweep_f{slug}/confirmatory/{family}/seed_{seed}/search",
        body,
        flags=re.M,
    )
    assert n == 1, source_name
    body, n = re.subn(
        r"^  final_directory: .*$",
        f"  final_directory: results/corrected_exploratory_fraction_sweep_f{slug}/confirmatory/{family}/seed_{seed}/final",
        body,
        flags=re.M,
    )
    assert n == 1, source_name
    return HEADER.format(source=source_name) + "\n" + body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the files on disk match.")
    args = parser.parse_args()
    TARGET_DIR.mkdir(exist_ok=True)
    drift = []
    for source_family, target_family in FAMILIES.items():
        for seed in SEEDS:
            source_name = f"{source_family}_seed_{seed}.yaml"
            source_text = (SOURCE_DIR / source_name).read_text()
            for fraction in FRACTIONS:
                slug = fraction.replace(".", "p")
                target = TARGET_DIR / f"{target_family}_f{slug}_seed_{seed}.yaml"
                text = render(
                    source_text,
                    source_name=source_name,
                    fraction=fraction,
                    family=target_family,
                    seed=seed,
                )
                if args.check:
                    if not target.is_file() or target.read_text() != text:
                        drift.append(str(target.relative_to(ROOT)))
                else:
                    target.write_text(text)
    if args.check:
        if drift:
            raise SystemExit("Sweep configs differ from their generator:\n  " + "\n  ".join(drift))
        print(f"OK: {len(FAMILIES) * len(SEEDS) * len(FRACTIONS)} sweep configs match the generator")
    else:
        print(f"wrote {len(FAMILIES) * len(SEEDS) * len(FRACTIONS)} configs to {TARGET_DIR}")


if __name__ == "__main__":
    main()
