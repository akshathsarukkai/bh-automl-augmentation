#!/usr/bin/env python
"""Run the validation-only candidate-scope reanalysis and oracle diagnostic.

This entry point produces the ``observed_only_condition_transfer``,
``globally_unmeasured_condition_transfer`` and ``withheld_cell_transfer_oracle``
development families.  It never materializes an outer-test outcome and never
claims an evaluation-registry identity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bh_augmentation.candidate_scope_reanalysis import (  # noqa: E402
    run_candidate_scope_reanalysis,
)
from bh_augmentation.utils.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Load the configuration and run one candidate-scope reanalysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/corrected_candidate_scope_reanalysis.yaml",
        help="Reanalysis configuration to execute.",
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help="Override the configured fresh output directory.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = run_candidate_scope_reanalysis(
        args.config,
        config,
        output_directory=args.output_directory,
    )
    print("Candidate-scope reanalysis complete.")
    for name, path in sorted(paths.items()):
        print(f"  {name:38s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
