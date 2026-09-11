"""Run validation-only scientific model-policy search."""

from __future__ import annotations

import argparse

from bh_augmentation.policy_search import run_policy_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    parser.add_argument(
        "--record-degenerate",
        action="store_true",
        help=(
            "Account for every typed policy's candidate pool before searching. "
            "If all typed policies accept zero synthetic rows, write "
            "degenerate_unit.json instead of a frozen policy and exit 0 without "
            "claiming any outer-test identity (preregistration section 6)."
        ),
    )
    args = parser.parse_args()
    paths = run_policy_search_command(
        args.config,
        output_directory=args.output_directory,
        record_degenerate=args.record_degenerate,
    )
    if "degenerate_unit" in paths:
        print(f"Degenerate unit: {paths['degenerate_unit']}")
        print("No frozen policy was written; skip final evaluation for this unit.")
        return
    print(f"Frozen policy: {paths['frozen_policy']}")
    print(f"Search manifest: {paths['search_manifest']}")
    if "pool_accounting" in paths:
        print(f"Pool accounting: {paths['pool_accounting']}")


if __name__ == "__main__":
    main()
