"""Run validation-only scientific model-policy search."""

from __future__ import annotations

import argparse

from bh_augmentation.policy_search import run_policy_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_policy_search_command(
        args.config,
        output_directory=args.output_directory,
    )
    print(f"Frozen policy: {paths['frozen_policy']}")
    print(f"Search manifest: {paths['search_manifest']}")


if __name__ == "__main__":
    main()
