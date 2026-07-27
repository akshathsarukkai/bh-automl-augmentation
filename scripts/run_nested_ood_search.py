"""Run validation-only nested group-aware OOD policy search."""

from __future__ import annotations

import argparse

from bh_augmentation.nested_ood_search import run_nested_ood_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_nested_ood_search(
        args.config,
        output_directory=args.output_directory,
    )
    print(f"Frozen policies: {paths['frozen_policies']}")
    print(f"Search manifest: {paths['search_manifest']}")


if __name__ == "__main__":
    main()
