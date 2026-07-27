"""Run one-time outer evaluation of frozen nested OOD policies."""

from __future__ import annotations

import argparse

from bh_augmentation.nested_ood_final import run_nested_ood_final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-policies", required=True)
    parser.add_argument("--search-manifest", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_nested_ood_final(
        args.config,
        frozen_policies_path=args.frozen_policies,
        search_manifest_path=args.search_manifest,
        output_directory=args.output_directory,
    )
    print(f"Outer metrics: {paths['outer_metrics']}")
    print(f"Final manifest: {paths['final_manifest']}")


if __name__ == "__main__":
    main()
