"""Run one frozen joint supervised-AE outer-test evaluation."""

from __future__ import annotations

import argparse

from bh_augmentation.ae_final_evaluation import run_ae_final_evaluation_command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frozen-policy", required=True)
    parser.add_argument("--search-manifest", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_ae_final_evaluation_command(
        args.config,
        frozen_policy_path=args.frozen_policy,
        search_manifest_path=args.search_manifest,
        output_directory=args.output_directory,
    )
    print(f"Final test metrics: {paths['final_test_metrics']}")
    print(f"Final manifest: {paths['final_evaluation_manifest']}")


if __name__ == "__main__":
    main()
