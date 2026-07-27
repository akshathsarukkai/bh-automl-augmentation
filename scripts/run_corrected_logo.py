"""Run corrected canonical product and reactant LOGO evaluation."""

from __future__ import annotations

import argparse

from bh_augmentation.logo_evaluation import run_corrected_logo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_corrected_logo(
        args.config,
        output_directory=args.output_directory,
    )
    print(f"Corrected LOGO outputs: {paths['output_directory']}")
    print(f"Product LOGO manifest: {paths['product_key_manifest']}")
    print(f"Reactant LOGO manifest: {paths['reactant_key_manifest']}")


if __name__ == "__main__":
    main()
