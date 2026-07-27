"""Run the matched canonical Phase 11 augmentation-control benchmark."""

from __future__ import annotations

import argparse

from bh_augmentation.augmentation_control_benchmark import (
    run_augmentation_control_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_augmentation_control_benchmark(
        args.config,
        output_directory=args.output_directory,
    )
    print(f"Validated augmentation-control benchmark: {paths['manifest']}")


if __name__ == "__main__":
    main()
