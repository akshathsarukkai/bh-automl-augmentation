"""Run the corrected low-complexity representation-learning benchmark."""

from __future__ import annotations

import argparse

from bh_augmentation.low_complexity_benchmark import (
    run_low_complexity_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_low_complexity_benchmark(
        args.config,
        output_directory=args.output_directory,
    )
    print(f"Validated low-complexity benchmark: {paths['manifest']}")


if __name__ == "__main__":
    main()
