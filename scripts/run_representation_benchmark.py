"""Run the matched canonical no-augmentation representation benchmark."""

from __future__ import annotations

import argparse

from bh_augmentation.representation_benchmark import run_representation_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_representation_benchmark(
        args.config,
        output_directory=args.output_directory,
    )
    print(f"Validated representation benchmark: {paths['manifest']}")


if __name__ == "__main__":
    main()
