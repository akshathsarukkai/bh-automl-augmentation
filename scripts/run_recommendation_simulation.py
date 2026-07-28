"""Run the Phase 18 retrospective recommendation simulation."""

from __future__ import annotations

import argparse

from bh_augmentation.recommendation_simulation import (
    run_recommendation_simulation,
    validate_recommendation_simulation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Replay and validate an existing output directory instead of running.",
    )
    args = parser.parse_args()
    if args.validate_only:
        if not args.output_directory:
            parser.error("--validate-only requires --output-directory")
        manifest = validate_recommendation_simulation(args.output_directory)
        print(f"Validated recommendation simulation: {manifest['manifest_hash']}")
        return
    paths = run_recommendation_simulation(
        args.config, output_directory=args.output_directory
    )
    print(f"Validated recommendation simulation: {paths['manifest']}")


if __name__ == "__main__":
    main()
