"""Build the Phase 18 prospective-experiment package for chemist review."""

from __future__ import annotations

import argparse
import json

from bh_augmentation.prospective_package import (
    build_prospective_package,
    summarize_prospective_package,
    validate_prospective_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Rebuild and validate an existing package instead of writing a new one.",
    )
    args = parser.parse_args()
    if args.validate_only:
        if not args.output_directory:
            parser.error("--validate-only requires --output-directory")
        manifest = validate_prospective_package(args.output_directory)
        print(f"Validated prospective package: {manifest['manifest_hash']}")
        return
    paths = build_prospective_package(args.config, output_directory=args.output_directory)
    summary = summarize_prospective_package(paths["output_directory"])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
