#!/usr/bin/env python
"""Run or validate the leakage-safe Phase 14 benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bh_augmentation.redesigned_ae_benchmark import (
    run_redesigned_ae_benchmark,
    validate_redesigned_ae_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument(
        "--evaluation-registry-directory",
        type=Path,
        help=(
            "Outer-test claim registry root. Defaults to the shared scientific "
            "registry. Pass a throwaway directory for smokes so that repeatable "
            "development runs never consume a real outer-test claim."
        ),
    )
    parser.add_argument("--validate", type=Path)
    args = parser.parse_args()
    if args.validate is not None:
        if (
            args.config is not None
            or args.output_directory is not None
            or args.evaluation_registry_directory is not None
        ):
            parser.error("--validate cannot be combined with run arguments")
        result = validate_redesigned_ae_benchmark(args.validate)
    else:
        if args.config is None:
            parser.error("--config is required unless --validate is used")
        result = {
            key: str(value)
            for key, value in run_redesigned_ae_benchmark(
                args.config,
                output_directory=args.output_directory,
                evaluation_registry_directory=args.evaluation_registry_directory,
            ).items()
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
