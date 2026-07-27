"""Run validation-calibrated uncertainty and hidden-measured transfer probes."""

from __future__ import annotations

import argparse

from bh_augmentation.hidden_measured_calibration import (
    run_hidden_measured_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory")
    args = parser.parse_args()
    paths = run_hidden_measured_calibration(
        args.config,
        output_directory=args.output_directory,
    )
    print(f"Validated hidden-measured calibration: {paths['manifest']}")


if __name__ == "__main__":
    main()
