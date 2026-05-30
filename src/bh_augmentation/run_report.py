"""Command-line report generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from bh_augmentation.reporting.make_report import generate_markdown_report


def run_report(results_dir: str | Path, output: str | Path) -> Path:
    """Generate a Markdown report from available result CSVs."""
    return generate_markdown_report(results_dir=results_dir, output_path=output)


def main() -> None:
    """CLI entry point for `python -m bh_augmentation.run_report`."""
    parser = argparse.ArgumentParser(description="Generate experiment report.")
    parser.add_argument("--results-dir", required=True, help="Directory containing result CSVs.")
    parser.add_argument("--output", required=True, help="Output Markdown path.")
    args = parser.parse_args()

    output_path = run_report(args.results_dir, args.output)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
