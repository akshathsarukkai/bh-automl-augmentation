"""Prove that a completed result bundle detects a single mutated output byte.

CI runs this against the bounded production-path smoke bundle so a regression
that silently disables hash verification cannot pass. The mutation is applied to
a throwaway copy of the bundle; the original result directory is never modified.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from bh_augmentation.utils.scientific_manifest import (  # noqa: E402
    ManifestError,
    verify_manifest,
)


def assert_tamper_evident(result_directory: str | Path, output_name: str) -> None:
    """Verify the bundle, then prove a one-byte mutation is rejected."""
    source = Path(result_directory)
    verify_manifest(source)
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / "bundle"
        shutil.copytree(source, copy)
        verify_manifest(copy)
        target = copy / output_name
        if not target.is_file():
            raise SystemExit(f"Cannot tamper with a missing output: {target}")
        payload = bytearray(target.read_bytes())
        if not payload:
            raise SystemExit(f"Cannot tamper with an empty output: {target}")
        payload[-1] ^= 0x01
        target.write_bytes(bytes(payload))
        try:
            verify_manifest(copy)
        except ManifestError:
            print(f"Tamper detected as required for {output_name}.")
            return
        raise SystemExit(
            "Manifest verification failed to detect a mutated output byte; "
            "result integrity checking is broken."
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tamper-evidence assertion."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result-directory", required=True)
    parser.add_argument("--output-name", default="summary.csv")
    args = parser.parse_args(argv)
    assert_tamper_evident(args.result_directory, args.output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
