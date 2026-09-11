#!/usr/bin/env python
"""Verify every declared output hash under a run root and index every file.

Factored out of ``scripts/run_lowdata_candidate_scope_reanalysis.sh`` step 8 so
a run interrupted before that step can be completed by
``scripts/resume_observed_only_confirmatory.sh`` with the identical procedure.

For every ``manifest.json`` below the run root, each entry in ``output_hashes``
is re-hashed and compared.  Then a single ``run_file_index.json`` listing the
SHA-256 of every file the run produced is written into the summary directory.
The index is never overwritten: an existing index is a record of a previous
completion and must be kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

INDEX_FILENAME = "run_file_index.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_declared_output_hashes(run_root: Path) -> int:
    """Return the number of verified declared outputs; raise on any mismatch."""
    verified = 0
    for manifest_path in sorted(run_root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        for name, expected in (manifest.get("output_hashes") or {}).items():
            target = manifest_path.parent / name
            if not target.is_file():
                raise SystemExit(f"Declared output missing for {manifest_path}: {name}")
            if sha256(target) != expected:
                raise SystemExit(f"Hash mismatch for {name} under {manifest_path.parent}")
            verified += 1
    return verified


def build_index(run_root: Path, *, exclude: Path | None = None) -> dict:
    files = {}
    for path in sorted(run_root.rglob("*")):
        if not path.is_file():
            continue
        if exclude is not None and path == exclude:
            continue
        files[str(path.relative_to(run_root))] = sha256(path)
    return {"run_root": str(run_root), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--summary-directory",
        type=Path,
        help="Where to write run_file_index.json (default: <run-root>/summary).",
    )
    args = parser.parse_args()
    run_root: Path = args.run_root
    if not run_root.is_dir():
        raise SystemExit(f"Run root does not exist: {run_root}")
    summary_dir = args.summary_directory or run_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    index_path = summary_dir / INDEX_FILENAME
    if index_path.exists():
        raise SystemExit(f"Refusing to overwrite an existing index: {index_path}")

    verified = verify_declared_output_hashes(run_root)
    print(f"verified {verified} declared output hashes")
    index = build_index(run_root, exclude=index_path)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"indexed {len(index['files'])} produced files -> {index_path}")


if __name__ == "__main__":
    sys.exit(main())
