"""Skip guard for tests that need the gitignored full-scale canonical inputs.

`data/processed/bh_canonical_roles_v1.csv` (the 12 MB canonical dataset) and
`results/corrected_canonical_splits/` (the group-safe split assignments) are
generated artifacts excluded by .gitignore; a clean checkout does not have them.
Tests that exercise the production runners on those inputs must skip with a
reason that names what is missing, rather than error at fixture setup, so a CI
log under ``pytest -ra`` shows exactly what did not run.  `docs/REPRODUCIBILITY.md`
section 4 gives the regeneration commands and the expected hashes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATASET = ROOT / "data/processed/bh_canonical_roles_v1.csv"
CANONICAL_SPLIT_DIRECTORY = ROOT / "results/corrected_canonical_splits"
CANONICAL_SPLIT_MANIFEST = CANONICAL_SPLIT_DIRECTORY / "split_manifest.json"


def require_canonical_artifacts(*paths: Path) -> None:
    """Skip the calling test when any of ``paths`` is absent from the checkout.

    With no arguments, the canonical dataset and the canonical split manifest
    are both required.
    """
    required = paths or (CANONICAL_DATASET, CANONICAL_SPLIT_MANIFEST)
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        pytest.skip(
            "Gitignored canonical artifacts are absent from this checkout: "
            + ", ".join(missing)
            + ". Regenerate them with the commands in docs/REPRODUCIBILITY.md section 4."
        )
